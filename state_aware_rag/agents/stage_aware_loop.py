import asyncio
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

from state_aware_rag.agents.prompts.decompose_and_answer import AnswerOutput, SubquestionOutput
from state_aware_rag.agents.prompts.extract import ExtractOutput
from state_aware_rag.agents.agents import GeneratorAgent, RetrievalAgent, EvaluatorAgent
from state_aware_rag.agents.utils import format_reasoning_trace, format_memory, format_context, format_extractor_messages, extract_info_from_text, format_reflection_context

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class StageAwareLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, retrieval_config_path, generator_config_path, evaluator_config_path, max_interations=5, **kwargs):
        if cls._class_initialized:
            return

        cls.tokenizer = tokenizer
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_iterations = max_interations

        if 'qwen3' in config.actor_rollout_ref.model.path.lower():
            cls.model_type = 'qwen3'
            cls.reasoning_parser = Qwen3ReasoningParser(tokenizer=tokenizer)
        else:
            logger.warning(f"Model {config.actor_rollout_ref.model.path} is not explicitly supported. Defaulting to Qwen3 settings.")
            cls.model_type = 'qwen3'
            cls.reasoning_parser = Qwen3ReasoningParser(tokenizer=tokenizer)

        retrieval_config = OmegaConf.load(retrieval_config_path)
        retrieval_config = OmegaConf.to_container(retrieval_config, resolve=True)
        cls.retriever_agent = RetrievalAgent(retrieval_config)

        generator_config = OmegaConf.load(generator_config_path)
        generator_config.generation_config.n = 1 # only generate one for training, TODO: make it configurable
        generator_config = OmegaConf.to_container(generator_config, resolve=True)
        cls.generator_agent = GeneratorAgent(generator_config)

        evaluator_config = OmegaConf.load(evaluator_config_path)
        evaluator_config = OmegaConf.to_container(evaluator_config, resolve=True)
        cls.evaluator_agent = EvaluatorAgent(evaluator_config)

        cls._class_initialized = True
        print("Performing class-level StageAwareLoop initialization")
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        question = kwargs["raw_prompt"]
        correct_answer = kwargs.get("correct_answer", None)
        assert correct_answer is not None, "correct_answer must be provided in kwargs for reward computation."
        agent_kwargs = kwargs.get("agent_kwargs", {})
        retrieval_kwargs = agent_kwargs.get("retrieval_agent", {})
        generator_kwargs = agent_kwargs.get("generator_agent", {})
        evaluator_kwargs = agent_kwargs.get("evaluator_agent", {})
        memory_data = []
        reasoning_traces = []
        final_answer = "Cannot answer the question based on the provided information."
        # AgentLoopOutput fields
        metrics = {}
        all_prompt_ids = []
        all_response_ids = []
        all_response_masks = []
        all_response_logprobs = []
        for i in range(self.max_iterations):
            # Step 1: Subquestion generation
            with simple_timer('subquestion_generation', metrics):
                subq_result = await self._subquestion_generation(question, memory_data, reasoning_traces, generator_kwargs)
            answerable = subq_result.get("answerable_main_question", False)
            sub_question = subq_result.get("subquestion", question)

            # Step 2: Retrieval
            with simple_timer('document_retrieval', metrics):
                retrieval_docs = await self._retrieve(sub_question, retrieval_kwargs)
            if not retrieval_docs:
                logger.warning(f"Retrieval returned no documents for question: {sub_question}")
                break
            
            # Step 3: Information consolidation
            extractor_messages = format_extractor_messages(question=[sub_question]*len(retrieval_docs), context=retrieval_docs)
            tokenize_tasks = []
            for messages in extractor_messages:
                tokenize_tasks.append(self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                ))
            extractor_prompt_ids_list = await asyncio.gather(*tokenize_tasks) # List of prompt_ids for each document
            consolidate_tasks = []
            for extractor_prompt_ids in extractor_prompt_ids_list:
                request_id = uuid4().hex
                consolidate_tasks.append(self.server_manager.generate(
                    request_id=request_id, prompt_ids=extractor_prompt_ids, sampling_params=sampling_params
                ))
            with simple_timer("extract_explored_data", metrics):
                consolidate_outputs = await asyncio.gather(*consolidate_tasks)
            all_extractions = []
            for output in consolidate_outputs:
                consolidate_ids = output.token_ids
                consolidate_txt = await self.loop.run_in_executor(None, self.tokenizer.decode, consolidate_ids)
                consolidate_entry = self.parse_extractor_outputs(consolidate_txt)
                if consolidate_entry:
                    all_extractions.extend(consolidate_entry.information)
            # Add all prompt and response ids for training
            for extractor_prompt_ids, output in zip(extractor_prompt_ids_list, consolidate_outputs):
                all_prompt_ids.append(extractor_prompt_ids) # List of token ids for the prompt
                all_response_ids.append(output.token_ids) # List of token ids for the response
                response_mask = [1] * len(output.token_ids)
                all_response_masks.append(response_mask)
                all_response_logprobs.append(output.log_probs if output.log_probs else None)
            
            # Step 4: Subanswer generation
            with simple_timer('subanswer_generation', metrics):
                sub_answer = await self._subanswer_generation(sub_question, all_extractions, memory_data, reasoning_traces, generator_kwargs)
            if answerable and sub_answer != "Answer generation failed." and sub_answer.strip():
                # If the main question is answerable, return the answer
                final_answer = sub_answer
                reasoning_traces.append(f"{sub_question}\n{sub_answer}")
                break

            # Step 5: Update memory and reasoning traces
            reasoning_traces.append(f"{sub_question}\n{sub_answer}")
            if all_extractions:
                extraction_text = format_memory(all_extractions)
                curent_memory = format_memory(memory_data)
                context = format_reflection_context(current_memory=curent_memory, explored_data=extraction_text)
                update_memory_messages = format_extractor_messages(question=question, context=context)
                update_memory_input_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        update_memory_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
                request_id = uuid4().hex
                with simple_timer("update_memory", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=update_memory_input_ids, sampling_params=sampling_params
                    )
                update_memory_ids = output.token_ids
                update_memory_txt = await self.loop.run_in_executor(None, self.tokenizer.decode, update_memory_ids)
                update_memory_entry = self.parse_extractor_outputs(update_memory_txt)
                if update_memory_entry and update_memory_entry.information:
                    memory_data = update_memory_entry.information
                # Add prompt and response ids for training
                all_prompt_ids.append(update_memory_input_ids) # List of token ids for the prompt
                all_response_ids.append(output.token_ids) # List of token ids for the response
                response_mask = [1] * len(output.token_ids)
                all_response_masks.append(response_mask)
                all_response_logprobs.append(output.log_probs if output.log_probs else None)
            else:
                logger.info(f"No new extractions to update memory at iteration {i}.")
        
        # Compute the Reward for the trajectory
        reward_tasks = []
        is_outcome_aware = False
        # Outcome-aware reward
        if final_answer and correct_answer:
            reward_tasks.append(self._evaluate_final_answer(question, final_answer, correct_answer, evaluator_kwargs))
            is_outcome_aware = True
        # Path-aware reward
        if reasoning_traces:
            sub_questions = [trace.split('\n')[0] for trace in reasoning_traces]
            sub_answers = [trace.split('\n')[1] if len(trace.split('\n'))>1 else "" for trace in reasoning_traces]
            for sub_q, sub_a in zip(sub_questions, sub_answers):
                reward_tasks.append(self._judge_answer(sub_q, sub_a, evaluator_kwargs))
            reward_tasks.append(self._evaluate_path(reasoning_traces, question, correct_answer, evaluator_kwargs))
        if reward_tasks:
            with simple_timer('reward_computation', metrics):
                rewards = await asyncio.gather(*reward_tasks)
            outcome_reward = rewards[0] if is_outcome_aware else None
            path_rewards = rewards[1:] if is_outcome_aware else rewards
            path_reward = sum(path_rewards) / len(path_rewards) if path_rewards else None
            if outcome_reward and path_reward:
                reward = 0.7*outcome_reward + 0.3*path_reward
            elif outcome_reward:
                reward = outcome_reward
            elif path_reward:
                reward = path_reward
            else:
                reward = 0.1
        else:
            logger.warning("No reward tasks were created. Setting total_reward to 0.1")
            reward = 0.1
        
        # Check validity
        assert len(all_prompt_ids) == len(all_response_ids) == len(all_response_masks) == len(all_response_logprobs), \
            f"Length mismatch: {len(all_prompt_ids)} prompts, {len(all_response_ids)} responses, {len(all_response_masks)} masks, {len(all_response_logprobs)} logprobs"
        # Truncate sequences to max lengths
        all_prompt_ids = [ids[:self.prompt_length] for ids in all_prompt_ids]
        all_response_ids = [ids[:self.response_length] for ids in all_response_ids]
        all_response_masks = [mask[:self.response_length] for mask in all_response_masks]
        all_response_logprobs = [logprobs[:self.response_length] if logprobs else None for logprobs in all_response_logprobs]
        all_outputs = []
        for i in range(len(all_prompt_ids)):
            all_outputs.append(AgentLoopOutput(
                prompt_ids=all_prompt_ids[i],
                response_ids=all_response_ids[i],
                response_mask=all_response_masks[i],
                response_logprobs=all_response_logprobs[i],
                multi_modal_data={},
                num_turns=1,
                reward_score=reward,
                metrics=metrics,
            ))

        return all_outputs
        
    def parse_extractor_outputs(self, text: str):
        reasoning_txt, output_txt = self.reasoning_parser.extract_reasoning_content(text, None)
        if output_txt:
            try:
                extractor_output = json.loads(output_txt)
                info = extractor_output.get("information", [])
                decision = extractor_output.get("decision", "not_relevant")
                extractor_output = ExtractOutput(
                    information=info,
                    decision=decision,
                    reasoning=""
                )
                is_valid = True
            except Exception as e:
                logger.error(f"Failed to decode extractor output: {e}")
                logger.error(f"Extractor output text: {output_txt}")
                is_valid = False
                keys = ExtractOutput.model_fields.keys()
                value_types = [field.annotation.__name__ for field in ExtractOutput.model_fields.values()]
                extractor_output = extract_info_from_text(text, keys, value_types)
                info = extractor_output.get("information", [])
                if isinstance(info, str):
                    info = [info]
                if not isinstance(info, list):
                    info = []
                decision = extractor_output.get("decision", "not_relevant")
                extractor_output = ExtractOutput(
                    information=info,
                    decision=decision,
                    reasoning=""
                )
        else:
            is_valid = False
            extractor_output = None
        return extractor_output
    
    async def _retrieve(self, query: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs.get("retrieval_agent", {})
            instance_id, _ = await self.retriever_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            paramenters = {'retrieval_query_list': [query]}
            retrieval_results, _, _ = await self.retriever_agent.execute(
                instance_id=instance_id,
                parameters=paramenters,
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ["Retrieval failed."]
        finally:
            if instance_id:
                await self.retriever_agent.release(instance_id=instance_id)
        
        retrieval_docs = retrieval_results.get("retrieval_docs", None)
        retrieval_docs = retrieval_results.get("retrieval_docs", None)
        if retrieval_docs:
            retrieval_docs = retrieval_docs[0] # List of documents for the single query
        
        if not isinstance(retrieval_docs, list):
            retrieval_docs = ["Retrieval failed."]
        return retrieval_docs
        
    async def _subquestion_generation(
            self,
            question: str,
            memory_data: list[str],
            reasoning_trace: list[str],
            agent_kwargs: dict[str, Any],
            **kwargs
        ):
        instance_id = None
        memory = format_memory(memory_data)
        reasoning_trace = format_reasoning_trace(reasoning_trace)
        context = format_context(memory=memory, reasoning_trace=reasoning_trace)
        try:
            kwargs = agent_kwargs.get("generator_agent", {})
            instance_id, _ = await self.generator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                "generate_fn": "generate_subquestion",
                "question_list": [question],
                "context_list": [context],
                "run_kwargs": kwargs.get("run_kwargs", {}),
            }
            generation_result, _, _ = await self.generator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            )
            subquestion_output = generation_result.get("output", None)
            if subquestion_output:
                if isinstance(subquestion_output, list):
                    subquestion_output = subquestion_output[0]
            assert isinstance(subquestion_output, SubquestionOutput), f"subquestion_output is not of type SubquestionOutput: {subquestion_output}"
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            # Fail-safe: return the original question as subquestion and mark it as not answerable
            return {"answerable_main_question": False, "subquestion": question}
        finally:
            if instance_id:
                await self.generator_agent.release(instance_id=instance_id)
        subquestion = subquestion_output.subquestion
        reasoning = subquestion_output.reasoning
        is_answerable = subquestion_output.answerable_main_question
        if is_answerable:
            subquestion = question
        if not subquestion:
            if not reasoning:
                subquestion = question
            else:
                subquestion = f"Based on the reasoning: {reasoning}, what is a relevant follow-up question to ask?"
        return {"answerable_main_question": is_answerable, "subquestion": subquestion}

    async def _subanswer_generation(
            self,
            question: str,
            explored_data: list[str],
            memory_data: list[str],
            reasoning_trace: list[str],
            agent_kwargs: dict[str, Any],
            **kwargs
        ):
        instance_id = None
        memory = format_memory(memory_data)
        reasoning_trace = format_reasoning_trace(reasoning_trace)
        explored_data = format_memory(explored_data)
        context = format_context(memory, reasoning_trace, explored_data)
        try:
            kwargs = agent_kwargs.get("generator_agent", {})
            instance_id, _ = await self.generator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                "generate_fn": "generate_answer",
                "question_list": [question],
                "context_list": [context],
                "run_kwargs": kwargs.get("run_kwargs", {}),
            }
            generation_result, _, _ = await self.generator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            ) # AnswerOutput
            answer_output = generation_result.get("output", None)
            if answer_output:
                if isinstance(answer_output, list):
                    answer_output = answer_output[0]
            assert isinstance(answer_output, AnswerOutput), f"answer_output is not of type AnswerOutput: {answer_output}"
            assert answer_output.answer or answer_output.detailed_answer, f"Both answer and detailed_answer are empty in answer_output: {answer_output}"
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return "Answer generation failed."
        finally:
            if instance_id:
                await self.generator_agent.release(instance_id=instance_id)
        full_answer = ""
        answer = answer_output.answer
        if answer:
            full_answer += f"{answer}\n"
        detailed_answer = answer_output.detailed_answer
        if detailed_answer and detailed_answer != answer:
            full_answer += f"{detailed_answer}\n"
        reasoning = answer_output.reasoning
        if reasoning:
            full_answer += f"Reasoning: {reasoning}"
        full_answer = full_answer.strip()
        if not full_answer:
            full_answer = "Answer generation failed."
        return full_answer
       
    async def _evaluate_final_answer(self, question: str, predicted_answer: str, correct_answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs.get("evaluator_agent", {})
            instance_id, _ = await self.evaluator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                'evaluate_fn': 'evaluate_final_answer',
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
            }
            evaluation_result, _, _ = await self.evaluator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            )
            assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
            evaluation_result = evaluation_result[0]
            assert isinstance(evaluation_result, int, float), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0.0
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        return reward

    async def _judge_answer(self, question: str, answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs.get("evaluator_agent", {})
            instance_id, _ = await self.evaluator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                'evaluate_fn': 'judge_answer',
                'user_question': question,
                'system_answer': answer,
            }
            evaluation_result, _, _ = await self.evaluator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            )
            assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
            evaluation_result = evaluation_result[0]
            assert isinstance(evaluation_result, int, float), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result) / 10 # Scale to [0, 1]
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        return reward

    async def _evaluate_path(self, reasoning_trace: list[str], question: str, correct_answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs.get("evaluator_agent", {})
            instance_id, _ = await self.evaluator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                'evaluate_fn': 'evaluate_path',
                'main_question': question,
                'ground_truth_answer': correct_answer,
                'reasoning_path': reasoning_trace,
            }
            evaluation_result, _, _ = await self.evaluator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            )
            assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
            evaluation_result = evaluation_result[0]
            assert isinstance(evaluation_result, int, float), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0.0
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        return reward


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from finetune.rl.verl.tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
    from verl.protocol import DataProto
    import ray
    import numpy as np

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    with initialize_config_dir(config_dir=os.path.abspath("finetune/rl/verl/verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )

    model_path = "Hieuman/Extractor-Qwen3-4B-SFT-v1"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 8192
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 5
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    # Agent loop specific configs
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = "configs/train/state_aware.yaml"

    agent_loop_manager = init_agent_loop_manager(config)

    raw_prompts = [
        "What is the capital of France?", 
        "Who is the president of the United States?", 
        "Which magazine was started first 'Arthur's Magazine' or 'First for Women'?",
        "Who wrote the play 'Romeo and Juliet'?"
        ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array(["state_aware"]*len(raw_prompts)),
            "data_source": np.array(["test"]*len(raw_prompts)),
        }
    )
    n = config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    results = agent_loop_manager.generate_sequences(prompts=batch)

    results.print_size()
    
    ray.shutdown()

    

