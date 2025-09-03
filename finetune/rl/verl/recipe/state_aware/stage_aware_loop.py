import asyncio
import copy
import json
import logging
import os
import random
from typing import Any, Optional
from uuid import uuid4
from omegaconf import OmegaConf
from transformers import AutoConfig
try:
    # Hydra may change the working directory; use its helpers if available
    from hydra.utils import get_original_cwd, to_absolute_path  # type: ignore
except Exception:  # pragma: no cover - hydra might not be present in some contexts
    get_original_cwd = None
    to_absolute_path = None
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.utils.profiler import simple_timer
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from state_aware_rag.agents.prompts.decompose_and_answer import AnswerOutput, SubquestionOutput
from state_aware_rag.agents.prompts.extract import ExtractOutput
from state_aware_rag.agents.agents import GeneratorAgent, RetrievalAgent, EvaluatorAgent
from state_aware_rag.agents.prompts.finalize import FinalizeOutput
from state_aware_rag.agents.utils import format_reasoning_trace, format_memory, format_context, format_extractor_messages, extract_info_from_text, format_reflection_context

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@register("state_aware")
class StageAwareLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, retrieval_config_path, generator_config_path, evaluator_config_path, max_iterations=5, **kwargs):
        if getattr(cls, "_class_initialized", False):
            return
        print("Performing class-level StageAwareLoop initialization")
        cls.tokenizer = tokenizer
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        # prefer explicit arg, fallback to legacy hydra key if present
        cls.max_iterations = max_iterations if max_iterations is not None else kwargs.get("max_interations", 5)

        model_config = AutoConfig.from_pretrained(config.actor_rollout_ref.model.path, trust_remote_code=True)
        if model_config.model_type == 'qwen3':
            cls.model_type = 'qwen3'
            cls.reasoning_parser = DeepSeekR1ReasoningParser(tokenizer=tokenizer)
        else:
            logger.warning(f"Model {config.actor_rollout_ref.model.path} is not explicitly supported. Defaulting to Qwen3 settings.")
            cls.model_type = 'qwen3'
            cls.reasoning_parser = DeepSeekR1ReasoningParser(tokenizer=tokenizer)

        retrieval_config = OmegaConf.load(retrieval_config_path)
        retrieval_config = OmegaConf.to_container(retrieval_config, resolve=True)
        cls.retriever_agent = RetrievalAgent(retrieval_config)

        generator_config = OmegaConf.load(generator_config_path)
        generator_config.generation_config.n = 1  # only generate one for training, TODO: make it configurable
        generator_config = OmegaConf.to_container(generator_config, resolve=True)
        cls.generator_agent = GeneratorAgent(generator_config)

        evaluator_config = OmegaConf.load(evaluator_config_path)
        evaluator_config = OmegaConf.to_container(evaluator_config, resolve=True)
        cls.evaluator_agent = EvaluatorAgent(evaluator_config)

        cls._class_initialized = True
        print("Class-level StageAwareLoop initialization complete.")
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        # Prefer an explicit question field; otherwise try to infer from raw_prompt
        question = kwargs.get("question")
        example_id = kwargs.get("uid")
        assert example_id is not None, "Each dataset row must have a unique 'uid' field."
        if question is None:
            raw_prompt = kwargs.get("raw_prompt")
            if isinstance(raw_prompt, list):
                for msg in reversed(raw_prompt):
                    if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        question = msg["content"].strip()
                        break
            elif isinstance(raw_prompt, str):
                question = raw_prompt.strip()
        assert question is not None and isinstance(question, str) and question.strip(), (
            "StageAwareLoop requires a 'question' string in dataset row, or a 'raw_prompt' containing user message."
        )

        correct_answer = kwargs.get("correct_answer", None)
        logger.debug(f"Starting StageAwareLoop for question: {question} with correct_answer: {correct_answer}")
        assert correct_answer is not None, "correct_answer must be provided in kwargs for reward computation."

        agent_kwargs = kwargs.get("agent_kwargs", {})
        retrieval_kwargs = agent_kwargs.get("retrieval_agent", {})
        generator_kwargs = agent_kwargs.get("generator_agent", {})
        evaluator_kwargs = agent_kwargs.get("evaluator_agent", {})

        memory_data: list[str] = []
        reasoning_traces: list[str] = []
        final_answer = "Cannot answer the question based on the provided information."

        # Metrics
        metrics_model = AgentLoopMetrics()
        _timings: dict[str, float] = {}

        # Accumulate multiple training examples per input
        all_prompt_ids: list[list[int]] = []
        all_response_ids: list[list[int]] = []
        all_response_masks: list[list[int]] = []
        all_response_logprobs: list[Optional[list[float]]] = []
        structure_reward = []

        step_data = []

        extracted_cache = {}

        for i in range(self.max_iterations):
            # Step 1: Subquestion generation
            with simple_timer('subquestion_generation', _timings):
                subq_result = await self._subquestion_generation(
                    question, memory_data, reasoning_traces, generator_kwargs
                )
            answerable = subq_result.get("answerable_main_question", False)
            sub_question = subq_result.get("subquestion", question)

            # Step 2: Retrieval
            with simple_timer('document_retrieval', _timings):
                retrieval_docs = await self._retrieve(sub_question, retrieval_kwargs)
            if not retrieval_docs:
                logger.warning(f"Retrieval returned no documents for question: {sub_question}")
                break

            # Step 3: Information consolidation
            all_extractions: list[str] = []
            should_extract_documents = []
            for doc in retrieval_docs:
                consolidate_entry = extracted_cache.get((sub_question, doc), None)
                if consolidate_entry is None:
                    should_extract_documents.append(doc)
                else:
                    if consolidate_entry.information:
                        all_extractions.extend(consolidate_entry.information)

            extractor_messages = format_extractor_messages(
                question=[sub_question] * len(should_extract_documents), context=should_extract_documents
            )
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
                consolidate_tasks.append(
                    self.server_manager.generate(
                        request_id=request_id, prompt_ids=extractor_prompt_ids, sampling_params=sampling_params
                    )
                )
            with simple_timer("extract_explored_data", _timings):
                consolidate_outputs = await asyncio.gather(*consolidate_tasks)
            
            all_is_valid = []
            for output in consolidate_outputs:
                consolidate_ids = output.token_ids
                consolidate_txt = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(consolidate_ids, skip_special_tokens=True)
                )
                consolidate_entry, is_valid = self.parse_extractor_outputs(consolidate_txt)
                all_is_valid.append(is_valid)
                extracted_cache[(sub_question, doc)] = consolidate_entry
                if consolidate_entry and consolidate_entry.information:
                    all_extractions.extend(consolidate_entry.information)
            all_extractions = list(set(all_extractions))  # Deduplicate
            # Add prompt and response ids for training for each consolidation
            for extractor_prompt_ids, output, is_valid in zip(extractor_prompt_ids_list, consolidate_outputs, all_is_valid):
                all_prompt_ids.append(extractor_prompt_ids)
                all_response_ids.append(output.token_ids)
                all_response_masks.append([1] * len(output.token_ids))
                all_response_logprobs.append(output.log_probs if output.log_probs else None)
                structure_reward.append(1 if is_valid else 0)

            # Step 4: Subanswer generation
            with simple_timer('subanswer_generation', _timings):
                sub_answer = await self._subanswer_generation(
                    sub_question, all_extractions, memory_data, reasoning_traces, generator_kwargs, answerable=answerable
                )
            if (sub_question, sub_answer) not in step_data:
                step_data.append((sub_question, sub_answer))
            if answerable and sub_answer != "Answer generation failed." and sub_answer.strip():
                final_answer = sub_answer
                reasoning_traces.append(f"{sub_question}\n{sub_answer}")
                break

            # Step 5: Update memory and reasoning traces
            reasoning_step = f"{sub_question}\n{sub_answer}"
            if reasoning_step not in reasoning_traces:
                reasoning_traces.append(reasoning_step)
                same_reasoning = False
            else:
                same_reasoning = True
            if all_extractions:
                extraction_text = format_memory(all_extractions)
                current_memory = format_memory(memory_data)
                context = format_reflection_context(current_memory=current_memory, explored_data=extraction_text)
                update_memory_entry = extracted_cache.get((question, context), None)
                if update_memory_entry is None:
                    update_memory_messages = format_extractor_messages(question=question, context=context)
                    update_memory_input_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            update_memory_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                        ),
                    )
                    request_id = uuid4().hex
                    with simple_timer("update_memory", _timings):
                        output = await self.server_manager.generate(
                            request_id=request_id, prompt_ids=update_memory_input_ids, sampling_params=sampling_params
                        )
                    update_memory_ids = output.token_ids
                    update_memory_txt = await self.loop.run_in_executor(
                        None, 
                        lambda: self.tokenizer.decode(update_memory_ids, skip_special_tokens=True)
                    )
                    update_memory_entry, is_valid = self.parse_extractor_outputs(update_memory_txt)
                    if is_valid:
                        extracted_cache[(question, context)] = update_memory_entry
                    structure_reward.append(1 if is_valid else 0)
                    all_prompt_ids.append(update_memory_input_ids)
                    all_response_ids.append(output.token_ids)
                    all_response_masks.append([1] * len(output.token_ids))
                    all_response_logprobs.append(output.log_probs if output.log_probs else None)
                    
                if update_memory_entry and update_memory_entry.information:
                    _memory_data = update_memory_entry.information
                    _memory_data = list(set(_memory_data))  # Deduplicate
                    if _memory_data == memory_data and same_reasoning:
                        break
                    memory_data = _memory_data
            else:
                if same_reasoning:
                    break # No new information and same reasoning, break the loop

        # Compute rewards for the trajectory
        reward_tasks = []
        is_outcome_aware = False
        # Outcome-aware reward
        if final_answer and correct_answer:
            reward_tasks.append(
                self._evaluate_final_answer(question, final_answer, correct_answer, evaluator_kwargs)
            )
            is_outcome_aware = True
        # Path-aware reward
        if reasoning_traces:
            sub_questions = [q for q, a in step_data]
            sub_answers = [a for q, a in step_data]
            for sub_q, sub_a in zip(sub_questions, sub_answers):
                reward_tasks.append(self._judge_answer(sub_q, sub_a, evaluator_kwargs))
            reward_tasks.append(
                self._evaluate_path(reasoning_traces, question, correct_answer, evaluator_kwargs)
            )
        if reward_tasks:
            with simple_timer('reward_computation', _timings):
                rewards = await asyncio.gather(*reward_tasks)
            outcome_reward = rewards[0] if is_outcome_aware else None
            path_rewards = rewards[1:] if is_outcome_aware else rewards
            path_reward = sum(path_rewards) / len(path_rewards) if path_rewards else None
            if outcome_reward and path_reward:
                reward = 0.7 * outcome_reward + 0.3 * path_reward
            elif outcome_reward:
                reward = outcome_reward
            elif path_reward:
                reward = path_reward
            else:
                reward = 0.001
        else:
            logger.warning("No reward tasks were created. Setting total_reward to 0.1")
            reward = 0.001

        # Pack outputs
        assert len(all_prompt_ids) == len(all_response_ids) == len(all_response_masks) == len(all_response_logprobs), (
            f"Length mismatch: {len(all_prompt_ids)} prompts, {len(all_response_ids)} responses, {len(all_response_masks)} masks, {len(all_response_logprobs)} logprobs"
        )
        assert len(all_prompt_ids) == len(structure_reward), (
            f"Length mismatch: {len(all_prompt_ids)} prompts, {len(structure_reward)} structure rewards"
        )
        all_prompt_ids = [ids[: self.prompt_length] for ids in all_prompt_ids]
        all_response_ids = [ids[: self.response_length] for ids in all_response_ids]
        all_response_masks = [mask[: self.response_length] for mask in all_response_masks]
        all_response_logprobs = [logprobs[: self.response_length] if logprobs else None for logprobs in all_response_logprobs]

        outputs: list[AgentLoopOutput] = []
        for i in range(len(all_prompt_ids)):
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=all_prompt_ids[i],
                    response_ids=all_response_ids[i],
                    response_mask=all_response_masks[i],
                    response_logprobs=all_response_logprobs[i],
                    multi_modal_data={},
                    num_turns=2,
                    reward_score=reward if structure_reward[i]==1 else 0.2*reward,  # downweight invalid structure
                    metrics=metrics_model,
                    extra_fields={'uid': example_id}
                )
            )
        if outputs:
            logger.debug(f"Example AgentLoopOutput: {outputs[0]}")
        else:
            logger.warning("No outputs were generated in the Agent Loop.")
        if len(outputs) % 8 != 0:
            # Add duplicates to make the batch size a multiple of 8
            n_to_add = 8 - (len(outputs) % 8)
            outputs.extend([copy.deepcopy(outputs[-1]) for _ in range(n_to_add)]) # add duplicates of the last element, i.e., the memory update
        return outputs
        
    def parse_extractor_outputs(self, text: str):
        reasoning_txt, output_txt = self.reasoning_parser.extract_reasoning_content(text, None)
        # logger.debug(f"Extractor Reasoning Text: {reasoning_txt}")
        logger.debug(f"Extractor Output Text: {output_txt}")
        if output_txt:
            try:
                extractor_output = ExtractOutput.model_validate_json(output_txt)
                is_valid = True
            except Exception as e:
                # logger.warning(f"Failed to decode extractor output: {e}")
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
                if decision not in ["relevant", "not_relevant"]:
                    decision = "not_relevant"
                extractor_output = ExtractOutput(
                    information=info,
                    decision=decision,
                    reasoning=""
                )
        else:
            is_valid = False
            extractor_output = ExtractOutput(
                information=[],
                decision="not_relevant",
                reasoning=""
            )
        logger.debug(f"Extractor Output Valid: {is_valid}")
        logger.debug(f"Extractor Output: {extractor_output}")
        return extractor_output, is_valid
    
    async def _retrieve(self, query: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs or {}
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
        if retrieval_docs:
            retrieval_docs = retrieval_docs[0] # List of documents for the single query
        
        if not isinstance(retrieval_docs, list):
            retrieval_docs = ["Retrieval failed."]
        logger.debug(f"Retrieved {len(retrieval_docs)} documents for query: {query}")
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
            kwargs = agent_kwargs or {}
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
        logger.debug(f"Generated Subquestion: {subquestion}, Answerable: {is_answerable}, Reasoning: {reasoning}")
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
        answerable = kwargs.get("answerable", False)
        try:
            kwargs = agent_kwargs or {}
            instance_id, _ = await self.generator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                "generate_fn": "generate_answer" if not answerable else "finalize",
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
            assert isinstance(answer_output, (AnswerOutput, FinalizeOutput)), f"answer_output is not of type AnswerOutput or FinalizeOutput: {answer_output}"
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
        logger.debug(f"Generated Answer: {full_answer}, Reasoning: {reasoning}")
        return full_answer
       
    async def _evaluate_final_answer(self, question: str, predicted_answer: str, correct_answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs or {}
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
            assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0.
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        logger.debug(f"Final Answer Evaluation Reward: {reward}")
        return reward

    async def _judge_answer(self, question: str, answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs or {}
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
            assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0.
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        logger.debug(f"Judge Reward: {reward}")
        return reward

    async def _evaluate_path(self, reasoning_trace: list[str], question: str, correct_answer: str, agent_kwargs: dict[str, Any]):
        instance_id = None
        reasoning_trace = format_reasoning_trace(reasoning_trace)
        try:
            kwargs = agent_kwargs or {}
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
            assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
            reward = float(evaluation_result)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            reward = 0.
        finally:
            if instance_id:
                await self.evaluator_agent.release(instance_id=instance_id)
        logger.debug(f"Path Evaluation Reward: {reward}")
        return reward
