import asyncio
import copy
from hashlib import sha256
import json
import logging
import os
import random
from typing import Any, List, Optional
from uuid import uuid4
from omegaconf import OmegaConf
from transformers import AutoConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.utils.profiler import simple_timer
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from state_aware_rag.agents.prompts import decompose_and_answer, extract, finalize
from state_aware_rag.agents.agents import GeneratorAgent, RetrievalAgent, EvaluatorAgent
from state_aware_rag.agents.utils import format_reasoning_trace, format_memory, format_context, format_extractor_messages, extract_info_from_text, format_reflection_context

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

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
        # prefer explicit arg, fallback to correctly spelled legacy kwarg if present
        cls.max_iterations = max_iterations if max_iterations is not None else kwargs.get("max_iterations", 5)

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
        retrieval_config['top_k'] = 1 # only retrieve one document for training, TODO: make it configurable
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
        final_answer = None
        # Metrics
        metrics_model = AgentLoopMetrics()
        # Accumulate multiple training examples per input
        all_train_data = []

        for i in range(self.max_iterations):
            # Step 1: Subquestion generation
            subq_result = await self._subquestion_generation(
                question, memory_data, reasoning_traces, generator_kwargs
            )
            answerable = subq_result.get("answerable_main_question", False)
            sub_question = subq_result.get("subquestion", question)
            if not sub_question or not sub_question.strip():
                logger.warning(f"Subquestion generation returned empty subquestion at iteration {i}. Ending loop.")
                sub_question = question
                answerable = True

            # Step 2: Retrieval
            retrieval_docs = await self._retrieve(sub_question, retrieval_kwargs)
            if not retrieval_docs:
                logger.warning(f"Retrieval returned no documents for question: {sub_question}")
                final_answer = format_memory(reasoning_traces) if reasoning_traces else None
                break

            # Step 3: Information consolidation
            all_extractions: list[str] = []
            is_valid: list[bool] = []
            outputs = []
            prompt_ids = []
            consolidate_outputs, extractor_prompt_ids_list = await self._extract(
                sub_question, retrieval_docs, sampling_params
            )
            for output, extractor_prompt_ids in zip(consolidate_outputs, extractor_prompt_ids_list, strict=True):
                consolidate_txt = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                consolidate_entry, _is_valid = self.parse_extractor_outputs(consolidate_txt)
                prompt_ids.append(extractor_prompt_ids)
                outputs.append(output)
                is_valid.append(_is_valid)
                if consolidate_entry.information:
                    all_extractions.extend(consolidate_entry.information)

            all_extractions = list(set(all_extractions))  # Deduplicate
            train_data = {
                'sub_question': sub_question,
                'explore_input_ids_list': prompt_ids,
                'explore_output_list': outputs,
                'explore_is_valid': is_valid,
            }

            # Reflection if not answerable
            if answerable or not memory_data:
                all_reflections = memory_data
            else:
                memory_str = format_memory(memory_data) if memory_data else ""
                reflection_output, reflection_prompt_ids = await self._extract(
                    sub_question, [memory_str], sampling_params
                )
                reflection_output = reflection_output[0]
                reflection_prompt_ids = reflection_prompt_ids[0]
                reflection_txt =  self.tokenizer.decode(reflection_output.token_ids, skip_special_tokens=True)
                reflection_entry, _is_valid = self.parse_extractor_outputs(reflection_txt)
                train_data['reflection_input_ids'] = [reflection_prompt_ids]
                train_data['reflection_output'] = [reflection_output]
                train_data['reflection_is_valid'] = [_is_valid]
                all_reflections = reflection_entry.information

            # Step 4: Subanswer generation
            sub_answer = await self._subanswer_generation(
                question=sub_question if not answerable else question,
                explored_data=all_extractions,
                memory_data=all_reflections,
                reasoning_trace=reasoning_traces,
                main_question_answerable=answerable,
                agent_kwargs=generator_kwargs,
            )
            if sub_answer is None or not sub_answer.strip():
                logger.warning(f"Subanswer generation returned empty answer at iteration {i}. Ending loop.")
                all_train_data.append(train_data)
                final_answer = format_memory(reasoning_traces) if reasoning_traces else None
                break

            if answerable:
                final_answer = sub_answer
                all_train_data.append(train_data)
                break
            else:
                train_data["sub_answer"] = sub_answer

            # Step 5: Update memory and reasoning traces
            reasoning_step = f"{sub_question}\n{sub_answer}"
            if reasoning_step not in reasoning_traces:
                reasoning_traces.append(reasoning_step)
            else:
                logger.warning(f"Detected repeated reasoning step at iteration {i}. Ending loop to prevent cycles.")
                all_train_data.append(train_data)
                final_answer = format_memory(reasoning_traces) if reasoning_traces else None
                break

            if all_extractions:
                extraction_text = format_memory(all_extractions)
                current_memory = format_memory(memory_data)
                context = format_reflection_context(current_memory=current_memory, explored_data=extraction_text)
                update_memory_output, update_memory_prompt_ids = await self._extract(
                    question, [context], sampling_params
                )
                update_memory_output = update_memory_output[0]
                update_memory_prompt_ids = update_memory_prompt_ids[0]
                update_memory_txt = self.tokenizer.decode(update_memory_output.token_ids, skip_special_tokens=True)
                update_memory_entry, is_valid = self.parse_extractor_outputs(update_memory_txt)
                train_data['update_memory_input_ids'] = [update_memory_prompt_ids]
                train_data['update_memory_output'] = [update_memory_output]
                train_data['update_memory_is_valid'] = [is_valid]
                _memory_data = update_memory_entry.information
                _memory_data = list(set(_memory_data))  # Deduplicate
                _memory_data.sort()
                memory_data = _memory_data
            all_train_data.append(train_data)

        # Compute rewards for the trajectory
        reward_tasks = []
        is_outcome_aware = False
        # Outcome-aware reward
        if final_answer and correct_answer:
            reward_tasks.append(
                self._evaluate_final_answer(question, final_answer, correct_answer, evaluator_kwargs)
            )
            is_outcome_aware = True
        
        logger.debug(f"Total training steps collected: {len(all_train_data)}")
        for i, data in enumerate(all_train_data):
            assert "sub_question" in data, f"Training data at step {i} missing 'sub_question'."
            if "sub_answer" in data:
                reward_tasks.append(
                    self._judge_answer(data["sub_question"], data["sub_answer"], evaluator_kwargs)
                )
            else:
                assert data['sub_question'] == question, "If no sub_answer, sub_question must equal original question."
                reward_tasks.append(
                    self._judge_answer(data["sub_question"], final_answer if final_answer else "", evaluator_kwargs)
                )
        if reasoning_traces:
            reward_tasks.append(
                self._evaluate_path(reasoning_traces, question, correct_answer, evaluator_kwargs)
            )

        all_prompt_ids = []
        all_response_ids = []
        all_response_masks = []
        all_response_logprobs = []
        instance_id = []
        reward = []
        rewards = await asyncio.gather(*reward_tasks)
        outcome_reward = rewards[0] if is_outcome_aware else None
        rewards = rewards[1:] if is_outcome_aware else rewards
        whole_path_reward = rewards[-1] if reasoning_traces else None
        rewards = rewards[:-1] if reasoning_traces else rewards
        assert len(rewards) == len(all_train_data), (
            f"Number of rewards {len(rewards)} does not match number of training steps {len(all_train_data)}"
        )
        for idx, r in enumerate(rewards):
            all_train_data[idx]['step_reward'] = r

        avg_step_reward = sum(rewards) / len(rewards) 
        for data in all_train_data:
            if "explore_input_ids_list" in data and "explore_output_list" in data:
                qa_pair = f"{example_id}-{data['sub_question']}-explore"
                _id = int(sha256(qa_pair.encode('utf-8')).hexdigest(), 16)
                instance_id.extend([_id] * len(data["explore_input_ids_list"]))
                all_prompt_ids.extend(data["explore_input_ids_list"])
                all_response_ids.extend([output.token_ids for output in data["explore_output_list"]])
                all_response_masks.extend([[1]*len(output.token_ids) for output in data["explore_output_list"]])
                all_response_logprobs.extend([output.log_probs if output.log_probs else None for output in data["explore_output_list"]])
                structure_reward = [1 if is_valid else 0.2 for is_valid in data["explore_is_valid"]]
                step_reward = data["step_reward"]
                if whole_path_reward:
                    step_reward = 0.75 * step_reward + 0.25 * whole_path_reward
                logger.debug(f"Structure rewards for exploration: {structure_reward}")
                logger.debug(f"Step reward for exploration: {step_reward}")
                reward.extend([step_reward * sr for sr in structure_reward])
            if "reflection_input_ids" in data and "reflection_output" in data:
                qa_pair = f"{example_id}-{data['sub_question']}-reflection"
                _id = int(sha256(qa_pair.encode('utf-8')).hexdigest(), 16)
                instance_id.append(_id)
                all_prompt_ids.append(data["reflection_input_ids"][0])
                all_response_ids.append(data["reflection_output"][0].token_ids)
                all_response_masks.append([1]*len(data["reflection_output"][0].token_ids))
                all_response_logprobs.append(data["reflection_output"][0].log_probs if data["reflection_output"][0].log_probs else None)
                structure_reward = [1 if data["reflection_is_valid"][0] else 0.2]
                step_reward = data["step_reward"]
                if whole_path_reward:
                    step_reward = 0.75 * step_reward + 0.25 * whole_path_reward
                logger.debug(f"Structure reward for reflection: {structure_reward[0]}")
                logger.debug(f"Step reward for reflection: {step_reward}")
                reward.append(step_reward * structure_reward[0])
            if "update_memory_input_ids" in data and "update_memory_output" in data:
                qa_pair = f"{example_id}-{question}-memory_update"
                _id = int(sha256(qa_pair.encode('utf-8')).hexdigest(), 16)
                instance_id.append(_id)
                if outcome_reward is None:
                    if whole_path_reward:
                        outcome_reward = whole_path_reward
                    else:
                        outcome_reward = avg_step_reward
                if outcome_reward is None:
                    continue
                else:
                    if whole_path_reward:
                        reward_value = 0.9 * outcome_reward + 0.1 * whole_path_reward
                    else:
                        reward_value = outcome_reward
                logger.debug(f"Using reward {reward_value} for memory update.")
                all_prompt_ids.append(data["update_memory_input_ids"][0])
                all_response_ids.append(data["update_memory_output"][0].token_ids)
                all_response_masks.append([1]*len(data["update_memory_output"][0].token_ids))
                all_response_logprobs.append(data["update_memory_output"][0].log_probs if data["update_memory_output"][0].log_probs else None)
                structure_reward = [1 if data["update_memory_is_valid"][0] else 0.2]
                logger.debug(f"Structure reward for memory update: {structure_reward[0]}")
                reward.append(reward_value * structure_reward[0])

        assert len(all_prompt_ids) == len(all_response_ids) == len(all_response_masks) == len(all_response_logprobs), (
            f"Length mismatch: {len(all_prompt_ids)} prompts, {len(all_response_ids)} responses, {len(all_response_masks)} masks, {len(all_response_logprobs)} logprobs"
        )
        assert len(all_prompt_ids) == len(reward), f"Number of rewards {len(reward)} does not match number of training examples {len(all_prompt_ids)}"
        assert len(all_prompt_ids) == len(instance_id), f"Number of instance IDs {len(instance_id)} does not match number of training examples {len(all_prompt_ids)}"
        all_prompt_ids = [ids[: self.prompt_length] for ids in all_prompt_ids]
        all_response_ids = [ids[: self.response_length] for ids in all_response_ids]
        all_response_masks = [mask[: self.response_length] for mask in all_response_masks]
        all_response_logprobs = [logprobs[: self.response_length] if logprobs else None for logprobs in all_response_logprobs]
        logger.debug(f"Generated {len(all_prompt_ids)} training examples for question: {question}")
        logger.debug(f"Rewards: {reward}")
        outputs: list[AgentLoopOutput] = []
        for i in range(len(all_prompt_ids)):
            reward_score = reward[i] 
            _id = instance_id[i] 
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=all_prompt_ids[i],
                    response_ids=all_response_ids[i],
                    response_mask=all_response_masks[i],
                    response_logprobs=all_response_logprobs[i],
                    multi_modal_data={},
                    num_turns=2,
                    reward_score=reward_score,
                    metrics=metrics_model,
                    extra_fields={'uid': example_id, '_id': _id}
                )
            )
        return outputs
        
    def parse_extractor_outputs(self, text: str):
        reasoning_txt, output_txt = self.reasoning_parser.extract_reasoning_content(text, None)
        if output_txt:
            try:
                extractor_output = extract.ExtractOutput.model_validate_json(output_txt)
                is_valid = True
            except Exception as e:
                is_valid = False
                keys = extract.ExtractOutput.model_fields.keys()
                value_types = [field.annotation.__name__ for field in extract.ExtractOutput.model_fields.values()]
                extractor_output = extract_info_from_text(text, keys, value_types)
                info = extractor_output.get("information", [])
                if isinstance(info, str):
                    info = [info]
                if not isinstance(info, list):
                    info = []
                decision = extractor_output.get("decision", "not_relevant")
                if decision not in ["relevant", "not_relevant"]:
                    decision = "not_relevant"
                extractor_output = extract.ExtractOutput(
                    information=info,
                    decision=decision,
                    reasoning=""
                )
        else:
            logger.warning(f"Extractor output parsing failed for text: {text}, returning empty extraction.")
            return extract.ExtractOutput(information=[], decision="not_relevant", reasoning=""), False
        return extractor_output, is_valid
    
    async def _extract(
            self, 
            question: str,
            should_extract_documents: list[str],
            sampling_params: dict[str, Any]
        ):
        extractor_messages = format_extractor_messages(
                question=[question] * len(should_extract_documents), context=should_extract_documents
            )
        extractor_prompt_ids_list = []
        for messages in extractor_messages:
            input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                )
            extractor_prompt_ids_list.append(input_ids)
        consolidate_tasks = []
        for extractor_prompt_ids in extractor_prompt_ids_list:
            request_id = uuid4().hex
            consolidate_tasks.append(
                self.server_manager.generate(
                    request_id=request_id, prompt_ids=extractor_prompt_ids, sampling_params=sampling_params
                )
            )
        consolidate_outputs = await asyncio.gather(*consolidate_tasks)
        assert len(consolidate_outputs) == len(extractor_prompt_ids_list), "Mismatch between outputs and prompt ID sets"
        return consolidate_outputs, extractor_prompt_ids_list

    async def _retrieve(self, query: str, agent_kwargs: dict[str, Any]):
        instance_id, _ = await self.retriever_agent.create()
        parameters = {'retrieval_query_list': [query], 'run_kwargs': agent_kwargs}
        retrieval_results, _, _ = await self.retriever_agent.execute(
            instance_id=instance_id,
            parameters=parameters,
        )
        await self.retriever_agent.release(instance_id)
        retrieval_docs = retrieval_results.get("retrieval_docs", None)
        if retrieval_docs:
            retrieval_docs = retrieval_docs[0] # List of documents for the single query
        if not isinstance(retrieval_docs, list):
            logger.warning(f"Retrieval returned invalid documents for query: {query}")
            retrieval_docs = []
        logger.debug(f"Retrieved {len(retrieval_docs)} documents for query: {query}")
        return retrieval_docs
        
    async def _subquestion_generation(
            self,
            user_question: str,
            memory_data: list[str],
            reasoning_trace: list[str],
            agent_kwargs: dict[str, Any],
            **kwargs
        ):
        memory_str = format_memory(memory_data) if memory_data else ""
        reasoning_trace = format_reasoning_trace(reasoning_trace)
        context = format_context(memory=memory_str, reasoning_trace=reasoning_trace)
        logger.debug(f"Decomposing question: {user_question}")
        logger.debug(f"Context for decomposition:\n{context}")
        agent_input = {
            'generate_fn': 'generate_subquestion',
            'question_list': user_question,
            'context_list': context,
            'run_kwargs': agent_kwargs,
        }
        instance_id, _ = await self.generator_agent.create()
        response, _, _ = await self.generator_agent.execute(instance_id, agent_input)
        await self.generator_agent.release(instance_id)
        response: Optional[List[decompose_and_answer.SubquestionOutput]] = response.get('output', None)
        if not response:
            logger.warning(f"No subquestion output from generator for question: {user_question}")
            return {"answerable_main_question": True, "subquestion": user_question}
        logger.debug(f"Decomposition output: {response}")
        response: decompose_and_answer.SubquestionOutput = response[0]
        answerable_main_question = response.answerable_main_question
        sub_question = response.subquestion.strip()
        if not sub_question:
            logger.warning(f"No valid subquestion generated for question: {user_question}")
            return {"answerable_main_question": True, "subquestion": user_question}
        return {"answerable_main_question": answerable_main_question, "subquestion": sub_question}

    async def _subanswer_generation(
            self,
            question: str,
            explored_data: list[str],
            memory_data: list[str],
            reasoning_trace: list[str],
            main_question_answerable: bool,
            agent_kwargs: dict[str, Any],
            **kwargs
        ):
        if main_question_answerable:
            memory_str = format_memory(memory_data) if memory_data else ""
            explored_str = format_memory(explored_data) if explored_data else ""
            reasoning_trace = format_reasoning_trace(reasoning_trace)
            context = format_context(memory=memory_str, reasoning_trace=reasoning_trace, explored_data=explored_str)
            logger.debug(f"Generating final answer for question: {question}")
            logger.debug(f"Context for final answer:\n{context}")
            if not question or not question.strip():
                logger.warning(f"Empty user question for final answer generation")
                return None

            agent_input = {
                'generate_fn': 'finalize',
                'question_list': question,
                'context_list': context,
                'run_kwargs': agent_kwargs,
            }
            instance_id, _ = await self.generator_agent.create()
            response, _, _ = await self.generator_agent.execute(instance_id, agent_input)
            await self.generator_agent.release(instance_id)
            response: Optional[List[finalize.FinalizeOutput]] = response.get('output', None)
            logger.debug(f"Final answer generation output: {response}")
            if not response:
                logger.warning(f"No final answer output from generator for question: {question}")
                return None
            response: finalize.FinalizeOutput = response[0]
            answer = response.answer.strip() if response.answer else ""
            detailed_answer = response.detailed_answer.strip() if response.detailed_answer else ""
            reasoning = response.reasoning.strip() if response.reasoning else ""
            if not reasoning or not reasoning.strip():
                reasoning = reasoning_trace
            if not answer and not detailed_answer:
                answer = detailed_answer = reasoning
            elif not answer and detailed_answer:
                answer = detailed_answer
            elif not detailed_answer and answer:
                detailed_answer = answer
            return detailed_answer
        else:
            reflection_str = format_memory(memory_data) if memory_data else ""
            exploration_str = format_memory(explored_data) if explored_data else ""
            important_info = format_context(memory=reflection_str, reasoning_trace=None, explored_data=exploration_str)
            logger.debug(f"Generating subQA for rephrased question: {question}")
            logger.debug(f"Important info for subQA:\n{important_info}")
            instance_id, _ = await self.generator_agent.create()
            agent_input = {
                'generate_fn': 'generate_answer',
                'question_list': question,
                'context_list': important_info,
                'run_kwargs': agent_kwargs,
            }
            response, _, _ = await self.generator_agent.execute(instance_id, agent_input)
            await self.generator_agent.release(instance_id)
            response: Optional[List[decompose_and_answer.AnswerOutput]] = response.get('output', None)
            logger.debug(f"SubQA generation output: {response}")
            if not response:
                logger.warning(f"No subQA output from generator for rephrased question: {question}")
                return None
            response: decompose_and_answer.AnswerOutput = response[0]
            sub_answer = response.answer.strip() if response.answer else ""
            reasoning = response.reasoning.strip() if response.reasoning else ""
            if not sub_answer and reasoning:
                sub_answer = reasoning
            elif not reasoning and sub_answer:
                reasoning = sub_answer
            return sub_answer
       
    async def _evaluate_final_answer(self, question: str, predicted_answer: str, correct_answer: str, agent_kwargs: dict[str, Any]):
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
        await self.evaluator_agent.release(instance_id)
        assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
        evaluation_result = evaluation_result[0]
        assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
        reward = float(evaluation_result)
        logger.debug(f"Final Answer Evaluation Reward: {reward}")
        return reward

    async def _judge_answer(self, question: str, answer: str, agent_kwargs: dict[str, Any]):
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
        await self.evaluator_agent.release(instance_id)
        assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
        evaluation_result = evaluation_result[0]
        assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
        reward = float(evaluation_result)
        logger.debug(f"QA Pair Evaluation Reward: {reward}")
        return reward

    async def _evaluate_path(self, reasoning_trace: list[str], question: str, correct_answer: str, agent_kwargs: dict[str, Any]):
        reasoning_trace = format_reasoning_trace(reasoning_trace)
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
        await self.evaluator_agent.release(instance_id)
        assert isinstance(evaluation_result, list), f"evaluation_result is not a list: {evaluation_result}"
        evaluation_result = evaluation_result[0]
        assert isinstance(evaluation_result, (int, float)), f"evaluation_result is not a number: {evaluation_result}"
        reward = float(evaluation_result)
        logger.debug(f"Path Evaluation Reward: {reward}")
        return reward
