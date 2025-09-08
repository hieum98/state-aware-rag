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
        retrieval_config.top_k = 1 # only retrieve one for training
        retrieval_config = OmegaConf.to_container(retrieval_config, resolve=True)
        retrieval_config['top_k'] = 1 # only retrieve one document for training, TODO: make it configurable
        cls.retriever_agent = RetrievalAgent(retrieval_config)

        generator_config = OmegaConf.load(generator_config_path)
        generator_config.generation_config.n = 1  # only generate one for training
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
        # assert correct_answer is not None, "correct_answer must be provided in kwargs for reward computation."
        # Deterministically sample based on example_id
        _seed = int.from_bytes(sha256(str(example_id).encode("utf-8")).digest(), "big")
        _types = ['exploration', 'reflection', 'memory_update']
        extractor_type = _types[_seed % len(_types)]
        iteration = _seed % self.max_iterations  
        logger.debug(f"Starting StageAwareLoop for question: {question} with correct_answer: {correct_answer}")
        logger.debug(f"Generating training data using extractor type: {extractor_type} at iteration {iteration}")
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

        # Training example
        prompt_ids = []
        output = None
        structure_reward = 0.
        to_eval_subquestion = ""
        to_eval_subanswer = ""
        step_reward = 0.
        outcome_reward = 0.
        reasoning_reward = 0.

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
                iteration = i  # Current iteration

            # Step 2: Retrieval
            retrieval_docs = await self._retrieve(sub_question, retrieval_kwargs)
            skip_extraction = False
            if not retrieval_docs:
                logger.warning(f"No documents retrieved for subquestion: {sub_question}")
                extractor_type = 'memory_update'  # Force to use memory update as training signal
                skip_extraction = True
            
            extraction_info = []
            if not skip_extraction:
                # Step 3: Information consolidation
                consolidate_output, extractor_prompt_ids = await self._extract(
                    sub_question, retrieval_docs, sampling_params
                )
                consolidate_output = consolidate_output[0]
                extractor_prompt_ids = extractor_prompt_ids[0]
                consolidate_txt = self.tokenizer.decode(consolidate_output.token_ids, skip_special_tokens=True)
                consolidate_entry, _is_valid = self.parse_extractor_outputs(consolidate_txt)
                extraction_info = consolidate_entry.information if consolidate_entry.information else []
                extraction_info = list(set(extraction_info))  # Deduplicate
                if iteration == i and extractor_type == 'exploration':
                    prompt_ids = extractor_prompt_ids
                    output = consolidate_output
                    structure_reward = 1.0 if _is_valid else 0.0

            # reflection
            if answerable or not memory_data:
                # If the main question is answerable or no memory, skip reflection
                reflection_info = memory_data
                # Defer the training signal to memory update step
                extractor_type = 'memory_update'
            else:
                memory_str = format_memory(memory_data) if memory_data else ""
                reflection_output, reflection_prompt_ids = await self._extract(
                    sub_question, [memory_str], sampling_params
                )
                reflection_output = reflection_output[0]
                reflection_prompt_ids = reflection_prompt_ids[0]
                reflection_txt =  self.tokenizer.decode(reflection_output.token_ids, skip_special_tokens=True)
                reflection_entry, _is_valid = self.parse_extractor_outputs(reflection_txt)
                reflection_info = reflection_entry.information
                reflection_info = list(set(reflection_info))  # Deduplicate
                if iteration == i and extractor_type == 'reflection':
                    prompt_ids = reflection_prompt_ids
                    output = reflection_output
                    structure_reward = 1.0 if _is_valid else 0.0

            # Step 4: Subanswer generation
            should_break = False
            sub_answer = await self._subanswer_generation(
                question=sub_question if not answerable else question,
                explored_data=extraction_info,
                memory_data=reflection_info,
                reasoning_trace=reasoning_traces,
                main_question_answerable=answerable,
                agent_kwargs=generator_kwargs,
            )
            if sub_answer is None or not sub_answer.strip():
                logger.warning(f"Subanswer generation returned empty answer at iteration {i}. Ending loop.")
                should_break = True
                final_answer = format_memory(reasoning_traces) if reasoning_traces else None
            if iteration == i:
                to_eval_subquestion = sub_question
                to_eval_subanswer = sub_answer if sub_answer else None

            # Step 5: Update memory and reasoning traces
            reasoning_step = f"{sub_question}\n{sub_answer}"
            if reasoning_step not in reasoning_traces:
                reasoning_traces.append(reasoning_step)
            else:
                final_answer = format_memory(reasoning_traces) if reasoning_traces else None
                should_break = True

            extraction_text = format_memory(extraction_info) if extraction_info else ""
            current_memory = format_memory(memory_data)
            context = format_reflection_context(current_memory=current_memory, explored_data=extraction_text)
            update_memory_output, update_memory_prompt_ids = await self._extract(
                question, [context], sampling_params
            )
            update_memory_output = update_memory_output[0]
            update_memory_prompt_ids = update_memory_prompt_ids[0]
            update_memory_txt = self.tokenizer.decode(update_memory_output.token_ids, skip_special_tokens=True)
            update_memory_entry, is_valid = self.parse_extractor_outputs(update_memory_txt)
            if iteration == i and extractor_type == 'memory_update':
                prompt_ids = update_memory_prompt_ids
                output = update_memory_output
                structure_reward = 1.0 if is_valid else 0.0
            if update_memory_entry.information:
                memory_data = list(set(update_memory_entry.information))  # Deduplicate
            else:
                logger.info(f"No new memory extracted at iteration {i}. Keeping existing memory.")
            
            if should_break:
                break
        
        # Reasoning reward
        if reasoning_traces:
            reasoning_reward = await self._evaluate_path(reasoning_traces, question, correct_answer, evaluator_kwargs)
        else:
            reasoning_reward = None
            logger.warning(f"Reasoning traces are empty. Setting reasoning reward to None.")
        # Outcome-aware reward
        if final_answer:
            outcome_reward = await self._evaluate_final_answer(question, final_answer, correct_answer, evaluator_kwargs)
        else:
            outcome_reward = reasoning_reward
            logger.warning(f"Final answer is missing. Setting outcome reward to None.")
        # Path-aware reward
        if to_eval_subquestion and to_eval_subanswer:
            step_reward = await self._judge_answer(to_eval_subquestion, to_eval_subanswer, evaluator_kwargs)
        else:
            step_reward = None
            logger.warning(f"Subquestion or subanswer for step reward is missing. Setting step reward to None.")
        reward = [structure_reward] 
        if step_reward is not None and extractor_type in ['exploration', 'reflection']:
            reward.append(step_reward)
        if outcome_reward is not None:
            reward.append(outcome_reward)
        reward = sum(reward) / len(reward) if reward else 0.0
        logger.info(f"Computed rewards - Structure: {structure_reward}, Step: {step_reward}, Outcome: {outcome_reward}, Reasoning: {reasoning_reward}, Combined: {reward}")

        response_ids = output.token_ids
        response_mask = [1] * len(response_ids)
        response_logprobs = output.log_probs if output.log_probs else None
        return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    multi_modal_data={},
                    num_turns=2,
                    reward_score=reward,
                    metrics=metrics_model
                )
        
    def parse_extractor_outputs(self, text: str):
        reasoning_txt, output_txt = self.reasoning_parser.extract_reasoning_content(text, None)
        if output_txt:
            try:
                extractor_output = extract.ExtractOutput.model_validate_json(output_txt)
                is_valid = True
            except Exception as e:
                is_valid = True
                keys = extract.ExtractOutput.model_fields.keys()
                value_types = [field.annotation.__name__ for field in extract.ExtractOutput.model_fields.values()]
                extractor_output = extract_info_from_text(text, keys, value_types)
                info = extractor_output.get("information", [])
                if isinstance(info, str):
                    info = [info]
                if not isinstance(info, list):
                    is_valid = False
                    info = []
                decision = extractor_output.get("decision", "not_relevant")
                if decision not in ["relevant", "not_relevant"]:
                    is_valid = False
                    decision = "not_relevant"
                if not info and decision == "relevant":
                    is_valid = False
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
