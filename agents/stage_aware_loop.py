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

from agents.prompts.decompose_and_answer import AnswerOutput, SubquestionOutput
from agents.prompts.extract import ExtractOutput
from agents.agents import GeneratorAgent, RetrievalAgent
from agents.utils import format_reasoning_trace, format_memory, format_context, format_extractor_messages, extract_info_from_text, format_reflection_context

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class StageAwareLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def init_class(cls, config, tokenizer, retrieval_config_path, generator_config_path, **kwargs):
        if cls._class_initialized:
            return

        cls.tokenizer = tokenizer
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_iterations = config.agent_loop.get("max_iterations", 3)

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

        cls._class_initialized = True
        print("Performing class-level StageAwareLoop initialization")
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        question = kwargs["raw_prompt"]
        metrics = {}
        agent_kwargs = kwargs.get("agent_kwargs", {})
        retrieval_kwargs = agent_kwargs.get("retrieval_agent", {})
        generator_kwargs = agent_kwargs.get("generator_agent", {})
        memory_data = []
        reasoning_traces = []
        final_answer = None
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
            
            # Step 4: Subanswer generation
            with simple_timer('subanswer_generation', metrics):
                sub_answer = await self._subanswer_generation(sub_question, all_extractions, memory_data, reasoning_traces, generator_kwargs)
            if answerable and sub_answer != "Answer generation failed.":
                # If the main question is answerable, return the answer
                final_answer = sub_answer
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
            else:
                logger.info(f"No new extractions to update memory at iteration {i}.")
            
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
       

            




