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

from agents.agents import GeneratorAgent, RetrievalAgent
from agents.utils import format_reasoning_trace, format_memory, format_context, format_extractor_messages

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
        questions = list(kwargs["raw_prompt"])
        bs = len(questions)
        metrics = {}
        agent_kwargs = kwargs.get("agent_kwargs", {})
        retrieval_kwargs = agent_kwargs.get("retrieval_agent", {})
        generator_kwargs = agent_kwargs.get("generator_agent", {})
        memory_data = [[]] * bs
        reasoning_traces = [[] for _ in range(bs)]
        active_indices = list(range(bs))
        train_data = [{
            "question": questions[i],
            "final_answer": None,
            "steps": [],
        } for i in range(bs)]
        for _ in range(self.max_iterations):
            if len(active_indices) == 0:
                break

            # Subquestion generation
            to_run_questions = [questions[i] for i in active_indices]
            _memory_data = [memory_data[i] for i in active_indices]
            _reasoning_traces = [reasoning_traces[i] for i in active_indices]
            subq_tasks = []
            for q, m, r in zip(to_run_questions, _memory_data, _reasoning_traces):
                subq_tasks.append(self._subquestion_generation(
                    questions=[q],
                    memory_data=[m],
                    reasoning_traces=[r],
                    agent_kwargs=generator_kwargs,
                ))
            with simple_timer('subquestion_generation', metrics):
                subq_results = await asyncio.gather(*subq_tasks)
            sub_questions = {}
            answerable_indices = []
            for idx, res in zip(active_indices, subq_results):
                try:
                    sub_q = res[questions[idx]].get("subquestion", None)
                    is_answerable = res[questions[idx]].get("answerable_main_question", False)
                    if is_answerable:
                        answerable_indices.append(idx)
                        sub_q = questions[idx]
                    if sub_q:
                        sub_questions[idx] = sub_q
                except Exception as e:
                    logger.warning(f"Error in processing {res}: {e}")
            active_indices = list(sub_questions.keys())
            if len(active_indices) == 0:
                break

            # Document retrieval
            to_retrieve_queries = [sub_questions[idx] for idx in active_indices] # Get the sub-questions for retrieval
            retrieval_outputs = await self._retrieve(
                queries=to_retrieve_queries,
                agent_kwargs=retrieval_kwargs,
            )
            if retrieval_outputs is None:
                logger.warning(f"Retrieval failed for queries: {to_retrieve_queries}")
                break
            all_retrieved_docs = []
            to_extract_questions = []
            for idx in active_indices:
                q = sub_questions[idx]
                docs = retrieval_outputs.get(q, [])
                if not docs: # Empty retrieval results, skip this query
                    active_indices.remove(idx)
                    continue
                all_retrieved_docs.extend(docs)
                to_extract_questions.extend([q] * len(docs))
            if len(active_indices) == 0:
                break
            request_id = uuid4().hex
            extractor_batch = format_extractor_messages(question=to_extract_questions, context=all_retrieved_docs)
            extractor_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    extractor_batch, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            with simple_timer("extract_explored_data", metrics):
                extractor_output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=extractor_prompt_ids, sampling_params=sampling_params
                )
            extractor_texts = self.tokenizer.decode_batch(extractor_output.token_ids, skip_special_tokens=True)
            
                

    
    async def _retrieve(self, queries: list[str], agent_kwargs: dict[str, Any]):
        instance_id = None
        try:
            kwargs = agent_kwargs.get("retrieval_agent", {})
            instance_id, _ = await self.retriever_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            paramenters = {'retrieval_query_list': queries}
            retrieval_results, _, _ = await self.retriever_agent.execute(
                instance_id=instance_id,
                parameters=paramenters,
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return None
        finally:
            if instance_id:
                await self.retriever_agent.release(instance_id=instance_id)
        
        retrieval_docs = retrieval_results.get("retrieval_docs", None)
        queries = retrieval_results.get("queries", None)
        results = {}
        if isinstance(retrieval_docs, list) and len(retrieval_docs) == len(queries):
            for docs, query in zip(retrieval_docs, queries):
                results[query] = docs
        else:
            logger.warning(f"Retrieval results are not in expected format: {retrieval_results}")
            logger.warning(f"queries: {queries}")
            logger.warning(f"len(retrieval_docs): {len(retrieval_docs) if retrieval_docs else 'N/A'}")
            logger.warning(f"len(queries): {len(queries)}")
            return None
        return results
    
    async def _subquestion_generation(
            self,
            questions: list[str],
            memory_data: list[list[str]],
            reasoning_traces: list[list[str]],
            agent_kwargs: dict[str, Any],
            **kwargs
        ):
        instance_id = None
        memory = [format_memory(x) for x in memory_data]
        reasoning_traces = [format_reasoning_trace(x) for x in reasoning_traces]
        contexts = [format_context(memory=m, reasoning_trace=r) for m, r in zip(memory, reasoning_traces)]
        try:
            kwargs = agent_kwargs.get("generator_agent", {})
            instance_id, _ = await self.generator_agent.create(create_kwargs=kwargs.get("create_kwargs", {}))
            parameters = {
                "generate_fn": "generate_subquestion",
                "question_list": questions,
                "context_list": contexts,
                "run_kwargs": kwargs.get("run_kwargs", {}),
            }
            generation_results, _, _ = await self.generator_agent.execute(
                instance_id=instance_id,
                parameters=parameters,
            ) # List of SubquestionOutput
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return None
        finally:
            if instance_id:
                await self.generator_agent.release(instance_id=instance_id)
        all_outputs = generation_results.get("outputs", None)
        questions = generation_results.get("input", None)
        results = {}
        if isinstance(all_outputs, list) and len(all_outputs) == len(questions):
            for output, question in zip(all_outputs, questions):
                results[question] = {}
                if hasattr(output, "subquestion"):
                    results[question]["subquestion"] = output.subquestion
                if hasattr(output, "answerable_main_question"):
                    results[question]["answerable_main_question"] = output.answerable_main_question
        else:
            logger.warning(f"Generation results are not in expected format: {generation_results}")
            logger.warning(f"questions: {questions}")
            logger.warning(f"len(all_outputs): {len(all_outputs) if all_outputs else 'N/A'}")
            logger.warning(f"len(questions): {len(questions)}")
            return None
        return results
            




