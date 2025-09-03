import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar
from pydantic import BaseModel
from uuid import uuid4
import ray
# from verl.utils.rollout_trace import rollout_trace_op

from state_aware_rag.agents.retriever_agents import RetrieverAgent
from state_aware_rag.agents.roles.generator import Generator
from state_aware_rag.agents.roles.extractor import Extractor
from state_aware_rag.agents.roles.evaluator import Evaluator
from state_aware_rag.agents.utils import convert_confidence_to_score
from state_aware_rag.agents.prompts import (
    decompose_and_answer,
    evaluate,
    extract,
    finalize,
    rephase_question,
    self_correct,
    synthesize
    )

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2

@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class AgentExecutionWorker:
    """Deprecated: Previously executed callables under a global rate limit.

    Keeping the class for backward compatibility, but we no longer pass
    bound methods into Ray to avoid serialization issues with locks.
    """

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = (
            TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)
            if enable_global_rate_limit
            else None
        )

    def ping(self):
        return True

    # NOTE: Do not use this in new code; execute bound functions locally instead.
    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return fn(*fn_args, **fn_kwargs)
        return fn(*fn_args, **fn_kwargs)


def init_agent_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """Initialize the (legacy) execution pool actor.

    New code should not rely on this actor for executing callables.
    """
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(AgentExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class BaseAgent:
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "base_agent")
        
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 300)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        # Global rate limiter actor (shared by name across processes)
        self.rate_limit_worker = (
            TokenBucketWorker.options(name=f"rate-limiter-{self.name}", 
                                      get_if_exists=True).remote(self.rate_limit)
            if self.enable_global_rate_limit
            else None
        )

        self._instance_dict: Dict[str, Dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, Dict[str, Any]]:
        """Create a agent instance.

        Args:
            instance_id: The instance id of the agent.

        Returns:
            The instance id of the agent.
            agent_creation_response: The response of the agent when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "results": [],
            "metadata": {},
        }
        return instance_id, {}
        
    def run(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        return {}, {}

    # @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[Dict[str, Any], float, dict]:
        """Execute the agent

        Args:
            instance_id: The instance id of the agent
            parameters: The parameters for the agent

        Returns: agent_response, agent_reward_score, agent_metrics:
            agent_response: The response of the agent.
            agent_reward_score: The step reward score of the agent.
            agent_metrics: The metrics of the agent.
        """
        timeout = self.timeout
        try:
            # Acquire global token if enabled, then execute locally to avoid
            # sending bound methods (which may capture locks) to Ray.
            if self.rate_limit_worker:
                with ExitStack() as stack:
                    stack.callback(self.rate_limit_worker.release.remote)
                    # Block until we get a token
                    ray.get(self.rate_limit_worker.acquire.remote())
                    results, metadata = self.run(instance_id, parameters, timeout=timeout, **kwargs)
            else:
                results, metadata = self.run(instance_id, parameters, timeout=timeout, **kwargs)

            self._instance_dict[instance_id] = {
                "results": results,
                "metadata": metadata,
            }
            metrics = {}
            # Update metrics with metadata entries that are startwith "metric/"
            for k, v in metadata.items():
                if k.startswith("metric/"):
                    metrics[k[7:]] = v
            return results, 0.0, metrics
        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[EnvAgent] Execution failed: {e}")
            return {"error": error_result}, 0.0, {'error': str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the agent.

        Args:
            instance_id: The instance id of the agent.

        Returns:
            The reward of the agent.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the instance.

        Args:
            instance_id: The instance id of the agent
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class RetrievalAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize the retriever agent
        self.online_retrieval_config = config.get("online_retrieval_config", None)
        self.offline_retrieval_config = config.get("offline_retrieval_config", None)
        self.top_k = config.get("top_k", None)
        assert self.online_retrieval_config or self.offline_retrieval_config, "At least one of online_retrieval_config or offline_retrieval_config must be provided."
        self.agent = RetrieverAgent(
            online_kwargs=self.online_retrieval_config,
            offline_kwargs=self.offline_retrieval_config,
            )

    def run(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        retrieval_query_list = parameters.get("retrieval_query_list", [])
        if isinstance(retrieval_query_list, str):
            retrieval_query_list = [retrieval_query_list]
        if not retrieval_query_list or not isinstance(retrieval_query_list, list):
            error_msg = "Error: 'retrieval_query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[RetrievalAgent] {error_msg} Received parameters: {parameters}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        top_k = parameters.get("top_k", self.top_k if self.top_k is not None else 5)
        instruction = parameters.get("instruction", None)
        response = None
        error_msg = None
        try:
            response = self.agent.search(query=retrieval_query_list, top_k=top_k, instruction=instruction)
        except Exception as e:
            error_msg = f"Search execution failed: {e}"
            logger.error(f"[RetrievalAgent] {error_msg}")
        logger.debug(f"Search response: {response} for instance_id: {instance_id}")
        metadata = {
            "metric/query_count": len(retrieval_query_list),
            "queries": retrieval_query_list,
            "metric/api_request_error": error_msg,
            "api_response": None,
            "metric/status": "unknown",
            "metric/total_results": 0,
        }
        if response is None:
            logger.error(f"[RetrievalAgent] No response received from search.")
            resutls = {"retrieval_docs": []}
            metadata["metric/status"] = "error"
            return resutls, metadata
        
        logger.debug(f"[RetrievalAgent] Search response: {response}")
        metadata["api_response"] = response
        try:
            retrieval_docs = response.get("retrieved_docs", [])
            if not retrieval_docs:
                metadata["metric/status"] = "no_results"
                results = {"retrieval_docs": []}
                logger.info("[RetrievalAgent] No results found.")
                return results, metadata
            if len(retrieval_docs) != len(retrieval_query_list):
                logger.warning(f"[RetrievalAgent] Mismatch in number of queries and results: {len(retrieval_query_list)} queries but {len(retrieval_docs)} results.")
                metadata["metric/status"] = "mismatch_results"
                return {"retrieval_docs": []}, metadata
            if any(not isinstance(docs, list) for docs in retrieval_docs):
                logger.warning(f"[RetrievalAgent] One or more results are not lists.")
                metadata["metric/status"] = "invalid_results"
                return {"retrieval_docs": []}, metadata
            if any(len(docs) == 0 for docs in retrieval_docs):
                logger.info(f"[RetrievalAgent] One or more queries returned zero results.")
            all_retrieval_docs = []
            all_retrieval_urls = []
            all_retrieval_ids = []
            for docs in retrieval_docs:
                all_retrieval_docs.append([doc.get("contents", "") for doc in docs])
                all_retrieval_urls.append([doc.get("url", "") for doc in docs])
                all_retrieval_ids.append([doc.get("id", "") for doc in docs])
            results = {
                "queries": retrieval_query_list,
                "retrieval_docs": all_retrieval_docs,
                "retrieval_urls": all_retrieval_urls,
                "retrieval_ids": all_retrieval_ids,
            }
            total_results = sum(len(docs) for docs in retrieval_docs if isinstance(docs, list))
            metadata["metric/status"] = "success"
            metadata["metric/total_results"] = total_results
            logger.info(f"[RetrievalAgent] Successful search, got {total_results} total results.")
            return results, metadata
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            logger.error(f"[RetrievalAgent] {error_msg}")
            metadata["metric/status"] = "processing_error"
            return {"retrieval_docs": []}, metadata


class GeneratorAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client_kwargs = config.get("client_kwargs", None)
        self.generation_config = config.get("generation_config", None)
        self.use_cache = config.get("use_cache", False)
        model_name = self.client_kwargs['model_name']
        cache_dir = config.get("cache_dir", 'cache/generator')
        self.cache_dir = os.path.join(cache_dir, model_name)
        assert self.client_kwargs is not None, "client_kwargs must be provided in config."
        assert self.generation_config is not None, "generation_config must be provided in config."
        self.agent = Generator(
            client_kwargs=self.client_kwargs,
            generate_kwargs=self.generation_config,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )

    def run(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        # check parameters
        generate_fn = parameters.pop("generate_fn", None)
        # Check if generate_fn is method of self.agent
        if not generate_fn or not hasattr(self.agent, generate_fn) or not callable(getattr(self.agent, generate_fn)):
            error_msg = f"Error: 'generate_fn' is missing or invalid in parameters. Received: {parameters}"
            logger.error(f"[GeneratorAgent] {error_msg}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        question_list = parameters.get("question_list", [])
        if isinstance(question_list, str):
            question_list = [question_list]
        if not question_list or not isinstance(question_list, list):
            error_msg = "Error: 'questions' is missing, empty, or not a list in parameters."
            logger.error(f"[GeneratorAgent] {error_msg} Received parameters: {parameters}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        context_list = parameters.get("context_list", None)
        if generate_fn in ['generate_answer', 'generate_subquestion', 'generate_synthesis', 'finalize', 'self_correct']:
            if context_list is not None and isinstance(context_list, str):
                context_list = [context_list]
            if context_list is not None and (not isinstance(context_list, list) or len(context_list) != len(question_list)):
                error_msg = "'context_list' must be a list of the same length as 'question_list' if provided."
                logger.error(f"[GeneratorAgent] {error_msg} Received parameters: {parameters}")
                return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        current_answer_list = None
        if generate_fn in ['self_correct']:
            current_answer_list = parameters.get("current_answer_list", None)
            if current_answer_list is not None and isinstance(current_answer_list, str):
                current_answer_list = [current_answer_list]
            if current_answer_list is not None and (not isinstance(current_answer_list, list) or len(current_answer_list) != len(question_list)):
                error_msg = "'current_answer_list' must be a list of the same length as 'question_list' if provided."
                logger.error(f"[GeneratorAgent] {error_msg} Received parameters: {parameters}")
                return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        inputs_kwargs = parameters.get("run_kwargs", {}).copy()
        # Explicitly add fields expected by role methods
        inputs_kwargs.update({
            "question": question_list,
            "context": context_list,
            "current_answer": current_answer_list,
        })
        # Allow top-level kwargs to override
        inputs_kwargs.update(kwargs)
        
        try:
            generate_method = getattr(self.agent, generate_fn)
            response = generate_method(**inputs_kwargs) # List of BaseModel as output
        except Exception as e:
            error_msg = f"Generation execution failed: {e}"
            logger.error(f"[GeneratorAgent] {error_msg}")
            return {"error": error_msg}, {"metric/status": "execution_error"}
        
        logger.debug(f"[GeneratorAgent] Generation response: {response} for instance_id: {instance_id}")
        metadata = {
            "metric/question_count": len(question_list),
            "questions": question_list,
            "generate_fn": generate_fn,
            "api_response": None,
            "metric/status": "unknown",
        }
        if response is None:
            logger.error(f"[GeneratorAgent] No response received from generation.")
            results = {"output": None}
            metadata["metric/status"] = "error"
            return results, metadata
        is_batch = True if len(question_list) > 1 else False
        if is_batch:
            if len(response) != len(question_list):
                logger.warning(f"[GeneratorAgent] Mismatch in number of questions and results: {len(question_list)} questions but {len(response)} results.")
                metadata["metric/status"] = "mismatch_results"
                return {"output": None}, metadata
        results = {
            "input": question_list,
            "output": response,
            "is_batch": True if len(question_list) > 1 else False,
        }
        metadata["api_response"] = response
        metadata["metric/status"] = "success"
        logger.info(f"[GeneratorAgent] Successful generation for {len(question_list)} questions.")
        return results, metadata
    

class ExtractorAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client_kwargs = config.get("client_kwargs", None)
        self.generation_config = config.get("generation_config", None)
        self.use_cache = config.get("use_cache", False)
        model_name = self.client_kwargs['model_name']
        cache_dir = config.get("cache_dir", 'cache/extractor')
        self.cache_dir = os.path.join(cache_dir, model_name)
        assert self.client_kwargs is not None, "client_kwargs must be provided in config."
        assert self.generation_config is not None, "generation_config must be provided in config."

        self.agent = Extractor(
            client_kwargs=self.client_kwargs,
            generate_kwargs=self.generation_config,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )

    def run(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        # check parameters
        question = parameters.get("question", [])
        if isinstance(question, str):
            question = [question]
        if not question or not isinstance(question, list):
            error_msg = "Error: 'question' is missing, empty, or not a list in parameters."
            logger.error(f"[ExtractorAgent] {error_msg} Received parameters: {parameters}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        document = parameters.get("document", None)
        if document is not None and isinstance(document, str):
            document = [document]
        if document is not None and (not isinstance(document, list) or len(document) != len(question)):
            error_msg = "'document' must be a list of the same length as 'question' if provided."
            logger.error(f"[ExtractorAgent] {error_msg} Received parameters: {parameters}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        
        inputs_kwargs = parameters.get("run_kwargs", {}).copy()
        inputs_kwargs.update({
            "question": question,
            "document": document,
        })
        inputs_kwargs.update(kwargs)
        
        try:
            responses = self.agent.extract(**inputs_kwargs) # List of dict as output
            assert isinstance(responses, list), "Extractor output should be a list."
            assert all(isinstance(x, extract.ExtractOutput) for x in responses), "Each item in extractor output should be of type ExtractOutput."
        except Exception as e:
            error_msg = f"Extraction execution failed: {e}"
            logger.error(f"[ExtractorAgent] {error_msg}")
            return {"error": error_msg}, {"metric/status": "execution_error"}
        
        logger.debug(f"[ExtractorAgent] Extraction response: {responses} for instance_id: {instance_id}")
        metadata = {
            "metric/question_count": len(question),
            "questions": question,
            "api_response": None,
            "metric/status": "unknown",
        }
        
        is_batch = True if len(question) > 1 else False
        if is_batch:
            if len(responses) != len(question):
                logger.warning(f"[ExtractorAgent] Mismatch in number of questions and results: {len(question)} questions but {len(responses)} results.")
                metadata["metric/status"] = "mismatch_results"
                return {"extracted_info": None}, metadata
        results = {
            "input": question,
            "extracted_info": responses,
            "is_batch": is_batch,
        }
        metadata["api_response"] = responses
        metadata["metric/status"] = "success"
        logger.info(f"[ExtractorAgent] Successful extraction for {len(question)} questions.")
        return results, metadata


class EvaluatorAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client_kwargs = config.get("client_kwargs", None)
        self.generation_config = config.get("generation_config", None)
        self.use_cache = config.get("use_cache", False)
        model_name = self.client_kwargs['model_name']
        cache_dir = config.get("cache_dir", 'cache/evaluator')
        self.cache_dir = os.path.join(cache_dir, model_name)
        assert self.client_kwargs is not None, "client_kwargs must be provided in config."
        assert self.generation_config is not None, "generation_config must be provided in config."

        self.agent = Evaluator(
            client_kwargs=self.client_kwargs,
            generate_kwargs=self.generation_config,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            verbose = config.get("verbose", False),
        )

    def run(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        evaluate_fn = parameters.pop("evaluate_fn", None)
        metadata: Dict[str, Any] = {"metric/status": "unknown"}
        
        if evaluate_fn == "evaluate_final_answer":
            question = parameters.get("question", [])
            if isinstance(question, str):
                question = [question]
            correct_answer = parameters.get("correct_answer", [])
            if correct_answer == []:
                correct_answer = parameters.get("golen_answer", [])
            if isinstance(correct_answer, str):
                correct_answer = [correct_answer]
            predicted_answer = parameters.get("predicted_answer", [])
            if predicted_answer == []:
                predicted_answer = parameters.get("answer", [])
            if isinstance(predicted_answer, str):
                predicted_answer = [predicted_answer]
            assert len(question) == len(correct_answer) == len(predicted_answer), "'question', 'correct_answer', and 'predicted_answer' must be lists of the same length."
            inputs_kwargs = parameters.get("run_kwargs", {}).copy()
            inputs_kwargs.update({
                "question": question,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
            })
            inputs_kwargs.update(kwargs)
            try:
                responses = self.agent.evaluate_final_answer(**inputs_kwargs) # List of dict as output
                results = []
                for resp in responses:
                    decision = resp.get("decision", 0.1)
                    if decision == False:
                        decision = 0.1
                    confidence = resp.get("confidence", "low")
                    confidence = convert_confidence_to_score(confidence)
                    score = decision
                    results.append(score)
                assert len(results) == len(question)
                metadata.update({
                    "metric/status": "success",
                    "metric/question_count": len(question),
                })
                return results, metadata
            except Exception as e:
                error_msg = f"Evaluation execution failed: {e}"
                logger.error(f"[EvaluatorAgent] {error_msg}")
                return {"error": error_msg}, {"metric/status": "execution_error"}
        elif evaluate_fn == "judge_answer":
            user_question = parameters.get("user_question", [])
            if user_question == []:
                user_question = parameters.get("question", [])
            if isinstance(user_question, str):
                user_question = [user_question]
            system_answer = parameters.get("system_answer", [])
            if system_answer == []:
                system_answer = parameters.get("answer", [])
            if isinstance(system_answer, str):
                system_answer = [system_answer]
            assert len(user_question) == len(system_answer), "'user_question' and 'system_answer' must be lists of the same length."
            correct_answer = parameters.get("correct_answer", None)
            if correct_answer == None:
                correct_answer = parameters.get("golen_answer", None)
            if isinstance(correct_answer, str):
                correct_answer = [correct_answer]
            if correct_answer:
                assert len(correct_answer) == len(user_question), "'correct_answer' must be a list of the same length as 'user_question' if provided."
            
            inputs_kwargs = parameters.get("run_kwargs", {}).copy()
            inputs_kwargs.update({
                "user_question": user_question,
                "system_answer": system_answer,
                "correct_answer": correct_answer,
            })
            inputs_kwargs.update(kwargs)
            try:
                results = self.agent.judge_answer(**inputs_kwargs) # List of float as output
                assert len(results) == len(user_question)
                metadata.update({
                    "metric/status": "success",
                    "metric/question_count": len(user_question),
                })
                return results, metadata
            except Exception as e:
                error_msg = f"Evaluation execution failed: {e}"
                logger.error(f"[EvaluatorAgent] {error_msg}")
                return {"error": error_msg}, {"metric/status": "execution_error"}
        elif evaluate_fn == "evaluate_path":
            main_question = parameters.get("main_question", [])
            if main_question == []:
                main_question = parameters.get("question", [])
            if isinstance(main_question, str):
                main_question = [main_question]
            reasoning_path = parameters.get("reasoning_path", [])
            if isinstance(reasoning_path, str):
                reasoning_path = [reasoning_path]
            ground_truth_answer = parameters.get("ground_truth_answer", [])
            if ground_truth_answer == []:
                ground_truth_answer = parameters.get("golden_answer", [])
            if isinstance(ground_truth_answer, str):
                ground_truth_answer = [ground_truth_answer]
            assert len(main_question) == len(reasoning_path) == len(ground_truth_answer), "'main_question', 'reasoning_path', and 'ground_truth_answer' must be lists of the same length."
            inputs_kwargs = parameters.get("run_kwargs", {}).copy()
            inputs_kwargs.update({
                "main_question": main_question,
                "reasoning_path": reasoning_path,
                "ground_truth_answer": ground_truth_answer,
            })
            inputs_kwargs.update(kwargs)
            try:
                results = self.agent.evaluate_path(**inputs_kwargs)
                assert len(results) == len(main_question)
                metadata.update({
                    "metric/status": "success",
                    "metric/question_count": len(main_question),
                })
                return results, metadata
            except Exception as e:
                error_msg = f"Evaluation execution failed: {e}"
                logger.error(f"[EvaluatorAgent] {error_msg}")
                return {"error": error_msg}, {"metric/status": "execution_error"}
        elif evaluate_fn == "majority_vote":
            question = parameters.get("question", None)
            assert isinstance(question, str), "'question' must be a string."
            answer_lists = parameters.get("answer_lists", [])
            if isinstance(answer_lists, str):
                logger.warning("[EvaluatorAgent] 'answer_lists' is a string; majority vote is trivial.")
                return answer_lists, {"metric/status": "single_answer"}
            assert isinstance(answer_lists, list) and all(isinstance(x, str) for x in answer_lists), "'answer_lists' must be a list of strings."
            if len(answer_lists) == 1:
                logger.warning("[EvaluatorAgent] Only one answer provided in 'answer_lists'; majority vote is trivial.")
                return answer_lists[0], {"metric/status": "single_answer"}
            assert len(answer_lists) > 1, "'answer_lists' must contain at least two answers for majority voting."
            inputs_kwargs = parameters.get("run_kwargs", {}).copy()
            inputs_kwargs.update({
                "question": question,
                "answers": answer_lists,
            })
            inputs_kwargs.update(kwargs)
            try:
                results = self.agent.majority_vote(**inputs_kwargs)
                metadata.update({
                    "metric/status": "success",
                    "metric/num_answers": len(answer_lists),
                })
                return results, metadata
            except Exception as e:
                error_msg = f"Majority vote execution failed: {e}"
                logger.error(f"[EvaluatorAgent] {error_msg}")
                return {"error": error_msg}, {"metric/status": "execution_error"}
        elif evaluate_fn == "synthesize_final_answer":
            question = parameters.get("question", None)
            assert isinstance(question, str), "'question' must be a string."
            answers = parameters.get("answers", [])
            if isinstance(answers, str):
                logger.warning("[EvaluatorAgent] 'answers' is a string; synthesis is trivial.")
                return {'final_answer': answers, "final_reasoning": answers}, {"metric/status": "single_answer"}
            assert isinstance(answers, list) and all(isinstance(x, str) for x in answers), "'answers' must be a list of strings."
            assert len(answers) > 0, "'answers' must contain at least one answer for synthesis."
            if len(answers) == 1:
                logger.warning("[EvaluatorAgent] Only one answer provided in 'answers'; synthesis is trivial.")
                return {'final_answer': answers[0], "final_reasoning": answers[0]}, {"metric/status": "single_answer"}
            inputs_kwargs = parameters.get("run_kwargs", {}).copy()
            inputs_kwargs.update({
                "question": question,
                "reasoning_paths": answers,
            })
            inputs_kwargs.update(kwargs)
            try:
                results = self.agent.synthesize_final_answer(**inputs_kwargs)
                metadata.update({
                    "metric/status": "success",
                    "metric/num_answers": len(answers),
                })
                return results, metadata
            except Exception as e:
                error_msg = f"Synthesis execution failed: {e}"
                logger.error(f"[EvaluatorAgent] {error_msg}")
                return {"error": error_msg}, {"metric/status": "execution_error"}
        else:
            error_msg = f"Error: 'evaluate_fn' is missing or invalid in parameters. Received: {parameters}"
            logger.error(f"[EvaluatorAgent] {error_msg}")
            return {"error": error_msg}, {"metric/status": "invalid_parameters"}
        

if __name__=='__main__':
    import asyncio
    import ray

    ray.init()
    retrieval_config = {
        "name": "retrieval_agent",
        "num_workers": 64,
        "rate_limit": 32,
        "timeout": 300,
        "enable_global_rate_limit": True,
        "online_retrieval_config": {
            "url": "http://ip-10-4-225-181:5000/search",
            "retrieval_topk": 5,
            "query_instruction": None,
        }
    }
    agent = RetrievalAgent(retrieval_config)
    instance_id, _ = asyncio.run(agent.create())
    print(f"Created agent instance: {instance_id}")
    parameters = {
        "retrieval_query_list": ["What is the capital of France?", "Who is the president of the United States?"],
        "top_k": 3,
    }
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    # breakpoint() # with ray debug, breakpoint() does not work or hang



        
            


