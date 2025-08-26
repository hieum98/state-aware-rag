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
from verl.utils.rollout_trace import rollout_trace_op

from agents.retriever_agents import RetrieverAgent

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
    """Worker for executing agent operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)
        
def init_agent_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """Initialize execution pool."""
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
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_agent_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
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

    @rollout_trace_op
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
            results, metadata = await self.execution_pool.execute.remote(
                self.run, instance_id, parameters, timeout=timeout
            )
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
        top_k = parameters.get("top_k", 3)
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
            resutls = {"retrieval_docs": None}
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
                return {"retrieval_docs": None}, metadata
            if any(not isinstance(docs, list) for docs in retrieval_docs):
                logger.warning(f"[RetrievalAgent] One or more results are not lists.")
                metadata["metric/status"] = "invalid_results"
                return {"retrieval_docs": None}, metadata
            if any(len(docs) == 0 for docs in retrieval_docs):
                logger.info(f"[RetrievalAgent] One or more queries returned zero results.")
            results = {"retrieval_docs": retrieval_docs}
            total_results = sum(len(docs) for docs in retrieval_docs if isinstance(docs, list))
            metadata["metric/status"] = "success"
            metadata["metric/total_results"] = total_results
            logger.info(f"[RetrievalAgent] Successful search, got {total_results} total results.")
            return results, metadata
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            logger.error(f"[RetrievalAgent] {error_msg}")
            metadata["metric/status"] = "processing_error"
            return {"retrieval_docs": error_msg}, metadata
                


if __name__=='__main__':
    import asyncio
    import ray

    ray.init()
    retrieval_config = {
        "name": "retrieval_agent",
        "num_workers": 10,
        "rate_limit": 10,
        "timeout": 20,
        "enable_global_rate_limit": True,
        "retriever_online_kwargs": {
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
    breakpoint()



        
            


