import asyncio
import omegaconf
import pytest
import ray

from state_aware_rag.agents.agents import RetrievalAgent


@pytest.fixture(scope="module")
def ray_cluster():
    # Lightweight Ray setup for actor-based execute() calls
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
    try:
        yield
    finally:
        ray.shutdown()

@pytest.fixture()
def base_config():
    # Load generator config and make it safe for tests (no global rate limiter actor)
    cfg = omegaconf.OmegaConf.load("configs/retriever.yaml")
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["enable_global_rate_limit"] = False
    cfg_dict["num_workers"] = 4
    cfg_dict["timeout"] = 30
    cfg_dict["verbose"] = True
    return cfg_dict
    
def test_retrieval(ray_cluster, base_config):
    agent = RetrievalAgent(config=base_config)
    parameters = {
        "retrieval_query_list": "Who is the president of the United States?",
    }
    
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    results = results["retrieval_docs"]
    assert all(isinstance(o, str) for o in results)

