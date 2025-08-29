import asyncio
import omegaconf
import pytest
import ray

from state_aware_rag.agents.agents import ExtractorAgent
from state_aware_rag.agents.prompts import extract


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
    cfg = omegaconf.OmegaConf.load("configs/extractor.yaml")
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["enable_global_rate_limit"] = False
    cfg_dict["num_workers"] = 4
    cfg_dict["timeout"] = 30
    cfg_dict["verbose"] = True
    return cfg_dict


def test_extract(ray_cluster, base_config):
    agent = ExtractorAgent(config=base_config)
    parameters = {
        "question": "Who is the president of the United States?",
        "document": "As of 2024, the president of the United States is Joe Biden.",
    }
    
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    results = results["extracted_info"]
    assert all(isinstance(o, extract.ExtractOutput) for o in results)