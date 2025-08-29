import asyncio
import omegaconf
import pytest
import ray

from state_aware_rag.agents.agents import EvaluatorAgent
from state_aware_rag.agents.prompts import evaluate


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
    cfg = omegaconf.OmegaConf.load("configs/evaluator.yaml")
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["enable_global_rate_limit"] = False
    cfg_dict["num_workers"] = 4
    cfg_dict["timeout"] = 30
    cfg_dict["verbose"] = True
    # Reduce flakiness: deterministic single sample, low temp
    try:
        cfg_dict["generation_config"]["temperature"] = 0.0
        cfg_dict["generation_config"]["n"] = 1
        cfg_dict["client_kwargs"]["concurrency"] = min(4, cfg_dict["client_kwargs"].get("concurrency", 4))
    except Exception:
        pass
    return cfg_dict


def test_evaluate_final_answer(ray_cluster, base_config):
    agent = EvaluatorAgent(config=base_config)
    parameters = {
        "evaluate_fn": "evaluate_final_answer",
        "question": "What is the capital of France?",
        "predicted_answer": "The capital of France is Paris.",
        "correct_answer": "Paris",
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    # Should be our pydantic model
    assert all(isinstance(o, float) for o in results)
    assert all(0.0 <= o <= 1.0 for o in results)

def test_judge_answer(ray_cluster, base_config):
    agent = EvaluatorAgent(config=base_config)
    parameters = {
        "evaluate_fn": "judge_answer",
        "user_question": "What is the capital of France?",
        "system_answer": "The capital of France is Paris.",
        "correct_answer": "Paris",
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    # Should be our pydantic model
    assert all(isinstance(o, float) for o in results)
    assert all(0.0 <= o <= 1.0 for o in results)

def test_evaluate_path(ray_cluster, base_config):
    agent = EvaluatorAgent(config=base_config)
    parameters = {
        "evaluate_fn": "evaluate_path",
        "main_question": "What is the capital of France?",
        "ground_truth_answer": "Paris",
        "reasoning_path": "['France is a country in Europe.', 'The capital of France is Paris.']",
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    assert all(isinstance(o, float) for o in results)
    assert all(0.0 <= o <= 1.0 for o in results)
    
