import asyncio
import omegaconf
import pytest
import ray

from state_aware_rag.agents.agents import GeneratorAgent
from state_aware_rag.agents.prompts import (
    decompose_and_answer,
    synthesize,
    finalize,
    self_correct,
    rephase_question,
    )



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
    cfg = omegaconf.OmegaConf.load("configs/generator.yaml")
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["enable_global_rate_limit"] = False
    cfg_dict["num_workers"] = 4
    cfg_dict["timeout"] = 30
    # Reduce flakiness: deterministic single sample, low temp
    try:
        cfg_dict["generation_config"]["temperature"] = 0.0
        cfg_dict["generation_config"]["n"] = 1
        cfg_dict["client_kwargs"]["concurrency"] = min(4, cfg_dict["client_kwargs"].get("concurrency", 4))
    except Exception:
        pass
    return cfg_dict


def test_generate_answer(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "generate_answer",
        "question_list": ["What is the capital of France?"],
        "context_list": ["France's capital is Paris."],
    }

    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, decompose_and_answer.AnswerOutput) for o in out)

def test_generate_subquestion(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "generate_subquestion",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
        "context_list": ["The New York Times was founded in 1851"]
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, decompose_and_answer.SubquestionOutput) for o in out)

def test_generate_synthesis(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "generate_synthesis",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
        "context_list": ["The New York Times was founded in 1851, Wall Street Journal was founded in 1889"]
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, synthesize.SynthesizeOutput) for o in out)

def test_generate_final_answer(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "finalize",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
        "context_list": ["The New York Times was founded in 1851"],
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, finalize.FinalizeOutput) for o in out)

def test_generate_self_correction(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "self_correct",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
        "context_list": ["The New York Times was founded in 1851"],
        "current_answer_list": ["The Wall Street Journal was founded earlier."],
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, self_correct.SelfCorrectOutput) for o in out)

def test_rephase_question(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "rephase_question",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, rephase_question.RephraseQuestionOutput) for o in out)

def test_generate_queries(ray_cluster, base_config):
    agent = GeneratorAgent(config=base_config)
    parameters = {
        "generate_fn": "generate_queries_for_retriever",
        "question_list": ["which magazine was founded earlier, The New York Times or Wall Street Journal?"],
    }
    instance_id, _ = asyncio.run(agent.create())
    results, reward, metrics = asyncio.run(agent.execute(instance_id, parameters))
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    out = results["output"]
    # Should be our pydantic model
    assert all(isinstance(o, decompose_and_answer.QueriesGenerationOutput) for o in out)






