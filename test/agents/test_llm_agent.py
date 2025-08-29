import os
from typing import Any, Dict, List, Tuple
import omegaconf
import pydantic

# Import after defining stubs if needed
from state_aware_rag.agents.llm_agents import LLMAgent


class OutputSchema(pydantic.BaseModel):
    answer: str
    confidence: str


def test_role_execute_with_structured_output_single():
    config_path = "configs/generator.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    # Convet OmegaConf to dict for LLMAgent
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    client_config = config_dict.pop("client_kwargs", {})
    generation_config = config_dict.pop("generation_config", {})
    # overwrite generation_config to only generate single response
    generation_config["n"] = 1
    llm = LLMAgent(client_kwargs=client_config, generate_kwargs=generation_config, **config_dict)

    batch = [[{"role": "user", "content": "What is the capital of France?"}]]
    results = llm.role_execute(batch, output_schema=OutputSchema)
    # Check that results is a list with one item matching the schema
    # Print results for debugging
    print("Results:", results)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == 1, f"Expected list of length 1, got {len(results)}"
    assert isinstance(results[0], dict), f"Expected dict, got {type(results[0])}"
    assert "answer" in results[0], f"Expected 'answer' key in result, got {results[0].keys()}"
    assert "confidence" in results[0], f"Expected 'confidence' key in result, got {results[0].keys()}"

def test_role_execute_with_unstructured_output_and_batch():
    config_path = "configs/generator.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    # Convet OmegaConf to dict for LLMAgent
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    client_config = config_dict.pop("client_kwargs", {})
    generation_config = config_dict.pop("generation_config", {})
    # overwrite generation_config to only generate single response
    generation_config["n"] = 1
    llm = LLMAgent(client_kwargs=client_config, generate_kwargs=generation_config, **config_dict)

    batch = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Who won the World Series in 2020?"}],
    ]
    results = llm.role_execute(batch, output_schema=OutputSchema)

    # Print results for debugging
    print("Results:", results)
    # role_execute returns the first item per batch or {} when empty
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == 2, f"Expected list of length 2, got {len(results)}"
    assert isinstance(results[0], dict), f"Expected dict, got {type(results[0])}"
    assert "answer" in results[0], f"Expected 'answer' key in result, got {results[0].keys()}"
    assert "confidence" in results[0], f"Expected 'confidence' key in result, got {results[0].keys()}"

def test_generate_uses_cache_on_second_call():
    config_path = "configs/generator.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    # Convet OmegaConf to dict for LLMAgent
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    client_config = config_dict.pop("client_kwargs", {})
    generation_config = config_dict.pop("generation_config", {})
    llm = LLMAgent(client_kwargs=client_config, generate_kwargs=generation_config, **config_dict)

    messages = ([{"role": "user", "content": "What is the capital of France?"}], 0)

    # First call should hit the client and write cache
    idx1, out1 = llm.generate(messages, output_schema=OutputSchema, any_other_info={"k": "v"})
    # Print out the output for debugging purposes
    print("First call output:", out1)

    assert idx1 == 0
    assert isinstance(out1, list) and len(out1) == 1

    # Second call should load from cache and not call client again
    idx2, out2 = llm.generate(messages, output_schema=OutputSchema)
    assert idx2 == 0
    assert out2 == out1  # identical payload
