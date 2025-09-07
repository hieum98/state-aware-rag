import os
from hydra import compose, initialize_config_dir
import pytest
from verl.protocol import DataProto
import ray
import numpy as np

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager

@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(ignore_reinit_error=True, 
             include_dashboard=False, 
             runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO",
                    "VLLM_USE_V1": "1",
                    "VERL_LOGGING_LEVEL": "DEBUG",
                }
            })
    try:
        yield
    finally:
        ray.shutdown()

@pytest.fixture()
def base_config():
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )

    model_path = "Hieuman/Extractor-Qwen3-4B-SFT-v1"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 5
    config.actor_rollout_ref.rollout.agent.num_workers = 1
    # Agent loop specific configs
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = "/fsx/ubuntu/users/hieuman/state-aware-rag/finetune/rl/verl/recipe/state_aware/config/state_aware.yaml"
    return config


def test_state_aware_loop(ray_cluster, base_config):
    agent_loop_manager = init_agent_loop_manager(base_config)
    raw_prompts = [
        "Which magazine was started first 'First for Women' or 'Arthur's Magazine'?",
        "Who wrote the play 'Romeo and Juliet'?",
        "In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?"
        ]
    correct_answers = [
        "'Arthur's Magazine'",
        "The play 'Romeo and Juliet' was written by William Shakespeare.",
        "Alexis Sanchez"
    ]
    # Create a batch of prompts
    batch = DataProto(
        non_tensor_batch={
            'uid': np.array([f"test-{i}" for i in range(len(raw_prompts))], dtype=object),
            "correct_answer": np.array(correct_answers, dtype=object),
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array(["state_aware"]*len(raw_prompts)),
            "data_source": np.array(["test"]*len(raw_prompts)),
        }
    )
    # n = base_config.actor_rollout_ref.rollout.n
    # batch = batch.repeat(n)
    print("Batch:", batch)
    results = agent_loop_manager.generate_sequences(prompts=batch)
    print("Results:", results)
    print("Rewards:", results.batch.get("rm_scores"))
    

