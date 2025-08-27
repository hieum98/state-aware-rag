from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from agents.agents import GeneratorAgent, RetrievalAgent


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

        retrieval_config = OmegaConf.load(retrieval_config_path)
        retrieval_config = OmegaConf.to_container(retrieval_config, resolve=True)
        cls.retriever_agent = RetrievalAgent(retrieval_config)

        generator_config = OmegaConf.load(generator_config_path)
        generator_config = OmegaConf.to_container(generator_config, resolve=True)
        cls.generator_agent = GeneratorAgent(generator_config)

        cls._class_initialized = True
        print("Performing class-level StageAwareLoop initialization")
