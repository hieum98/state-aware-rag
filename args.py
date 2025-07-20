from dataclasses import dataclass, field


@dataclass
class LLMAgentArguments:
    model_name: str = field(
        default="openai/qwen3-8B",
        metadata={"help": "Path to the model or model name from Hugging Face."}
    )
    url: str = field(
        default="http://ip-10-4-228-30:30000/v1",
        metadata={"help": "URL for the model API."}
    )
    api_key: str = field(
        default="your_api_key_here",
        metadata={"help": "API key for accessing the model."}
    )
    client_type: str = field(
        default="openai",
        metadata={"help": "Type of client to use (e.g., 'openai', 'litellm')."}
    )
    concurrency: int = field(
        default=64,
        metadata={"help": "Number of concurrent requests."}
    )

@dataclass
class GenerationArguments:
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for generation."}
    )
    n: int = field(
        default=1,
        metadata={"help": "Number of responses to generate."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p sampling value."}
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens to generate."}
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling value."}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use caching for generation."}
    )
    cache_dir: str = field(
        default="agent_cache",
        metadata={"help": "Directory to store cache files."}
    )
    reasoning_effort: str = field(
        default="medium",
        metadata={
            "help": "Level of reasoning effort: 'high', 'medium', or 'low'."
        }
    )


@dataclass
class RetrieverArguments:
    url: str = field(
        default="http://ip-10-4-228-30:5000/search",
        metadata={"help": "URL for the retriever API."}
    )
    retrieval_topk: int = field(
        default=64,
        metadata={"help": "Number of top documents to retrieve."}
    )
    query_instruction: str = field(
        default="query: ",
        metadata={"help": "Instruction for the retriever to follow."}
    )


@dataclass
class MCTSArguments:
    max_depth: int = field(
        default=15,
        metadata={"help": "Maximum depth for MCTS search."}
    )
    num_rollouts: int = field(
        default=100,
        metadata={"help": "Number of rollouts for MCTS."}
    )
    use_golden_answer: bool = field(
        default=False,
        metadata={"help": "Whether to use the golden answer in MCTS."}
    )
    save_tree: bool = field(
        default=False,
        metadata={"help": "Whether to save the MCTS tree."}
    )
    save_dir: str = field(
        default="mcts_data",
        metadata={"help": "Directory to save MCTS data."}
    )

