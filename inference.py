import dataclasses
import os
import pprint
import datasets
from typing import Any, Dict, List, Optional, Union
from argparse import ArgumentParser
from transformers import HfArgumentParser
from types import SimpleNamespace

from state_aware_rag.agents.roles.evaluator import Evaluator
from state_aware_rag.agents.roles.extractor import Extractor
from state_aware_rag.agents.roles.generator import Generator
from state_aware_rag.agents.retriever_agents import RetrieverAgent
from state_aware_rag.planners.MCTS.utils import clear_agent_cache 
from state_aware_rag.preprocess.utils import simple_preprocess
from args import SearchArguments, GenerationArguments, RetrieverArguments, LLMAgentArguments

# Per-process lazy state for workers
_WORKER_STATE = {
    "initialized": False,
    "generator": None,
    "evaluator": None,
    "extractor": None,
    "retriever": None,
    "search_config": None,
}


def _build_process_fn(
    gen_client_kwargs: Dict[str, Any],
    gen_generate_kwargs: Dict[str, Any],
    eval_client_kwargs: Dict[str, Any],
    eval_generate_kwargs: Dict[str, Any],
    ext_client_kwargs: Dict[str, Any],
    ext_generate_kwargs: Dict[str, Any],
    retriever_kwargs: Dict[str, Any],
    search_config_dict: Dict[str, Any],
    use_cot: bool,
    use_mcts: bool,
    data_name: Optional[str],
):
    """Return a worker-safe map function that lazily initializes agents once per process."""

    def _ensure_workers_initialized():
        if not _WORKER_STATE["initialized"]:
            _WORKER_STATE["generator"] = Generator(
                client_kwargs=gen_client_kwargs,
                generate_kwargs=gen_generate_kwargs,
                use_cache=gen_generate_kwargs.get("use_cache", False),
                cache_dir=gen_generate_kwargs.get("cache_dir"),
            )
            _WORKER_STATE["evaluator"] = Evaluator(
                client_kwargs=eval_client_kwargs,
                generate_kwargs=eval_generate_kwargs,
                use_cache=eval_generate_kwargs.get("use_cache", False),
                cache_dir=eval_generate_kwargs.get("cache_dir"),
            )
            _WORKER_STATE["extractor"] = Extractor(
                client_kwargs=ext_client_kwargs,
                generate_kwargs=ext_generate_kwargs,
                use_cache=ext_generate_kwargs.get("use_cache", False),
                cache_dir=ext_generate_kwargs.get("cache_dir"),
            )
            _WORKER_STATE["retriever"] = RetrieverAgent(online_kwargs=retriever_kwargs)
            # SimpleNamespace to mimic attribute access in SearchArguments
            _WORKER_STATE["search_config"] = SimpleNamespace(**search_config_dict)
            _WORKER_STATE["initialized"] = True

    def process_example(example: Dict[str, Any], idx: int):
        _ensure_workers_initialized()
        res = generate_answer(
            question=example["question"],
            generator=_WORKER_STATE["generator"],
            evaluator=_WORKER_STATE["evaluator"],
            extractor=_WORKER_STATE["extractor"],
            retriever=_WORKER_STATE["retriever"],
            question_id=f"{data_name}_{example['id']}" if data_name is not None and "id" in example else None,
            golden_answer=example.get("golden_answers"),
            config=_WORKER_STATE["search_config"],
            use_cot=use_cot,
            use_mcts=use_mcts,
        )
        return res

    return process_example


def generate_answer(
        question: Union[str, List[str]],
        generator: Generator,
        evaluator: Evaluator,
        extractor: Extractor,
        retriever: RetrieverAgent,
        # Optional parameters
        question_id: Optional[str] = None,
        golden_answer: Optional[Union[str, List[str]]] = None,
        use_cot: bool = False,
        use_mcts: bool = True,
        config: Optional[SearchArguments] = None,
    ):
    if use_mcts:
        # Use absolute import so running as module works
        from state_aware_rag.planners.MCTS.utils import search
    else:
        assert use_cot, "Either MCTS or CoT must be used for inference."
        from state_aware_rag.planners.CoT.utils import search
    final_answer, final_reasoning, reasoning_paths = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id=question_id,
        golden_answer=golden_answer,
        # MCTS parameters
        max_depth=getattr(config, "max_depth", 5) if config else 5,
        num_rollouts=getattr(config, "num_rollouts", 1) if config else 1,
        top_k=getattr(config, "top_k", 5) if config else 5,
        use_golden_answer=getattr(config, "use_golden_answer", False) if config else False,
        save_tree=getattr(config, "save_tree", False) if config else False,
        save_dir=getattr(config, "save_dir", "mcts_data") if config else "mcts_data",
    )
    detailed_answer = f"{final_reasoning}\n\nFinal Answer: {final_answer}" 
    return {
        "pred": final_answer,
        "detailed_answer": detailed_answer,
        "all_candidates_answers": reasoning_paths,
    }


def main():
    parser = ArgumentParser(description="Run MCTS inference for question answering.")
    parser.add_argument("--question", type=str, default=None, help="The question to answer.")
    parser.add_argument("--data_name", type=str, default=None, help="Name of the dataset to use.")
    parser.add_argument("--question_id", type=str, default=None, help="Optional question ID.")
    parser.add_argument("--golden_answer", type=str, default=None, help="Optional golden answer.")
    parser.add_argument("--generator_config", type=str, default="configs/infer/generator_config.yaml", help="Path to generator configuration file.")
    parser.add_argument("--evaluator_config", type=str, default="configs/infer/evaluator_config.yaml", help="Path to evaluator configuration file.")
    parser.add_argument("--extractor_config", type=str, default="configs/infer/extractor_config.yaml", help="Path to extractor configuration file.")
    parser.add_argument("--retriever_config", type=str, default="configs/infer/retriever_config.yaml", help="Path to retriever configuration file.")
    parser.add_argument("--search_config", type=str, default=None, help="Path to search configuration file.")
    parser.add_argument("--use_cot", action="store_true", help="Whether to use CoT for inference.")
    parser.add_argument("--use_mcts", action="store_true", help="Whether to use MCTS for inference.")
    parser.add_argument("--extractor_server_url", type=str, default=None, help="Optional extractor server address.")
    parser.add_argument("--extractor_model_name", type=str, default=None, help="Optional extractor model name.")
    parser.add_argument("--results_dir", type=str, default="results/mcts", help="Directory to save results.")
    parser.add_argument("--num_proc", type=int, default=64, help="Number of processes to use for parallel processing.")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of rollouts")

    args = parser.parse_args()
    assert args.question is not None or args.data_name is not None, "Either question or data_name must be provided."
    generator_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    llm_args, generation_args = generator_hf_parser.parse_yaml_file(args.generator_config)
    extractor_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    extractor_llm_args, extractor_generation_args = extractor_hf_parser.parse_yaml_file(args.extractor_config)
    if args.extractor_server_url:
        extractor_llm_args.url = args.extractor_server_url
    if args.extractor_model_name:
        extractor_llm_args.model_name = args.extractor_model_name
    evaluator_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    evaluator_llm_args, evaluator_generation_args = evaluator_hf_parser.parse_yaml_file(args.evaluator_config)
    retriever_hf_parser = HfArgumentParser(RetrieverArguments)
    retriever_args = retriever_hf_parser.parse_yaml_file(args.retriever_config)[0]
    search_hf_parser = HfArgumentParser(SearchArguments)
    if args.use_mcts:
        search_args = search_hf_parser.parse_yaml_file(args.search_config)[0]
        if args.n_rollouts is not None:
            search_args.num_rollouts = args.n_rollouts
        use_cot = False
        use_mcts = True
        results_dir = os.path.join(args.results_dir, 'mcts', f"Generator_{llm_args.model_name}", f"Retriever_4B-wiki23", f"Extractor_{extractor_llm_args.model_name}")
    else:
        search_args = search_hf_parser.parse_yaml_file(args.search_config)[0]
        use_cot = True
        use_mcts = False
        results_dir = os.path.join(args.results_dir, 'cot', f"Generator_{llm_args.model_name}", f"Retriever_4B-wiki23", f"Extractor_{extractor_llm_args.model_name}")
    search_args.save_dir = os.path.join(results_dir, 'result_trees')
    # Ensure the directory exists
    os.makedirs(search_args.save_dir, exist_ok=True)
    if args.n_rollouts is not None:
        search_args.num_rollouts = args.n_rollouts

    # Print the configurations
    print("Generator Config:")
    pprint.pprint(llm_args)
    pprint.pprint(generation_args)
    print("Extractor Config:")
    pprint.pprint(extractor_llm_args)
    pprint.pprint(extractor_generation_args)
    print("Evaluator Config:")
    pprint.pprint(evaluator_llm_args)
    pprint.pprint(evaluator_generation_args)
    print("Retriever Config:")
    pprint.pprint(retriever_args)
    print("Search Config:")
    if use_mcts:
        print("Using MCTS for inference.")
    else:
        print("Using CoT for inference.")
    pprint.pprint(search_args)

    # Initialize the generator, evaluator, extractor, and retriever agents for single-question mode
    if args.question is not None and not args.data_name:
        generator = Generator(
            client_kwargs=dataclasses.asdict(llm_args),
            generate_kwargs=dataclasses.asdict(generation_args),
            use_cache=generation_args.use_cache,
            cache_dir=generation_args.cache_dir
        )
        evaluator = Evaluator(
            client_kwargs=dataclasses.asdict(evaluator_llm_args),
            generate_kwargs=dataclasses.asdict(evaluator_generation_args),
            use_cache=evaluator_generation_args.use_cache,
            cache_dir=evaluator_generation_args.cache_dir
        )
        extractor = Extractor(
            client_kwargs=dataclasses.asdict(extractor_llm_args),
            generate_kwargs=dataclasses.asdict(extractor_generation_args),
            use_cache=extractor_generation_args.use_cache,
            cache_dir=extractor_generation_args.cache_dir
        )
        retriever = RetrieverAgent(online_kwargs=dataclasses.asdict(retriever_args))

    # Load the dataset if data_name is provided
    if args.data_name:
        if args.data_name == '2wiki':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', '2wikimultihopqa', split='dev')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == '2wiki-eval':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', '2wikimultihopqa', split='dev')
            # Randomly select 100 samples from the dataset for evaluation
            dataset = dataset.shuffle(seed=42).select(range(100))
        elif args.data_name == 'hotpotqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa', split='dev')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'musique':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'musique', split='dev')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'simpleqa':
            dataset = datasets.load_dataset('basicv8vc/SimpleQA', split='test')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
            dataset = dataset.map(lambda x, idx: {
                'id': idx,
                'question': x['problem'],
                'golden_answers': [x['answer']]
            }, with_indices=True, remove_columns=dataset.column_names)
        elif args.data_name == 'multi-hop-rag':
            print("You are evaluating on the Multi-hop RAG dataset. Please ensure that you deploy the retriever model with the multi-hop RAG corpus.")
            dataset = datasets.load_dataset('yixuantt/MultiHopRAG', split='train')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
            dataset = dataset.map(lambda x, idx: {
                'id': idx,
                'question': x['query'],
                'golden_answers': [x['answer']]
            }, with_indices=True, remove_columns=dataset.column_names)
        elif args.data_name == 'bamboogle':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'bamboogle', split='test')
        elif args.data_name == 'frames':
            dataset = datasets.load_dataset('google/frames-benchmark', split='test')
            dataset = dataset.map(lambda x, idx: {
                'id': idx,
                'question': x['Prompt'],
                'golden_answers': [x['Answer']]
            }, with_indices=True, remove_columns=dataset.column_names)
        elif args.data_name == 'solutionbench':
            print("You are evaluating on the SolutionBench dataset. Please ensure that you deploy the retriever model with the SolutionBench corpus.")
            dataset = datasets.load_dataset('lzq2021/SolutionBench', 'datas')
            # Concatenate all splits into a single dataset
            splits = dataset.keys()
            dataset = datasets.concatenate_datasets([dataset[split] for split in splits])
            dataset = dataset.map(lambda x, idx: {
                'id': idx,
                'question': f"Request: {x['title']}\nDetails: {x['requirement']}",
                'golden_answers': [x['solution']]
            }, with_indices=True, remove_columns=dataset.column_names)
        elif args.data_name == 'nq':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', split='test')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'triviaqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'triviaqa', split='test')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'popqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'popqa', split='test')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'mmlu':
            print("You are evaluating on the MMLU dataset. Please select the appropriate metric for evaluation.")
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'mmlu', split='test')
            # Randomly select 1000 samples from the dataset for testing
            dataset = dataset.shuffle(seed=42).select(range(1000))
        elif args.data_name == 'training_data_small':
            dataset = datasets.load_from_disk('data/small_data_with_support')
            dataset = dataset.rename_column("answer", "golden_answers")
        elif args.data_name == 'training_data':
            dataset = datasets.load_from_disk('data/train_data')
            # Filter out examples with empty golden answers
            dataset = dataset.filter(lambda x: x['golden_answers'] and len(x['golden_answers']) > 0, num_proc=64)
            dataset = dataset.map(lambda x: {'golden_answers': [ans for ans in x['golden_answers'] if ans]}, num_proc=64)
            dataset = dataset.filter(lambda x: x['golden_answers'] and len(x['golden_answers']) > 0, num_proc=64)
            # Filter out examples with empty questions
            dataset = dataset.filter(lambda x: x['question'] and len(x['question'].strip()) > 0, num_proc=64)
        else:
            raise ValueError(f"Unsupported dataset name: {args.data_name}")

        # Normlize the question and golden answer in the dataset
        dataset = dataset.map(lambda x: {
            'question': simple_preprocess(x['question']),
            'golden_answers': [simple_preprocess(ans) for ans in x['golden_answers']] if isinstance(x['golden_answers'], list) else [simple_preprocess(x['golden_answers'])]
        })

        # Build a safe worker function that initializes agents per process
        process_fn = _build_process_fn(
            gen_client_kwargs=dataclasses.asdict(llm_args),
            gen_generate_kwargs=dataclasses.asdict(generation_args),
            eval_client_kwargs=dataclasses.asdict(evaluator_llm_args),
            eval_generate_kwargs=dataclasses.asdict(evaluator_generation_args),
            ext_client_kwargs=dataclasses.asdict(extractor_llm_args),
            ext_generate_kwargs=dataclasses.asdict(extractor_generation_args),
            retriever_kwargs=dataclasses.asdict(retriever_args),
            search_config_dict=dataclasses.asdict(search_args),
            use_cot=use_cot,
            use_mcts=use_mcts,
            data_name=args.data_name,
        )

        # Cap processes to available CPUs to avoid oversubscription/races in pebble
        effective_num_proc = max(1, min(args.num_proc, (os.cpu_count() or args.num_proc)))
        if effective_num_proc != args.num_proc:
            print(f"Adjusting num_proc from {args.num_proc} to {effective_num_proc} (CPU count)")

        dataset = dataset.map(
            process_fn,
            with_indices=True,
            num_proc=effective_num_proc,
        )

        dataset.save_to_disk(f"{results_dir}")
        print(f"Results saved to {results_dir}")
    
    # clear_agent_cache(generator, extractor, evaluator)


if __name__ == "__main__":
    main()





