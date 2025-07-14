import dataclasses
import pprint
import datasets
from typing import Any, Dict, List, Optional, Union
from argparse import ArgumentParser
from transformers import HfArgumentParser

from agents.roles.evaluator import Evaluator
from agents.roles.extractor import Extractor
from agents.roles.generator import Generator
from agents.retriever_agents import RetrieverAgent
from planners.MCTS.utils import search, clear_agent_cache
from args import MCTSArguments, GenerationArguments, RetrieverArguments, LLMAgentArguments


def generate_answer(
        question: Union[str, List[str]],
        generator: Generator,
        evaluator: Evaluator,
        extractor: Extractor,
        retriever: RetrieverAgent,
        # Optional parameters
        question_id: Optional[str] = None,
        golden_answer: Optional[Union[str, List[str]]] = None,
        config: Optional[MCTSArguments] = None
    ):
    # breakpoint()
    final_answer, solutions = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id=question_id,
        golden_answer=golden_answer,
        # MCTS parameters
        max_depth=config.max_depth if config else 15,
        num_rollouts=config.num_rollouts if config else 100,
        use_golden_answer=config.use_golden_answer if config else False,
        save_tree=config.save_tree if config else False,
        save_dir=config.save_dir if config else "mcts_data"
    )
    return {
        "final_answer": final_answer,
        "solutions": solutions
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Run MCTS inference for question answering.")
    parser.add_argument("--question", type=str, default=None, help="The question to answer.")
    parser.add_argument("--data_name", type=str, default=None, help="Name of the dataset to use.")
    parser.add_argument("--question_id", type=str, default=None, help="Optional question ID.")
    parser.add_argument("--golden_answer", type=str, default=None, help="Optional golden answer.")
    parser.add_argument("--generator_config", type=str, default="configs/eval_config/generator_config.yaml", help="Path to generator configuration file.")
    parser.add_argument("--evaluator_config", type=str, default="configs/eval_config/evaluator_config.yaml", help="Path to evaluator configuration file.")
    parser.add_argument("--extractor_config", type=str, default="configs/eval_config/extractor_config.yaml", help="Path to extractor configuration file.")
    parser.add_argument("--retriever_config", type=str, default="configs/eval_config/retriever_config.yaml", help="Path to retriever configuration file.")
    parser.add_argument("--mcts_config", type=str, default="configs/eval_config/MCTS_config.yaml", help="Path to MCTS configuration file.")

    args = parser.parse_args()
    assert args.question is not None or args.data_name is not None, "Either question or data_name must be provided."
    generator_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    llm_args, generation_args = generator_hf_parser.parse_yaml_file(args.generator_config)
    extractor_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    extractor_llm_args, extractor_generation_args = extractor_hf_parser.parse_yaml_file(args.extractor_config)
    evaluator_hf_parser = HfArgumentParser((LLMAgentArguments, GenerationArguments))
    evaluator_llm_args, evaluator_generation_args = evaluator_hf_parser.parse_yaml_file(args.evaluator_config)
    retriever_hf_parser = HfArgumentParser(RetrieverArguments)
    retriever_args = retriever_hf_parser.parse_yaml_file(args.retriever_config)[0]
    mcts_hf_parser = HfArgumentParser(MCTSArguments)
    mcts_args = mcts_hf_parser.parse_yaml_file(args.mcts_config)[0]

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
    print("MCTS Config:")
    pprint.pprint(mcts_args)

    # Initialize the generator, evaluator, extractor, and retriever agents
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
        # breakpoint()
        dataset = dataset.map(lambda x: generate_answer(
            question=x['question'],
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            question_id=f"{args.data_name}_{x['id']}",
            golden_answer=x['golden_answers'],
            config=mcts_args
            ), num_proc=128)
        breakpoint()
    
    clear_agent_cache(generator, extractor, evaluator)



    

