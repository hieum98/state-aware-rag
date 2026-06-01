import os
from typing import Any, Dict, List, Optional, Union

import datasets
import hydra
from hydra import utils as hy_utils
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from state_aware_rag.agents.agents import (
    EvaluatorAgent,
    ExtractorAgent,
    GeneratorAgent,
    RetrievalAgent,
)
from state_aware_rag.preprocess.utils import simple_preprocess

# Lazy, per-process agent cache for HF datasets multiprocessing workers
_AGENTS_CACHE: Dict[str, Any] = {}

def _get_or_init_agents(agent_cfg: Dict[str, Any]):
    global _AGENTS_CACHE
    if _AGENTS_CACHE:
        return _AGENTS_CACHE
    # Initialize fresh agent instances in this worker process
    _AGENTS_CACHE = {
        "generator": GeneratorAgent(config=agent_cfg["generator"]),
        "retriever": RetrievalAgent(config=agent_cfg["retriever"]),
        "extractor": ExtractorAgent(config=agent_cfg["extractor"]),
        "evaluator": EvaluatorAgent(config=agent_cfg["evaluator"]),
    }
    return _AGENTS_CACHE


def generate_answer(
    question: Union[str, List[str]],
    generator: GeneratorAgent,
    evaluator: EvaluatorAgent,
    extractor: ExtractorAgent,
    retriever: RetrievalAgent,
    question_id: Optional[str] = None,
    golden_answer: Optional[Union[str, List[str]]] = None,
    mode: str = "mcts",
    search: Optional[Dict[str, Any]] = None,
    ):
    use_mcts = mode == "mcts"
    if use_mcts:
        from state_aware_rag.planners.MCTS.utils import search as planner_search
    else:
        from state_aware_rag.planners.CoT.utils import search as planner_search
    search = search or {}
    final_answer, final_reasoning = planner_search(
        # Agents
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Inputs
        user_question=question,
        question_id=question_id,
        golden_answer=golden_answer,
        # Search parameters
        max_depth=search.get("max_depth", 5),
        num_rollouts=search.get("num_rollouts", 1),
        top_k=search.get("top_k", 5),
        exploration_weight=search.get("exploration_weight", 1.0),
        use_golden_answer=search.get("use_golden_answer", False),
        save_tree=search.get("save_tree", False),
        save_dir=search.get("tree_dir", "mcts_trees" if use_mcts else "cot_trees"),
        verbose=search.get("verbose", False),
    )
    detailed_answer = f"{final_reasoning}\n\nFinal Answer: {final_answer}"
    return {
        "pred": str(final_answer),
        "detailed_answer": detailed_answer,
    }


def _load_dataset(cfg: DictConfig):
    name = cfg.name
    split = cfg.get("split", None)
    limit = cfg.get("limit", None)

    if name == "2wiki" or name == "2wiki-eval":
        if not split:
            split = "dev"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "hotpotqa":
        if not split:
            split = "dev"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "musique":
        if not split:
            split = "dev"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "simpleqa":
        if not split:
            split = "test"
        ds = datasets.load_dataset("basicv8vc/SimpleQA", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
        ds = ds.map(
            lambda x, idx: {
                "id": idx,
                "question": x["problem"],
                "golden_answers": [x["answer"]],
            },
            with_indices=True,
            remove_columns=ds.column_names,
        )
    elif name == "multi-hop-rag":
        print(
            "You are evaluating on the Multi-hop RAG dataset. Please ensure that you deploy the retriever model with the Multi-hop RAG corpus."
        )
        ds = datasets.load_dataset("yixuantt/MultiHopRAG", split="train")
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
        ds = ds.map(
            lambda x, idx: {
                "id": idx,
                "question": x["query"],
                "golden_answers": [x["answer"]],
            },
            with_indices=True,
            remove_columns=ds.column_names,
        )
    elif name == "bamboogle":
        if not split:
            split = "test"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "bamboogle", split=split)
    elif name == "frames":
        if not split:
            split = "test"
        ds = datasets.load_dataset("google/frames-benchmark", split=split)
        ds = ds.map(
            lambda x, idx: {
                "id": idx,
                "question": x["Prompt"],
                "golden_answers": [x["Answer"]],
            },
            with_indices=True,
            remove_columns=ds.column_names,
        )
    elif name == "solutionbench":
        print(
            "You are evaluating on the SolutionBench dataset. Please ensure that you deploy the retriever model with the SolutionBench corpus."
        )
        ds = datasets.load_dataset("lzq2021/SolutionBench", "datas")
        ds = datasets.concatenate_datasets([ds[s] for s in ds.keys()])
        ds = ds.map(
            lambda x, idx: {
                "id": idx,
                "question": f"Request: {x['title']}\nDetails: {x['requirement']}",
                "golden_answers": [x["solution"]],
            },
            with_indices=True,
            remove_columns=ds.column_names,
        )
    elif name == "nq":
        if not split:
            split = "test"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "triviaqa":
        if not split:
            split = "test"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "triviaqa", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "popqa":
        if not split:
            split = "test"
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "popqa", split=split)
        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))
    elif name == "training_data_small":
        ds = datasets.load_from_disk("data/small_data_with_support")
        ds = ds.rename_column("answer", "golden_answers")
    elif name == "training_data":
        ds = datasets.load_from_disk("data/train_data")
        ds = ds.filter(lambda x: x["golden_answers"] and len(x["golden_answers"]) > 0, num_proc=64)
        ds = ds.map(lambda x: {"golden_answers": [ans for ans in x["golden_answers"] if ans]}, num_proc=64)
        ds = ds.filter(lambda x: x["golden_answers"] and len(x["golden_answers"]) > 0, num_proc=64)
        ds = ds.filter(lambda x: x["question"] and len(x["question"].strip()) > 0, num_proc=64)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    # Normalize question and answers
    ds = ds.map(
        lambda x: {
            "question": simple_preprocess(x["question"]),
            "golden_answers": [
                simple_preprocess(ans)
                for ans in (x["golden_answers"] if isinstance(x["golden_answers"], list) else [x["golden_answers"]])
            ],
        }
    )
    return ds


def _compute_results_dir(cfg: DictConfig) -> str:
    mode = cfg.mode
    data_name = cfg.get("data", {}).get("name", "unknown_data")
    gen_model = cfg.agents.generator.client_kwargs.model_name.replace("/", "-")
    ext_model = cfg.agents.extractor.client_kwargs.model_name.replace("/", "-")
    eva_model = cfg.agents.evaluator.client_kwargs.model_name.replace("/", "-")
    base = os.path.join(cfg.results_dir, mode, f"Generator_{gen_model}", f"Extractor_{ext_model}", f'Evaluator_{eva_model}', data_name)
    return base


def _maybe_load_agent_cfg(agent_entry: Any) -> Dict[str, Any]:
    """Accept a dict or a string path to YAML; return a plain python dict."""
    if isinstance(agent_entry, (dict, DictConfig)):
        return OmegaConf.to_container(agent_entry, resolve=True)  # type: ignore[arg-type]
    if isinstance(agent_entry, str):
        path = agent_entry
        if not os.path.isabs(path):
            path = os.path.join(hy_utils.get_original_cwd(), path)
        conf = OmegaConf.load(path)
        return OmegaConf.to_container(conf, resolve=True)  # type: ignore[return-value]
    raise TypeError(f"Unsupported agent config type: {type(agent_entry)}")


@hydra.main(config_path="configs/infer", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Keep working directory stable (avoid hydra changing CWD)
    print("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    mode = cfg.mode  # "mcts" or "cot"
    # Build output dirs (resolve relative to original cwd)
    results_dir = os.path.join(hy_utils.get_original_cwd(), _compute_results_dir(cfg))
    tree_dir = os.path.join(results_dir, "result_trees")
    os.makedirs(tree_dir, exist_ok=True)

    # Prepare agent configs and search config
    agent_cfg = {
        "generator": _maybe_load_agent_cfg(cfg.agents.generator),
        "evaluator": _maybe_load_agent_cfg(cfg.agents.evaluator),
        "extractor": _maybe_load_agent_cfg(cfg.agents.extractor),
        "retriever": _maybe_load_agent_cfg(cfg.agents.retriever),
    }
    search_cfg = OmegaConf.to_container(cfg.search, resolve=True)  
    search_cfg["tree_dir"] = tree_dir

    # Single-question mode
    if cfg.question:
        generator = GeneratorAgent(config=agent_cfg["generator"])  
        retriever = RetrievalAgent(config=agent_cfg["retriever"])  
        extractor = ExtractorAgent(config=agent_cfg["extractor"])  
        evaluator = EvaluatorAgent(config=agent_cfg["evaluator"])  
        res = generate_answer(
            question=cfg.question,
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            question_id="single_question",
            golden_answer=cfg.get("golden_answer"),
            mode=mode,
            search=search_cfg,
        )
        print("Final Answer:", res["pred"])
        print("Final Reasoning:", res["detailed_answer"])

    # Dataset mode
    ds = _load_dataset(cfg.data)

    # Map helper that lazily initializes agents per worker process
    def _map_generate(ex, idx, *, agent_cfg: Dict[str, Any], search_cfg: Dict[str, Any], mode: str, data_name: Optional[str]):
        agents = _get_or_init_agents(agent_cfg)
        res = generate_answer(
            question=ex["question"],
            generator=agents["generator"],
            evaluator=agents["evaluator"],
            extractor=agents["extractor"],
            retriever=agents["retriever"],
            question_id=f"{data_name}_{ex['id']}" if data_name is not None and "id" in ex else None,
            golden_answer=ex.get("golden_answers"),
            mode=mode,
            search=search_cfg,
        )
        return res

    num_proc = max(1, min(cfg.num_proc, os.cpu_count() or 1))
    print(f"Running datasets.map with num_proc={num_proc}...")
    ds = ds.map(
        _map_generate,
        with_indices=True,
        fn_kwargs={
            "agent_cfg": agent_cfg,
            "search_cfg": search_cfg,
            "mode": mode,
            "data_name": cfg.data.name,
        },
        num_proc=num_proc,
        desc="inference",
    )
    print("Inference completed.")
    ds.save_to_disk(results_dir)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()





