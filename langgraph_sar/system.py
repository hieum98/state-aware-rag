"""Top-level orchestrator for ``langgraph_sar`` (IMPLEMENTATION_PLAN §9)."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .config import SARConfig
from .graphs.mcts import build_mcts_graph
from .graphs.mcts_state import SARNodeType, make_tree_node
from .graphs.socratic import build_socratic_graph
from .llm import RoleModelRegistry
from .explicit import raise_explicit, warn_explicit
from .tools.retrieval import init_retrieval_pipeline
from .tools.web import init_web_search, reset_web_research_session

VALID_STRATEGIES = ("socratic", "mcts")
EVAL_MODES = ("judge_only", "teacher_forced")


@dataclass
class AnswerResult:
    """Strategy-agnostic answer envelope returned by :func:`answer_question`."""

    question: str
    pred: str
    detailed_answer: str = ""
    strategy: str = ""
    reward: Optional[float] = None
    eval_mode: str = "judge_only"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        *,
        strategy: str,
        eval_mode: str = "judge_only",
    ) -> "AnswerResult":
        state = state or {}
        metadata: Dict[str, Any] = {"strategy": strategy, "eval_mode": eval_mode}
        for key in ("iteration", "best_value", "text_memory", "reward_details"):
            if key in state and state[key] is not None:
                metadata[key] = state[key]
        history = state.get("iteration_history")
        if isinstance(history, list):
            metadata["num_iterations"] = len(history)

        final = str(state.get("final_answer") or "")
        detailed = str(state.get("detailed_answer") or "")
        if not detailed and final:
            detailed = final

        reward = state.get("reward")
        if reward is not None:
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                reward = None

        return cls(
            question=str(state.get("question", "") or ""),
            pred=final,
            detailed_answer=detailed,
            strategy=strategy,
            reward=reward,
            eval_mode=eval_mode,
            metadata=metadata,
        )


def _normalize_golden_answer(
    golden_answer: Optional[Union[str, Sequence[str]]],
) -> Optional[str]:
    if golden_answer is None:
        return None
    if isinstance(golden_answer, str):
        text = golden_answer.strip()
        return text or None
    parts = [str(x).strip() for x in golden_answer if str(x).strip()]
    return parts[0] if len(parts) == 1 else "\n".join(parts) if parts else None


def _run_config(
    config: SARConfig,
    *,
    strategy: str,
    golden_answer: Optional[str],
    eval_mode: str,
) -> Dict[str, Any]:
    limit = (
        config.search.mcts_recursion_limit
        if strategy == "mcts"
        else config.search.socratic_recursion_limit
    )
    return {
        "recursion_limit": int(limit),
        "configurable": {
            "golden_answer": golden_answer,
            "eval_mode": eval_mode,
        },
    }


def _init_runtime(config: SARConfig) -> None:
    """Idempotent tool init (retrieval + optional web). No LLM response cache (§4)."""
    try:
        init_retrieval_pipeline(config.retriever)
    except Exception as exc:
        raise_explicit(
            "Failed to initialize FAISS retrieval pipeline. "
            "Set SAR_CORPUS_INDEX_PATH or build a local index (see langgraph_sar/tests/phase1/create_toy_corpus.py).",
            component="system.init_runtime",
            cause=exc,
        )
    if config.web_search.enabled:
        try:
            init_web_search(config.web_search)
        except Exception as exc:
            raise_explicit(
                "web_search.enabled is true but web search initialization failed.",
                component="system.init_runtime",
                cause=exc,
            )


def _initial_socratic_state(question: str, config: SARConfig) -> Dict[str, Any]:
    return {
        "question": question,
        "max_depth": int(config.search.max_depth),
        "depth": 0,
        "is_answerable": False,
        "sub_question": "",
        "reasoning_trace": [],
        "text_memory": [],
        "iteration_history": [],
        "final_answer": "",
        "detailed_answer": "",
    }


def _initial_mcts_state(question: str, config: SARConfig) -> Dict[str, Any]:
    root_id = "root"
    root = make_tree_node(
        SARNodeType.USER_QUESTION,
        None,
        {"question": question},
        use_prior=config.search.use_node_type_prior,
    )
    root["node_id"] = root_id
    return {
        "question": question,
        "num_rollouts": int(config.search.num_rollouts),
        "max_depth": int(config.search.max_depth),
        "iteration": 0,
        "tree": {root_id: root},
        "root_id": root_id,
        "selected_path": [],
        "current_path": [],
        "expanded_node_ids": [],
        "simulation_result": {},
        "reward": 0.0,
        "text_memory": [],
        "best_value": 0.0,
        "iterations_without_improvement": 0,
        "final_answer": "",
        "detailed_answer": "",
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def answer_question(
    question: str,
    *,
    config: Optional[SARConfig] = None,
    golden_answer: Optional[Union[str, Sequence[str]]] = None,
    eval_mode: str = "judge_only",
) -> AnswerResult:
    """Answer one question via the configured Socratic or MCTS graph."""
    cfg = config or SARConfig.from_yaml()
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unknown eval_mode {eval_mode!r}; expected one of {EVAL_MODES}")

    strategy = cfg.search.strategy
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown search strategy {strategy!r}; expected one of {VALID_STRATEGIES}"
        )

    _init_runtime(cfg)
    # Per-question reset of the web research session. The session lives in a ContextVar,
    # and asyncio.gather runs each batch item in its own copied context, so this reset is
    # isolated per question even under concurrent answer_questions_batch (no shared-state race).
    reset_web_research_session()

    registry = RoleModelRegistry(cfg.llm)
    gold = _normalize_golden_answer(golden_answer)
    run_cfg = _run_config(cfg, strategy=strategy, golden_answer=gold, eval_mode=eval_mode)

    if strategy == "mcts":
        graph = build_mcts_graph(registry, cfg)
        initial_state = _initial_mcts_state(question, cfg)
    else:
        graph = build_socratic_graph(registry, cfg)
        initial_state = _initial_socratic_state(question, cfg)

    result = await graph.ainvoke(initial_state, config=run_cfg)
    return AnswerResult.from_state(result or {}, strategy=strategy, eval_mode=eval_mode)


async def answer_questions_batch(
    questions: Sequence[str],
    *,
    config: Optional[SARConfig] = None,
    golden_answers: Optional[Sequence[Optional[Union[str, Sequence[str]]]]] = None,
    eval_modes: Optional[Sequence[str]] = None,
    max_workers: int = 4,
) -> List[AnswerResult]:
    """Answer many questions concurrently; results preserve input order."""
    if golden_answers is not None and len(golden_answers) != len(questions):
        raise ValueError("golden_answers length must match questions")
    if eval_modes is not None and len(eval_modes) != len(questions):
        raise ValueError("eval_modes length must match questions")

    sem = asyncio.Semaphore(max(1, int(max_workers or 1)))

    async def _one(idx: int, question: str) -> AnswerResult:
        gold = None if golden_answers is None else golden_answers[idx]
        mode = "judge_only" if eval_modes is None else eval_modes[idx]
        async with sem:
            return await answer_question(
                question,
                config=config,
                golden_answer=gold,
                eval_mode=mode,
            )

    return list(await asyncio.gather(*[_one(i, q) for i, q in enumerate(questions)]))


async def answer_records_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Optional[SARConfig] = None,
    max_workers: int = 4,
    question_key: str = "question",
    golden_key: str = "golden_answers",
    eval_mode_key: str = "eval_mode",
) -> List[Dict[str, Any]]:
    """Run inference over dataset-like records; attach pred + eval_mode per row."""
    questions = [str(r[question_key]) for r in records]
    golds: List[Optional[Union[str, Sequence[str]]]] = []
    modes: List[str] = []
    for record in records:
        golds.append(record.get(golden_key))
        modes.append(str(record.get(eval_mode_key, "judge_only")))

    results = await answer_questions_batch(
        questions,
        config=config,
        golden_answers=golds,
        eval_modes=modes,
        max_workers=max_workers,
    )

    out: List[Dict[str, Any]] = []
    for record, result in zip(records, results):
        row = dict(record)
        row["pred"] = result.pred
        row["detailed_answer"] = result.detailed_answer
        row["eval_mode"] = result.eval_mode
        row["strategy"] = result.strategy
        if result.reward is not None:
            row["reward"] = result.reward
        out.append(row)
    return out


def filter_records_for_metrics(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Drop ``teacher_forced`` rows before metric aggregation (§11.3b)."""
    return [
        dict(r)
        for r in records
        if str(r.get("eval_mode", "judge_only")) != "teacher_forced"
    ]


def records_to_metric_dataset(records: Sequence[Mapping[str, Any]]):
    """Build a HuggingFace ``Dataset`` from inference rows for ``evaluate.py`` metrics."""
    import datasets

    filtered = filter_records_for_metrics(records)
    if not filtered:
        return datasets.Dataset.from_dict(
            {"question": [], "pred": [], "golden_answers": [], "eval_mode": []}
        )
    return datasets.Dataset.from_list(filtered)


def compute_metrics_from_records(
    records: Sequence[Mapping[str, Any]],
    metrics: Optional[Sequence[str]] = None,
    *,
    dataset_name: str = "default",
    llm_setting: Optional[Any] = None,
) -> tuple[Dict[str, Any], Any]:
    """Filter teacher-forced rows, then delegate to legacy ``evaluate.compute_metrics``."""
    from evaluate import compute_metrics

    ds = records_to_metric_dataset(records)
    return compute_metrics(
        ds,
        metrics=list(metrics) if metrics else ["sub_em"],
        dataset_name=dataset_name,
        llm_setting=llm_setting,
    )
