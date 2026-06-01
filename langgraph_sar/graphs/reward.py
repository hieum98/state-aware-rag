"""Phase 2 reward helpers for Socratic/MCTS evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.runnables import RunnableConfig

from ..config import SARConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import OUTCOME_JUDGE, PATH_JUDGE, OutcomeJudgeInput, PathJudgeInput
from .actions import confidence_to_score, format_memory, format_reasoning_trace

_PATH_SCORE = {"poor": 0.0, "fair": 1.0 / 3.0, "good": 2.0 / 3.0, "excellent": 1.0}


def get_configurable(config: Optional[RunnableConfig]) -> Dict[str, Any]:
    if not config:
        return {}
    return dict(config.get("configurable") or {})


def path_score(path_output: Any) -> float:
    values = [
        _PATH_SCORE.get(str(getattr(path_output, key, "poor")).lower(), 0.0)
        for key in ("relevance", "sufficiency", "coherence", "factuality")
    ]
    return sum(values) / len(values)


def path_details(path_output: Any) -> Dict[str, Any]:
    return {
        key: getattr(path_output, key, None)
        for key in (
            "relevance",
            "relevance_details",
            "sufficiency",
            "sufficiency_details",
            "coherence",
            "coherence_details",
            "factuality",
            "factuality_details",
        )
    }


async def evaluate_outcome(
    registry: RoleModelRegistry,
    *,
    question: str,
    answer: str,
    golden_answer: Optional[str] = None,
) -> Dict[str, Any]:
    out, _ = await execute_role_lc(
        registry,
        OUTCOME_JUDGE,
        OutcomeJudgeInput(
            user_question=question,
            system_answer=answer,
            correct_answer=golden_answer or "Not provided",
        ),
    )
    rating = float(getattr(out, "total_rating", 0.0))
    normalized = max(0.0, min(1.0, rating / 10.0))
    return {"reward": normalized, "rating": rating, "reasoning": getattr(out, "reasoning", "")}


async def evaluate_path_step(
    registry: RoleModelRegistry,
    *,
    main_question: str,
    reasoning_trace: List[str],
    sub_question: str,
    selected_information: Iterable[str],
    generated_answer: str,
) -> Dict[str, Any]:
    out, _ = await execute_role_lc(
        registry,
        PATH_JUDGE,
        PathJudgeInput(
            main_question=main_question,
            reasoning_trace=format_reasoning_trace(reasoning_trace),
            sub_question=sub_question,
            selected_information=format_memory(selected_information),
            generated_answer=generated_answer,
        ),
    )
    return {"reward": path_score(out), "details": path_details(out)}


async def evaluate_socratic_result(
    registry: RoleModelRegistry,
    config: SARConfig,
    *,
    question: str,
    final_answer: str,
    iteration_history: List[Dict[str, Any]],
    golden_answer: Optional[str] = None,
) -> Dict[str, Any]:
    outcome = await evaluate_outcome(
        registry,
        question=question,
        answer=final_answer,
        golden_answer=golden_answer,
    )

    if config.search.reward_mode == "confidence_blend":
        confidences = [confidence_to_score(step.get("confidence")) for step in iteration_history]
        confidence_reward = sum(confidences) / len(confidences) if confidences else 0.1
        reward = 0.5 * outcome["reward"] + 0.5 * confidence_reward
        return {"reward": reward, "outcome": outcome, "path_rewards": [], "mode": "confidence_blend"}

    path_rewards: List[Dict[str, Any]] = []
    trace_prefix: List[str] = []
    for step in iteration_history:
        if step.get("node_type") != "SUB_QA":
            continue
        judged = await evaluate_path_step(
            registry,
            main_question=question,
            reasoning_trace=trace_prefix,
            sub_question=str(step.get("sub_question", "")),
            selected_information=step.get("distilled_info", []),
            generated_answer=str(step.get("answer", "")),
        )
        path_rewards.append(judged)
        trace_prefix.append(str(step.get("trace_entry", step.get("answer", ""))))

    # No SUB_QA steps (direct conclude) → fall back to R_o rather than halving it.
    lam = config.search.reward_lambda
    if path_rewards:
        mean_path = sum(item["reward"] for item in path_rewards) / len(path_rewards)
        reward = (mean_path + lam * outcome["reward"]) / (1.0 + lam)
    else:
        reward = outcome["reward"]
    return {"reward": reward, "outcome": outcome, "path_rewards": path_rewards, "mode": "dual"}


def _factuality_score(details: Dict[str, Any]) -> float:
    raw = details.get("factuality")
    if raw is None:
        return 0.0
    return _PATH_SCORE.get(str(raw).lower(), 0.0)


def collect_judge_critiques(outcome: Dict[str, Any], path_details_list: List[Dict[str, Any]]) -> List[str]:
    critiques: List[str] = []
    reasoning = outcome.get("reasoning")
    if reasoning:
        critiques.append(str(reasoning))
    for details in path_details_list:
        for key in (
            "relevance_details",
            "sufficiency_details",
            "coherence_details",
            "factuality_details",
        ):
            text = details.get(key)
            if text:
                critiques.append(str(text))
    return critiques


def should_commit_memory(config: SARConfig, *, reward: float, last_factuality: float) -> bool:
    gate = config.search.memory_commit_gate
    threshold = config.search.memory_commit_threshold
    if gate == "off":
        return True
    if gate == "reward":
        return reward >= threshold
    return last_factuality >= threshold


def _node_path_judge_fields(node: Dict[str, Any]) -> tuple[str, str, List[str]] | None:
    content = node.get("content") or {}
    ntype = node.get("node_type")
    if ntype == "SUB_QA":
        return (
            str(content.get("sub_question", "")),
            str(content.get("answer", "")),
            list(content.get("distilled_info") or []),
        )
    if ntype == "SYNTHESIS":
        synthesis = str(content.get("synthesis") or content.get("reasoning") or "")
        return (str(content.get("sub_question", "synthesis")), synthesis, [])
    if ntype == "SELF_CORRECTED":
        return (
            str(content.get("sub_question", "")),
            str(content.get("answer", "")),
            list(content.get("distilled_info") or []),
        )
    return None


async def evaluate_mcts_iteration(
    registry: RoleModelRegistry,
    config: SARConfig,
    *,
    question: str,
    tree: Dict[str, Any],
    selected_path: List[str],
    terminal_answer: str,
    golden_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Dual (or confidence_blend) reward for one MCTS iteration with per-node R_p memoization."""
    tree_updates: Dict[str, Any] = {}
    path_rewards: List[float] = []
    path_details_list: List[Dict[str, Any]] = []
    trace_prefix: List[str] = []

    # confidence_blend is the cost escape hatch (§11.2): skip PATH_JUDGE entirely.
    # Running the loop would pay the full per-step judge cost only to discard r_p.
    if config.search.reward_mode != "confidence_blend":
        judged_types = {"SUB_QA", "SYNTHESIS", "SELF_CORRECTED"}
        for nid in selected_path:
            node = dict(tree.get(nid) or {})
            if node.get("node_type") not in judged_types:
                continue
            if node.get("r_p") is not None:
                path_rewards.append(float(node["r_p"]))
                path_details_list.append(dict(node.get("r_p_details") or {}))
                fields = _node_path_judge_fields(node)
                if fields:
                    sq, ans, _ = fields
                    trace_prefix.append(f"Q: {sq}\nA: {ans}")
                continue

            fields = _node_path_judge_fields(node)
            if not fields:
                continue
            sub_q, answer, distilled = fields
            judged = await evaluate_path_step(
                registry,
                main_question=question,
                reasoning_trace=trace_prefix,
                sub_question=sub_q,
                selected_information=distilled,
                generated_answer=answer,
            )
            node["r_p"] = judged["reward"]
            node["r_p_details"] = judged["details"]
            tree_updates[nid] = node
            path_rewards.append(judged["reward"])
            path_details_list.append(judged["details"])
            trace_prefix.append(f"Q: {sub_q}\nA: {answer}")

    outcome = await evaluate_outcome(
        registry,
        question=question,
        answer=terminal_answer,
        golden_answer=golden_answer,
    )

    if config.search.reward_mode == "confidence_blend":
        confidences: List[float] = []
        for nid in selected_path:
            node = tree.get(nid) or {}
            conf = (node.get("content") or {}).get("confidence")
            if conf is not None:
                confidences.append(confidence_to_score(conf))
        confidence_reward = sum(confidences) / len(confidences) if confidences else 0.1
        reward = 0.5 * outcome["reward"] + 0.5 * confidence_reward
        critiques = collect_judge_critiques(outcome, [])
        last_factuality = _factuality_score(path_details_list[-1]) if path_details_list else 0.0
        return {
            "reward": reward,
            "outcome": outcome,
            "path_rewards": path_rewards,
            "path_details_list": path_details_list,
            "judge_critiques": critiques,
            "last_factuality": last_factuality,
            "tree_updates": tree_updates,
            "mode": "confidence_blend",
        }

    # When the selected path has no judged reasoning step yet (e.g. iteration 1 from
    # root), fall back to the outcome reward instead of treating mean_path as 0.0,
    # which would otherwise halve the reward (R_o/2 at λ=1) for shallow expansions.
    lam = config.search.reward_lambda
    if path_rewards:
        mean_path = sum(path_rewards) / len(path_rewards)
        reward = (mean_path + lam * outcome["reward"]) / (1.0 + lam)
    else:
        reward = outcome["reward"]
    critiques = collect_judge_critiques(outcome, path_details_list)
    last_factuality = _factuality_score(path_details_list[-1]) if path_details_list else 0.0
    return {
        "reward": reward,
        "outcome": outcome,
        "path_rewards": path_rewards,
        "path_details_list": path_details_list,
        "judge_critiques": critiques,
        "last_factuality": last_factuality,
        "tree_updates": tree_updates,
        "mode": "dual",
    }
