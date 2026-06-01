"""MCTSGraph for ``langgraph_sar`` (IMPLEMENTATION_PLAN §8.5)."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..config import SARConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import ANSWER_SYNTHESIZER, MAJORITY_VOTER, AnswerSynthesizerInput, MajorityVoterInput
from .actions import (
    answer_subquestion,
    conclude_question,
    dedup_preserve_order,
    format_memory,
    format_reasoning_trace,
    generate_subquestion,
    refine_step,
    rephrase_step,
    synthesize_step,
)
from .consolidate import build_consolidate_graph
from .memory_update import build_memory_update_graph
from .mcts_state import (
    MCTSState,
    MCTSTreeNode,
    SARNodeType,
    dict_merge,
    is_terminal,
    make_tree_node,
    node_type_str,
)
from ..explicit import warn_explicit
from .reward import evaluate_mcts_iteration, get_configurable, should_commit_memory
from .socratic import build_socratic_graph

logger = logging.getLogger(__name__)


def puct_score(
    child: MCTSTreeNode,
    parent_visits: int,
    *,
    exploration_weight: float,
) -> float:
    visits = int(child.get("visits") or 0)
    q = (float(child.get("value") or 0.0) / visits) if visits > 0 else 0.0
    prior = float(child.get("prior") or 1.0)
    u = exploration_weight * prior * math.sqrt(parent_visits + 1.0) / (1.0 + visits)
    return q + u


def _trace_from_path(tree: Dict[str, MCTSTreeNode], path: List[str]) -> List[str]:
    trace: List[str] = []
    for nid in path:
        node = tree.get(nid) or {}
        content = node.get("content") or {}
        ntype = node.get("node_type")
        if ntype == SARNodeType.SUB_QA.value:
            trace.append(f"Q: {content.get('sub_question', '')}\nA: {content.get('answer', '')}")
        elif ntype == SARNodeType.SYNTHESIS.value:
            trace.append(str(content.get("synthesis") or content.get("reasoning") or ""))
        elif ntype == SARNodeType.SELF_CORRECTED.value:
            trace.append(f"Q: {content.get('sub_question', '')}\nA: {content.get('answer', '')}")
        elif ntype == SARNodeType.FINAL_ANSWER.value:
            trace.append(f"Final Answer: {content.get('detailed_answer') or content.get('answer', '')}")
    return [t for t in trace if t.strip()]


def _content_to_tree_node(parent_id: str, content: dict, *, use_prior: bool) -> MCTSTreeNode:
    ntype = SARNodeType(content.get("node_type", SARNodeType.SUB_QA.value))
    return make_tree_node(ntype, parent_id, content, use_prior=use_prior)


def _format_synthesis_paths(candidates: List[str], scores: List[float]) -> str:
    blocks: List[str] = []
    for idx, (answer, score) in enumerate(zip(candidates, scores), 1):
        blocks.append(f"Candidate {idx} (search score={score:.3f}):\n{answer}")
    return "\n\n".join(blocks)


def build_mcts_graph(registry: RoleModelRegistry, config: SARConfig):
    sar_config = config
    consolidate_graph = build_consolidate_graph(registry, sar_config)
    memory_update_graph = build_memory_update_graph(registry, sar_config)
    # Rollout-only: simulate discards the dual reward, so skip the evaluate node
    # (avoids a discarded OUTCOME_JUDGE + per-step PATH_JUDGE on every rollout).
    socratic_graph = build_socratic_graph(registry, sar_config, include_evaluate=False)

    workflow = StateGraph(MCTSState)

    async def select(state: MCTSState) -> Dict[str, Any]:
        tree = dict(state.get("tree") or {})
        root_id = state.get("root_id") or "root"
        updates: Dict[str, MCTSTreeNode] = {}

        if root_id not in tree:
            root = make_tree_node(
                SARNodeType.USER_QUESTION,
                None,
                {"question": state.get("question", "")},
                use_prior=sar_config.search.use_node_type_prior,
            )
            root["node_id"] = root_id
            tree[root_id] = root
            updates[root_id] = root

        node_id = root_id
        path = [node_id]
        explore = sar_config.search.exploration_weight

        while True:
            node = tree[node_id]
            children = node.get("children_ids") or []
            if not children or is_terminal(node):
                break
            parent_visits = int(node.get("visits") or 0)
            best_id = max(children, key=lambda cid: puct_score(tree[cid], parent_visits, exploration_weight=explore))
            node_id = best_id
            path.append(node_id)

        out: Dict[str, Any] = {"selected_path": path, "current_path": path}
        if updates:
            out["tree"] = updates
        return out

    async def _expand_a1(parent_id: str, state: MCTSState, ctx_memory: List[str], trace: List[str]) -> List[MCTSTreeNode]:
        question = state["question"]
        out = await generate_subquestion(registry, question, ctx_memory, trace)
        if bool(getattr(out, "answerable_main_question", False)):
            content, _ = await conclude_question(
                registry,
                consolidate_graph,
                user_question=question,
                text_memory=ctx_memory,
                reasoning_trace=trace,
            )
            return [_content_to_tree_node(parent_id, content, use_prior=sar_config.search.use_node_type_prior)]

        sub_q = (getattr(out, "subquestion", "") or "").strip()
        if not sub_q:
            return []
        content, _ = await answer_subquestion(
            registry,
            sar_config,
            consolidate_graph,
            user_question=question,
            sub_question=sub_q,
            text_memory=ctx_memory,
            reasoning_trace=trace,
        )
        return [_content_to_tree_node(parent_id, content, use_prior=sar_config.search.use_node_type_prior)]

    async def expand(state: MCTSState) -> Dict[str, Any]:
        tree = dict(state.get("tree") or {})
        path = state.get("current_path") or []
        if not path:
            return {"expanded_node_ids": []}

        leaf_id = path[-1]
        leaf = dict(tree[leaf_id])
        if is_terminal(leaf):
            return {"expanded_node_ids": []}

        ntype = node_type_str(leaf)
        parent_id = leaf_id
        memory = list(state.get("text_memory") or [])
        trace = _trace_from_path(tree, path)
        children: List[MCTSTreeNode] = []

        if len(path) - 1 >= int(state.get("max_depth", sar_config.search.max_depth)):
            content, _ = await conclude_question(
                registry,
                consolidate_graph,
                user_question=state["question"],
                text_memory=memory,
                reasoning_trace=trace,
            )
            children.append(_content_to_tree_node(parent_id, content, use_prior=sar_config.search.use_node_type_prior))
        elif ntype == SARNodeType.USER_QUESTION.value:
            a1_nodes, a4_content, a5_content = await asyncio.gather(
                _expand_a1(parent_id, state, memory, trace),
                rephrase_step(registry, question=state["question"]),
                conclude_question(
                    registry,
                    consolidate_graph,
                    user_question=state["question"],
                    text_memory=memory,
                    reasoning_trace=trace,
                ),
            )
            a4_node = _content_to_tree_node(parent_id, a4_content[0], use_prior=sar_config.search.use_node_type_prior)
            a5_node = _content_to_tree_node(parent_id, a5_content[0], use_prior=sar_config.search.use_node_type_prior)
            children = list(a1_nodes) + [a4_node, a5_node]
        elif ntype == SARNodeType.SUB_QA.value:
            content = leaf.get("content") or {}
            sq = str(content.get("sub_question", ""))
            sa = str(content.get("answer", ""))
            a1_nodes = await _expand_a1(parent_id, state, memory, trace)
            a2_content, _ = await synthesize_step(
                registry,
                user_question=state["question"],
                text_memory=memory,
                reasoning_trace=trace,
            )
            a3_content, _ = await refine_step(
                registry,
                sar_config,
                consolidate_graph,
                user_question=state["question"],
                sub_question=sq,
                sub_answer=sa,
                text_memory=memory,
                reasoning_trace=trace,
            )
            a4_content, _ = await rephrase_step(registry, question=sq or state["question"])
            children = a1_nodes + [
                _content_to_tree_node(parent_id, a2_content, use_prior=sar_config.search.use_node_type_prior),
                _content_to_tree_node(parent_id, a3_content, use_prior=sar_config.search.use_node_type_prior),
                _content_to_tree_node(parent_id, a4_content, use_prior=sar_config.search.use_node_type_prior),
            ]
        elif ntype == SARNodeType.REPHRASED_QUESTION.value:
            rephrased = str((leaf.get("content") or {}).get("rephrased_question") or state["question"])
            out = await generate_subquestion(registry, rephrased, memory, trace)
            sub_q = (getattr(out, "subquestion", "") or rephrased).strip()
            content, _ = await answer_subquestion(
                registry,
                config,
                consolidate_graph,
                user_question=state["question"],
                sub_question=sub_q,
                text_memory=memory,
                reasoning_trace=trace,
            )
            children = [_content_to_tree_node(parent_id, content, use_prior=sar_config.search.use_node_type_prior)]
        elif ntype == SARNodeType.SELF_CORRECTED.value:
            a1_nodes = await _expand_a1(parent_id, state, memory, trace)
            a2_content, _ = await synthesize_step(
                registry,
                user_question=state["question"],
                text_memory=memory,
                reasoning_trace=trace,
            )
            children = a1_nodes + [
                _content_to_tree_node(parent_id, a2_content, use_prior=sar_config.search.use_node_type_prior)
            ]
        elif ntype == SARNodeType.SYNTHESIS.value:
            children = await _expand_a1(parent_id, state, memory, trace)
        else:
            children = []

        if not children:
            return {"expanded_node_ids": []}

        updates: Dict[str, MCTSTreeNode] = {c["node_id"]: c for c in children}
        parent = dict(leaf)
        parent["children_ids"] = list(parent.get("children_ids") or []) + [c["node_id"] for c in children]
        updates[parent_id] = parent
        return {"tree": updates, "expanded_node_ids": [c["node_id"] for c in children]}

    async def simulate(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        path = list(state.get("current_path") or [])
        expanded = state.get("expanded_node_ids") or []

        start_id = next((cid for cid in expanded if not is_terminal(tree[cid])), None)
        if start_id is None:
            if expanded:
                path = path + [expanded[0]]
            terminal = tree[path[-1]] if path else {}
            content = terminal.get("content") or {}
            answer = str(content.get("answer") or content.get("detailed_answer") or "")
            return {
                "current_path": path,
                "simulation_result": {
                    "terminal_answer": answer,
                    "rollout_distilled": [],
                    "rollout_trace": _trace_from_path(tree, path),
                },
            }

        fork_memory = list(state.get("text_memory") or [])
        start_node = tree[start_id]
        start_content = start_node.get("content") or {}
        trace = _trace_from_path(tree, path)
        if start_node.get("node_type") == SARNodeType.SUB_QA.value:
            trace = trace + [
                f"Q: {start_content.get('sub_question', '')}\nA: {start_content.get('answer', '')}"
            ]

        socratic_out = await socratic_graph.ainvoke(
            {
                "question": state["question"],
                "max_depth": sar_config.search.max_depth,
                "depth": 0,
                "text_memory": fork_memory,
                "reasoning_trace": trace,
                "iteration_history": [],
            }
        )

        rollout_distilled: List[str] = []
        for step in socratic_out.get("iteration_history") or []:
            rollout_distilled.extend(step.get("distilled_info") or [])
        rollout_distilled = dedup_preserve_order(rollout_distilled)

        terminal_answer = str(socratic_out.get("final_answer") or "")
        rollout_trace = list(trace) + [
            f"Final Answer: {socratic_out.get('detailed_answer') or terminal_answer}"
        ]

        new_path = path + [start_id]
        return {
            "current_path": new_path,
            "simulation_result": {
                "terminal_answer": terminal_answer,
                "rollout_distilled": rollout_distilled,
                "rollout_trace": rollout_trace,
            },
        }

    async def evaluate(state: MCTSState, config: RunnableConfig) -> Dict[str, Any]:  # noqa: ARG001
        tree = dict(state.get("tree") or {})
        sim = dict(state.get("simulation_result") or {})
        terminal_answer = str(sim.get("terminal_answer") or "")
        if not terminal_answer:
            path = state.get("current_path") or []
            if path:
                content = (tree.get(path[-1]) or {}).get("content") or {}
                terminal_answer = str(content.get("answer") or content.get("detailed_answer") or "")

        configurable = get_configurable(config)
        # Use current_path (selected_path + the expanded child chosen for rollout) so
        # the freshly expanded persistent node is path-judged in its creating iteration.
        # All elements are persistent tree nodes; the rollout terminal is not appended.
        judged_path = list(state.get("current_path") or state.get("selected_path") or [])
        details = await evaluate_mcts_iteration(
            registry,
            sar_config,
            question=state["question"],
            tree=tree,
            selected_path=judged_path,
            terminal_answer=terminal_answer,
            golden_answer=configurable.get("golden_answer"),
        )
        tree.update(details.get("tree_updates") or {})
        return {
            "tree": tree,
            "reward": details["reward"],
            "judge_critiques": details.get("judge_critiques") or [],
            "last_factuality": details.get("last_factuality", 0.0),
        }

    async def backprop(state: MCTSState) -> Dict[str, Any]:
        tree = dict(state.get("tree") or {})
        path = state.get("current_path") or []
        reward = float(state.get("reward", 0.0) or 0.0)
        updates: Dict[str, MCTSTreeNode] = {}

        for nid in path:
            node = dict(updates.get(nid) or tree.get(nid) or {})
            node["visits"] = int(node.get("visits") or 0) + 1
            node["value"] = float(node.get("value") or 0.0) + reward
            updates[nid] = node

        best = float(state.get("best_value", 0.0) or 0.0)
        no_imp = int(state.get("iterations_without_improvement", 0) or 0)
        if reward > best:
            best, no_imp = reward, 0
        else:
            no_imp += 1

        return {
            "tree": updates,
            "iteration": int(state.get("iteration", 0) or 0) + 1,
            "best_value": best,
            "iterations_without_improvement": no_imp,
        }

    async def mem_update(state: MCTSState) -> Dict[str, Any]:
        reward_val = float(state.get("reward", 0.0) or 0.0)
        factuality = float(state.get("last_factuality", 0.0) or 0.0)
        if not should_commit_memory(
            sar_config,
            reward=reward_val,
            last_factuality=factuality,
        ):
            warn_explicit(
                "Memory commit gate blocked; rollout facts were not written to global text_memory.",
                component="mcts.mem_update",
                details={
                    "gate": sar_config.search.memory_commit_gate,
                    "threshold": sar_config.search.memory_commit_threshold,
                    "reward": reward_val,
                    "last_factuality": factuality,
                },
            )
            return {"judge_critiques": state.get("judge_critiques") or []}

        tree = state.get("tree") or {}
        distilled: List[str] = []
        for nid in state.get("expanded_node_ids") or []:
            content = (tree.get(nid) or {}).get("content") or {}
            distilled.extend(content.get("distilled_info") or [])

        sim = state.get("simulation_result") or {}
        distilled.extend(sim.get("rollout_distilled") or [])
        distilled = dedup_preserve_order(distilled)
        if not distilled and not (state.get("judge_critiques") or []):
            warn_explicit(
                "mem_update skipped: no distilled_info or judge_critiques to merge.",
                component="mcts.mem_update",
            )
            return {}

        # Thread the expanded node's real (sub_question, answer) so the reconciler sees
        # the latest QA step; None when the leaf carries no QA (e.g. a SYNTHESIS node).
        qa_pair = None
        path = state.get("current_path") or []
        if path:
            leaf_content = (tree.get(path[-1]) or {}).get("content") or {}
            q = str(leaf_content.get("sub_question") or "")
            a = str(leaf_content.get("answer") or leaf_content.get("detailed_answer") or "")
            if q or a:
                qa_pair = (q, a)

        mem_out = await memory_update_graph.ainvoke(
            {
                "user_question": state["question"],
                "current_memory": list(state.get("text_memory") or []),
                "distilled_info": distilled,
                "qa_pair": qa_pair,
                "intermediate_conclusions": [],
                "judge_critiques": list(state.get("judge_critiques") or []),
            }
        )
        return {
            "text_memory": dedup_preserve_order(mem_out.get("updated_memory", state.get("text_memory"))),
        }

    async def synthesize(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        candidates: List[str] = []
        scores: List[float] = []

        for node in tree.values():
            if node.get("node_type") != SARNodeType.FINAL_ANSWER.value:
                continue
            content = node.get("content") or {}
            text = str(content.get("detailed_answer") or content.get("answer") or "")
            visits = int(node.get("visits") or 0)
            value = float(node.get("value") or 0.0)
            score = (value / visits) if visits > 0 else 0.0
            candidates.append(text)
            scores.append(score)

        memory = list(state.get("text_memory") or [])
        if not candidates:
            warn_explicit(
                "No FINAL_ANSWER nodes in tree; synthesizing from text_memory only.",
                component="mcts.synthesize",
                details={"memory_facts": len(memory)},
            )
            final_answer = format_memory(memory) if memory else "No answer available."
            return {"final_answer": final_answer, "detailed_answer": final_answer}

        reasoning_paths = _format_synthesis_paths(candidates, scores)
        # Memory-aware synthesis (§5/§8.5): the legacy SYNTHESIZE_FINAL_ANSWER_PROMPT has
        # no context slot, so fold the working memory into the existing reasoning_paths
        # field rather than altering the trained prompt shape.
        memory_block = format_memory(memory)
        if memory_block == "Not provided":
            memory_block = format_reasoning_trace(
                _trace_from_path(tree, state.get("selected_path") or [])
            )
        if memory_block and memory_block != "Not provided":
            reasoning_paths = (
                "Working memory facts (grounding evidence — prefer claims consistent with these):\n"
                f"{memory_block}\n\n"
                f"Candidate reasoning paths:\n{reasoning_paths}"
            )

        try:
            out, _ = await execute_role_lc(
                registry,
                ANSWER_SYNTHESIZER,
                AnswerSynthesizerInput(question=state["question"], reasoning_paths=reasoning_paths),
            )
            final = (getattr(out, "final_answer", "") or "").strip()
            detailed = (getattr(out, "final_reasoning_path", "") or final).strip()
            if final:
                return {"final_answer": final, "detailed_answer": detailed}
            warn_explicit(
                "ANSWER_SYNTHESIZER returned an empty final_answer; falling back to majority vote.",
                component="mcts.synthesize",
                details={"num_candidates": len(candidates)},
            )
        except Exception as exc:
            warn_explicit(
                "ANSWER_SYNTHESIZER failed; falling back to majority vote.",
                component="mcts.synthesize",
                details={"error": str(exc)},
            )

        try:
            out, _ = await execute_role_lc(
                registry,
                MAJORITY_VOTER,
                MajorityVoterInput(
                    question=state["question"],
                    answers="\n".join(f"- {c}" for c in candidates),
                ),
            )
            final = (getattr(out, "final_answer", "") or candidates[0]).strip()
            return {"final_answer": final, "detailed_answer": final}
        except Exception as exc:
            best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
            warn_explicit(
                "MAJORITY_VOTER failed; using highest visit-value FINAL_ANSWER candidate.",
                component="mcts.synthesize",
                details={"error": str(exc), "candidate_index": best_idx},
            )
            return {"final_answer": candidates[best_idx], "detailed_answer": candidates[best_idx]}

    def route_after_iteration(state: MCTSState) -> str:
        iteration = int(state.get("iteration", 0) or 0)
        max_iters = int(state.get("num_rollouts", sar_config.search.num_rollouts) or sar_config.search.num_rollouts)
        if iteration >= max_iters:
            return "synthesize"
        if float(state.get("best_value", 0.0) or 0.0) >= sar_config.search.high_confidence:
            return "synthesize"
        if int(state.get("iterations_without_improvement", 0) or 0) >= sar_config.search.patience:
            return "synthesize"
        return "select"

    workflow.add_node("select", select)
    workflow.add_node("expand", expand)
    workflow.add_node("simulate", simulate)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("backprop", backprop)
    workflow.add_node("mem_update", mem_update)
    workflow.add_node("synthesize", synthesize)

    workflow.add_edge(START, "select")
    workflow.add_edge("select", "expand")
    workflow.add_edge("expand", "simulate")
    workflow.add_edge("simulate", "evaluate")
    workflow.add_edge("evaluate", "backprop")
    workflow.add_edge("backprop", "mem_update")
    workflow.add_conditional_edges("mem_update", route_after_iteration, ["select", "synthesize"])
    workflow.add_edge("synthesize", END)

    return workflow.compile()
