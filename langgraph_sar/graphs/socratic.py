"""Phase 2 SocraticGraph for linear State-Aware RAG inference."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..config import SARConfig
from ..llm import RoleModelRegistry
from .actions import (
    answer_subquestion,
    conclude_question,
    dedup_preserve_order,
    generate_subquestion,
)
from .consolidate import build_consolidate_graph
from .memory_update import build_memory_update_graph
from .reward import evaluate_socratic_result, get_configurable


class SocraticState(TypedDict, total=False):
    question: str
    max_depth: int
    depth: int
    is_answerable: bool
    sub_question: str
    reasoning_trace: List[str]
    text_memory: List[str]
    iteration_history: List[Dict[str, Any]]
    final_answer: str
    detailed_answer: str
    reward: float
    reward_details: Dict[str, Any]


def _initial_list(state: SocraticState, key: str) -> List[Any]:
    return list(state.get(key, []))


def build_socratic_graph(
    registry: RoleModelRegistry,
    config: SARConfig,
    *,
    include_evaluate: bool = True,
):
    """Build the linear Socratic graph.

    ``include_evaluate=False`` produces a **rollout-only** variant (``conclude → END``)
    used by MCTS ``simulate``: the rollout only needs ``final_answer``/``iteration_history``
    and computing the discarded dual reward per rollout would defeat the §11.8 per-node
    ``R_p`` memoization (one extra OUTCOME_JUDGE + per-step PATH_JUDGE per rollout).
    """
    sar_config = config
    consolidate_graph = build_consolidate_graph(registry, sar_config)
    memory_update_graph = build_memory_update_graph(registry, sar_config)
    workflow = StateGraph(SocraticState)

    async def gen_subq(state: SocraticState):
        depth = int(state.get("depth", 0))
        out = await generate_subquestion(
            registry,
            state["question"],
            _initial_list(state, "text_memory"),
            _initial_list(state, "reasoning_trace"),
        )
        return {
            "depth": depth,
            "max_depth": int(state.get("max_depth", sar_config.search.max_depth)),
            "is_answerable": bool(getattr(out, "answerable_main_question", False)),
            "sub_question": (getattr(out, "subquestion", "") or "").strip(),
            "reasoning_trace": _initial_list(state, "reasoning_trace"),
            "text_memory": _initial_list(state, "text_memory"),
            "iteration_history": _initial_list(state, "iteration_history"),
        }

    def route_after_subq(state: SocraticState) -> str:
        depth = int(state.get("depth", 0))
        max_depth = int(state.get("max_depth", sar_config.search.max_depth))
        if state.get("is_answerable") or depth >= max_depth or not state.get("sub_question"):
            return "conclude"
        return "answer"

    async def answer(state: SocraticState):
        memory = _initial_list(state, "text_memory")
        trace = _initial_list(state, "reasoning_trace")
        content, explored_info = await answer_subquestion(
            registry,
            config,
            consolidate_graph,
            user_question=state["question"],
            sub_question=state["sub_question"],
            text_memory=memory,
            reasoning_trace=trace,
        )
        mem_out = await memory_update_graph.ainvoke(
            {
                "user_question": state["question"],
                "current_memory": memory,
                "distilled_info": explored_info,
                "qa_pair": (state["sub_question"], content["answer"]),
                "intermediate_conclusions": [],
                "judge_critiques": [],
            }
        )
        updated_memory = dedup_preserve_order(mem_out.get("updated_memory", memory))
        trace_entry = f"Q: {state['sub_question']}\nA: {content['answer']}"
        history_entry = {**content, "trace_entry": trace_entry, "depth": int(state.get("depth", 0)) + 1}
        return {
            "depth": int(state.get("depth", 0)) + 1,
            "reasoning_trace": trace + [trace_entry],
            "text_memory": updated_memory,
            "iteration_history": _initial_list(state, "iteration_history") + [history_entry],
        }

    async def conclude(state: SocraticState):
        memory = _initial_list(state, "text_memory")
        trace = _initial_list(state, "reasoning_trace")
        content, explored_info = await conclude_question(
            registry,
            consolidate_graph,
            user_question=state["question"],
            text_memory=memory,
            reasoning_trace=trace,
        )
        history = _initial_list(state, "iteration_history") + [{**content, "trace_entry": f"Final Answer: {content['detailed_answer']}"}]
        return {
            "final_answer": content["answer"],
            "detailed_answer": content["detailed_answer"],
            "iteration_history": history,
        }

    async def evaluate(state: SocraticState, config: RunnableConfig):
        configurable = get_configurable(config)
        details = await evaluate_socratic_result(
            registry,
            sar_config,
            question=state["question"],
            final_answer=state.get("final_answer", ""),
            iteration_history=_initial_list(state, "iteration_history"),
            golden_answer=configurable.get("golden_answer"),
        )
        return {"reward": details["reward"], "reward_details": details}

    workflow.add_node("gen_subq", gen_subq)
    workflow.add_node("answer", answer)
    workflow.add_node("conclude", conclude)

    workflow.add_edge(START, "gen_subq")
    workflow.add_conditional_edges("gen_subq", route_after_subq, {"answer": "answer", "conclude": "conclude"})
    workflow.add_edge("answer", "gen_subq")

    if include_evaluate:
        workflow.add_node("evaluate", evaluate)
        workflow.add_edge("conclude", "evaluate")
        workflow.add_edge("evaluate", END)
    else:
        workflow.add_edge("conclude", END)

    return workflow.compile()
