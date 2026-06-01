"""MCTS state schemas and reducers for ``langgraph_sar`` (IMPLEMENTATION_PLAN §7.5)."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

class SARNodeType(str, Enum):
    USER_QUESTION = "USER_QUESTION"
    SUB_QA = "SUB_QA"
    SYNTHESIS = "SYNTHESIS"
    SELF_CORRECTED = "SELF_CORRECTED"
    REPHRASED_QUESTION = "REPHRASED_QUESTION"
    FINAL_ANSWER = "FINAL_ANSWER"


# Action-type priors for PUCT (§11.11). USER_QUESTION is never a UCB candidate.
NODE_TYPE_PRIOR: Dict[SARNodeType, float] = {
    SARNodeType.SUB_QA: 0.60,
    SARNodeType.SYNTHESIS: 0.45,
    SARNodeType.SELF_CORRECTED: 0.50,
    SARNodeType.REPHRASED_QUESTION: 0.40,
    SARNodeType.FINAL_ANSWER: 0.30,
}


def dict_merge(
    left: Dict[str, "MCTSTreeNode"],
    right: Dict[str, "MCTSTreeNode"],
) -> Dict[str, "MCTSTreeNode"]:
    """Rightward dict union; right wins on key collisions."""
    return {**(left or {}), **(right or {})}


class MCTSTreeNode(TypedDict, total=False):
    node_id: str
    parent_id: Optional[str]
    children_ids: List[str]
    node_type: str
    content: Dict[str, Any]
    visits: int
    value: float
    prior: float
    r_p: Optional[float]
    r_p_details: Optional[Dict[str, Any]]
    r_o: Optional[float]


class MCTSState(TypedDict, total=False):
    question: str
    num_rollouts: int
    iteration: int
    max_depth: int
    tree: Annotated[Dict[str, MCTSTreeNode], dict_merge]
    root_id: str
    selected_path: List[str]
    current_path: List[str]
    expanded_node_ids: List[str]
    simulation_result: Dict[str, Any]
    reward: float
    last_factuality: float
    judge_critiques: List[str]
    text_memory: List[str]
    best_value: float
    iterations_without_improvement: int
    final_answer: str
    detailed_answer: str


def new_node_id() -> str:
    return uuid.uuid4().hex[:8]


def make_tree_node(
    node_type: SARNodeType,
    parent_id: Optional[str],
    content: Dict[str, Any],
    *,
    use_prior: bool = True,
) -> MCTSTreeNode:
    prior = NODE_TYPE_PRIOR.get(node_type, 1.0) if use_prior else 1.0
    return {
        "node_id": new_node_id(),
        "parent_id": parent_id,
        "children_ids": [],
        "node_type": node_type.value,
        "content": content,
        "visits": 0,
        "value": 0.0,
        "prior": prior,
        "r_p": None,
        "r_p_details": None,
        "r_o": None,
    }


def is_terminal(node: MCTSTreeNode) -> bool:
    return node.get("node_type") == SARNodeType.FINAL_ANSWER.value


def node_type_str(node: MCTSTreeNode) -> str:
    value = node.get("node_type", "")
    return getattr(value, "value", value)


__all__ = [
    "SARNodeType",
    "NODE_TYPE_PRIOR",
    "dict_merge",
    "MCTSTreeNode",
    "MCTSState",
    "new_node_id",
    "make_tree_node",
    "is_terminal",
    "node_type_str",
]
