"""State schemas and reducers for langgraph_sar."""

from typing import Annotated, Any, List, Optional, Tuple, TypedDict

def append_or_clear(left: Optional[List[str]], right: Any) -> List[str]:
    """Reducer for lists that can be appended to or completely cleared.
    If right is the sentinel "CLEAR", returns empty list.
    Otherwise appends right to left. right can be a single item or a list.
    """
    left = left or []
    if right == "CLEAR":
        return []
    if isinstance(right, list):
        return left + right
    return left + [right]

class ConsolidateState(TypedDict, total=False):
    query: str                       # q_i (sub-question or user question)
    user_question: str               # x (anchor for relevance lens)
    text_memory: List[str]           # M_{i-1} (for reflect mode)
    mode: str                        # "explore" (retrieve) | "reflect" (memory-only)
    
    # scratch
    queries: List[str]               # QUERY_GENERATOR output
    retrieved_docs: Annotated[List[str], append_or_clear]   # truncated to search.top_k
    
    # output
    distilled_info: List[str]        # I_i  (relevant==True extractions, deduped)

class MemoryUpdateState(TypedDict, total=False):
    user_question: str               # x
    current_memory: List[str]        # M_{i-1}
    distilled_info: List[str]        # I_i
    qa_pair: Tuple[str, str]         # (q_i, r_i)
    intermediate_conclusions: List[str]   # optional (A2 path)
    judge_critiques: List[str]       # optional (MCTS)
    
    # output
    updated_memory: List[str]        # M_i
