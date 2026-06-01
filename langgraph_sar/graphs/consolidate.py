"""ConsolidateGraph for ``langgraph_sar``.

Phase 1 implements the paper's ``E_consolidate`` primitive on LangGraph:
retrieve (or reflect over memory), run the extractor per document, then keep only
per-document ``decision == relevant`` facts. Query expansion and document joining
are explicit cost/recall knobs; the default path uses the sub-question directly.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List

from langgraph.graph import END, START, StateGraph

from ..config import SARConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import EXTRACTOR, QUERY_GENERATOR, ExtractorInput, QueryGeneratorInput
from ..tools import corpus_search
from .state import ConsolidateState


def _dedup_preserve_order(items: Iterable[str], *, limit: int | None = None) -> List[str]:
    """Return non-empty strings deduped by exact normalized text, preserving order."""
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _split_oversized_doc(doc: str, max_chars: int) -> List[str]:
    """Split one oversized document into extractor-sized chunks; never merge docs."""
    text = doc.strip()
    if not text:
        return []
    max_chars = max(1, max_chars)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _join_docs_into_blocks(docs: Iterable[str], max_chars: int) -> List[str]:
    """Optional OOD cost lever: merge docs into char-budget blocks."""
    max_chars = max(1, max_chars)
    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for doc in docs:
        chunks = _split_oversized_doc(doc, max_chars)
        for chunk in chunks:
            extra_len = len(chunk) + (2 if current else 0)
            if current and current_len + extra_len > max_chars:
                blocks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(chunk)
            current_len += extra_len

    if current:
        blocks.append("\n\n".join(current))
    return blocks


def _format_memory(memory: Iterable[str]) -> str:
    facts = _dedup_preserve_order(memory)
    return "\n".join(f"- {fact}" for fact in facts)


def build_consolidate_graph(registry: RoleModelRegistry, config: SARConfig):
    workflow = StateGraph(ConsolidateState)

    async def gen_queries(state: ConsolidateState):
        out, _ = await execute_role_lc(
            registry,
            QUERY_GENERATOR,
            QueryGeneratorInput(question=state["query"]),
        )
        queries = _dedup_preserve_order(getattr(out, "queries", []) or [state["query"]])
        return {"queries": queries or [state["query"]]}

    async def retrieve(state: ConsolidateState):
        queries = _dedup_preserve_order(state.get("queries") or [state["query"]])
        if not queries:
            return {"retrieved_docs": []}

        tasks = [
            corpus_search.ainvoke({"query": query, "top_k": config.search.top_k})
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        docs = _dedup_preserve_order(
            (doc for query_docs in results for doc in query_docs),
            limit=config.search.top_k,
        )
        return {"retrieved_docs": docs}

    async def extract(state: ConsolidateState):
        mode = state.get("mode", "explore")
        max_chars = config.memory.extractor_max_input_chars

        if mode == "reflect":
            memory_doc = _format_memory(state.get("text_memory", []))
            docs_to_process = _split_oversized_doc(memory_doc, max_chars)
        else:
            retrieved_docs = _dedup_preserve_order(state.get("retrieved_docs", []), limit=config.search.top_k)
            if config.search.extractor_join:
                docs_to_process = _join_docs_into_blocks(retrieved_docs, max_chars)
            else:
                docs_to_process = [
                    chunk
                    for doc in retrieved_docs
                    for chunk in _split_oversized_doc(doc, max_chars)
                ]

        if not docs_to_process:
            return {"distilled_info": []}

        # Use q_i as the extractor question: for A1 this is the sub-question, and
        # for A5 it is the original user question. ``user_question`` remains the
        # outer anchor available to callers but must not replace the local query.
        extractor_inputs = [
            ExtractorInput(question=state["query"], document=doc)
            for doc in docs_to_process
        ]
        results, _ = await execute_role_lc(registry, EXTRACTOR, extractor_inputs, n=1)
        if not isinstance(results, list):
            results = [results]

        distilled = _dedup_preserve_order(
            info
            for result in results
            if getattr(result, "decision", None) == "relevant"
            for info in getattr(result, "information", [])
        )
        return {"distilled_info": distilled}

    def route_from_start(state: ConsolidateState) -> str:
        if state.get("mode", "explore") == "reflect":
            return "extract"
        if config.search.query_expansion:
            return "gen_queries"
        return "retrieve"

    workflow.add_node("gen_queries", gen_queries)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("extract", extract)

    workflow.add_conditional_edges(
        START,
        route_from_start,
        {"extract": "extract", "gen_queries": "gen_queries", "retrieve": "retrieve"},
    )
    workflow.add_edge("gen_queries", "retrieve")
    workflow.add_edge("retrieve", "extract")
    workflow.add_edge("extract", END)

    return workflow.compile()
