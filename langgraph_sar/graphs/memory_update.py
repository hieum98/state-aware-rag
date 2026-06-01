"""MemoryUpdateGraph for ``langgraph_sar``.

The updater is intentionally still one EXTRACTOR call over a reflection document,
matching the legacy prompt shape while asking the extractor to reconcile memory
instead of append facts blindly.
"""

from __future__ import annotations

from typing import Iterable, List

from langgraph.graph import END, START, StateGraph

from ..config import SARConfig
from ..explicit import warn_explicit
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import EXTRACTOR, ExtractorInput
from .state import MemoryUpdateState


def _dedup_preserve_order(items: Iterable[str], *, limit: int | None = None) -> List[str]:
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


def _format_bullets(items: Iterable[str]) -> str:
    facts = _dedup_preserve_order(items)
    return "\n".join(f"- {fact}" for fact in facts)


def _qa_pair_text(state: MemoryUpdateState) -> str:
    qa_pair = state.get("qa_pair")
    if not qa_pair:
        return ""
    question, answer = qa_pair
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if not question and not answer:
        return ""
    return f"Question: {question}\nAnswer: {answer}"


def _build_update_context(state: MemoryUpdateState, config: SARConfig) -> str:
    parts: List[str] = [
        "Memory update task: reconcile the evidence below into a compact set of "
        "self-contained facts useful for answering the user question. Merge duplicates, "
        "drop superseded or contradicted claims, and keep provenance/confidence caveats "
        "attached to the affected fact when warranted."
    ]

    current_memory = _format_bullets(state.get("current_memory", []))
    if current_memory:
        parts.append(f"Current memory knowledge:\n{current_memory}")

    qa_text = _qa_pair_text(state)
    if qa_text:
        parts.append(f"Latest QA step:\n{qa_text}")

    distilled_info = _format_bullets(state.get("distilled_info", []))
    if distilled_info:
        parts.append(f"Information from external KB:\n{distilled_info}")

    conclusions = _format_bullets(state.get("intermediate_conclusions", []))
    if conclusions:
        parts.append(f"Intermediate conclusions:\n{conclusions}")

    critiques = _format_bullets(state.get("judge_critiques", []))
    if critiques:
        parts.append(f"Judge critiques to account for:\n{critiques}")

    parts.append(f"Memory fact budget: keep at most {config.memory.max_memory_facts} facts.")
    if config.memory.confidence_tags:
        parts.append(
            "When a fact is uncertain or contested, preserve that caveat inline with the fact."
        )

    return "\n\n----------\n\n".join(part.strip() for part in parts if part.strip())


def build_memory_update_graph(registry: RoleModelRegistry, config: SARConfig):
    workflow = StateGraph(MemoryUpdateState)

    async def update(state: MemoryUpdateState):
        prior_memory = _dedup_preserve_order(state.get("current_memory", []))
        has_new_evidence = (
            any(
                state.get(key)
                for key in ("distilled_info", "intermediate_conclusions", "judge_critiques")
            )
            or bool(_qa_pair_text(state))
        )
        if not has_new_evidence:
            return {"updated_memory": prior_memory}

        context = _build_update_context(state, config)
        if not context.strip():
            return {"updated_memory": prior_memory}

        # Reconciliation (dedup / contradiction-resolution) must see the whole context
        # in a single EXTRACTOR call. If it exceeds the extractor budget, truncate (the
        # tail holds the lowest-priority sections) and warn, rather than chunk-and-union
        # which would make cross-chunk reconciliation impossible.
        max_chars = max(1, int(config.memory.extractor_max_input_chars))
        if len(context) > max_chars:
            warn_explicit(
                "Reconciliation context exceeds extractor_max_input_chars; truncating the "
                "tail (lowest-priority evidence) to keep a single-call reconciliation.",
                component="memory_update",
                details={
                    "context_chars": len(context),
                    "max_chars": max_chars,
                    "user_question": state.get("user_question", ""),
                },
            )
            context = context[:max_chars]

        result, _ = await execute_role_lc(
            registry,
            EXTRACTOR,
            ExtractorInput(question=state["user_question"], document=context),
            n=1,
        )
        results = result if isinstance(result, list) else [result]

        updated = _dedup_preserve_order(
            info
            for result in results
            if getattr(result, "decision", None) == "relevant"
            for info in getattr(result, "information", [])
        )
        if not updated:
            warn_explicit(
                "Extractor returned no relevant facts; keeping previous memory unchanged.",
                component="memory_update",
                details={
                    "user_question": state.get("user_question", ""),
                    "prior_memory_size": len(prior_memory),
                },
            )
            updated = prior_memory

        return {"updated_memory": updated[: config.memory.max_memory_facts]}

    workflow.add_node("update", update)
    workflow.add_edge(START, "update")
    workflow.add_edge("update", END)
    return workflow.compile()
