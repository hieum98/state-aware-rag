"""Phase 2 SAR action helpers (A1/A5) for Socratic inference."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from ..config import SARConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import (
    ANSWER_GENERATOR,
    FINALIZER,
    QUESTION_REPHRASER,
    SELF_CORRECTOR,
    SUBQUESTION_GENERATOR,
    SYNTHESIZER,
    AnswerGeneratorInput,
    FinalizerInput,
    QuestionRephraserInput,
    SelfCorrectorInput,
    SubquestionGeneratorInput,
    SynthesizerInput,
)


def confidence_to_score(confidence: str | float | int | None) -> float:
    if isinstance(confidence, (int, float)):
        return max(0.0, min(1.0, float(confidence)))
    value = (confidence or "low").lower()
    if value == "high":
        return 1.0
    if value == "medium":
        return 0.5
    return 0.1


def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def format_memory(memory: Iterable[str] | None) -> str:
    facts = dedup_preserve_order(memory or [])
    return "\n".join(f"- {fact}" for fact in facts) if facts else "Not provided"


def format_reasoning_trace(trace: Iterable[str] | None) -> str:
    steps = dedup_preserve_order(trace or [])
    return "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, 1)) if steps else "Not provided"


def format_context(*, memory: Iterable[str] | None = None, reasoning_trace: Iterable[str] | None = None, distilled_info: Iterable[str] | None = None) -> str:
    parts: List[str] = []
    mem = format_memory(memory)
    if mem != "Not provided":
        parts.append(f"Working memory:\n{mem}")
    trace = format_reasoning_trace(reasoning_trace)
    if trace != "Not provided":
        parts.append(f"Reasoning trace:\n{trace}")
    info = format_memory(distilled_info)
    if info != "Not provided":
        parts.append(f"Newly retrieved/distilled information:\n{info}")
    return "\n\n".join(parts) if parts else "Not provided"


async def generate_subquestion(registry: RoleModelRegistry, question: str, memory: List[str], reasoning_trace: List[str]):
    context = format_context(memory=memory, reasoning_trace=reasoning_trace)
    out, _ = await execute_role_lc(
        registry,
        SUBQUESTION_GENERATOR,
        SubquestionGeneratorInput(question=question, context=context),
    )
    return out


async def answer_subquestion(
    registry: RoleModelRegistry,
    config: SARConfig,
    consolidate_graph,
    *,
    user_question: str,
    sub_question: str,
    text_memory: List[str],
    reasoning_trace: List[str],
) -> Tuple[dict, List[str]]:
    explore = await consolidate_graph.ainvoke(
        {"query": sub_question, "user_question": user_question, "mode": "explore"}
    )
    explored_info = dedup_preserve_order(explore.get("distilled_info", []))

    reflected_info: List[str] = []
    if len(text_memory) > config.search.reflect_threshold_facts:
        reflect = await consolidate_graph.ainvoke(
            {
                "query": sub_question,
                "user_question": user_question,
                "mode": "reflect",
                "text_memory": text_memory,
            }
        )
        reflected_info = dedup_preserve_order(reflect.get("distilled_info", []))

    context_memory = reflected_info if reflected_info else text_memory
    answer_context = format_context(
        memory=context_memory,
        reasoning_trace=reasoning_trace,
        distilled_info=explored_info,
    )
    answer, _ = await execute_role_lc(
        registry,
        ANSWER_GENERATOR,
        AnswerGeneratorInput(question=sub_question, context=answer_context),
    )

    answer_text = (getattr(answer, "answer", "") or getattr(answer, "detailed_answer", "") or getattr(answer, "reasoning", "")).strip()
    detailed = (getattr(answer, "detailed_answer", "") or answer_text).strip()
    reasoning = (getattr(answer, "reasoning", "") or detailed or answer_text).strip()
    confidence = confidence_to_score(getattr(answer, "confidence", "low"))

    content = {
        "node_type": "SUB_QA",
        "sub_question": sub_question,
        "answer": answer_text,
        "detailed_answer": detailed,
        "reasoning": reasoning,
        "confidence": confidence,
        "distilled_info": explored_info,
        "reflected_info": reflected_info,
    }
    return content, explored_info


async def synthesize_step(
    registry: RoleModelRegistry,
    *,
    user_question: str,
    text_memory: List[str],
    reasoning_trace: List[str],
) -> Tuple[dict, List[str]]:
    context = format_context(memory=text_memory, reasoning_trace=reasoning_trace)
    out, _ = await execute_role_lc(
        registry,
        SYNTHESIZER,
        SynthesizerInput(question=user_question, context=context),
        n=1,
    )
    synthesis = (getattr(out, "synthesis", "") or getattr(out, "reasoning", "") or "").strip()
    answerable = bool(getattr(out, "answerable_main_question", False))
    confidence = confidence_to_score(getattr(out, "confidence", "low"))
    if answerable:
        content = {
            "node_type": "FINAL_ANSWER",
            "answer": synthesis,
            "detailed_answer": synthesis,
            "reasoning": synthesis,
            "confidence": confidence,
            "distilled_info": [],
        }
    else:
        content = {
            "node_type": "SYNTHESIS",
            "synthesis": synthesis,
            "reasoning": synthesis,
            "confidence": confidence,
            "distilled_info": [],
        }
    return content, []


async def refine_step(
    registry: RoleModelRegistry,
    config: SARConfig,
    consolidate_graph,
    *,
    user_question: str,
    sub_question: str,
    sub_answer: str,
    text_memory: List[str],
    reasoning_trace: List[str],
) -> Tuple[dict, List[str]]:
    verify_query = f"Verify and improve this answer: {sub_answer}"
    explore = await consolidate_graph.ainvoke(
        {"query": verify_query, "user_question": user_question, "mode": "explore"}
    )
    explored_info = dedup_preserve_order(explore.get("distilled_info", []))

    reflected_info: List[str] = []
    if len(text_memory) > config.search.reflect_threshold_facts:
        reflect = await consolidate_graph.ainvoke(
            {
                "query": verify_query,
                "user_question": user_question,
                "mode": "reflect",
                "text_memory": text_memory,
            }
        )
        reflected_info = dedup_preserve_order(reflect.get("distilled_info", []))

    context = format_context(
        memory=reflected_info or text_memory,
        reasoning_trace=reasoning_trace,
        distilled_info=explored_info,
    )
    corrected, _ = await execute_role_lc(
        registry,
        SELF_CORRECTOR,
        SelfCorrectorInput(question=sub_question, answer=sub_answer, context=context),
    )
    reanswer = (getattr(corrected, "reanswer", "") or sub_answer).strip()
    reasoning = (getattr(corrected, "reasoning", "") or reanswer).strip()
    confidence = confidence_to_score(getattr(corrected, "confidence", "low"))
    content = {
        "node_type": "SELF_CORRECTED",
        "sub_question": sub_question,
        "answer": reanswer,
        "reasoning": reasoning,
        "confidence": confidence,
        "distilled_info": explored_info,
    }
    return content, explored_info


async def rephrase_step(
    registry: RoleModelRegistry,
    *,
    question: str,
) -> Tuple[dict, List[str]]:
    out, _ = await execute_role_lc(
        registry,
        QUESTION_REPHRASER,
        QuestionRephraserInput(question=question),
    )
    rephrased = (getattr(out, "rephrased_question", "") or question).strip()
    content = {
        "node_type": "REPHRASED_QUESTION",
        "rephrased_question": rephrased,
        "distilled_info": [],
    }
    return content, []


async def conclude_question(
    registry: RoleModelRegistry,
    consolidate_graph,
    *,
    user_question: str,
    text_memory: List[str],
    reasoning_trace: List[str],
) -> Tuple[dict, List[str]]:
    explore = await consolidate_graph.ainvoke(
        {"query": user_question, "user_question": user_question, "mode": "explore"}
    )
    explored_info = dedup_preserve_order(explore.get("distilled_info", []))
    context = format_context(
        memory=text_memory,
        reasoning_trace=reasoning_trace,
        distilled_info=explored_info,
    )
    final, _ = await execute_role_lc(
        registry,
        FINALIZER,
        FinalizerInput(question=user_question, context=context),
    )

    answer = (getattr(final, "answer", "") or getattr(final, "detailed_answer", "") or getattr(final, "reasoning", "")).strip()
    detailed = (getattr(final, "detailed_answer", "") or answer).strip()
    reasoning = (getattr(final, "reasoning", "") or detailed or answer).strip()
    confidence = confidence_to_score(getattr(final, "confidence", "low"))
    content = {
        "node_type": "FINAL_ANSWER",
        "answer": answer,
        "detailed_answer": detailed,
        "reasoning": reasoning,
        "confidence": confidence,
        "distilled_info": explored_info,
    }
    return content, explored_info
