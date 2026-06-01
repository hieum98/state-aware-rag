"""Role registry for ``langgraph_sar``.

Each :class:`Role` bundles a name, a **full legacy prompt template**, a Pydantic
input model (whose fields fill the template's ``{placeholders}``), and the legacy
Pydantic output model. ``llm.RoleModelRegistry`` maps the role name to a tier
(``generator`` / ``extractor`` / ``evaluator``); see ``config.py``.

**Prompt fidelity (IMPLEMENTATION_PLAN.md §4, §11.1).** Prompts and output models
are *imported verbatim* from ``state_aware_rag/agents/prompts/`` — not copied —
so the port stays byte-identical to what the RL-trained extractor saw. The legacy
agents send each prompt as a **single ``{'role': 'user'}`` message** with no system
turn (``state_aware_rag/agents/roles/extractor.py:54``). ``llm.format_messages``
reproduces that exactly: one ``HumanMessage`` holding ``template.format(**fields)``.
Splitting into system+user would add a chat-template turn the trained checkpoint
never saw — the train/inference misalignment §11.1 rejects.

``golden_answer`` never reaches a generator/extractor input model: only
``OutcomeJudgeInput.correct_answer`` carries a reference, and that role is on the
evaluator/reward path (§11.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import pydantic

# --- Legacy prompt templates + output models (imported, never copied) ----------
from state_aware_rag.agents.prompts.decompose_and_answer import (
    GENERATE_SUBQUESTION_PROMPT,
    ANSWER_PROMPT,
    GENERATE_QUERIES_FOR_RETRIEVER,
    SubquestionOutput,
    AnswerOutput,
    QueriesGenerationOutput,
)
from state_aware_rag.agents.prompts.finalize import FINALIZE_PROMPT, FinalizeOutput
from state_aware_rag.agents.prompts.self_correct import (
    SELF_CORRECT_PROMPT,
    SelfCorrectOutput,
)
from state_aware_rag.agents.prompts.rephase_question import (
    REPHRASE_QUESTION_PROMPT,
    RephraseQuestionOutput,
)
from state_aware_rag.agents.prompts.synthesize import SYNTHESIZE_PROMPT, SynthesizeOutput
from state_aware_rag.agents.prompts.extract import EXTRACT_PROMPT, ExtractOutput
from state_aware_rag.agents.prompts.evaluate import (
    JUDGE_ANSWER_PROMPT,
    PATH_AWARE_PROMPT,
    SYNTHESIZE_FINAL_ANSWER_PROMPT,
    MAJORITY_VOTE_PROMPT,
    JudgeAnswerOutput,
    PathAwareOutput,
    SynthesizeFinalAnswerOutput,
    MajorityVoteOutput,
)


_NO_EXAMPLES = "No examples provided."
_NOT_PROVIDED = "Not provided"


# =============================================================================
# Role dataclass
# =============================================================================


@dataclass
class Role:
    name: str
    prompt_template: str
    input_model: Type[pydantic.BaseModel]
    output_model: Type[pydantic.BaseModel]


class PromptInput(pydantic.BaseModel):
    """Base input model. ``prompt_fields`` returns the dict used to ``.format`` the
    role's template. Override only when a field needs custom rendering (e.g. a list
    joined into a block). Extra keys are harmless — ``str.format`` ignores them."""

    def prompt_fields(self) -> Dict[str, Any]:
        return self.model_dump()


# =============================================================================
# Input models — fields map 1:1 to each legacy template's {placeholders}
# =============================================================================


class SubquestionGeneratorInput(PromptInput):
    # GENERATE_SUBQUESTION_PROMPT: {examples, question, context}
    question: str
    context: str = _NOT_PROVIDED
    examples: str = _NO_EXAMPLES


class AnswerGeneratorInput(PromptInput):
    # ANSWER_PROMPT: {examples, question, context}
    question: str
    context: str = _NOT_PROVIDED
    examples: str = _NO_EXAMPLES


class QueryGeneratorInput(PromptInput):
    # GENERATE_QUERIES_FOR_RETRIEVER: {examples, question}
    question: str
    examples: str = _NO_EXAMPLES


class FinalizerInput(PromptInput):
    # FINALIZE_PROMPT: {examples, question, context}
    question: str
    context: str = _NOT_PROVIDED
    examples: str = _NO_EXAMPLES


class SelfCorrectorInput(PromptInput):
    # SELF_CORRECT_PROMPT: {examples, question, answer, context}
    question: str
    answer: str
    context: str = _NOT_PROVIDED
    examples: str = _NO_EXAMPLES


class QuestionRephraserInput(PromptInput):
    # REPHRASE_QUESTION_PROMPT: {examples, question}
    question: str
    examples: str = _NO_EXAMPLES


class SynthesizerInput(PromptInput):
    # SYNTHESIZE_PROMPT: {examples, question, context}
    question: str
    context: str
    examples: str = _NO_EXAMPLES


class ExtractorInput(PromptInput):
    # EXTRACT_PROMPT: {examples, question, document}
    question: str
    document: str
    examples: str = _NO_EXAMPLES


class OutcomeJudgeInput(PromptInput):
    # JUDGE_ANSWER_PROMPT: {examples, user_question, system_answer, correct_answer}
    # correct_answer is the ONLY reference field across all roles — judge-only (§11.3).
    user_question: str
    system_answer: str
    correct_answer: str = _NOT_PROVIDED
    examples: str = _NO_EXAMPLES


class PathJudgeInput(PromptInput):
    # PATH_AWARE_PROMPT: {main_question, reasoning_trace, sub_question,
    #                     selected_information, generated_answer} — no {examples}
    main_question: str
    reasoning_trace: str
    sub_question: str
    selected_information: str
    generated_answer: str


class AnswerSynthesizerInput(PromptInput):
    # SYNTHESIZE_FINAL_ANSWER_PROMPT: {examples, question, reasoning_paths}
    question: str
    reasoning_paths: str
    examples: str = _NO_EXAMPLES


class MajorityVoterInput(PromptInput):
    # MAJORITY_VOTE_PROMPT: {examples, question, answers}
    question: str
    answers: str
    examples: str = _NO_EXAMPLES


# =============================================================================
# Role instances (12) — names match config.role_tiers keys
# =============================================================================


# Generation roles (generator tier)
SUBQUESTION_GENERATOR = Role(
    "subquestion_generator", GENERATE_SUBQUESTION_PROMPT, SubquestionGeneratorInput, SubquestionOutput
)
ANSWER_GENERATOR = Role(
    "answer_generator", ANSWER_PROMPT, AnswerGeneratorInput, AnswerOutput
)
QUERY_GENERATOR = Role(
    "query_generator", GENERATE_QUERIES_FOR_RETRIEVER, QueryGeneratorInput, QueriesGenerationOutput
)
FINALIZER = Role(
    "finalizer", FINALIZE_PROMPT, FinalizerInput, FinalizeOutput
)
SELF_CORRECTOR = Role(
    "self_corrector", SELF_CORRECT_PROMPT, SelfCorrectorInput, SelfCorrectOutput
)
QUESTION_REPHRASER = Role(
    "question_rephraser", REPHRASE_QUESTION_PROMPT, QuestionRephraserInput, RephraseQuestionOutput
)
SYNTHESIZER = Role(
    "synthesizer", SYNTHESIZE_PROMPT, SynthesizerInput, SynthesizeOutput
)

# The one curated component (extractor tier)
EXTRACTOR = Role(
    "extractor", EXTRACT_PROMPT, ExtractorInput, ExtractOutput
)

# Dual-reward judges + terminal synthesis (evaluator tier)
OUTCOME_JUDGE = Role(
    "outcome_judge", JUDGE_ANSWER_PROMPT, OutcomeJudgeInput, JudgeAnswerOutput
)
PATH_JUDGE = Role(
    "path_judge", PATH_AWARE_PROMPT, PathJudgeInput, PathAwareOutput
)
ANSWER_SYNTHESIZER = Role(
    "answer_synthesizer", SYNTHESIZE_FINAL_ANSWER_PROMPT, AnswerSynthesizerInput, SynthesizeFinalAnswerOutput
)
MAJORITY_VOTER = Role(
    "majority_voter", MAJORITY_VOTE_PROMPT, MajorityVoterInput, MajorityVoteOutput
)


ALL_ROLES: List[Role] = [
    SUBQUESTION_GENERATOR,
    ANSWER_GENERATOR,
    QUERY_GENERATOR,
    FINALIZER,
    SELF_CORRECTOR,
    QUESTION_REPHRASER,
    SYNTHESIZER,
    EXTRACTOR,
    OUTCOME_JUDGE,
    PATH_JUDGE,
    ANSWER_SYNTHESIZER,
    MAJORITY_VOTER,
]

ROLES_BY_NAME: Dict[str, Role] = {r.name: r for r in ALL_ROLES}
