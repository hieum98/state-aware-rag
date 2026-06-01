"""LangGraph port of State-Aware RAG.

See ``IMPLEMENTATION_PLAN.md`` for the architecture design and phased plan.

Phase 0 (Foundation) public surface: configuration, the tiered LLM registry,
the role registry, and the corpus/web tools.
"""

from .config import SARConfig
from .explicit import SARExplicitFallbackWarning, raise_explicit, warn_explicit
from .llm import RoleModelRegistry, execute_role_lc, format_messages
from .roles import Role, ALL_ROLES, ROLES_BY_NAME
from .system import (
    AnswerResult,
    answer_question,
    answer_questions_batch,
    answer_records_batch,
    compute_metrics_from_records,
    filter_records_for_metrics,
    records_to_metric_dataset,
)

__all__ = [
    "SARConfig",
    "SARExplicitFallbackWarning",
    "warn_explicit",
    "raise_explicit",
    "RoleModelRegistry",
    "execute_role_lc",
    "format_messages",
    "Role",
    "ALL_ROLES",
    "ROLES_BY_NAME",
    "AnswerResult",
    "answer_question",
    "answer_questions_batch",
    "answer_records_batch",
    "filter_records_for_metrics",
    "records_to_metric_dataset",
    "compute_metrics_from_records",
]
