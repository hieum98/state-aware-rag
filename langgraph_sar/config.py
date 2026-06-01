"""Configuration for ``langgraph_sar`` — the State-Aware RAG LangGraph port.

Mirrors ``wemg/langgraph_coe/config.py`` but collapses the tier set to the
paper's three-model split (one trainable). See ``IMPLEMENTATION_PLAN.md`` §4.

Tiers (one ``ChatLiteLLM`` per tier, built lazily in ``llm.RoleModelRegistry``):
  - ``generator``  — frozen Qwen3-8B, samples the reasoning/answer roles.
  - ``extractor``  — TRAINABLE Qwen3-4B; swap to the RL checkpoint by editing
    this tier's ``model_name``/``api_base`` only (zero graph-code changes).
  - ``evaluator``  — judge (claude / any litellm-routable model), temp 0.

Endpoints default to the SSH-tunnelled SGLang servers on the Talapas cluster
(``localhost:30172`` → ``n0172:30000`` LLM, ``localhost:30164`` → ``n0164:30000``
embedder). Override in ``config.yaml`` or via env for other deployments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# LLM tiers & role routing
# ──────────────────────────────────────────────────────────────────────────────


class TierConfig(BaseModel):
    """LLM configuration for one tier (one served model)."""

    model_name: str = "openai/Qwen/Qwen3-8B"
    # OpenAI-compatible vLLM/SGLang endpoint. ``None`` lets litellm pick the
    # provider default (e.g. the Anthropic API for a ``claude-*`` evaluator).
    api_base: Optional[str] = "http://localhost:30172/v1"
    api_key: Optional[str] = None  # inherits from top-level llm.api_key when unset
    temperature: float = 0.7
    max_tokens: int = 4096
    max_input_tokens: int = 32768
    top_p: float = 0.95
    # Anti-repetition sampling knobs. ``None`` = omit (the served model's own default),
    # so non-SGLang providers (Anthropic) never receive an unsupported kwarg. SGLang/vLLM
    # support all three: ``frequency_penalty``/``presence_penalty`` (OpenAI-style, additive)
    # and ``repetition_penalty`` (multiplicative, >1.0 suppresses; passed through extra body).
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    enable_thinking: bool = False
    # Qwen3/SGLang exposes the thinking toggle via ``chat_template_kwargs``.
    # Non-SGLang providers (Anthropic) reject that kwarg, so gate its injection.
    supports_thinking_toggle: bool = True
    max_retries: int = 3
    timeout: int = 300


def _default_tiers() -> Dict[str, TierConfig]:
    return {
        # Frozen generator — samples the seven generation roles (temp 0.7 gives
        # MCTS its branching entropy; do not add a response cache, see §4).
        "generator": TierConfig(
            model_name="openai/Qwen/Qwen3-8B",
            api_base="http://localhost:30172/v1",
            temperature=0.7,
        ),
        # TRAINABLE extractor — the single curated component. No Qwen3-4B chat
        # endpoint is deployed yet, so this points at the same Qwen3-8B server as
        # a stand-in; swap model_name/api_base to the RL Qwen3-4B checkpoint here.
        "extractor": TierConfig(
            model_name="openai/Qwen/Qwen3-8B",
            api_base="http://localhost:30172/v1",
            temperature=0.2,
            # Suppress the degenerate sentence-level loops that truncate a long
            # ``information`` array and abort the run (the extractor is the only role
            # asked to emit large verbatim lists). Modest value: enough to break loops
            # without discouraging legitimately repeated tokens in extracted text.
            frequency_penalty=0.3,
        ),
        # Judge — deterministic. Routed to the Anthropic API by litellm (no
        # api_base); needs ANTHROPIC_API_KEY. chat_template_kwargs is suppressed.
        "evaluator": TierConfig(
            model_name="claude-3-7-sonnet-20250219",
            api_base=None,
            temperature=0.0,
            enable_thinking=False,
            supports_thinking_toggle=False,
        ),
    }


def _default_role_tiers() -> Dict[str, str]:
    return {
        # Generation roles (frozen generator tier)
        "subquestion_generator": "generator",
        "answer_generator": "generator",
        "query_generator": "generator",
        "finalizer": "generator",
        "self_corrector": "generator",
        "question_rephraser": "generator",
        "synthesizer": "generator",
        # The one curated component
        "extractor": "extractor",
        # Dual-reward judges + terminal synthesis (evaluator tier)
        "outcome_judge": "evaluator",    # R_o
        "path_judge": "evaluator",       # R_p
        "answer_synthesizer": "evaluator",  # MCTS terminal synthesis
        "majority_voter": "evaluator",   # synthesis fallback
    }


class LLMConfig(BaseModel):
    """Tier-based model selection. ``role → role_tiers[role] → tiers[tier]``."""

    api_key: Optional[str] = None  # SGLang/OpenAI-compatible key for the Qwen tiers
    tiers: Dict[str, TierConfig] = Field(default_factory=_default_tiers)
    role_tiers: Dict[str, str] = Field(default_factory=_default_role_tiers)


# ──────────────────────────────────────────────────────────────────────────────
# Search / memory / retriever / web blocks
# ──────────────────────────────────────────────────────────────────────────────


class SearchConfig(BaseModel):
    """Strategy selection + MCTS/Socratic knobs (``IMPLEMENTATION_PLAN.md`` §4)."""

    strategy: str = "socratic"          # "socratic" | "mcts"
    top_k: int = 5                      # retriever cap; sub-question used directly as query
    query_expansion: bool = False       # gate QUERY_GENERATOR; off = cost-optimized default
    extractor_join: bool = False        # off = per-doc extraction (paper-faithful)
    reflect_threshold_facts: int = 8    # run reflect-focus only when |memory| > this (§11.9)
    # socratic
    max_depth: int = 5
    # mcts
    exploration_weight: float = 1.4     # PUCT c; tuned to normalized reward R∈[0,1] (§11.10)
    use_node_type_prior: bool = True    # PUCT prior from NODE_TYPE_PRIOR; False → plain UCT
    num_rollouts: int = 8
    reward_lambda: float = 1.0          # λ in R = (mean R_p + λ·R_o)/(1+λ)
    reward_mode: str = "dual"           # "dual" (paper) | "confidence_blend" (legacy)
    high_confidence: float = 0.9        # early-stop on normalized reward
    patience: int = 3
    # global-memory commit gate (MCTS) — §7.1 / §8.5 / §11.7
    memory_commit_gate: str = "factuality"   # "factuality" | "reward" | "off"
    memory_commit_threshold: float = 0.5
    # LangGraph superstep budgets (guard layered over max_depth / num_rollouts)
    socratic_recursion_limit: int = 75
    mcts_recursion_limit: int = 150


class MemoryConfig(BaseModel):
    """Working-memory limits for ConsolidateGraph / MemoryUpdateGraph."""

    extractor_max_input_chars: int = 24_000   # per-doc overflow splitter (split, never merge)
    max_memory_facts: int = 64                # global-memory size cap; E_update evicts beyond
    confidence_tags: bool = True              # inline ⟨confidence/provenance⟩ markers


class EmbedderConfig(BaseModel):
    """Embedding endpoint (OpenAI-compatible) for the FAISS corpus retriever."""

    model_name: str = "Qwen/Qwen3-Embedding-4B"
    url: str = "http://localhost:30164/v1"
    api_key: Optional[str] = None
    # Qwen3-Embedding is asymmetric: passages are embedded raw (build_index.py) but
    # QUERIES must carry this instruction prefix ({query} placeholder). Mirrors the
    # legacy servers/retriever.py qwen3 default; empty string disables it.
    query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: {query}"
    )


class CorpusConfig(BaseModel):
    """Local FAISS corpus + the embedder used to query it (coe pattern, no reranker)."""

    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    # Path to the FAISS index. Two layouts are supported:
    #   1. LangChain bundle  — ``<name>.faiss`` + ``<name>.pkl`` sidecar docstore
    #      (e.g. the toy index); loaded via ``FAISS.load_local``.
    #   2. Raw HF-datasets index — a bare ``<name>.faiss`` produced by
    #      ``state_aware_rag/servers/build_index.py`` (datasets.save_faiss_index),
    #      whose passage text lives in a separate HF dataset (``corpus_dataset``).
    # Override with env SAR_CORPUS_INDEX_PATH when the corpus lives elsewhere.
    index_path: str = "data/wiki23-Qwen3-4B-Emb-Indexed/index.faiss"
    search_k: int = 10  # FAISS fetch depth before search.top_k truncation downstream

    # Raw-index layout only (no ``.pkl`` sidecar): HF dataset name or local
    # ``load_from_disk`` path supplying passage text. Row order MUST match the
    # vector order the index was built from. Override with env SAR_CORPUS_DATASET.
    corpus_dataset: Optional[str] = None
    corpus_split: str = "train"
    text_column: str = "contents"


class WebSearchConfig(BaseModel):
    """Optional ``web_search`` tool (paper's "+Google Search" variant). Off by default."""

    enabled: bool = False
    api_key: Optional[str] = None  # Serper key; DuckDuckGo fallback when null
    top_k: int = 5
    crawl_full_text: bool = False
    max_crawl_requests_per_second: float = 2.0


class RetrieverConfig(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Root config
# ──────────────────────────────────────────────────────────────────────────────


class SARConfig(BaseModel):
    """Root settings for ``langgraph_sar`` loaded from ``config.yaml``."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)

    @staticmethod
    def default_yaml_path() -> Path:
        return Path(__file__).resolve().parent / "config.yaml"

    @classmethod
    def from_yaml(
        cls,
        path: Optional[Path | str] = None,
        *,
        merge_api_key_env: bool = True,
    ) -> "SARConfig":
        """Load from YAML; when *path* is omitted use ``default_yaml_path()`` if present.

        Env merge (when ``merge_api_key_env``):
          - ``API_KEY`` / ``OPENAI_API_KEY`` → ``llm.api_key`` + corpus embedder key
            (the SGLang/OpenAI-compatible Qwen endpoints).
          - ``ANTHROPIC_API_KEY`` → any ``claude``/``anthropic`` tier's key.
          - ``SAR_CORPUS_INDEX_PATH`` → ``retriever.corpus.index_path``.
          - ``SAR_CORPUS_DATASET`` → ``retriever.corpus.corpus_dataset`` (raw-index
            layout: HF dataset / local path supplying passage text).
        """
        p = Path(path) if path is not None else cls.default_yaml_path()
        raw: Optional[Dict[str, Any]] = None
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        merged = cls.model_validate(raw or {})

        if merge_api_key_env:
            sglang_key = (
                merged.llm.api_key
                or os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "EMPTY"
            )
            merged.llm.api_key = sglang_key
            if not merged.retriever.corpus.embedder.api_key:
                merged.retriever.corpus.embedder.api_key = sglang_key

            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            for tier in merged.llm.tiers.values():
                name = tier.model_name.lower()
                if ("claude" in name or "anthropic" in name) and not tier.api_key:
                    tier.api_key = anthropic_key

            env_index = os.environ.get("SAR_CORPUS_INDEX_PATH")
            if env_index:
                merged.retriever.corpus.index_path = env_index

            env_corpus = os.environ.get("SAR_CORPUS_DATASET")
            if env_corpus:
                merged.retriever.corpus.corpus_dataset = env_corpus

        return merged
