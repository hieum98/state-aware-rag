"""LLM execution layer for ``langgraph_sar`` — tier-based model selection.

Ports ``wemg/langgraph_coe/llm.py`` and retargets it to the SAR roles. Two
deliberate SAR differences:

  1. **Single user message.** ``format_messages`` renders the role's full legacy
     template into one ``HumanMessage`` (no system turn) — byte-identical to how
     the trained extractor was prompted (``roles.py`` docstring; §11.1).
  2. **Default tier is ``generator``** (coe defaulted to ``heavy``), and the
     Qwen3/SGLang ``chat_template_kwargs`` thinking toggle is injected only when
     the tier sets ``supports_thinking_toggle`` (Anthropic judges reject it).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from .config import LLMConfig
from .explicit import raise_explicit, warn_explicit
from .parsing import extract_info_from_text, extraction_type_from_annotation
from .roles import Role

logger = logging.getLogger(__name__)

_DEFAULT_TIER = "generator"
_LLM_DEBUG = os.environ.get("LANGGRAPH_SAR_LLM_DEBUG", "").lower() in ("1", "true", "yes")


def _safe_debug_repr(value: Any) -> str:
    try:
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2)
        if hasattr(value, "model_dump"):
            return pprint.pformat(value.model_dump(), width=120)
        return pprint.pformat(value, width=120)
    except Exception as exc:
        return f"<failed to render {type(value).__name__}: {exc}; repr={value!r}>"


def _print_llm_debug(role_name: str, item_idx: int, sample_idx: int, label: str, value: Any) -> None:
    if not _LLM_DEBUG:
        return
    print(
        f"\n[llm_debug] role={role_name!r} item={item_idx} sample={sample_idx} {label} "
        f"type={type(value).__name__}\n{_safe_debug_repr(value)}\n[/llm_debug]",
        flush=True,
    )


def format_messages(role: Role, item: BaseModel) -> List[Any]:
    fields = item.prompt_fields() if hasattr(item, "prompt_fields") else item.model_dump()
    content = role.prompt_template.format(**fields)
    return [HumanMessage(content=content)]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype in ("thinking", "reasoning", "redacted_thinking"):
                continue
            for key in ("text", "content", "value"):
                if key in block and isinstance(block[key], str):
                    parts.append(block[key])
                    break
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _apply_role_defaults(role: Role, parsed_dict: Dict[str, Any]) -> None:
    """Fill known-safe defaults only after an explicit warning per field."""
    if role.name == "extractor":
        if parsed_dict.get("reasoning") is None:
            warn_explicit(
                "Extractor response omitted 'reasoning'; using placeholder text.",
                component="llm.parse_fallback",
                details={"role": role.name, "field": "reasoning"},
            )
            parsed_dict["reasoning"] = "No reasoning field returned by model."
        return

    if role.name == "subquestion_generator":
        if not parsed_dict.get("gap_type"):
            warn_explicit(
                "Subquestion response omitted 'gap_type'; defaulting to 'factual'.",
                component="llm.parse_fallback",
                details={"role": role.name, "field": "gap_type", "default": "factual"},
            )
            parsed_dict["gap_type"] = "factual"
        return

    if role.name == "outcome_judge":
        if parsed_dict.get("total_rating") is None:
            warn_explicit(
                "Outcome judge response omitted 'total_rating'; defaulting to 5.0 (neutral).",
                component="llm.parse_fallback",
                details={"role": role.name, "field": "total_rating", "default": 5.0},
            )
            parsed_dict["total_rating"] = 5.0
        if not parsed_dict.get("reasoning"):
            warn_explicit(
                "Outcome judge response omitted 'reasoning'; using placeholder text.",
                component="llm.parse_fallback",
                details={"role": role.name, "field": "reasoning"},
            )
            parsed_dict["reasoning"] = "No reasoning field returned by model."
        return

    if role.name == "path_judge":
        defaults = {
            "relevance": "fair",
            "relevance_details": "",
            "sufficiency": "fair",
            "sufficiency_details": "",
            "coherence": "fair",
            "coherence_details": "",
            "factuality": "fair",
            "factuality_details": "",
        }
        for field, default in defaults.items():
            if field in parsed_dict and not parsed_dict.get(field):
                warn_explicit(
                    f"Path judge response omitted or emptied '{field}'; defaulting.",
                    component="llm.parse_fallback",
                    details={"role": role.name, "field": field, "default": default},
                )
                parsed_dict[field] = default


def parse_fallback(role: Role, raw_text: Any, *, parsing_error: Any = None) -> BaseModel:
    """Regex/JSON recovery when ``with_structured_output`` fails — never silent."""
    text = _content_to_text(raw_text)
    warn_explicit(
        "Structured output failed; recovering via JSON/regex parse_fallback.",
        component="llm.parse_fallback",
        details={
            "role": role.name,
            "raw_chars": len(text),
            "parsing_error": str(parsing_error)[:200] if parsing_error else None,
        },
    )

    keys = list(role.output_model.model_fields.keys())
    specs = [extraction_type_from_annotation(f.annotation) for f in role.output_model.model_fields.values()]
    value_types = [s[0] for s in specs]
    field_optional = [s[1] for s in specs]

    if role.name == "extractor" and "reasoning" in keys:
        field_optional[keys.index("reasoning")] = True
    if role.name == "subquestion_generator" and "gap_type" in keys:
        field_optional[keys.index("gap_type")] = True
    if role.name == "outcome_judge":
        for field in ("total_rating", "reasoning"):
            if field in keys:
                field_optional[keys.index(field)] = True
    if role.name == "path_judge":
        for field in keys:
            if field.endswith("_details") or field in (
                "relevance",
                "sufficiency",
                "coherence",
                "factuality",
            ):
                field_optional[keys.index(field)] = True

    try:
        parsed_dict = extract_info_from_text(text, keys, value_types, field_optional=field_optional)
    except Exception as exc:
        preview = text if len(text) < 4000 else f"{text[:2000]}\n…[truncated]…\n{text[-2000:]}"
        raise_explicit(
            f"parse_fallback could not extract required fields {keys} from model output.",
            component="llm.parse_fallback",
            cause=exc,
        )

    _apply_role_defaults(role, parsed_dict)
    return role.output_model(**parsed_dict)


class RoleModelRegistry:
    def __init__(self, llm_config: LLMConfig):
        self._tiers = llm_config.tiers
        self._role_tiers = llm_config.role_tiers
        self._api_key = llm_config.api_key
        self._instances: Dict[str, ChatLiteLLM] = {}

    def _get_tier(self, role_name: str) -> str:
        return self._role_tiers.get(role_name, _DEFAULT_TIER)

    def get_model_by_tier(self, tier: str) -> ChatLiteLLM:
        if tier not in self._tiers:
            warn_explicit(
                f"Tier {tier!r} not in config; falling back to {_DEFAULT_TIER!r}.",
                component="llm.registry",
                details={"requested_tier": tier, "fallback_tier": _DEFAULT_TIER},
            )
            tier = _DEFAULT_TIER

        if tier not in self._instances:
            cfg = self._tiers[tier]
            model_kwargs: Dict[str, Any] = {"top_p": cfg.top_p}
            if cfg.supports_thinking_toggle:
                model_kwargs["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}

            kwargs: Dict[str, Any] = dict(
                model=cfg.model_name,
                api_key=cfg.api_key or self._api_key,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                max_retries=cfg.max_retries,
                timeout=cfg.timeout,
                model_kwargs=model_kwargs,
            )
            if cfg.api_base:
                kwargs["api_base"] = cfg.api_base
            elif cfg.model_name.lower().startswith("openai/"):
                raise_explicit(
                    f"Tier {tier!r} uses {cfg.model_name!r} but api_base is unset; "
                    "point api_base at your SGLang/OpenAI-compatible server in config.yaml.",
                    component="llm.registry",
                )
            else:
                warn_explicit(
                    "Tier has no api_base; LiteLLM will use default provider routing for this model.",
                    component="llm.registry",
                    details={"tier": tier, "model_name": cfg.model_name},
                )

            self._instances[tier] = ChatLiteLLM(**kwargs)
        return self._instances[tier]

    def get_model(self, role_name: str) -> ChatLiteLLM:
        return self.get_model_by_tier(self._get_tier(role_name))

    def get_structured(self, role: Role) -> Runnable:
        return self.get_model(role.name).with_structured_output(role.output_model)


async def execute_role_lc(
    registry: RoleModelRegistry,
    role: Role,
    input_data: Union[BaseModel, List[BaseModel]],
    n: int = 1,
    tier_override: Optional[str] = None,
) -> Tuple[Union[BaseModel, List[BaseModel], List[List[BaseModel]]], Dict]:
    is_single = isinstance(input_data, BaseModel)
    items = [input_data] if is_single else input_data

    if tier_override:
        model = registry.get_model_by_tier(tier_override)
    else:
        model = registry.get_model(role.name)

    chain = model.with_structured_output(role.output_model, include_raw=True)

    all_results: List[Any] = []
    log_entries: List[Tuple[str, str]] = []

    for item_idx, item in enumerate(items):
        messages = format_messages(role, item)
        tasks = [chain.ainvoke(messages) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parsed: List[BaseModel] = []
        for sample_idx, r in enumerate(results):
            _print_llm_debug(role.name, item_idx, sample_idx, "raw_result", r)
            if isinstance(r, Exception):
                warn_explicit(
                    "LLM sample raised an exception; sample discarded.",
                    component="llm.execute_role_lc",
                    details={"role": role.name, "sample": sample_idx, "error": str(r)},
                )
                continue
            if isinstance(r, dict):
                raw = r.get("raw")
                parsed_obj = r.get("parsed")
                parsing_error = r.get("parsing_error")
                if raw is not None:
                    _print_llm_debug(role.name, item_idx, sample_idx, "raw_message", raw)
                    if hasattr(raw, "content"):
                        _print_llm_debug(role.name, item_idx, sample_idx, "raw_content", raw.content)
                if parsed_obj is not None:
                    _print_llm_debug(role.name, item_idx, sample_idx, "structured_parsed", parsed_obj)
                if parsing_error is not None:
                    _print_llm_debug(role.name, item_idx, sample_idx, "parsing_error", parsing_error)

                if isinstance(parsed_obj, role.output_model):
                    parsed.append(parsed_obj)
                elif raw is not None and hasattr(raw, "content"):
                    parsed.append(parse_fallback(role, raw.content, parsing_error=parsing_error))
                elif parsing_error is not None:
                    raise_explicit(
                        "Structured output failed and no raw message was returned for recovery.",
                        component="llm.execute_role_lc",
                        cause=parsing_error if isinstance(parsing_error, BaseException) else None,
                    )
            elif isinstance(r, role.output_model):
                parsed.append(r)

        if not parsed:
            errors = [r for r in results if isinstance(r, Exception)]
            raise_explicit(
                f"All {n} completion(s) failed for role {role.name!r}.",
                component="llm.execute_role_lc",
                cause=errors[0] if errors else None,
            )

        if n == 1:
            all_results.append(parsed[0])
            log_entries.append((str(item), str(parsed[0])))
        else:
            all_results.append(parsed)
            log_entries.append((str(item), str(parsed[0])))

    return (all_results[0] if is_single else all_results), {role.name: log_entries}
