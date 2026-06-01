"""Hydra CLI for LangGraph SAR inference (mirrors root ``inference.py``).

Usage (from repo root)::

    python -m langgraph_sar.inference question="Who founded Apple?"
    python -m langgraph_sar.inference strategy=mcts data.name=2wiki data.limit=32
    python -m langgraph_sar.inference mode=cot data.limit=10   # legacy alias for socratic
"""

from __future__ import annotations

import asyncio
import logging
import os

# Quiet LiteLLM's per-call INFO banner ("LiteLLM completion() model=..."). LiteLLM reads
# LITELLM_LOG when first imported (transitively via .system below), so set it before that.
os.environ.setdefault("LITELLM_LOG", "WARNING")

from pathlib import Path
from typing import Any, Dict, Optional

import datasets
import hydra
from hydra import utils as hy_utils
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .config import SARConfig
from .system import answer_question, answer_question_streaming, answer_records_batch

_STRATEGY_ALIASES = {"cot": "socratic", "socratic": "socratic", "mcts": "mcts"}


def _resolve_strategy(cfg: DictConfig) -> str:
    if cfg.get("strategy"):
        strategy = str(cfg.get("strategy"))
    elif cfg.get("mode"):
        mode = str(cfg.get("mode"))
        strategy = _STRATEGY_ALIASES.get(mode, mode)
    else:
        strategy = "socratic"
    if strategy not in ("socratic", "mcts"):
        raise ValueError(
            f"Unknown strategy {strategy!r}; use strategy=socratic|mcts or mode=cot|mcts"
        )
    return strategy


def _resolve_sar_yaml_path(cfg: DictConfig) -> Optional[Path]:
    raw = cfg.get("sar_config")
    if raw is None or raw == "":
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(hy_utils.get_original_cwd()) / path
    return path


def _merge_model_section(model: Any, overrides: Optional[Dict[str, Any]]) -> Any:
    if not overrides:
        return model
    return model.__class__.model_validate({**model.model_dump(), **overrides})


def _section_overrides(cfg: DictConfig, section: str) -> Optional[Dict[str, Any]]:
    if section not in cfg or cfg[section] is None:
        return None
    raw = cfg[section]
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, dict) or not raw:
        return None
    return raw


def build_sar_config(cfg: DictConfig) -> SARConfig:
    """Load ``SARConfig`` from YAML and apply Hydra overrides."""
    sar = SARConfig.from_yaml(_resolve_sar_yaml_path(cfg))
    sar.search.strategy = _resolve_strategy(cfg)

    for section in ("search", "memory", "llm", "retriever", "web_search"):
        overrides = _section_overrides(cfg, section)
        if overrides:
            setattr(sar, section, _merge_model_section(getattr(sar, section), overrides))

    return sar


def _compute_results_dir(cfg: DictConfig, sar: SARConfig) -> str:
    strategy = sar.search.strategy
    data_name = cfg.get("data", {}).get("name", "unknown_data")
    gen_model = sar.llm.tiers["generator"].model_name.replace("/", "-")
    ext_model = sar.llm.tiers["extractor"].model_name.replace("/", "-")
    eva_model = sar.llm.tiers["evaluator"].model_name.replace("/", "-")
    return os.path.join(
        cfg.results_dir,
        strategy,
        f"Generator_{gen_model}",
        f"Extractor_{ext_model}",
        f"Evaluator_{eva_model}",
        data_name,
    )


def _record_eval_mode(record: Dict[str, Any], default_mode: str) -> Dict[str, Any]:
    row = dict(record)
    row.setdefault("eval_mode", default_mode)
    return row


async def _run_single_question(
    cfg: DictConfig,
    sar: SARConfig,
    *,
    eval_mode: str,
) -> None:
    if bool(cfg.get("stream", False)):
        print(f"Streaming {sar.search.strategy!r} steps for: {cfg.question}")
        result = await answer_question_streaming(
            str(cfg.question),
            config=sar,
            golden_answer=cfg.get("golden_answer"),
            eval_mode=eval_mode,
            on_step=lambda msg: print(f"  {msg}", flush=True),
        )
    else:
        result = await answer_question(
            str(cfg.question),
            config=sar,
            golden_answer=cfg.get("golden_answer"),
            eval_mode=eval_mode,
        )
    print("Final Answer:", result.pred)
    print("Final Reasoning:", result.detailed_answer)


async def _run_dataset(
    cfg: DictConfig,
    sar: SARConfig,
    results_dir: str,
    *,
    eval_mode: str,
) -> None:
    from inference import _load_dataset

    ds = _load_dataset(cfg.data)
    max_workers = max(1, int(cfg.get("max_workers", 4) or 4))
    batch_size = max(1, int(cfg.get("batch_size", 32) or 32))

    all_rows: list[Dict[str, Any]] = []
    n = len(ds)
    print(
        f"Running langgraph_sar inference: n={n}, max_workers={max_workers}, "
        f"batch_size={batch_size}, strategy={sar.search.strategy!r}..."
    )
    for start in tqdm(range(0, n, batch_size), desc="inference"):
        end = min(start + batch_size, n)
        chunk = ds.select(range(start, end))
        records = [_record_eval_mode(dict(ex), eval_mode) for ex in chunk]
        out = await answer_records_batch(
            records,
            config=sar,
            max_workers=max_workers,
        )
        all_rows.extend(out)

    out_ds = datasets.Dataset.from_list(all_rows)
    out_ds.save_to_disk(results_dir)
    print(f"Results saved to {results_dir}")


async def run_inference(cfg: DictConfig) -> None:
    sar = build_sar_config(cfg)
    eval_mode = str(cfg.get("eval_mode", "judge_only"))
    results_dir = os.path.join(
        hy_utils.get_original_cwd(),
        _compute_results_dir(cfg, sar),
    )
    os.makedirs(results_dir, exist_ok=True)

    # question= is an exclusive single-shot (matches the module docstring); only
    # fall through to the dataset run when no single question was given.
    if cfg.get("question"):
        await _run_single_question(cfg, sar, eval_mode=eval_mode)
    else:
        await _run_dataset(cfg, sar, results_dir, eval_mode=eval_mode)


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "infer_langgraph"


@hydra.main(
    config_path=str(_CONFIG_DIR),
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Hydra reconfigures the root logger at INFO inside the decorator, which is why the
    # LiteLLM banner reappears as "[...][LiteLLM][INFO]". Pin the noisy LLM-stack loggers
    # to WARNING here (after that reconfiguration) so only WARNING+ survives.
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)
    print("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    asyncio.run(run_inference(cfg))


if __name__ == "__main__":
    main()
