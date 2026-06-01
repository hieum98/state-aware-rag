#!/usr/bin/env python3
"""Live end-to-end smoke test for ``langgraph_sar`` against real SGLang endpoints.

Prerequisites (see ``langgraph_sar/config.yaml``):
  - LLM tunnel:   localhost:30172 -> Qwen3-8B
  - Embedder:     localhost:30164 -> Qwen3-Embedding-4B
  - FAISS corpus: ``SAR_CORPUS_INDEX_PATH`` or ``data/toy-index/index.faiss``

Usage:
  python -m langgraph_sar.scripts.smoke_live
  python -m langgraph_sar.scripts.smoke_live --strategy mcts --num-rollouts 1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import httpx

from langgraph_sar.config import SARConfig
from langgraph_sar.system import answer_question


async def _reachable(base_url: str | None) -> bool:
    if not base_url:
        return True
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            resp = await client.get(f"{root}/health")
        return resp.status_code < 500
    except Exception:
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(description="langgraph_sar live smoke test")
    parser.add_argument(
        "--strategy",
        choices=("socratic", "mcts"),
        default="socratic",
        help="Search strategy (default: socratic)",
    )
    parser.add_argument(
        "--question",
        default="Who founded the company that created the iPhone?",
        help="Multi-hop question for the smoke run",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Socratic/MCTS depth cap")
    parser.add_argument("--num-rollouts", type=int, default=1, help="MCTS rollouts (mcts only)")
    parser.add_argument(
        "--corpus",
        default=os.environ.get("SAR_CORPUS_INDEX_PATH", "data/toy-index/index.faiss"),
        help="Path to FAISS index bundle",
    )
    args = parser.parse_args()

    os.environ.setdefault("SAR_CORPUS_INDEX_PATH", args.corpus)
    if not os.path.exists(args.corpus):
        print(f"ERROR: FAISS index not found at {args.corpus!r}", file=sys.stderr)
        print("Run: python langgraph_sar/tests/phase1/create_toy_corpus.py", file=sys.stderr)
        return 1

    cfg = SARConfig.from_yaml()
    cfg.search.strategy = args.strategy
    cfg.search.max_depth = args.max_depth
    cfg.search.num_rollouts = args.num_rollouts
    cfg.search.socratic_recursion_limit = 40
    cfg.search.mcts_recursion_limit = 60
    # Keep smoke runs short; cap tokens to avoid Qwen3 runaway whitespace after JSON.
    for tier_name in ("generator", "extractor", "evaluator"):
        tier = cfg.llm.tiers[tier_name]
        tier.temperature = 0.2 if tier_name == "generator" else 0.0
        tier.max_tokens = 768 if tier_name == "generator" else 512
        tier.enable_thinking = False

    llm_url = cfg.llm.tiers["generator"].api_base
    emb_url = cfg.retriever.corpus.embedder.url
    if not await _reachable(llm_url):
        print(f"ERROR: LLM endpoint unreachable at {llm_url}", file=sys.stderr)
        return 1
    if not await _reachable(emb_url):
        print(f"ERROR: Embedder unreachable at {emb_url}", file=sys.stderr)
        return 1

    print("=== langgraph_sar live smoke ===")
    print(f"strategy={args.strategy} max_depth={args.max_depth} corpus={args.corpus}")
    print(f"llm={llm_url} embedder={emb_url}")
    print(f"question: {args.question}")

    t0 = time.perf_counter()
    result = await answer_question(
        args.question,
        config=cfg,
        golden_answer=["Steve Jobs", "Apple"],
        eval_mode="judge_only",
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- result ({elapsed:.1f}s) ---")
    print(f"pred: {result.pred}")
    print(f"detailed: {result.detailed_answer[:500]}{'...' if len(result.detailed_answer) > 500 else ''}")
    if result.reward is not None:
        print(f"reward: {result.reward:.3f}")
    print(f"strategy: {result.strategy} eval_mode: {result.eval_mode}")

    if not (result.pred or "").strip():
        print("ERROR: empty prediction", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
