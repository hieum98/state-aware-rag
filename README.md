State-Aware RAG 
=================================

State-Aware RAG is a modular multi-hop Retrieval-Augmented Generation (RAG) framework featuring two complementary search paradigms:

1. **MCTS** (Monte Carlo Tree Search) — exploratory, branching multi-step reasoning
2. **Socratic / CoT** — lightweight linear (chain-of-thought) reasoning

The system performs iterative **decompose → retrieve → extract → evaluate → synthesize** cycles while maintaining an explicit evolving working memory. Trees and intermediate artifacts can be persisted, re-loaded, and post-analyzed.

Two inference stacks coexist in this repository:

| Stack | Path | Status |
|-------|------|--------|
| **Legacy** | `state_aware_rag/` + `inference.py` | Production training/eval path; Hydra configs |
| **LangGraph port** | `langgraph_sar/` | Inference/serving port on LangGraph (Phases 0–4 complete) |

> Design goals: transparency, reproducibility, pluggability (LLMs & retriever), and scalable batch inference.

Table of Contents
-----------------

- Features
- LangGraph Port (`langgraph_sar`)
  - Hydra CLI (`python -m langgraph_sar.inference`)
  - `eval_mode` (judge_only vs teacher_forced)
- Architecture Overview
- Quick Start (legacy)
  - Environment
  - Minimal Run (Single Question)
  - Dataset Inference
- Configuration System
- Components & Roles
- Search Modes (MCTS vs CoT)
- Caching & Performance Tips
- Evaluation & Metrics
- Directory Structure
- Extending the Framework
- Troubleshooting
- Roadmap / Ideas
- License & Citation

Features
--------

- Multi-hop reasoning with explicit search state (reasoning nodes + memory)
- Two planners: MCTS (branching) and CoT (linear)
- Modular role agents (Generator / Retriever / Extractor / Evaluator)
- **LangGraph port** (`langgraph_sar/`) with tiered `ChatLiteLLM` registry, PUCT MCTS, and Socratic loop
- Local FAISS + Qwen embedding retrieval (no separate retriever HTTP server in the port)
- Pluggable LLM backends via OpenAI-compatible or LiteLLM interface
- Online (HTTP API) and optional offline (FlashRAG) retrieval in the legacy stack
- Structured output extraction with fallback text parsing
- Concurrency + multi-processing + deterministic caching
- Reasoning tree export & reload (resumable search)
- Evaluation suite: F1, Exact Match, Sub EM, Retrieval Recall, LLM Judge
- Hydra + YAML config for reproducible experiment control

---
LangGraph Port (`langgraph_sar`)
--------------------------------

The LangGraph port reproduces the paper's inference modes on **LangGraph** primitives: shared curated text memory, per-doc extractor fan-out, dual path/outcome reward, and a cost-optimized default (no query expansion, `top_k` truncation). 

### Strategies

| `search.strategy` | Graph | Actions |
|-------------------|-------|---------|
| `socratic` | `SocraticGraph` | A1 + A5 linear loop (≈ legacy CoT) |
| `mcts` | `MCTSGraph` | A1–A5 tree search + Socratic rollouts |

Configure in `langgraph_sar/config.yaml` (tiers, endpoints, `search.*`, `memory.*`).

### Prerequisites

1. **Python ≥ 3.10** and `pip install -e .` (plus LangGraph / LangChain deps used by the port).
2. **SGLang**
3. **FAISS corpus** — full wiki index at `data/wiki23-Qwen3-4B-Emb-Indexed/index.faiss`
4. **API key** — `export API_KEY=EMPTY` (or your SGLang key) for Qwen tiers; evaluator defaults to the same local endpoint as the generator.

### Hydra CLI (`python -m langgraph_sar.inference`)

Mirrors legacy `python -m inference`: Hydra config at `configs/infer_langgraph/base.yaml`, model tiers and corpus in `langgraph_sar/config.yaml` (override with `sar_config=/path/to.yaml`).

```bash
# Single question
python -m langgraph_sar.inference \
  question="Who founded the company that created the iPhone?"

# Dataset batch (same dataset names as legacy infer)
python -m langgraph_sar.inference \
  strategy=mcts \
  data.name=2wiki \
  data.limit=32 \
  max_workers=4 \
  search.max_depth=6 \
  search.num_rollouts=8

# Legacy planner alias: mode=cot → strategy=socratic
python -m langgraph_sar.inference mode=cot data.limit=10
```

| Override | Meaning |
|----------|---------|
| `strategy` | `socratic` or `mcts` (default: `socratic`) |
| `mode` | Optional alias: `cot` → `socratic`, `mcts` → `mcts` |
| `sar_config` | Path to SAR YAML; default `langgraph_sar/config.yaml` |
| `search.*` | Merged into `SARConfig.search` (depth, rollouts, `top_k`, …) |
| `max_workers` | Async concurrency per batch (default: 4) |
| `batch_size` | Records per `answer_records_batch` chunk (default: 32) |
| `eval_mode` | `judge_only` (default) or `teacher_forced` — see below |
| `golden_answer` | Optional reference for single-question runs |

Results are saved as a Hugging Face dataset under:

```text
results/<strategy>/Generator_<model>/Extractor_<model>/Evaluator_<model>/<dataset>/
```

Like legacy `inference.py`, if `question` is set the CLI prints one answer and still runs the configured dataset pass.

### `eval_mode`

Controls how **golden answers** interact with the run and whether rows count toward benchmark metrics:

| Value | Use |
|-------|-----|
| **`judge_only`** (default) | Reference answer is for **outcome reward** (`OUTCOME_JUDGE` / \(R_o\)) and metrics only. It is passed via LangGraph `config.configurable`, not generator/extractor prompts (leak-safe eval). |
| **`teacher_forced`** | Intended for **SFT / trajectory synthesis** where gold may steer the loop. Rows are tagged and **excluded** from aggregated metrics (`filter_records_for_metrics`). |

Set globally: `eval_mode=judge_only`. Per record in batch API: `"eval_mode": "judge_only"` on each row.

### Single-question inference (Python API)

```python
import asyncio
from langgraph_sar import SARConfig, answer_question

async def main():
    cfg = SARConfig.from_yaml()
    cfg.search.strategy = "socratic"  # or "mcts"
    cfg.search.max_depth = 3

    result = await answer_question(
        "Who founded the company that created the iPhone?",
        config=cfg,
        golden_answer=["Steve Jobs"],  # optional; judge-only by default
        eval_mode="judge_only",
    )
    print(result.pred)
    print(result.detailed_answer)

asyncio.run(main())
```

### Batch inference

```python
from langgraph_sar import answer_records_batch, SARConfig

rows = [
    {"question": "...", "golden_answers": ["..."], "eval_mode": "judge_only"},
]
out = await answer_records_batch(rows, config=SARConfig.from_yaml(), max_workers=4)
```
---
Architecture Overview
---------------------

High-level loop per question:

1. Generator proposes sub-questions, answers, synthesis attempts, rephrasings, or self-corrections using prompt templates under `state_aware_rag/agents/prompts/`.
2. Retriever fetches context passages (HTTP retriever API and/or local index).
3. Extractor pulls fine-grained answer spans / structured fields from retrieved context.
4. Evaluator scores candidate answers, ranks or synthesizes final answers, may perform majority vote or reasoning synthesis.
5. Planner (MCTS or CoT) orchestrates iterative expansion until termination conditions (depth, rollouts, convergence).
6. Final answer + reasoning chain saved to a Hugging Face dataset directory under `results/` with optional `result_trees/` JSONL.

Key files:

- `inference.py` – legacy Hydra entry (single question or dataset map)
- `langgraph_sar/inference.py` – LangGraph Hydra entry (`python -m langgraph_sar.inference`)
- `evaluate.py` – metric computation and judged scoring
- `state_aware_rag/agents/` – agent abstractions & role implementations
- `state_aware_rag/planners/` – search logic (`MCTS` and `CoT`)
- `configs/infer/` – Hydra configs (legacy)
- `configs/infer_langgraph/` – Hydra configs (LangGraph port)
- `cache/` – per-model LLM call cache (legacy)

Each node logs: node type, memory snapshot, confidence, content (sub-question, answer, synthesis, final answer), plus lineage. Trees can be reloaded to continue or analyze.

Quick Start (legacy `inference.py`)
-----------------------------------

### 1. Environment

Python >= 3.10

Install (editable):
```bash
pip install -e .
```

You will need:

- An OpenAI-compatible endpoint (e.g. local server or provider) OR models accessible via LiteLLM routing
- A retriever service URL (see `scripts/deploy_retriever_server.sh`) or FlashRAG index
- API keys exported (e.g. `OPENAI_API_KEY`) if required by your LLM backend

Optional (FlashRAG offline retrieval):

```bash
pip install flashrag
```

#### Server Deployment Guide

See the full deployment steps (LLM + embedding + retriever + evaluation) in `docs/server_deployment.md`.

Minimal quick start (local stack):
 
```bash
# Start LLM
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --host 0.0.0.0 --port 30000 --dtype bfloat16 &
# Start embedding server
python -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-4B --is-embedding --host 0.0.0.0 --port 8000 --dtype bfloat16 &
# Start retriever
python -m state_aware_rag.servers.retriever --config configs/servers/retriever-Qwen3-4B-wiki-23.yaml --port 5000 --workers 2 --mmap_index &
# Inference
python -m inference mode=mcts question="Who wrote the novel Dune?" \
  agents.generator=configs/generator.yaml agents.extractor=configs/extractor.yaml \
  agents.evaluator=configs/evaluator.yaml agents.retriever=configs/retriever.yaml
```

### 2. Minimal Single-Question Inference

```bash
python -m inference \
  mode=mcts \
  question="Who founded the company that created the iPhone?" \
  agents.generator=configs/generator.yaml \
  agents.extractor=configs/extractor.yaml \
  agents.evaluator=configs/evaluator.yaml \
  agents.retriever=configs/retriever.yaml
```

Hydra merges overrides; `mode=cot` switches planner. Add `search.save_tree=true` to persist the reasoning tree.

### 3. Dataset Inference

Example (2Wiki dev subset, MCTS):
```bash
python -m inference \
  mode=mcts \
  data.name=2wiki \
  data.limit=32 \
  num_proc=4 \
  search.max_depth=6 \
  search.num_rollouts=8 \
  search.save_tree=true
```

Results saved under:

```text
results/mcts/Generator_<model>/Extractor_<model>/Evaluator_<model>/<dataset>/
```

### 4. Evaluation

```bash
python -m evaluate \
  mode=mcts \
  data.name=2wiki \
  data.metrics='["all"]' \
  agents.evaluator_metric=configs/evaluator.yaml
```

Creates `<results_path>_with_scores` with metric annotations + `evaluation_results.json`.

---
Configuration System
--------------------

Hydra config trees (simplified):

**Legacy** — `configs/infer/base.yaml`:

```text
  ├─ mode: mcts | cot
  ├─ results_dir: results
  ├─ num_proc: <int>
  ├─ agents: generator | extractor | evaluator | retriever (YAML paths)
  ├─ search: max_depth, num_rollouts, exploration_weight, top_k, save_tree, ...
  └─ data: name, split, limit
```

**LangGraph port** — `configs/infer_langgraph/base.yaml` + `langgraph_sar/config.yaml`:

```text
infer_langgraph/base.yaml          langgraph_sar/config.yaml
  ├─ strategy: socratic | mcts       ├─ llm.tiers (generator / extractor / evaluator)
  ├─ mode: (alias for strategy)     ├─ search.* (defaults; overridable from CLI)
  ├─ sar_config: null               ├─ retriever.corpus.index_path
  ├─ max_workers, batch_size         └─ memory.*, web_search.*
  ├─ eval_mode: judge_only | teacher_forced
  ├─ search: (CLI overrides)
  └─ data: name, split, limit
```

Each agent YAML defines:

```yaml
name: generator
client_kwargs:
  client_type: openai|litellm
  model_name: <model-id>
  api_base: <endpoint-url>
generation_config:
  temperature: 0.2
  max_tokens: 512
use_cache: true
concurrency: 8
```

Retriever config may include:

```yaml
online_retrieval_config:
  url: http://localhost:8000/search
  timeout: 30
offline_retrieval_config:
  index_path: data/wiki23-Qwen3-4B-Emb-Indexed/
top_k: 5
```

---
Components & Roles
------------------

- GeneratorAgent – calls LLM to produce sub-questions, candidate answers, rephrasings, self-corrections, synthesis segments.
- RetrievalAgent – queries HTTP retriever or offline index; returns passages + metadata.
- ExtractorAgent – extracts structured spans or normalized answers from context.
- EvaluatorAgent – scores candidates, performs majority vote or synthesizes final answer reasoning.

All wrap a shared caching layer (`agents/llm_agents.py`) keyed by (messages + params hash) → JSON.

---
Search Modes
------------

MCTS (`state_aware_rag/planners/MCTS`):

- Expands reasoning tree via rollouts.
- Nodes: USER_QUESTION, REPHASED_QUESTION, SUB_QA, SYNTHESIS, FINAL_ANSWER, SELF_CORRECTED.
- Selection guided by UCT (exploration_weight).
- Optional tree persistence: JSONL per node with visits & reward.

CoT (`state_aware_rag/planners/CoT`):

- Linear chain expansion (no branching) – lower compute, faster baselines.

---
Caching & Performance
---------------------

Tips:

1. Enable per-model cache: `use_cache: true` in agent configs – avoids repeat LLM costs.
2. Adjust `concurrency` / `num_workers` to match endpoint QPS capacity; start conservative (e.g. 8–16).
3. Use `num_proc` (datasets map) ≤ physical cores.
4. Trim search space: lower `max_depth` / `num_rollouts` for quick iteration.
5. Clear outdated cache if prompt changes: delete `cache/<role>/<model_name>/`.

---
Evaluation & Metrics
--------------------

Implemented in `evaluate.py` and `state_aware_rag/utils/metrics.py`.

Available metrics:

- F1 (token overlap)
- Exact Match (EM)
- Sub Exact Match (Sub-EM; partial multi-answer coverage)
- LLM Judge (configurable judging model)


---
Directory Structure (selected)
------------------------------

```text
inference.py                  # Legacy inference entry (Hydra)
langgraph_sar/
  inference.py                # LangGraph inference entry (Hydra)
  system.py                   # answer_question / batch / metric filtering
  config.yaml                 # Default SGLang endpoints & search knobs
  scripts/smoke_live.py       # Live end-to-end smoke test
evaluate.py                   # Metrics (shared by legacy + langgraph_sar batch glue)
configs/
  infer/                      # Hydra — legacy stack
  infer_langgraph/            # Hydra — LangGraph port
state_aware_rag/              # Legacy agents & planners (training reference)
results/                      # Saved HF datasets of predictions
cache/                        # Legacy LLM response caches
scripts/                      # Deploy / eval shell helpers
```

---
Extending the Framework
-----------------------
New LLM backend:
1. Implement a client in `agents/llm_agents.py` (follow `LiteLLMClient` pattern).
2. Add `client_type` selection logic.

Custom retriever:
1. Adapt `agents/retriever_agents.py` expectation: POST `{query, top_k, return_score?, instruction?}` → `{retrieved_docs: [[{id, contents, url?}, ...], ...]}`.
2. Update your retriever server or index loader.

New planner:
1. Create directory under `planners/<Name>/` with a `search` function signature mirroring existing planners.
2. Wire selection in `inference.generate_answer` based on `mode`.

Additional node / role:

- Extend `NodeType` & `ReasoningNode` logic, ensure serialization fields updated.

---
Troubleshooting
---------------
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty `retrieval_docs` | Retriever URL wrong or no results | Verify API endpoint & corpus |
| LangGraph `AuthenticationError` on judge | Evaluator `api_base: null` routes to OpenAI | Set evaluator tier to `http://localhost:30172/v1` in `langgraph_sar/config.yaml` |
| LangGraph parse errors after JSON | Qwen hit `max_tokens` with trailing whitespace | Lower temperature; parsing strips first JSON object; watch `SARExplicitFallbackWarning` from `langgraph_sar/llm.py` |
| `FAISS index not found` | Missing local corpus | `SAR_CORPUS_INDEX_PATH` or run `create_toy_corpus.py` |
| Cache not updating | Prompt or params changed but same hash path reused | Manually clear `cache/<role>/<model>` |
| MCTS very slow | Large `num_rollouts` * `max_depth` | Reduce both; enable `verbose=false` |
| OOM / rate errors | Concurrency too high | Lower `num_workers` & `concurrency` |

Logging: set `LOGGING_LEVEL=INFO` (default) or adjust per run.

---
Roadmap / Ideas
---------------

- Retrieval-conditioned adaptive rollouts (stop early on convergence)
- Graph-based memory 
- Tool invocation / function calling integration
- Multi-corpus hybrid retrieval (dense + sparse fusion)

Contributions via issues / PRs welcome.

---
License & Citation
------------------
This project is released under the MIT License (see `pyproject.toml`; some bundled third-party components retain their original licenses under `finetune/rl/verl/`).

<!-- If you use State-Aware RAG in academic or industrial work, please cite (placeholder):

```bibtex
@software{state_aware_rag_2025,
  title = {State-Aware RAG: Multi-Hop Reasoning with MCTS and CoT Planners},
  author = {Hieu M. and Contributors},
  year = {2025},
  url = {https://github.com/hieum98/state-aware-rag}
}
``` -->

---
FAQ
---
Q: Can I resume a previous MCTS run?  
A: Yes, set `search.save_tree=true`. If the JSONL already exists for `question_id`, it will reload instead of recomputing.

Q: How do I change the model for the Generator only?  
A: Point `agents.generator` to a different YAML or override `agents.generator.client_kwargs.model_name=...`.

Q: Do I need golden answers?  
A: No for serving. In the **legacy** stack, gold is optional and gated by `search.use_golden_answer`. In the **LangGraph** port, gold feeds outcome reward when provided; with `eval_mode=judge_only` (default) it never enters generator/extractor prompts.

Q: What is `eval_mode` on the LangGraph CLI?  
A: `judge_only` (default) = reference for judges/metrics only. `teacher_forced` = synthesis runs excluded from reported metrics; graph branching for teacher forcing is not fully implemented yet.

---
Support
-------
Please open an issue for bugs or feature requests.

---
Happy reasoning!
