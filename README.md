State-Aware RAG 
=================================

State-Aware RAG is a modular multi-hop Retrieval-Augmented Generation (RAG) framework featuring two complementary search paradigms:

1. MCTS (Monte Carlo Tree Search) planner for exploratory, branching multi-step reasoning
2. CoT (Chain-of-Thought) linear planner for lightweight sequential reasoning

The system performs iterative (decompose → retrieve → extract → evaluate → synthesize) cycles while maintaining an explicit evolving reasoning state ("memory"). Trees and intermediate artifacts can be persisted, re-loaded, and post-analyzed.

> Core design goals: transparency, reproducibility, pluggability (LLMs & retrievers), and scalability (multi-processing + concurrency + caching).

Table of Contents
-----------------

- Features
- Architecture Overview
- Quick Start
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
- Pluggable LLM backends via OpenAI-compatible or LiteLLM interface
- Online (HTTP API) and optional offline (FlashRAG) retrieval
- Structured output extraction with fallback text parsing
- Concurrency + multi-processing + deterministic caching
- Reasoning tree export & reload (resumable search)
- Evaluation suite: F1, Exact Match, Sub EM, Retrieval Recall, LLM Judge
- Hydra + YAML config for reproducible experiment control

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

- `inference.py` – main entry (single question or dataset map)
- `evaluate.py` – metric computation and judged scoring
- `state_aware_rag/agents/` – agent abstractions & role implementations
- `state_aware_rag/planners/` – search logic (`MCTS` and `CoT`)
- `configs/infer/` – Hydra configs for experiments
- `cache/` – per-model LLM call cache

Reasoning State:

Each node logs: node type, memory snapshot, confidence, content (sub-question, answer, synthesis, final answer), plus lineage. Trees can be reloaded to continue or analyze.

Quick Start
-----------

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

Hydra config tree (simplified):

```text
configs/infer/base.yaml
  ├─ mode: mcts | cot
  ├─ results_dir: results
  ├─ num_proc: <int>
  ├─ agents:
  │   ├─ generator: (inline dict or path to YAML)
  │   ├─ extractor:
  │   ├─ evaluator:
  │   └─ retriever:
  ├─ search:
  │   ├─ max_depth
  │   ├─ num_rollouts
  │   ├─ exploration_weight
  │   ├─ top_k
  │   ├─ save_tree
  │   └─ verbose
  └─ data:
      ├─ name (2wiki | hotpotqa | musique | ...)
      ├─ split
      └─ limit
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
- Retrieval Recall
- LLM Judge (configurable judging model)

Evaluation annotates dataset with per-sample details (e.g. `f1_detail`, `llm_judge_detail`).

---
Directory Structure (selected)
------------------------------

```text
inference.py            # Main inference entry
evaluate.py             # Metrics & dataset annotation
configs/                # Hydra configs
state_aware_rag/
  agents/              # Agent abstractions & role logic
  planners/            # MCTS & CoT planners
  preprocess/          # Normalization utilities
  utils/               # Metrics & helpers
results/                # Saved HF datasets of predictions
cache/                  # LLM response caches
mcts_trees/             # (Optional) reasoning tree JSONLs
scripts/                # Helper shell scripts (deploy, eval)
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
| Cache not updating | Prompt or params changed but same hash path reused | Manually clear `cache/<role>/<model>` |
| MCTS very slow | Large `num_rollouts` * `max_depth` | Reduce both; enable `verbose=false` |
| OOM / rate errors | Concurrency too high | Lower `num_workers` & `concurrency` |
| Missing metric columns | Evaluation run before inference or wrong path | Point `to_eval_path` to saved dataset |
| Unicode / formatting mismatch | Preprocessing differences | Use `simple_preprocess` for normalization |

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
A: No. They are optional and only used when `use_golden_answer` is true (e.g., supervision / debugging modes).

---
Support
-------
Please open an issue for bugs or feature requests. Include:

- Config overrides used
- Console logs (trim secrets)
- Dataset name / sample

---
Happy reasoning!

