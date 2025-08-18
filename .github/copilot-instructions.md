# Copilot Instructions for state-aware-rag

Purpose and architecture
- This repo implements a state-aware multi-hop RAG system with two search modes:
  - MCTS search: planners/MCTS drives multi-step reasoning over retrieve–generate–evaluate cycles.
  - CoT search: planners/CoT runs linear chain-of-thought reasoning.
- Core agents live in `agents/roles/` and wrap the LLM (`Generator`, `Evaluator`, `Extractor`). Retrieval is abstracted by `agents/retriever_agents.py`.
- Cross-component flow (per question):
  1) Generator proposes sub-questions, answers and synthesis prompts (`agents/prompts/*`).
  2) Retriever fetches context via HTTP API (`APIRetrieverAgent`) or FlashRAG (optional).
  3) Evaluator scores candidates; Extractor parses/normalizes.
  4) Search (`planners/MCTS.utils.search` or `planners/CoT.utils.search`) orchestrates iterative steps.
  5) Results and (optionally) search trees are saved under `results/*` and `mcts_data/*`.

Key files and patterns
- `inference.py`: CLI entrypoint. Loads YAML configs for each agent, builds result directories, loads datasets (Hugging Face datasets), and maps `generate_answer` over samples.
  - Important: We initialize agents per-process when using `datasets.Dataset.map(num_proc>1)` to avoid pickling/shared-state issues.
- `agents/llm_agents.py`: LLM abstraction with two clients:
  - LiteLLMClient (generic models via litellm.completion) and OpenAIClient (OpenAI-compatible endpoints).
  - Caching: `LLMAgent` computes a hash of messages+kwargs and stores JSON under `cache/llm_agents/<model>/`. Use `use_cache` in configs.
  - Concurrency: client-level `batch_generate` uses a ThreadPool capped by `concurrency` in YAML.
- `agents/roles/{generator,evaluator,extractor}.py`: Thin role wrappers exposing typed methods and loading prompt templates from `agents/prompts/`.
- `agents/retriever_agents.py`: `RetrieverAgent` delegates to `APIRetrieverAgent` (HTTP JSON API) or FlashRAG (optional dependency).
- `evaluate.py`: Computes metrics (F1, EM, Sub-EM, Retrieval Recall, LLM Judge) and appends details to the dataset; saves results to disk.
- `args.py`: Dataclasses parsed by `transformers.HfArgumentParser` from YAML configs under `configs/infer/*.yaml`.

Developer workflows
- Inference (dataset mode):
  - Edit `configs/infer/*.yaml` as needed (model URLs, API keys, retriever URL, search params).
  - Run: `bash configs/infer/run_infer.sh` (adjust `--data_name`, `--results_dir`, `--use_mcts|--use_cot`).
  - Output: Hugging Face dataset saved under `results/<dataset>/<mode>/.../` with fields like `pred`, `detailed_answer`, and optional search artifacts under `result_trees/`.
- Inference (single question):
  - `python -m inference --question "..." --use_mcts --search_config configs/infer/search_config.yaml --generator_config ...`
- Evaluation over saved results:
  - `python -m evaluate --dataset_path results/... --metrics all`

Project-specific conventions
- YAML config via `HfArgumentParser.parse_yaml_file` for all agents. Keys map directly to dataclasses in `args.py`.
- Prompts are formatted as chat messages arrays: `[{"role":"user","content": prompt}]`; schema validation via Pydantic when models support response_schema.
- Multi-processing with datasets: avoid capturing agent instances in map closures; initialize per process (already handled in `inference.py`).
- Results are persisted as Hugging Face dataset directories; prefer `load_from_disk`/`save_to_disk` over ad-hoc JSON.

External services and dependencies
- Requires an OpenAI-compatible HTTP endpoint for LLMs (e.g., sglang server) and a retriever HTTP server (`servers/retriever.py` not used in inference configs by default; external retriever URL is required).
- Optional FlashRAG integration for offline retrieval (`pip install flashrag`).
- Uses `datasets`, `transformers`, `litellm`, `openai`, `pydantic`, `pebble`, `tqdm`.

Gotchas and tips
- High parallelism: set `configs/infer/*concurrency` to match endpoint capacity; set `--num_proc <= os.cpu_count()` for dataset map.
- Structured outputs: If the model doesn’t support response_schema, the system falls back to text parsing via `agents.utils.extract_info_from_text`.
- Caching: clear caches under `cache/` and `mcts_cache/` if prompts/kwargs change; cache key is a hash of messages+kwargs.
- Retrieval API contract: POST JSON {query, top_k, return_score, instruction}; response must include `retrieved_docs` (list or list[list]).

Examples
- Change generator model: edit `configs/infer/generator_config.yaml` (model_name, url, client_type) and rerun.
- Switch to CoT: pass `--use_cot` and ensure `planners/CoT/utils.py` is available with `search`.
- Evaluate 2wiki dev subset: run `bash configs/infer/run_infer.sh` (defaults to 2wiki, MCTS) then `python -m evaluate --dataset_path results/2wiki/... --metrics all`.

House rules for AI agents
- Don’t pass agent instances into multiprocessing workers; always lazily init per process.
- Keep edits within role classes and prompts; avoid changing planner interfaces unless updating both MCTS/CoT utils.
- Prefer saving intermediate artifacts via Hugging Face dataset columns rather than ad-hoc prints.
- Validate any API shape changes in `agents/retriever_agents.py` against `servers/retriever.py`.
