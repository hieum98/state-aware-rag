# Server Deployment Guide

This document consolidates all steps required to stand up the inference stack for State-Aware RAG.

## Overview
You will run two (sometimes three) logical services:

1. LLM Server (chat/completions) – used by Generator, Extractor, Evaluator, Judge.
2. Embedding + Retriever Stack – embedding model server + FastAPI retriever (FAISS index).
3. (Optional) Separate Judge model server (if you don't reuse the main LLM) for evaluation.

```
+-------------------+            +--------------------+
|   LLM Server      |<--HTTP---->|  StateAwareRAG.    |
| (sglang/vLLM)     |            |                    |
+-------------------+            +---------+----------+
                                          |
                                          v
                               +---------------------+
                               |  Retriever Server   |
                               | (FastAPI + FAISS)   |
                               +----------+----------+
                                          |
                                          v
                                +-------------------+
                                | Embedding Server  |
                                |  (sglang/vLLM)    |
                                +-------------------+
```

## 0. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# install sglang server, you can install vllm insteads
pip install uv
uv pip install "sglang[all]>=0.5.3rc0"
# install faiss for retriever server
pip install faiss-cpu

# Optional extras
pip install faiss-cpu
```

## 1. Start LLM Server
Minimal local launch:
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 --mem-fraction-static 0.8
```
You may want to deploy a separate server for extractor using model we fine-tuned at: https://huggingface.co/Hieuman/State-Aware-RAG-Extractor-4B.v0
```bash
python -m sglang.launch_server \
  --model-path Hieuman/State-Aware-RAG-Extractor-4B.v0 \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 --mem-fraction-static 0.8
```
Note the base URL: `http://<host>:30000/v1`

## 2. Start Embedding Server
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --mem-fraction-static 0.9
```
Base URL: `http://<host>:8000/v1`

## 3. Start Retriever Server
Edit (or verify) `configs/servers/retriever-Qwen3-4B-wiki-23.yaml`:
```yaml
encoder_model_base_url: http://<host>:8000/v1
encoder_model_api_key: YOUR_OPENAI_API_KEY   # or dummy if not enforced
encoder_name: qwen3
corpus_path: data/wiki23-Qwen3-4B-Emb-Indexed
index_path: data/wiki23-Qwen3-4B-Emb-Indexed/index.faiss
```
### Preprocessed Corpus & FAISS Index (Ready-to-Use)
We provide the full preprocessed Wikipedia 2023 subset (chunked + embedded with `Qwen/Qwen3-Embedding-4B`) **and** the matching FAISS index on Hugging Face:

https://huggingface.co/datasets/Hieuman/wiki23-processed/tree/main

After download, ensure your retriever config points to the directory (it should contain multiple `.arrow` files for corpus plus `index.faiss`).

Memory note: enabling `--mmap_index` is strongly recommended to avoid loading the full FAISS index into each worker's heap.

Launch:
```bash
python -m state_aware_rag.servers.retriever \
  --config configs/servers/retriever-Qwen3-4B-wiki-23.yaml \
  --port 5000 --workers 4 --mmap_index
```
Search test:
```bash
curl -X POST http://localhost:5000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "When was the Eiffel Tower constructed?", "top_k": 3}' | jq .
```


## 4. SLURM Scripts
- `scripts/deploy_llm_server.sh` – multi-GPU LLM server (auto DP/TP)
- `scripts/deploy_retriever_server.sh` – embedding + retriever
Adjust partition, ports, and model path as needed.

## 5. Configure Agents
Example `configs/generator.yaml` snippet:
```yaml
client_kwargs:
  client_type: openai
  model_name: "openai/qwen3-8B"
  url: "http://<host>:30000/v1"
  api_key: YOUR_KEY
  concurrency: 32
```
Example `configs/retriever.yaml`:
```yaml
online_retrieval_config:
  url: "http://<host>:5000/search"
  timeout: 300
```

## 6. Operational Tips
- Enable `--mmap_index` for large FAISS indexes to reduce RAM duplication.
- Align embedding model with index build (dimension & normalization).
- Use moderate `--workers` and tune `--max_batch_size` for throughput.
- Clear caches (`cache/<role>/<model>`) after prompt/template changes.


## 7. Tear Down
```bash
pkill -f sglang.launch_server || true
pkill -f state_aware_rag.servers.retriever || true
```

---
This guide should get you from environment setup to full multi-hop inference + evaluation.
