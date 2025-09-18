import os, sys, math
import numpy as np
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser
import faiss
import datasets
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
import uvicorn
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Notes on multiprocessing approach:
# 1. We scale throughput primarily by running multiple uvicorn workers (processes).
# 2. For large batch requests we optionally parallelize embedding calls by chunking
#    the input queries and issuing multiple embedding requests concurrently.
# 3. FAISS already supports batched search efficiently; we only need to batch
#    embedding requests when the remote embedding service has per-request throughput limits.

DEFAULT_MAX_BATCH_SIZE = 64  # number of queries per embedding API call
DEFAULT_EMBED_CONCURRENCY = 8  # threads per worker process for embedding calls

# Environment variable keys (set by CLI in main()) used by worker processes.
ENV_CONFIG_PATH = "RETRIEVER_CONFIG_PATH"
ENV_MMAP_INDEX = "RETRIEVER_MMAP_INDEX"
ENV_MAX_BATCH = "RETRIEVER_MAX_BATCH_SIZE"
ENV_EMBED_CONCURRENCY = "RETRIEVER_EMBED_CONCURRENCY"

app = FastAPI()

class SearchRequest(BaseModel):
    query: Union[str, List[str]]
    top_k: int = 16
    return_score: bool = False
    instruction: Optional[str] = None
    # Optional override for max batch size / concurrency per request
    max_batch_size: Optional[int] = None
    embed_concurrency: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    pid: int
    docs: int
    vectors: int


# Globals initialized in main when running as script. Each worker process will
# have its own copy (unless mmap is enabled for the FAISS index file).
corpus = None
index = None
encoder_model_client = None
embedding_model_id = None
default_instruction = None
server_args = None

@app.post('/search')
def search(request: SearchRequest):
    """Search endpoint with optional intra-request embedding parallelism.

    Accepts single string or list of queries. For large query lists we chunk +
    parallelize embedding calls to improve throughput (I/O bound on network).
    """
    global corpus, index, encoder_model_client, embedding_model_id, default_instruction

    query = request.query
    top_k = request.top_k
    return_score = request.return_score
    instruction = request.instruction or default_instruction
    max_batch_size = request.max_batch_size or getattr(server_args, 'max_batch_size', DEFAULT_MAX_BATCH_SIZE)
    embed_concurrency = request.embed_concurrency or getattr(server_args, 'embed_concurrency', DEFAULT_EMBED_CONCURRENCY)

    if not query:
        logging.warning("No query provided. Returning empty results.")
        return {"retrieved_docs": [], "scores": []} if return_score else {"retrieved_docs": []}
    if isinstance(query, str):
        query = [query]

    # Format with instruction
    if '{query}' in instruction:
        try:
            query = [instruction.format(query=q) for q in query]
        except Exception as e:
            logging.warning(f"Failed to format instruction, using raw query. Error: {e}")
    else:
        logging.debug("Instruction missing {query} placeholder; using raw queries.")

    # Helper: chunk list
    def chunk_list(items, size):
        for i in range(0, len(items), size):
            yield items[i:i+size]

    # Fetch embeddings (possibly multiple API calls in parallel)
    query_embeddings_list = []
    try:
        if len(query) <= max_batch_size:
            logging.info(f"Encoding {len(query)} queries in a single request (pid={os.getpid()}).")
            result = encoder_model_client.embeddings.create(input=query, model=embedding_model_id)
            query_embeddings_list.extend([item.embedding for item in result.data])
        else:
            chunks = list(chunk_list(query, max_batch_size))
            logging.info(f"Encoding {len(query)} queries via {len(chunks)} chunks (max_batch_size={max_batch_size}, concurrency={embed_concurrency}, pid={os.getpid()}).")
            with ThreadPoolExecutor(max_workers=embed_concurrency) as executor:
                future_to_idx = {executor.submit(encoder_model_client.embeddings.create, input=chunk, model=embedding_model_id): i for i, chunk in enumerate(chunks)}
                # Preserve order by storing results at chunk index
                chunk_results = [None]*len(chunks)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        r = future.result()
                        chunk_results[idx] = [item.embedding for item in r.data]
                    except Exception as e:
                        logging.error(f"Embedding chunk {idx} failed: {e}")
                        chunk_results[idx] = []
                # Flatten
                for cr in chunk_results:
                    query_embeddings_list.extend(cr)
        if len(query_embeddings_list) != len(query):
            logging.warning(f"Embedding count mismatch: expected {len(query)}, got {len(query_embeddings_list)}")
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return {"retrieved_docs": [], "scores": []} if return_score else {"retrieved_docs": []}

    # Convert to numpy and normalize
    query_embeddings = np.array(query_embeddings_list, dtype='float32')
    faiss.normalize_L2(query_embeddings)

    try:
        scores, doc_idxs = index.search(query_embeddings, top_k)
    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        return {"retrieved_docs": [], "scores": []} if return_score else {"retrieved_docs": []}

    scores_list = scores.tolist()
    doc_idxs_list = doc_idxs.tolist()
    all_retrieved_docs = []
    for item in doc_idxs_list:
        retrieved_docs = []
        for idx in item:
            try:
                retrieved_docs.append(corpus[int(idx)])
            except Exception:
                retrieved_docs.append({})
        all_retrieved_docs.append(retrieved_docs)

    payload = {'retrieved_docs': all_retrieved_docs}
    if return_score:
        payload['scores'] = scores_list
    return payload


@app.get('/health', response_model=HealthResponse)
def health():
    global corpus, index
    return HealthResponse(status='ok', pid=os.getpid(), docs=len(corpus) if corpus is not None else 0, vectors=index.ntotal if index is not None else 0)


def load_resources(config, mmap_index=False):
    """Load corpus, FAISS index, and embedding client based on config."""
    global corpus, index, encoder_model_client, embedding_model_id, default_instruction
    if config.encoder_name == 'e5':
        default_inst = "query: {query}"
    elif config.encoder_name == 'qwen3':
        default_inst = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}'
    else:
        default_inst = '{query}'
    default_instruction = default_inst

    corpus_path = config.corpus_path
    if corpus_path.endswith('.jsonl'):
        corpus_local = datasets.load_dataset('json', data_files=corpus_path, split='train')
    elif corpus_path.endswith('.parquet'):
        corpus_local = datasets.load_dataset('parquet', data_files=corpus_path, split='train')
        corpus_local = corpus_local.cast_column('image', datasets.Image())
    else:
        corpus_local = datasets.load_from_disk(corpus_path)

    # FAISS index
    index_flags = 0
    if mmap_index:
        # Memory map (read-only) can reduce per-worker RAM duplication for large indexes
        index_flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
    index_local = faiss.read_index(config.index_path, index_flags) if index_flags else faiss.read_index(config.index_path)

    # Embedding client
    client = openai.OpenAI(base_url=config.encoder_model_base_url, api_key=config.encoder_model_api_key)
    model_list = client.models.list()
    emb_model_id = model_list.data[0].id

    corpus = corpus_local
    index = index_local
    encoder_model_client = client
    embedding_model_id = emb_model_id

    logging.info(f"[pid={os.getpid()}] Loaded corpus ({len(corpus)} docs) and index ({index.ntotal} vectors). Using embedding model: {embedding_model_id}")


@app.on_event("startup")
def _startup_event():
    """Executed in every worker process after import; loads resources lazily."""
    global server_args
    if index is not None:
        # Already loaded (possible in single-process dev mode)
        return
    cfg_path = os.environ.get(ENV_CONFIG_PATH)
    if not cfg_path or not os.path.exists(cfg_path):
        logging.error("Startup: missing config path env variable; server will not function correctly.")
        return
    mmap_flag = os.environ.get(ENV_MMAP_INDEX, "0") == "1"
    try:
        with open(cfg_path, 'r') as f:
            cfg_raw = yaml.safe_load(f)
        config = SimpleNamespace(**cfg_raw)
        load_resources(config, mmap_index=mmap_flag)
    except Exception as e:
        logging.error(f"Failed to load resources at startup: {e}")
        return
    # Build server_args namespace from env so search() can read defaults
    server_args = SimpleNamespace(
        max_batch_size=int(os.environ.get(ENV_MAX_BATCH, DEFAULT_MAX_BATCH_SIZE)),
        embed_concurrency=int(os.environ.get(ENV_EMBED_CONCURRENCY, DEFAULT_EMBED_CONCURRENCY))
    )
    logging.info(f"Startup complete (pid={os.getpid()}) max_batch_size={server_args.max_batch_size} embed_concurrency={server_args.embed_concurrency}")


def main():
    """CLI entrypoint that sets environment variables then launches uvicorn with import string.

    Using an import string avoids the uvicorn warning and enables multiple workers.
    """
    global server_args
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = ArgumentParser(description="Retriever Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="Host interface")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2), help="Number of uvicorn worker processes")
    parser.add_argument("--mmap_index", action='store_true', help="Memory-map FAISS index read-only to reduce RAM per worker (Linux only)")
    parser.add_argument("--max_batch_size", type=int, default=DEFAULT_MAX_BATCH_SIZE, help="Max queries per embedding API call")
    parser.add_argument("--embed_concurrency", type=int, default=DEFAULT_EMBED_CONCURRENCY, help="Thread concurrency per worker for embedding calls")
    parser.add_argument("--disable_workers_import_string", action='store_true', help="(Debug) Run app object directly even with workers=1")
    args = parser.parse_args()
    server_args = args  # for single-process path

    # Export config for workers
    os.environ[ENV_CONFIG_PATH] = args.config
    os.environ[ENV_MMAP_INDEX] = "1" if args.mmap_index else "0"
    os.environ[ENV_MAX_BATCH] = str(args.max_batch_size)
    os.environ[ENV_EMBED_CONCURRENCY] = str(args.embed_concurrency)

    # For single worker we can still let startup load resources, but to reduce cold latency we may pre-load once.
    if args.workers == 1:
        try:
            with open(args.config, 'r') as f:
                cfg_raw = yaml.safe_load(f)
            config = SimpleNamespace(**cfg_raw)
            load_resources(config, mmap_index=args.mmap_index)
        except Exception as e:
            logging.error(f"Preload failed (will retry on startup): {e}")

    target = "state_aware_rag.servers.retriever:app"
    if args.workers == 1 and args.disable_workers_import_string:
        # Direct object path (mostly for debugging; still no workers>1)
        logging.info(f"Starting retriever server (single process) on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logging.info(f"Starting retriever server via import string {target} on {args.host}:{args.port} workers={args.workers}")
        uvicorn.run(target, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()

# Usage:
# python -m state_aware_rag.servers.retriever --config path/to/config.yaml --port 5000 --workers 4 --mmap_index