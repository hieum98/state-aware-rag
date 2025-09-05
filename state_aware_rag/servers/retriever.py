import os, sys
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

app = FastAPI()

class SearchRequest(BaseModel):
    query: Union[str, List[str]]
    top_k: int = 16
    return_score: bool = False
    instruction: Optional[str] = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = ArgumentParser(description="Retriever Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = SimpleNamespace(**config)
    if config.encoder_name == 'e5':
        default_instruction = "query: {query}"
    elif config.encoder_name == 'qwen3':
        default_instruction = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}'

    # Load the faiss index and the corpus
    index_path = config.index_path
    corpus_path = config.corpus_path
    if corpus_path.endswith(".jsonl"):
        corpus = datasets.load_dataset('json', data_files=corpus_path, split="train")
    elif corpus_path.endswith(".parquet"):
        corpus = datasets.load_dataset('parquet', data_files=corpus_path, split="train")
        corpus = corpus.cast_column('image', datasets.Image())
    else:
        corpus = datasets.load_from_disk(corpus_path)
    index = faiss.read_index(index_path)
    logging.info(f"Loaded corpus with {len(corpus)} documents from {corpus_path}.")
    logging.info(f"Loaded index with {index.ntotal} vectors from {index_path}.")

    encoder_model_base_url = config.encoder_model_base_url
    encoder_model_api_key = config.encoder_model_api_key
    encoder_model_client = openai.OpenAI(
        base_url=encoder_model_base_url,
        api_key=encoder_model_api_key
    )
    models = encoder_model_client.models.list()
    model = models.data[0].id

    @app.post('/search')
    def search(request: SearchRequest):
        query = request.query
        top_k = request.top_k
        return_score = request.return_score
        instruction = request.instruction
        if instruction is None:
            instruction = default_instruction
            logging.info(f"Using default instruction: {instruction}")
        if not query:
            logging.warning("No query provided. Returning empty results.")
            return {"retrieved_docs": [], "scores": []} if return_score else {"retrieved_docs": []}
        if isinstance(query, str):
            query = [query]

        try:
            assert '{query}' in instruction, "Instruction must contain a {query} placeholder. Falling back to raw query."
            query = [instruction.format(query=q) for q in query]
        except:
            logging.warning("Error formatting query with instruction. Using raw query.")
            query = query
        try:
            logging.info(f"Encoding query: {query}")
            query_embeddings = encoder_model_client.embeddings.create(
                input=query,
                model=model
            )
            query_embeddings = [item.embedding for item in query_embeddings.data]
            query_embeddings = np.array(query_embeddings).astype('float32')
            # Normalize the embeddings to unit length
            faiss.normalize_L2(query_embeddings)
            scores, doc_idxs = index.search(query_embeddings, top_k)
            scores = scores.tolist()
            doc_idxs = doc_idxs.tolist()
            all_retrieved_docs = []
            for item in doc_idxs:
                retrieved_docs = []
                for idx in item:
                    retrieved_docs.append(corpus[int(idx)])
                all_retrieved_docs.append(retrieved_docs)
            retrieved_docs = {
                'retrieved_docs': all_retrieved_docs,
                'scores': scores if return_score else None
            }
        except:
            logging.error("Error during retrieval. Check your query and index.")
            logging.error(f"Query: {query}")
            logging.error(f"Index path: {index_path}")
            logging.error(f"Corpus path: {corpus_path}")
            retrieved_docs = {"retrieved_docs": [], "scores": []} if return_score else {"retrieved_docs": []}

        return retrieved_docs

    uvicorn.run(app, host='0.0.0.0', port=args.port)

    # python -m servers.retriever --config path/to/config.yaml