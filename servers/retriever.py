import os, sys
import numpy as np
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser
from flask import Flask, request, jsonify
import faiss
import datasets
import openai

app = Flask(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="Retriever Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = SimpleNamespace(**config)

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
    print(f"Loaded corpus with {len(corpus)} documents.")
    print(f"Loaded index with {index.ntotal} vectors.")

    encoder_model_base_url = config.encoder_model_base_url
    encoder_model_api_key = config.encoder_model_api_key
    encoder_model_client = openai.OpenAI(
        base_url=encoder_model_base_url,
        api_key=encoder_model_api_key
    )
    models = encoder_model_client.models.list()
    model = models.data[0].id

    @app.route('/search', methods=['POST'])
    def search():
        data = request.get_json()
        query = data.get('query', None)
        top_k = data.get('top_k', 16)
        return_score = data.get('return_score', False)
        if not query:
            return jsonify({"error": "Query is required"}), 400
        if isinstance(query, str):
            query = [query]

        try:
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
            return jsonify({"error": "Failed to retrieve"}), 500

        # retrieved_docs = retriever.search(query, top_k=top_k, return_score=return_score, instruction=instruction)
        # if reranker:
        #     reranker_top_k = data.get('reranker_top_k', top_k)
        #     if reranker_top_k < top_k:
        #         docs = retrieved_docs['retrieved_docs']
        #         assert len(query) == len(docs), "Query and documents length mismatch"
        #         reranker_instruction = data.get('reranker_instruction', None)
        #         return_docs = []
        #         for q, d in zip(query, docs):
        #             reranker_docs = reranker.rerank(q, d, instruction=reranker_instruction, top_k=reranker_top_k)
        #             return_docs.append(reranker_docs)
        #         retrieved_docs['retrieved_docs'] = return_docs
        return jsonify(retrieved_docs)

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

    # The Flask app will run and listen for incoming requests on port 5000.
    # python -m servers.retriever --config path/to/config.yaml 