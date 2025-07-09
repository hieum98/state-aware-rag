import os, sys
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser
from flask import Flask, request, jsonify

from agents.reranker_agents import VLLMReranker
from agents.retriever_agents import FlashRAGRetrieverAgent

app = Flask(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="FlashRAG Retriever Server")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--reranker_model_name", type=str, default=None, help="Name of the model to use for reranking")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum length of the input sequence")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on")
    args = parser.parse_args()

    # Load configuration from YAML file into dictionary
    with open(args.config, 'r', encoding='utf-8') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    retriever = FlashRAGRetrieverAgent(**config_dict)
    if args.reranker_model_name:
        print(f"Using reranker model: {args.reranker_model_name}")
        gpu_memory_utilization = config_dict.get('gpu_memory_utilization', 0.9)
        reranker = VLLMReranker(model_name_or_path=args.reranker_model_name, max_length=args.max_length, num_gpus=args.num_gpus, gpu_memory_utilization=gpu_memory_utilization)
    else:
        reranker = None

    @app.route('/search', methods=['POST'])
    def search():
        data = request.get_json()
        query = data.get('query', None)
        top_k = data.get('top_k', 16)
        return_score = data.get('return_score', False)
        instruction = data.get('instruction', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400
        if isinstance(query, str):
            query = [query]

        retrieved_docs = retriever.search(query, top_k=top_k, return_score=return_score, instruction=instruction)
        if reranker:
            reranker_top_k = data.get('reranker_top_k', top_k)
            if reranker_top_k < top_k:
                docs = retrieved_docs['retrieved_docs']
                assert len(query) == len(docs), "Query and documents length mismatch"
                reranker_instruction = data.get('reranker_instruction', None)
                return_docs = []
                for q, d in zip(query, docs):
                    reranker_docs = reranker.rerank(q, d, instruction=reranker_instruction, top_k=reranker_top_k)
                    return_docs.append(reranker_docs)
                retrieved_docs['retrieved_docs'] = return_docs
        return jsonify(retrieved_docs)

    app.run(host='0.0.0.0', port=args.port, debug=False)

    # The Flask app will run and listen for incoming requests on port 5000.
    # python -m servers.retriever --config path/to/config.yaml --reranker_model_name Qwen/Qwen3-Reranker-4B --max_length 8192 --num_gpus 1 --port 5000

