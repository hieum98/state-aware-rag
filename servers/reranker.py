import os, sys
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser

from agents.reranker_agents import VLLMReranker
from flask import Flask, request, jsonify

app = Flask(__name__)
if __name__ == "__main__":
    parser = ArgumentParser(description="VLLM Reranker Server")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for reranking")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum length of the input sequence")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on")
    args = parser.parse_args()
    
    reranker = VLLMReranker(model_name_or_path=args.model_name, max_length=args.max_length, num_gpus=args.num_gpus)

    @app.route('/rerank', methods=['POST'])
    def rerank():
        data = request.get_json()
        query = data.get('query', None)
        documents = data.get('documents', [])
        top_k = data.get('top_k', 5)
        instruction = data.get('instruction', None)
        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not documents:
            return jsonify({"error": "Documents are required"}), 400
        
        reranked_docs = reranker.rerank(query, documents, instruction=instruction, top_k=top_k)
        return jsonify({"reranked_documents": reranked_docs})

    app.run(host='0.0.0.0', port=args.port, debug=False)

    # The Flask app will run and listen for incoming requests on port 5000.
    # python -m servers.reranker --model_name Qwen/Qwen3-Reranker-4B --max_length 8192 --num_gpus 8



