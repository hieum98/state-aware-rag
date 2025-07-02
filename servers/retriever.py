import os, sys
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser

from agents.retriever_agents import FlashRAGRetrieverAgent
from flask import Flask, request, jsonify

app = Flask(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="FlashRAG Retriever Server")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file into dictionary
    with open(args.config, 'r', encoding='utf-8') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    retriever = FlashRAGRetrieverAgent(**config_dict)

    @app.route('/search', methods=['POST'])
    def search():
        data = request.get_json()
        query = data.get('query', None)
        top_k = data.get('top_k', 5)
        return_score = data.get('return_score', False)
        instruction = data.get('instruction', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        retrieved_docs = retriever.search(query, top_k=top_k, return_score=return_score, instruction=instruction)
        return jsonify(retrieved_docs)

    app.run(host='0.0.0.0', port=5000, debug=False)

    # The Flask app will run and listen for incoming requests on port 5000.
    # python -m agents.servers.retriever --config path/to/config.yaml

