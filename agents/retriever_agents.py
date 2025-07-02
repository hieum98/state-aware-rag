import requests
import json
import time
from typing import List, Union
from flashrag.retriever.retriever import DenseRetriever
from flashrag.retriever.index_builder import Index_Builder


class FlashRAGRetrieverAgent:
    """
    A retriever agent that uses the FlashRAG DenseRetriever to retrieve relevant documents.
    """
    def __init__(
            self,
            retriever_method: str,
            retrieval_model_path: str,
            # Path settings
            corpus_path: str,
            index_path: str = None,
            # Indexing settings
            retrieval_pooling_method: str = "mean",
            faiss_gpu: bool = False,
            use_sentence_transformer: str = False,
            bm25_backend: str = "pyserini",
            # Retriever settings
            retrieval_topk: int = 5,
            retrieval_batch_size: int = 32,
            retrieval_use_fp16: bool = False,
            # Query settings
            retrieval_query_max_length: int = 2048,
            # Cache settings
            save_dir: str = "retrieval_cache",
            save_retrieval_cache: bool = True,
            use_retrieval_cache: bool = False,
            retrieval_cache_path: str = "retrieval_cache",
            # Reranker settings
            use_reranker: bool = False,
            rerank_model_name: str = None,
            rerank_model_path: str = None,
            **kwargs
            ):
        """
        Initializes the FlashRAGRetrieverAgent with the given configuration.
        Args:
            retrieval_method (str): The model to use for retrieval.
            retrieval_model_path (str): Path to the model used for retrieval.
            corpus_path (str): Path to the corpus to be indexed.
            index_path (str): Path to the index, if none compute the index.
            retrieval_pooling_method (str): Pooling method for the retrieval model.
            faiss_gpu (bool): Whether to use GPU for FAISS indexing.
            use_sentence_transformer (bool): Whether to use SentenceTransformer for embeddings.
            bm25_backend (str): Backend to use for BM25 retrieval.
            retrieval_topk (int): Number of top documents to retrieve.
            retrieval_batch_size (int): Batch size for retrieval.
            retrieval_use_fp16 (bool): Whether to use FP16 precision for retrieval.
            retrieval_query_max_length (int): Maximum length of the query for retrieval.
            save_retrieval_cache (bool): Whether to save the retrieval cache.
            use_retrieval_cache (bool): Whether to use a pre-existing retrieval cache.
            retrieval_cache_path (str): Path to the retrieval cache directory.
            use_reranker (bool): Whether to use a reranker after initial retrieval.
            rerank_model_name (str): Name of the rerank model, if applicable.
            rerank_model_path (str): Path to the rerank model, if applicable.
        """
        self.config = {
            "retrieval_method": retriever_method,
            "retrieval_model_path": retrieval_model_path,
            "corpus_path": corpus_path,
            "retrieval_pooling_method": retrieval_pooling_method,
            "faiss_gpu": faiss_gpu,
            "use_sentence_transformer": use_sentence_transformer,
            "bm25_backend": bm25_backend,
            "retrieval_topk": retrieval_topk,
            "retrieval_batch_size": retrieval_batch_size,
            "retrieval_use_fp16": retrieval_use_fp16,
            "instruction": "", # Always empty for retriever agent because we will add the instruction in the search method.
            "retrieval_query_max_length": retrieval_query_max_length,
            "save_retrieval_cache": save_retrieval_cache,
            "save_dir": save_dir,
            "use_retrieval_cache": use_retrieval_cache,
            "retrieval_cache_path": retrieval_cache_path,
            "use_reranker": use_reranker,
            "rerank_model_name": rerank_model_name,
            "rerank_model_path": rerank_model_path,
            "rerank_batch_size": retrieval_batch_size,
            "rerank_use_fp16": retrieval_use_fp16,
            "rerank_pooling_method": retrieval_pooling_method,
            "rerank_max_length": retrieval_query_max_length,
        }
        self.query_instruction = kwargs.get('query_instruction', "")
        if index_path is None:
            index_path = self.build_index(config=self.config, save_dir="indexes",)['index_save_path']
        self.config["index_path"] = index_path
        self.retriever = DenseRetriever(self.config)
    
    @staticmethod
    def build_index(
            config: dict,
            save_dir: str,
            max_length: int = 4096,
            batch_size: int = 32,
            instruction: str = "",
            faiss_type: str = 'Flat',
            save_embedding: bool = False,
            embedding_path: str = None
            ):
        index_builder = Index_Builder(
            retrieval_method=config["retrieval_method"],
            model_path=config["retrieval_model_path"],
            corpus_path=config["corpus_path"],
            save_dir=save_dir,
            max_length=max_length,
            batch_size=batch_size,
            use_fp16=False,
            pooling_method=config["retrieval_pooling_method"],
            instruction=instruction,
            faiss_type=faiss_type,
            embedding_path=embedding_path,
            save_embedding=save_embedding,
            faiss_gpu=config["faiss_gpu"],
            use_sentence_transformer=config["use_sentence_transformer"],
            bm25_backend=config["bm25_backend"],
            index_modal='all',
        )
        index_builder.build_index()
        return {
            'embedding_save_path': index_builder.embedding_save_path,
            'index_save_path': index_builder.index_save_path,
        }
    
    def search(self, query: Union[str, List[str]], top_k: Union[int, None] = None, return_score=False, instruction: str = ''):
        """
        Searches for the top-k relevant documents for a given query.
        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            top_k (int): Number of top documents to retrieve. Defaults to self.config["retrieval_topk"].
            return_score (bool): Whether to return the scores of the retrieved documents.
        Returns:
            dict: A dictionary containing the retrieved documents and their scores (if requested).
        """
        if top_k is None:
            top_k = self.config["retrieval_topk"]
        if instruction is None:
            instruction = self.query_instruction
        if isinstance(query, str):
            if instruction:
                query = f"{instruction} {query}"
            retrieved_docs, scores = [], []
            results = self.retriever.search(query, top_k, return_score=return_score)
            for item in results:
                retrieved_docs.append(item['contents'])
                scores.append(item.get('score', None))
        elif isinstance(query, list):
            if instruction:
                query = [f"{instruction} {q}" for q in query]
            retrieved_docs, scores = [], []
            results = self.retriever.batch_search(query, top_k, return_score=return_score)
            for item in results:
                retrieved_docs.append([x['contents'] for x in item])
                scores.append([x.get('score', None) for x in item])
        return {
            "retrieved_docs": retrieved_docs,
            "scores": scores if return_score else None
        }
    

class APIRetrieverAgent:
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.headers = {'Content-Type': 'application/json'}
        self.retrieval_topk = kwargs.get('retrieval_topk', 5)
        self.query_instruction = kwargs.get('query_instruction', "")

    def search(self, query: Union[str, List[str]], top_k: int = None, return_score=False, instruction: str = None):
        """
        Searches for the top-k relevant documents for a given query using an API.
        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            top_k (int): Number of top documents to retrieve.
            return_score (bool): Whether to return the scores of the retrieved documents.
            instruction (str): Additional instruction to prepend to the query.
        Returns:
            dict: A dictionary containing the retrieved documents and their scores (if requested).
        """
        if top_k is None:
            top_k = self.retrieval_topk
        if instruction is None:
            instruction = self.query_instruction
        data = json.dumps({
            "query": query,
            "top_k": top_k,
            "return_score": return_score,
            "instruction": instruction
        })
        begin_time = time.time()
        response = requests.post(self.url, headers=self.headers, data=data)
        end_time = time.time()
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        response_data = response.json()
        assert "retrieved_docs" in response_data, "Response does not contain 'retrieved_docs' key"
        return {
            "retrieved_docs": response_data.get("retrieved_docs", []),
            "scores": response_data.get("scores", []) if return_score else None,
            "response_time": end_time - begin_time
        }


class RetrieverAgent:
    def __init__(self, offline_kwargs=None, online_kwargs=None):
        assert offline_kwargs is not None or online_kwargs is not None, "Either offline_kwargs or online_kwargs must be provided"
        if online_kwargs is not None:
            self.agent = APIRetrieverAgent(**online_kwargs)
        else:
            self.agent = FlashRAGRetrieverAgent(**offline_kwargs)
    
    def search(self, query: Union[str, List[str]], top_k: int = None, return_score=False, instruction: str = ''):
        """
        Searches for the top-k relevant documents for a given query.
        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            top_k (int): Number of top documents to retrieve.
            return_score (bool): Whether to return the scores of the retrieved documents.
            instruction (str): Additional instruction to prepend to the query.
        Returns:
            dict: A dictionary containing the retrieved documents and their scores (if requested).
        """
        return self.agent.search(query, top_k=top_k, return_score=return_score, instruction=instruction)


if __name__ == "__main__":
    # Run on server side
    # python -m agents.servers.retriever --config path/to/config.yaml
    # Example usage
    # retriever_agent = FlashRAGRetrieverAgent(
    #     retriever_method="e5",
    #     retrieval_model_path="intfloat/e5-base-v2",
    #     corpus_path="data/wiki18_100w.jsonl",
    #     index_path="indexes/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index",  # Will build index if None
    #     retrieval_topk=5,
    #     retrieval_batch_size=32,
    #     retrieval_use_fp16=False,
    #     retrieval_query_max_length=2048,
    #     retrieval_pooling_method='mean',
    #     use_sentence_transformer=True,
    # )

    online_kwargs = {
        "url": "http://n0998.talapas.uoregon.edu:5000/search",
        "retrieval_topk": 5,
        "query_instruction": "query: ",
    }
    retriever_agent = RetrieverAgent(online_kwargs=online_kwargs)

    query = ["What is the capital of France?", "Who is the president of the United States?"]
    results = retriever_agent.search(query, top_k=5, instruction=None)
    breakpoint()