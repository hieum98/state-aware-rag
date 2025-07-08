import json
import math
import time
from typing import List, Union
import requests
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


class HFReranker:
    def __init__(self, model_name_or_path: str, max_length: int = 8192, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side='left')
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_length

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank(self, query: str, documents: List[str], instruction=None, top_k: int = 10):
        """ Rerank a list of documents based on a query and an optional instruction.
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to be reranked.
            instruction (str, optional): Instruction to guide the reranking. Defaults to None.
            top_k (int, optional): Number of top documents to return. Defaults to 10.
        Returns:
            List[str]: List of reranked documents.
        """
        pairs = [self.format_instruction(instruction, query, doc) for doc in documents]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs) 
        # Sort documents based on scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_documents = [documents[i] for i in sorted_indices]
        return sorted_documents[:top_k]


class VLLMReranker:
    def __init__(self, model_name_or_path: str, max_length: int = 8192, num_gpus: int = 1, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        number_of_gpu = min(num_gpus, torch.cuda.device_count())
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.8)
        self.model = LLM(model=self.model_name_or_path, tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length=max_length

        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(temperature=0, 
            max_tokens=1,
            logprobs=20, 
            allowed_token_ids=[self.true_token, self.false_token],
        )

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text

    def process_inputs(self, pairs, instruction, max_length, suffix_tokens):
        messages = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        messages =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:max_length] + suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages

    def compute_logits(self, messages):
        outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        for i in tqdm.tqdm(range(len(outputs)), desc="Computing logits"):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            token_count = len(outputs[i].outputs[0].token_ids)
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores
            
    def rerank(self, query: str, documents: List[str], instruction=None, top_k: int = 10):
        """ Rerank a list of documents based on a query and an optional instruction.
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to be reranked.
            instruction (str, optional): Instruction to guide the reranking. Defaults to None.
            top_k (int, optional): Number of top documents to return. Defaults to 10.
        Returns:
            List[str]: List of reranked documents.
        """
        pairs = [(query, doc) for doc in documents]
        inputs = self.process_inputs(pairs, instruction, self.max_length - len(self.suffix_tokens), self.suffix_tokens)
        scores = self.compute_logits(inputs) 
        # Sort documents based on scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_documents = [documents[i] for i in sorted_indices]
        return sorted_documents[:top_k]


class APIRerankerAgent:
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.headers = {'Content-Type': 'application/json'}
        self.retrieval_topk = kwargs.get('retrieval_topk', 5)
        self.query_instruction = kwargs.get('rerank_instruction', None)

    def rerank(self, query: str, documents: List[str], instruction=None, top_k: int = 10):
        if top_k is None:
            top_k = self.retrieval_topk
        if instruction is None:
            instruction = self.query_instruction
        data = json.dumps({
            "query": query,
            "documents": documents,
            "instruction": instruction,
            "top_k": top_k
        })
        begin_time = time.time()
        response = requests.post(self.url, headers=self.headers, data=data)
        end_time = time.time()
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        response_data = response.json()
        assert "reranked_documents" in response_data, "Response does not contain 'reranked_documents' key"
        return {
            "reranked_documents": response_data["reranked_documents"],
            "response_time": end_time - begin_time
        }


class RerankerAgent:
    def __init__(self, offline_kwargs=None, online_kwargs=None):
        assert offline_kwargs is not None or online_kwargs is not None, "Either offline_kwargs or online_kwargs must be provided"
        if online_kwargs is not None:
            self.agent = APIRerankerAgent(**online_kwargs)
            self.is_online = True
        else:
            self.agent = VLLMReranker(**offline_kwargs)
            self.is_online = False
    
    def rerank(self, query: str, documents: List[str], instruction=None, top_k: int = 10):
        """ Rerank a list of documents based on a query and an optional instruction.
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to be reranked.
            instruction (str, optional): Instruction to guide the reranking. Defaults to None.
            top_k (int, optional): Number of top documents to return. Defaults to 10.
        Returns:
            List[str]: List of reranked documents.
        """
        if self.is_online:
            response = self.agent.rerank(query, documents, instruction=instruction, top_k=top_k)
            if "reranked_documents" in response:
                return response["reranked_documents"]
            else:
                print("Warning: 'reranked_documents' not found in response, returning empty list")
                return []
        else:
            return self.agent.rerank(query, documents, instruction=instruction, top_k=top_k)


if __name__ == "__main__":
    # pip install flash_attn==2.7.4.post1 when encountering flash attention issues

    query = "What is the capital of France?"
    documents = [
        "The capital of France is Paris.",
        "France is a country in Europe.",
        "Paris is known for its art, fashion, and culture.",
        "The Eiffel Tower is located in Paris."
    ]
    online_kwargs = {
        "url": "http://n0998.talapas.uoregon.edu:5000/rerank",
        "retrieval_topk": 5,
        "rerank_instruction": None,
    }
    reranker_agent = RerankerAgent(online_kwargs=online_kwargs)
    reranked_docs = reranker_agent.rerank(query, documents)
    breakpoint()
    
    # reranker = HFReranker("Qwen/Qwen3-Reranker-4B")
    # ranked_docs = reranker.rerank(query, documents)
    # print("Ranked Documents (HFReranker):", ranked_docs)

    # vllm_reranker = VLLMReranker("Qwen/Qwen3-Reranker-4B")
    # ranked_docs_vllm = vllm_reranker.rerank(query, documents)
    # print("Ranked Documents (VLLMReranker):", ranked_docs_vllm)

