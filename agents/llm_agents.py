import copy
import os
import time
from typing import Any, Dict, List, Union, Callable
import json
from hashlib import sha256
from functools import partial
from pebble import ThreadPool
import openai
import pydantic
from litellm import completion
from litellm.utils import supports_response_schema
from tqdm import tqdm

from agents.utils import extract_info_from_text


class ModelClient:
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: Union[str, None] = None,
            concurrency: int = 64,
            **generate_kwargs: Dict[str, Any]
            ):
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        self.concurrency = concurrency

        ## Generate kwargs
        self.temperature = generate_kwargs.get('temperature', 0.7)
        self.num_samples = generate_kwargs.get('n', 1)
        self.top_p = generate_kwargs.get('top_p', 0.8)
        self.max_tokens = generate_kwargs.get('max_tokens', 8192) # default max tokens to generate
        self.top_k = generate_kwargs.get('top_k', 20)
        self.reasoning_effort = generate_kwargs.get('reasoning_effort', None)
        self.structure_output_supported = False
    
    def prepare_model_kwargs(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def completion(self, messages: List[Dict[str, str]], **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate(self, input_args,  **kwargs):
        messages, index = input_args
        should_return_reasoning = kwargs.get('should_return_reasoning', False)
        output_schema = kwargs.get('output_schema', None)
        if output_schema is not None:
            if self.structure_output_supported:
                assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            else:
                print(f"Model {self.model_name} does not support structured output. Ignoring output schema.")
        model_kwargs = self.prepare_model_kwargs(**kwargs)

        response = self.completion(messages=messages, **model_kwargs)
        all_outputs = []
        for choice in response.choices: # type: ignore
            reasoning = None
            if should_return_reasoning:
                try:
                    reasoning = choice.message.reasoning_content # type: ignore
                except:
                    print("No reasoning content found in the response. Returning None.")
            if output_schema is not None:
                try:
                    output = output_schema.model_validate_json(choice.message.content) # type: ignore
                except:
                    print(f"Failed to parse output with schema {output_schema}. Returning raw content.")
                    print(f"Raw content: {choice.message.content}") # type: ignore
                    output = choice.message.content # type: ignore
            else:
                output = choice.message.content # type: ignore
            all_outputs.append({
                'output': output,
                'reasoning': reasoning,
            })
        return index, all_outputs
    
    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs):
        batch = [(messages, i) for i, messages in enumerate(batch_messages)]
        num_workers = min(self.concurrency, len(batch))
        generate_fn = partial(self.generate, **kwargs)
        with ThreadPool(max_workers=num_workers) as pool:
            future = pool.map(generate_fn, batch)
            outputs = list(tqdm(future.result(), total=len(batch), desc=f"Generating responses from {self.model_name}"))
        assert len(outputs) == len(batch), "Batch generation did not return the expected number of outputs."
        outputs = sorted(outputs, key=lambda x: x[0])  # Sort by index
        return [output[1] for output in outputs]  # Return only the outputs

class LiteLLMClient(ModelClient):
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: Union[str, None] = None,
            concurrency: int = 64,
            **generate_kwargs: Dict[str, Any]
            ):
        super().__init__(model_name, url, api_key, concurrency, **generate_kwargs)
        self.structure_output_supported = supports_response_schema(model_name)
        print(f"Calling API via LiteLLM with model {self.model_name} at {self.url}")
    
    def prepare_model_kwargs(self, **kwargs):
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
        output_schema = kwargs.get('output_schema', None)
        if output_schema is not None:
            if self.structure_output_supported:
                assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            else:
                print(f"Model {self.model_name} does not support structured output. Ignoring output schema.")
        model_kwargs = {
            'model': self.model_name,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            "top_p": kwargs.get('top_p', self.top_p) if reasoning_effort is None else None,
            'n': kwargs.get('n', self.num_samples),
            "top_k": kwargs.get('top_k', self.top_k) if reasoning_effort is None else None,
            "api_key": self.api_key,
            "base_url": self.url,
            "response_format": output_schema,
            'reasoning_effort': reasoning_effort,
        }
        return model_kwargs
    
    def completion(self, messages: List[Dict[str, str]], **kwargs):
        assert isinstance(messages, list), "Messages must be a list of dictionaries with 'role' and 'content' keys."
        response = completion(
            messages=messages,
            **kwargs
        )
        return response


class OpenAIClient(ModelClient):
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: str = 'None',
            concurrency: int = 64,
            **generate_kwargs: Dict[str, Any]
        ):
        super().__init__(model_name, url, api_key, concurrency, **generate_kwargs)
        self.structure_output_supported = True # OpenAI supports structured output
        self.client = openai.Client(base_url=self.url, api_key=self.api_key)
        print(f"Calling OpenAI API with model {self.model_name} at {self.url}")
    
    def prepare_model_kwargs(self, **kwargs):
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
        output_schema = kwargs.get('output_schema', None)
        response_format = None
        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema()
                    }
            }
        model_kwargs = {
            'model': self.model_name,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'n': kwargs.get('n', self.num_samples),
            'response_format': response_format,
            'extra_body': {
                "top_k": kwargs.get('top_k', self.top_k),
                "chat_template_kwargs": {"enable_thinking": True if reasoning_effort else None},
            }
        }
        return model_kwargs
    
    def completion(self, messages: List[Dict[str, str]], **kwargs):
        assert isinstance(messages, list), "Messages must be a list of dictionaries with 'role' and 'content' keys."
        response = self.client.chat.completions.create(messages=messages, **kwargs) # type: ignore
        return response


class LLMAgent:
    def __init__(self, client_kwargs: Dict[str, Any], generate_kwargs: Dict[str, Any], 
                 use_cache: bool = True, cache_dir: str = './cache/llm_agents'):
        self.client_type = client_kwargs.pop('client_type', 'litellm')
        if self.client_type == 'litellm':
            self.client = LiteLLMClient(**client_kwargs, **generate_kwargs)
        elif self.client_type == 'openai':
            self.client = OpenAIClient(**client_kwargs, **generate_kwargs)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}. Supported types are 'litellm' and 'openai'.")
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, self.client.model_name)
        self.verbose = generate_kwargs.get('verbose', False)
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def save_to_cache(self, cache_file: str, responses: List[Dict[str, Any]]):
        with open(cache_file, 'w', encoding='utf-8') as f:
            to_save_data = []
            for item in responses:
                if isinstance(item['output'], pydantic.BaseModel):
                    to_save_data.append({
                        'output': item['output'].model_dump(),
                        'reasoning': item['reasoning']
                    })
                else:
                    to_save_data.append({
                        'output': item['output'],
                        'reasoning': item['reasoning']
                    })
            json.dump(to_save_data, f, indent=4)
    
    def load_from_cache(self, cache_file: str, output_schema) -> List[Dict[str, Any]]:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if output_schema is not None:
                assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
                for item in data:
                    try:
                        item['output'] = output_schema.model_validate(item['output'])
                    except:
                        print(f"Failed to parse output with schema {output_schema}. Returning raw content.")
                        print(f"Raw content: {item['output']}")
        return data
    
    def generate(self, input_args, **kwargs):
        use_cache = kwargs.get('use_cache', self.use_cache)
        if not use_cache:
            return self.client.generate(input_args, **kwargs)
        
        messages, index = input_args
        output_schema = kwargs.get('output_schema', None)
        input_str = f"Messages: {messages},  kwargs: {kwargs}"
        input_hash = sha256(input_str.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{input_hash}.json")
        try:
            response = self.load_from_cache(cache_file, output_schema)
            return index, response
        except FileNotFoundError:
            index, response = self.client.generate(input_args, **kwargs)
            self.save_to_cache(cache_file, response)
            return index, response
        
    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs):
        use_cache = kwargs.get('use_cache', self.use_cache)
        if not use_cache:
            return self.client.batch_generate(batch_messages, **kwargs)
        
        cached_responses = []
        to_compute_responses = []
        for i, messages in enumerate(batch_messages):
            input_str = f"Messages: {messages},  kwargs: {kwargs}"
            input_hash = sha256(input_str.encode('utf-8')).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{input_hash}.json")
            try:
                output_schema = kwargs.get('output_schema', None)
                response = self.load_from_cache(cache_file, output_schema)
                cached_responses.append((i, response))
            except FileNotFoundError:
                to_compute_responses.append((messages, i, cache_file))
        if to_compute_responses:
            messages = [item[0] for item in to_compute_responses]
            index = [item[1] for item in to_compute_responses]
            cache_files = [item[2] for item in to_compute_responses]
            responses = self.client.batch_generate(messages, **kwargs)
            for i, response, cache_file in zip(index, responses, cache_files):
                self.save_to_cache(cache_file, response)
                cached_responses.append((i, response))
        cached_responses = sorted(cached_responses, key=lambda x: x[0])
        assert len(cached_responses) == len(batch_messages), "Batch generation did not return the expected number of outputs."
        return [response for _, response in cached_responses]  # Return only the outputs

    def role_execute(self, batch: List[Dict[str, Any]], **kwargs: Any):
        """Execute the role with a batch of prompts and return the generated responses.
        Args:
            batch (List[Dict[str, Any]]): A list of prompts to be processed by the role.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[Dict[str, Any]]: A list of responses generated by the role for each prompt in the batch.
        """
        start_time = time.time()
        assert 'output_schema' in kwargs, "Output schema must be provided in kwargs."
        output_schema = kwargs['output_schema']
        assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel."
        responses = self.batch_generate(batch, **kwargs)
        end_time = time.time()
        if self.verbose:
            print(f"Generated {len(responses)} answers in {end_time - start_time:.2f} seconds.")
        batch_results = []
        for i, response in enumerate(responses):
            results = []
            for item in response:
                if self.verbose:
                    print(f"Processing response for question {i}: {item}")
                output_object = item.get('output', None)
                if isinstance(output_object, output_schema):
                    results.append(output_object.model_dump())
                else:
                    print("Warning: Output is not of type AnswerOutput, received:", output_object)
                    print("Trying to parse with regex...")
                    # Attempt to parse the output using regex
                    keys = output_schema.model_fields.keys()
                    value_types = [field.annotation.__name__ for field in output_schema.model_fields.values()]
                    extracted_info = extract_info_from_text(output_object, keys, value_types)
                    results.append(extracted_info)
            batch_results.append(results)
        if len(batch_results) == 1:
            return batch_results[0]
        return [x[0] for x in batch_results]  # Return the first result of each batch due to n=1
            

if __name__ == "__main__":
    ## Deploy the llm server via sglang
    ## python -m sglang.launch_server --host 0.0.0.0 --model-path Qwen/Qwen3-8B --reasoning-parser qwen3 # --port 30000 
    online_model_kwargs = {
        'model_name': 'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'url': None,  # Use default URL for the model
        'api_key': None,  # Set your API key if required
        # 'model_name': 'openai/qwen3-8B', 
        # 'url': 'http://n0998.talapas.uoregon.edu:30000/v1', 
        # 'api_key': 'your_api_key_here',  # Replace with your actual API key
        # 'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
        'concurrency': 64,
    }
    generate_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 1,  
        'n': 1, 
        'top_p': 0.9,
        'max_tokens': 4096,  
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    client = LLMAgent(client_kwargs=online_model_kwargs, generate_kwargs=generate_kwargs)

    question = "In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?"
    questions = [
        'Who is the current president of the United States?',
        'What is the capital of France?',
    ]
    answer = "Alexis Sánchez"
    messages = [{'role': 'user', 'content': question}]
    batch_messages = [[{'role': 'user', 'content': q}] for q in questions]
    class OutputSchema(pydantic.BaseModel):
        answer: str
        justification: str

    input_args = (messages, 0)  # Tuple of messages and index
    index, outputs = client.generate(input_args, output_schema=OutputSchema, should_return_reasoning=True)
    breakpoint()
    outputs = client.batch_generate(batch_messages, output_schema=OutputSchema, should_return_reasoning=True)
    breakpoint()
