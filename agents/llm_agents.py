import copy
import os
from typing import Any, Dict, List
import json
from hashlib import sha256
from functools import partial
import pydantic
from litellm import supports_response_schema, completion



class LiteLLMClient:
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: str = None,
            concurrency: int = 64,
            is_local_server: bool = False,
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
        if not is_local_server:
            self.structure_output_supported = supports_response_schema(model_name)
        else:
            print(f"Model {model_name} is a local model. Make sure it supports structured output by deploying it with the correct configuration.")
            self.structure_output_supported = True

    def generate(self, messages: List[Dict[str, str]], **kwargs):
        index = kwargs.get('index', 0)
        output_schema = kwargs.get('output_schema', None)
        if output_schema is not None:
            if self.structure_output_supported:
                assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            else:
                print(f"Model {self.model_name} does not support structured output. Ignoring output schema.")
        reasoning_effort = kwargs.get('reasoning_effort', self.reasoning_effort)
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
        should_return_reasoning = kwargs.get('should_return_reasoning', False)
        response = completion(messages=messages, **model_kwargs)
        all_outputs = []
        for choice in response.choices:
            if should_return_reasoning:
                try:
                    reasoning = choice.message.reasoning_content
                except:
                    print("No reasoning content found in the response. Returning None.")
                    reasoning = None
            if output_schema is not None:
                try:
                    output = output_schema.model_validate_json(choice.message.content)
                except:
                    print(f"Failed to parse output with schema {output_schema}. Returning raw content.")
                    print(f"Raw content: {choice.message.content}")
                    output = choice.message.content
            else:
                output = choice.message.content
            all_outputs.append({
                'output': output,
                'reasoning': reasoning,
            })
        return index, all_outputs



if __name__ == "__main__":
    online_model_kwargs = {
        'model_name': 'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'url': None,  # Use default URL for the model
        'api_key': None,  # Set your API key if required
        'concurrency': 64,
        'is_local_server': False,
    }
    generate_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 1,  
        'n': 1, 
        'top_p': 0.9,
        'max_tokens': 8192,
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    client = LiteLLMClient(**online_model_kwargs, **generate_kwargs)

    question = "In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?"
    answer = "Alexis Sánchez"
    messages = [{'role': 'user', 'content': question}]
    class OutputSchema(pydantic.BaseModel):
        answer: str
        step_by_step_reasoning: str

    index, outputs = client.generate(messages, output_schema=OutputSchema, should_return_reasoning=True)
    breakpoint()
