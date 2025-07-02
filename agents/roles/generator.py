import re
import time 
from typing import Any, Dict, List, Optional, Union

from agents.llm_agents import LLMAgent
from agents.prompts import (
    decompose_and_answer
    )


class Generator(LLMAgent):
    def __init__(
            self, 
            client_kwargs, 
            generate_kwargs, 
            use_cache = True, 
            cache_dir = './cache/llm_agents',
            verbose: bool = False,
            ):
        super().__init__(client_kwargs, generate_kwargs, use_cache, cache_dir)
        self.verbose = verbose

        # Initialize prompts
        self.generate_subquestion_prompt = decompose_and_answer.GENERATE_SUBQUESTION_PROMPT
        self.generate_subquestion_examples = None

        self.generate_answer_prompt = decompose_and_answer.ANSWER_PROMPT
        self.generate_answer_examples = None
    
    def generate_answer(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """        Generate answers for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate answers for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.AnswerOutput]: A list of AnswerOutput objects containing the generated answers and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.generate_answer_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_answer_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating answers for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")

        start_time = time.time()
        kwargs['output_schema'] = decompose_and_answer.AnswerOutput
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
                if isinstance(output_object, decompose_and_answer.AnswerOutput):
                    results.append(output_object)
                else:
                    print("Warning: Output is not of type AnswerOutput, received:", output_object)
                    print("Trying to parse with regex...")
                    # Attempt to parse the output using regex

            batch_results.append(results)
        if len(batch_results) == 1:
            return batch_results[0]
        return [x[0] for x in batch_results]  # Return the first result of each batch due to n=1

    def generate_subquestion(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate subquestions for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate subquestions for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SubquestionOutput]: A list of SubquestionOutput objects containing the generated subquestions and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.generate_subquestion_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_subquestion_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating subquestions for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")

        start_time = time.time()
        kwargs['output_schema'] = decompose_and_answer.SubquestionOutput
        responses = self.batch_generate(batch, **kwargs)
        end_time = time.time()
        if self.verbose:
            print(f"Generated {len(responses)} subquestions in {end_time - start_time:.2f} seconds.")
        batch_results = []
        for i, response in enumerate(responses):
            results = []
            for item in response:
                if self.verbose:
                    print(f"Processing response for question {i}: {item}")
                output_object = item.get('output', None)
                if isinstance(output_object, decompose_and_answer.SubquestionOutput):
                    results.append(output_object)
                else:
                    print("Warning: Output is not of type SubquestionOutput, received:", output_object)
                    print("Trying to parse with regex...")
                    # TODO: Attempt to parse the output using regex
            batch_results.append(results)
        if len(batch_results) == 1:
            return batch_results[0]
        return [x[0] for x in batch_results]  # Return the first result


if __name__ == "__main__":
    online_model_kwargs = {
        'model_name': 'openai/qwen3-8B', 
        'url': 'http://n0998.talapas.uoregon.edu:30000/v1', 
        'api_key': 'your_api_key_here',  # Replace with your actual API key
        'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
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
    generator = Generator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        verbose=True
    )

    question = 'In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?'
    batch = [question] + ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?"] + ["What is the capital of Japan?"]
    context = None
    # context = ["Arsenal is a football club based in London.", "The capital of France is Paris.", "Harper Lee wrote 'To Kill a Mockingbird'."]
    # results = generator.generate_answer(question=batch, context=context)
    # breakpoint()
    results = generator.generate_subquestion(question=batch, context=context)
    breakpoint()       
        
