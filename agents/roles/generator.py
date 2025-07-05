import time 
from typing import Any, Dict, List, Optional, Union

import pydantic

from agents.llm_agents import LLMAgent
from agents.prompts import (
    decompose_and_answer,
    synthesize,
    finalize,
    self_correct,
    rephase_question,
    )
from agents.utils import extract_info_from_text


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

        self.generate_synthesis_prompt = synthesize.SYNTHESIZE_PROMPT
        self.generate_synthesis_examples = None

        self.finalize_prompt = finalize.FINALIZE_PROMPT
        self.finalize_examples = None

        self.self_correct_prompt = self_correct.SELF_CORRECT_PROMPT
        self.self_correct_examples = None

        self.rephase_question_prompt = rephase_question.REPHRASE_QUESTION_PROMPT
        self.rephase_question_examples = None
    
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
        kwargs['output_schema'] = decompose_and_answer.AnswerOutput
        return self.role_execute(batch, **kwargs)

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

        kwargs['output_schema'] = decompose_and_answer.SubquestionOutput
        return self.role_execute(batch, **kwargs)

    def generate_synthesis(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate synthesis for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate synthesis for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SynthesisOutput]: A list of SynthesisOutput objects containing the generated synthesis and reasoning for each question.
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
            self.generate_synthesis_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_synthesis_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating synthesis for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = synthesize.SynthesizeOutput
        return self.role_execute(batch, **kwargs)

    def finalize(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate final answers for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate final answers for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.FinalizeOutput]: A list of FinalizeOutput objects containing the generated final answers and reasoning for each question.
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
            self.finalize_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.finalize_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating final answers for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = finalize.FinalizeOutput
        return self.role_execute(batch, **kwargs)
    
    def self_correct(
            self,
            question: Union[str, List[str]],
            current_answer: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Self-correct the current answer for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to self-correct answers for.
            current_answer (Union[str, List[str]]): A single answer or a list of answers corresponding to each question.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SelfCorrectOutput]: A list of SelfCorrectOutput objects containing the self-corrected answers and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(current_answer, str), "Current answer must be a string when question is a string."
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            current_answer = [current_answer]
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(current_answer) == len(context), "If context is provided, it must match the number of questions and answers."
        batch = [
            self.self_correct_prompt.format(
                question=q,
                answer=a,
                context=c if c else "No context provided.",
                examples=self.self_correct_examples or "No examples provided."
            ) for q, a, c in zip(question, current_answer, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Self-correcting answers for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = self_correct.SelfCorrectOutput
        return self.role_execute(batch, **kwargs)

    def rephase_question(
            self,
            question: Union[str, List[str]],
            **kwargs: Any
    ):
        """Rephrase the question for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to rephrase.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[rephase_question.RephraseQuestionOutput]: A list of RephraseQuestionOutput objects containing the rephrased questions and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
        if len(question) > 1:
            kwargs['n'] = 1
        batch = [
            self.rephase_question_prompt.format(
                question=q,
                examples=self.rephase_question_examples or "No examples provided."
            ) for q in question
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Rephrasing questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = rephase_question.RephraseQuestionOutput
        return self.role_execute(batch, **kwargs)

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
        
