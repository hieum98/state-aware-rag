import time 
from typing import Any, Dict, List, Optional, Union

import pydantic

from agents.llm_agents import LLMAgent
from agents.prompts import evaluate
from agents.utils import extract_info_from_text


class Evaluator(LLMAgent):
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

        self.evaluate_answer_prompt = evaluate.EVALUATE_ANSWER_PROMPT
        self.evaluate_answer_examples = None

        self.path_aware_prompt = evaluate.PATH_AWARE_PROMPT
        self.path_aware_examples = None

        self.outcome_aware_prompt = evaluate.OUTCOME_AWARE_PROMPT
        self.outcome_aware_examples = None

    def evaluate_final_answer(
            self,
            question: Union[str, List[str]],
            correct_answer: Union[str, List[str], List[List[str]]],
            predicted_answer: Union[str, List[str]],
            **kwargs: Any
    ):
        """
        Evaluate the final answer of a model against the correct answer.
        
        Args:
            question (Union[str, List[str]]): The question(s) to evaluate.
            correct_answer (Union[str, List[str], List[List[str]]]): The correct answer(s) or a list of correct answers.
            predicted_answer (Union[str, List[str]]): The predicted answer(s) from the model.
            **kwargs: Additional keyword arguments for the client execution.
        Returns:
            List[float]: A list of scores for the evaluation, where each score is a float between 0 and 1.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(predicted_answer, str), "predicted_answer must be a string when question is a string."
            predicted_answer = [predicted_answer]
            correct_answer = [correct_answer]
        if len(question) > 1:
            kwargs['n'] = 1
        
        assert len(question) == len(predicted_answer) == len(correct_answer), "The lengths of question, predicted_answer, and correct_answer must match."
        batch = [
            self.evaluate_answer_prompt.format(
                question=q,
                correct_answer=ca,
                predicted_answer=pa,
                examples=self.evaluate_answer_examples if self.evaluate_answer_examples else "Not provided."
            )
            for q, ca, pa in zip(question, correct_answer, predicted_answer)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating answers for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = evaluate.EvaluateAnswerOutput
        responses = self.role_execute(batch, **kwargs)
        results = []
        for response in responses:
            decision = response.get('decision', False)
            confidence = response.get('confidence', 'low')
            confidence = confidence.lower() if isinstance(confidence, str) else 'low'
            if confidence == 'high':
                results.append(decision*1.0)
            elif confidence == 'medium':
                results.append(decision*0.5)
            else:
                results.append(decision*0.1)
        return results
    
    def evaluate_path_step(
            self,
            main_question: Union[str, List[str]],
            reasoning_trace: Union[str, List[str]],
            sub_question: Union[str, List[str]],
            selected_information: Union[str, List[str]],
            generated_answer: Union[str, List[str]],
            **kwargs: Any
    ):
        """
        Evaluate a single step in the reasoning process of an agent.
        
        Args:
            main_question (Union[str, List[str]]): The main question being answered.
            reasoning_trace (Union[str, List[str]]): The full reasoning trace so far.
            sub_question (Union[str, List[str]]): The sub-question for the current step.
            selected_information (Union[str, List[str]]): The information selected for this step.
            generated_answer (Union[str, List[str]]): The generated answer for this step.
            **kwargs: Additional keyword arguments for the client execution.
        Returns:
            List[float]: A list of scores for the evaluation of the step, where each score is a float between 0 and 1.
        """
        if isinstance(main_question, str):
            main_question = [main_question]
            reasoning_trace = [reasoning_trace]
            sub_question = [sub_question]
            selected_information = [selected_information]
            generated_answer = [generated_answer]
        if len(main_question) > 1:
            kwargs['n'] = 1
        
        assert len(main_question) == len(reasoning_trace) == len(sub_question) == len(selected_information) == len(generated_answer), \
            "The lengths of main_question, reasoning_trace, sub_question, selected_information, and generated_answer must match."
        
        batch = [
            self.path_aware_prompt.format(
                main_question=mq,
                reasoning_trace=rt,
                sub_question=sq,
                selected_information=si,
                generated_answer=ga,
                examples=self.path_aware_examples if self.path_aware_examples else "Not provided."
            )
            for mq, rt, sq, si, ga in zip(main_question, reasoning_trace, sub_question, selected_information, generated_answer)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]
        if self.verbose:
            print("Generating evaluations for path steps:", main_question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = evaluate.PathAwareOutput
        response = self.role_execute(batch, **kwargs)
        results = []
        for res in response:
            relevance_score = res.get('relevance_score', 'poor')
            relevance_score = relevance_score.lower() if isinstance(relevance_score, str) else 'poor'
            if relevance_score == 'excellent':
                relevance_score = 1.0
            elif relevance_score == 'good':
                relevance_score = 0.75
            elif relevance_score == 'fair':
                relevance_score = 0.5
            else:
                relevance_score = 0.1
            sufficiency_score = res.get('sufficiency_score', 'poor')
            sufficiency_score = sufficiency_score.lower() if isinstance(sufficiency_score, str) else 'poor'
            if sufficiency_score == 'excellent':
                sufficiency_score = 1.0
            elif sufficiency_score == 'good':
                sufficiency_score = 0.75
            elif sufficiency_score == 'fair':
                sufficiency_score = 0.5
            else:
                sufficiency_score = 0.1
            coherence_score = res.get('coherence_score', 'poor')
            coherence_score = coherence_score.lower() if isinstance(coherence_score, str) else 'poor'
            if coherence_score == 'excellent':
                coherence_score = 1.0
            elif coherence_score == 'good':
                coherence_score = 0.75
            elif coherence_score == 'fair':
                coherence_score = 0.5
            else:
                coherence_score = 0.1
            factuality_score = res.get('factuality_score', 'poor')
            factuality_score = factuality_score.lower() if isinstance(factuality_score, str) else 'poor'
            if factuality_score == 'excellent':
                factuality_score = 1.0
            elif factuality_score == 'good':
                factuality_score = 0.75
            elif factuality_score == 'fair':
                factuality_score = 0.5
            else:
                factuality_score = 0.1
            score = (relevance_score + sufficiency_score + coherence_score + factuality_score) / 4.0
            results.append(score)
        return results
    
    def evaluate_path(
            self,
            main_question: Union[str, List[str]],
            reasoning_path: Union[str, List[str]],
            ground_truth_answer: Union[str, List[str], List[List[str]]],
            **kwargs: Any
    ):
        """
        Evaluate a complete path of reasoning steps.
        
        Args:
            main_question (Union[str, List[str]]): The main question being answered.
            reasoning_path (Union[str, List[str]]): The full reasoning path to evaluate.
            ground_truth_answer (Union[str, List[str], List[List[str]]]): The correct answer(s) or a list of correct answers.
            **kwargs: Additional keyword arguments for the client execution.
        Returns:
            List[float]: A list of scores for the evaluation of the path, where each score is a float between 0 and 1.
        """
        if isinstance(main_question, str):
            main_question = [main_question]
            reasoning_path = [reasoning_path]
            ground_truth_answer = [ground_truth_answer]
        if len(main_question) > 1:
            kwargs['n'] = 1
        
        assert len(main_question) == len(reasoning_path) == len(ground_truth_answer), \
            "The lengths of main_question, reasoning_path, and ground_truth_answer must match."
        
        batch = [
            self.outcome_aware_prompt.format(
                main_question=mq,
                ground_truth_answer=gt,
                reasoning_path=rp,
                examples=self.outcome_aware_examples if self.outcome_aware_examples else "Not provided."
            )
            for mq, gt, rp in zip(main_question, ground_truth_answer, reasoning_path)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]
        if self.verbose:
            print("Generating evaluations for paths:", main_question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = evaluate.OutcomeAwareOutput
        response = self.role_execute(batch, **kwargs)
        results = []
        for res in response:
            score = res.get('outcome_score', 'dead end')
            score = score.lower() if isinstance(score, str) else 'dead end'
            if score == 'highly promising':
                score = 1.0
            elif score == 'moderate potential':
                score = 0.75
            elif score == 'low potential':
                score = 0.5
            else:
                score = 0.1
            results.append(score)
        return results

            
    