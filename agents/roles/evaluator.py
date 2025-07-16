import json
import time 
from typing import Any, Dict, List, Optional, Union
import pydantic

from agents.llm_agents import LLMAgent
from agents.prompts import evaluate
from agents.utils import extract_info_from_text
from preprocess.utils import normalize_text


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

        self.majority_vote_prompt = evaluate.MAJORITY_VOTE_PROMPT
        self.majority_vote_examples = None

        self.llm_judge_metric_prompt = evaluate.JUDGE_METRIC_PROMPT
        self.llm_judge_metric_examples = None

    def evaluate_and_analyze_answer(
            self,
            question: Union[str, List[str]],
            correct_answer: Union[str, List[str], List[List[str]]],
            predicted_answer: Union[str, List[str]],
            **kwargs: Any
    ):
        if isinstance(question, str):
            question = [question]
            assert isinstance(predicted_answer, str), "predicted_answer must be a string when question is a string."
            predicted_answer = [predicted_answer]
            correct_answer = [correct_answer]
        if len(question) > 1:
            kwargs['n'] = 1
        
        assert len(question) == len(predicted_answer) == len(correct_answer), "The lengths of question, predicted_answer, and correct_answer must match."
        batch = [
            self.llm_judge_metric_prompt.format(
                question=q,
                correct_answer=ca,
                predicted_answer=pa,
                examples=self.llm_judge_metric_examples if self.llm_judge_metric_examples else "Not provided."
            )
            for q, ca, pa in zip(question, correct_answer, predicted_answer)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating evaluations for final answers:")
            print("Questions:", question)
            print("Correct Answers:", correct_answer)
            print("Predicted Answers:", predicted_answer)
        kwargs['output_schema'] = evaluate.LLMJudgeMetricOutput
        responses = self.role_execute(batch, **kwargs)
        if len(question) == 1 and len(responses) > 1:
            # Majority vote
            decision = [res.get('decision', False) for res in responses]
            confidence = [res.get('confidence', 0.1) for res in responses]
            decision = sum(decision) / len(decision) if len(decision) > 0 else 0.0
            confidence = sum(confidence) / len(confidence) if len(confidence) > 0 else 0.1
            decision = decision >= 0.5  # Convert to boolean
            error_type = [res.get('error_type', 'None') for res in responses]
            reasoning = [res.get('reasoning', '') for res in responses]
            # Convert error_type and reasoning to a single string to make it consistent. Make sure that we can reconstruct the original list later.
            error_type = json.dumps(error_type)
            reasoning = json.dumps(reasoning)
            responses = [{
                'decision': decision,
                'confidence': confidence,
                'error_type': error_type,
                'reasoning': reasoning
            }]
        return responses

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
            print("Generating evaluations for final answers:")
            print("Questions:", question)
            print("Correct Answers:", correct_answer)
            print("Predicted Answers:", predicted_answer)
        kwargs['output_schema'] = evaluate.EvaluateAnswerOutput
        responses = self.role_execute(batch, **kwargs)
        if len(question) == 1 and len(responses) > 1:
            # Majority vote
            decision = [res.get('decision', False) for res in responses]
            confidence = [res.get('confidence', 0.1) for res in responses]
            decision = sum(decision) / len(decision) if len(decision) > 0 else 0.0
            confidence = sum(confidence) / len(confidence) if len(confidence) > 0 else 0.1
            decision = decision >= 0.5
            return [decision * confidence]
        results = []
        for response in responses:
            score = response.get('decision', False) * response.get('confidence', 0.1)
            results.append(score)
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
            print("Generating evaluations for path steps:")
            print("Main Questions:", main_question)
            print("Reasoning Traces:", reasoning_trace)
            print("Sub Questions:", sub_question)
            print("Selected Information:", selected_information)
            print("Generated Answers:", generated_answer)
        kwargs['output_schema'] = evaluate.PathAwareOutput
        response = self.role_execute(batch, **kwargs)
        results = []
        for res in response:
            relevance_score = res.get('relevance', 'poor')
            relevance_score = relevance_score.lower() if isinstance(relevance_score, str) else 'poor'
            if relevance_score == 'excellent':
                relevance_score = 1.0
            elif relevance_score == 'good':
                relevance_score = 0.75
            elif relevance_score == 'fair':
                relevance_score = 0.5
            else:
                relevance_score = 0.1
            sufficiency_score = res.get('sufficiency', 'poor')
            sufficiency_score = sufficiency_score.lower() if isinstance(sufficiency_score, str) else 'poor'
            if sufficiency_score == 'excellent':
                sufficiency_score = 1.0
            elif sufficiency_score == 'good':
                sufficiency_score = 0.75
            elif sufficiency_score == 'fair':
                sufficiency_score = 0.5
            else:
                sufficiency_score = 0.1
            coherence_score = res.get('coherence', 'poor')
            coherence_score = coherence_score.lower() if isinstance(coherence_score, str) else 'poor'
            if coherence_score == 'excellent':
                coherence_score = 1.0
            elif coherence_score == 'good':
                coherence_score = 0.75
            elif coherence_score == 'fair':
                coherence_score = 0.5
            else:
                coherence_score = 0.1
            factuality_score = res.get('factuality', 'poor')
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
        if len(main_question) == 1 and len(results) > 1:
            # Majority vote
            results = [sum(results) / len(results)]  # Average the scores
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
                original_question=mq,
                correct_answer=gt,
                reasoning_path=rp,
                examples=self.outcome_aware_examples if self.outcome_aware_examples else "Not provided."
            )
            for mq, gt, rp in zip(main_question, ground_truth_answer, reasoning_path)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]
        if self.verbose:
            print("Generating evaluations for reasoning paths:")
            print("Main Questions:", main_question)
            print("Reasoning Paths:", reasoning_path)
            print("Ground Truth Answers:", ground_truth_answer)
        kwargs['output_schema'] = evaluate.OutcomeAwareOutput
        response = self.role_execute(batch, **kwargs)
        results = []
        for res in response:
            step_quality = res.get('step_quality', 'poor')
            step_quality = step_quality.lower() if isinstance(step_quality, str) else 'poor'
            if step_quality == 'excellent':
                step_quality = 1.0
            elif step_quality == 'good':
                step_quality = 0.75
            elif step_quality == 'fair':
                step_quality = 0.5
            else:
                step_quality = 0.1
            overall_quality = res.get('overall_quality', 'poor')
            overall_quality = overall_quality.lower() if isinstance(overall_quality, str) else 'poor'
            if overall_quality == 'excellent':
                overall_quality = 1.0
            elif overall_quality == 'good':
                overall_quality = 0.75
            elif overall_quality == 'fair':
                overall_quality = 0.5
            else:
                overall_quality = 0.1
            conclusion_quality = res.get('conclusion_quality', 'poor')
            conclusion_quality = conclusion_quality.lower() if isinstance(conclusion_quality, str) else 'poor'
            if conclusion_quality == 'excellent':
                conclusion_quality = 1.0
            elif conclusion_quality == 'good':
                conclusion_quality = 0.75
            elif conclusion_quality == 'fair':
                conclusion_quality = 0.5
            else:
                conclusion_quality = 0.1
            score = (step_quality + overall_quality + conclusion_quality) / 3.0
            results.append(score)
        if len(main_question) == 1 and len(results) > 1:
            # Majority vote
            results = [sum(results) / len(results)]
        return results

    # Modified from FlashRAG https://vscode.dev/github/RUC-NLPIR/FlashRAG/blob/main/flashrag/evaluator/metrics.py#L187
    def evaluate_with_em(self,
            correct_answer: Union[str, List[str], List[List[str]]],
            predicted_answer: Union[str, List[str]],
            **kwargs: Any
    ):
        if isinstance(predicted_answer, str):
            predicted_answer = [predicted_answer]
            correct_answer = [correct_answer]
        assert len(predicted_answer) == len(correct_answer), "The lengths of predicted_answer and correct_answer must match."
        em_scores = []  
        for pa, ca in zip(predicted_answer, correct_answer):
            if isinstance(ca, str):
                ca = [ca]
            assert isinstance(pa, str), "predicted_answer must be a string."
            pa = normalize_text(pa)
            ca = [normalize_text(c) for c in ca]
            score = 0.0
            for item in ca:
                if item in pa or pa in item:
                    score = 1.0
                    break
            em_scores.append(score)
        return em_scores

    def majority_vote(
            self,
            question: Union[str, List[str]],
            answers: Union[str, List[str], List[List[str]]],
            **kwargs: Any
    ):
        """
        Evaluate the majority vote of answers for a given question.
        """
        if isinstance(question, str):
            question = [question]
            answers = [answers]
        kwargs['n'] = 1
        
        assert len(question) == len(answers), "The lengths of question and answers must match."
        
        batch = [
            self.majority_vote_prompt.format(
                question=q,
                answers=a,
                examples=self.majority_vote_examples if self.majority_vote_examples else "Not provided."
            )
            for q, a in zip(question, answers)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]
        if self.verbose:
            print("Generating majority vote evaluations:")
            print("Questions:", question)
            print("Answers:", answers)
        kwargs['output_schema'] = evaluate.MajorityVoteOutput
        response = self.role_execute(batch, **kwargs)
        results = []
        for res in response:
            answer = res.get("final_answer")
            results.append(answer)
        return results

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
        'n': 4, 
        'top_p': 0.9,
        'max_tokens': 1024*16,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    generator = Evaluator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        verbose=True
    )

    question = 'In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?'
    correct_answer = ['Alexis Sánchez', 'Alexis Sanchez']
    predicted_answer = 'Alexis Sanchez'
    reasoning_trace = """
    1. The Saints beat Manchester United in the 1976 FA Cup Final.
    2. In 2018, Alexis Sánchez left Arsenal to join Manchester United.
    Final Answer: Alexis Sánchez
    """
    # analyze_answer = generator.evaluate_and_analyze_answer(question=question, correct_answer=correct_answer, predicted_answer=predicted_answer)
    evaluate_answer = generator.evaluate_final_answer(question=question, correct_answer=correct_answer, predicted_answer=predicted_answer)
    # em_scores = generator.evaluate_with_em(correct_answer, predicted_answer)
    # judge_score = generator.evaluate_final_answer(question=question, correct_answer=correct_answer, predicted_answer=predicted_answer)
    # path_score = generator.evaluate_path(main_question=question, reasoning_path=reasoning_trace, ground_truth_answer=correct_answer)
    breakpoint()

    question_2 = 'What is the capital of France?'
    correct_answer_2 = 'Paris'
    predicted_answer_2 = 'Paris'
    reasoning_trace_2 = """
    1. France is a country in Europe.
    2. The capital of France is Paris.
    Final Answer: Paris
    """
    correct_answer = [correct_answer, correct_answer_2]
    predicted_answer = [predicted_answer, predicted_answer_2]
    question = [question, question_2]
    reasoning_trace = [reasoning_trace, reasoning_trace_2]
    # # em_scores = generator.evaluate_with_em(correct_answer, predicted_answer)
    # judge_scores = generator.evaluate_final_answer(question=question, correct_answer=correct_answer, predicted_answer=predicted_answer)
    # path_score = generator.evaluate_path(main_question=question, reasoning_path=reasoning_trace, ground_truth_answer=correct_answer)
    breakpoint()