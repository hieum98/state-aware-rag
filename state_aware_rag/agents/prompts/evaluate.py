from enum import Enum
from typing import Optional, Union, Literal
import pydantic


EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. Your task is to assess the correctness of a model's generated output, which includes both its reasoning process and its final answer.

Instructions:
1.  Question Analysis:  Carefully read and understand the question. Identify key components and clarify what is being asked.
2.  Answer Evaluation: : First, compare the final conclusion of the predicted answer against the correct answers. Check for semantic equivalence, not just a literal match, a predicted answer is considered correct if it matches any one of the correct answers. If the final answer comprehensively and accurately matches the correct answer, the evaluation is complete. If the final answer is incorrect or incomplete, you must then analyze the full reasoning process. Search for any step or intermediate conclusion where the correct answer is explicitly stated or strongly implied.
3.  Final Decision : Based on your evaluation, determine the decision using these rules:
    - Mark as true (Correct) if: The model's final answer semantically matches the correct answer OR the correct answer is clearly present or implied in the model's reasoning steps, even if the final answer is different or flawed.
    - Mark as false (Incorrect) if: The correct answer is NOT found in the final answer OR anywhere in the reasoning path.

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerOutput(pydantic.BaseModel):
    decision: bool 
    reasoning: str  
    confidence: Literal['high', 'medium', 'low'] = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
    )


JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Your task is to provide a total rating on a scale of 0.0 to 10.0 for how well the system_answer resolves the user_question. Where 0.0 is completely unhelpful, irrelevant, or incorrect, and 10.0 is a perfect answer that is helpful, correct, and clear.

Evaluation Criteria:
    - Helpfulness & Relevance: How does the answer address the user's core need?
    - Correctness: Is the information accurate? If a correct_answer is provided and the system_answer matches it, you must give a rating of 10.0. If a correct_answer is not provided, use your own knowledge to judge correctness (you can do external research if needed).

Here are some examples: {examples}
Now, please evaluate the following question and answer:
User Question: {user_question}
System Answer: {system_answer}
Correct Answer (Optional): {correct_answer}
"""

class JudgeAnswerOutput(pydantic.BaseModel):
    total_rating: float = pydantic.Field(
        ...,
        ge=0.0,
        le=10.0,
        description="A float rating from 0 to 10 indicating how well the system answer addresses the user question."
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A detailed explanation of the rating, including specific aspects of the system answer that contributed to the score."
    )


PATH_AWARE_PROMPT = """You are an expert evaluator tasked with assessing the quality of a single step within a complex reasoning process. Your evaluation must be objective, critical, and strictly adhere to the provided rubric.

### Context:
An agent is attempting to answer a main question by breaking it down into a series of steps. You are provided with the agent's reasoning trace so far, and you must evaluate the quality of the  most recent step .

 Main Question: 
{main_question}

 Full Reasoning Trace (Prior Steps): 
{reasoning_trace}

---
### Step to Evaluate:

 Sub-Question: 
{sub_question}

 Information Selected for this Step: 
{selected_information}

 Generated Answer for this Step: 
{generated_answer}
---

### Task & Evaluation Rubric:
First, provide a step-by-step analysis based on the four criteria below. Then, assign a score from poor to excellent for each criterion:
1.   Relevance:  How relevant was the "Information Selected for this Step" to the "Sub-Question"?
2.   Sufficiency: How comprehensive was the information in addressing the sub-question?
3.   Logical Coherence:  How does the "Generated Answer" follow logically from the "Full Reasoning Trace"?
4.   Factuality: 
    *   How is the "Generated Answer" factually correct according to the "Information Selected for this Step"?
    *   Penalize any claims in the answer that are not supported by the selected information.
"""

class PathAwareOutput(pydantic.BaseModel):
    relevance: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
        description="The relevance of the selected information to the sub-question, based on how directly it addresses the question and its importance to the reasoning process."
    )
    relevance_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the relevance score, including specific examples from the selected information and how it relates to the sub-question."
    )
    sufficiency: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
        description="The sufficiency of the selected information in answering the sub-question, based on whether it provides enough detail and comprehensiveness."
    )
    sufficiency_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the sufficiency score, including whether the information was comprehensive or left gaps that needed further explanation."
    )
    coherence: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
        description="The logical coherence of the generated answer, based on whether it follows logically from the full reasoning trace and builds upon or contradicts previous conclusions."
    )
    coherence_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the coherence score, including how the generated answer follows logically from the full reasoning trace and whether it builds upon or contradicts previous conclusions."
    )
    factuality: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
        description="The factual accuracy of the generated answer, based on whether it is supported by the selected information and does not contain unsupported claims."
    )
    factuality_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the factuality score, including whether the generated answer is factually correct according to the selected information and whether it contains unsupported claims."
    )


OUTCOME_AWARE_PROMPT = """You are an expert assistant specializing in evaluating the quality of reasoning processes. You will be given: Original Question: The question the reasoning path attempts to answer; Reasoning Path: The sequence of steps, arguments, or inferences presented as the solution or explanation.; Correct Answer (Optional): The known correct answer to the Original Question. If not provided, the evaluation will focus solely on the intrinsic quality of the reasoning.
Please analyze the provided Reasoning Path based on the following criteria. Structure your evaluation to address each point clearly, providing specific examples or references to the steps in the Reasoning Path where appropriate.

## Instructions:
1.  Overall Path Evaluation:  You will then assess the Reasoning Path as a whole, considering the following criteria:
    - Coherence: Does the entire Reasoning Path demonstrate a logical and understandable flow? Do the steps connect smoothly and build upon each other in a cohesive manner? 
    - Completeness & Sufficiency: Does the path include all necessary intermediate steps, information, and considerations required to logically bridge the gap from the Original Question to the final conclusion? Are there any critical omissions? Conversely, are there any redundant or superfluous steps that do not add value
    - Consistency: Are there any internal contradictions or inconsistencies between different parts of the Reasoning Path?
2.  Conclusion Assessment:  Based on whether a Correct Answer is provided or not, you will evaluate the final conclusion of the Reasoning Path:
    - If a Correct Answer is provided: Does the Reasoning Path ultimately arrive at the Correct Answer? If the path's conclusion is incorrect, pinpoint the earliest step(s) where the error (logical, factual, calculational, misinterpretation, etc.) occurs that leads to the deviation. If the path's conclusion matches the Correct Answer, critically assess whether the reasoning process itself is sound, complete, and free of significant flaws. (It is possible to reach a correct answer through flawed reasoning).
    - If no Correct Answer is provided (focus on intrinsic quality):
        - Based solely on the structure, content, and evidence within the Reasoning Path, how convincing and well-supported is the stated conclusion?  
        - What are the most significant strengths of the reasoning in supporting its conclusion?
        - What are the most critical weaknesses or vulnerabilities in the reasoning that might undermine its conclusion?
        - Overall Summary and Recommendations (Optional but Encouraged):

Here are some examples: {examples}

Now, please evaluate the following reasoning process:
Original Question: {original_question}
Reasoning Path: {reasoning_path}
Correct Answer (Optional): {correct_answer}
"""

class OutcomeAwareOutput(pydantic.BaseModel):
    overall_quality: Literal['excellent', 'good', 'fair', 'poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor)$",
        description="The overall quality of the reasoning path, considering coherence, completeness, and consistency."
    )
    overall_quality_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the overall quality, including coherence, completeness, and consistency."
    )
    conclusion_quality: Literal['excellent', 'good', 'fair', 'poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor)$",
        description="The quality of the conclusion reached by the reasoning path, based on its correctness and the soundness of the reasoning process."
    )
    conclusion_quality_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the conclusion quality, including correctness, soundness of reasoning, and any critical flaws."
    )
   

MAJORITY_VOTE_PROMPT = """You are an expert assistant specializing in evaluating the answers to questions. Given a question and a set of answers, your task is to determine the final answer based on majority voting.

## Instructions:
1. Question Analysis: Carefully read and understand the question. Identify key components and clarify what is being asked.
2. Identify the underlying consensus: Determine the most frequent and correct answer, even if the wording varies across the different responses.
3. Synthesize the final answer: Formulate a single, directed, consolidated, and accurate answer based on the majority consensus for the question.

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Answers: 
{answers}
"""

class MajorityVoteOutput(pydantic.BaseModel):
    final_answer: str = pydantic.Field(
        ...,
        description="The final answer determined by majority voting based on the provided answers."
    )
    confidence: Literal['high', 'medium', 'low'] = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
        description="The confidence level in the final answer, indicating how strongly the majority consensus supports it."
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A detailed explanation of how the final answer was derived from the majority of the provided answers."
    )


SYNTHESIZE_FINAL_ANSWER_PROMPT = """You are an expert in argumentative synthesis and logical reasoning. Your task is to act as an impartial adjudicator and synthesizer. You will not generate a new answer from scratch, but will instead construct a superior answer by critically analyzing and integrating the provided candidate answers. To synthesize a single, comprehensive, and robust final answer and its corresponding reasoning path from a set of candidate reasoning paths. The synthesized answer should be more accurate, coherent, and well-supported than any individual candidate.

## Instructions:
Phase I: Deconstruction and Quality Assessment:
    - Deconstruct Each Candidate: For each candidate answer, systematically break it down into its core components: 
        - Conclusion: What is the final claim or answer?
        - Premises: What are the individual pieces of evidence, facts, or assumptions used?
        - Reasoning Path: What are the logical steps or inferences used to connect the premises to the conclusion?
    - Assess Individual Quality: Evaluate each candidate's argument independently based on these criteria :
        - Factual Accuracy: Are the premises verifiably true and from reliable sources?
        - Logical Soundness: Is the reasoning path logical and free from fallacies?
        - Sufficiency: Is the evidence provided strong enough and sufficient to justify the conclusion?
Phase II: Conflict Mapping and Adjudication: 
    - Identify Points of Convergence and Divergence: Compare the deconstructed arguments. Create a mental "synthesis matrix" to map out where the candidates agree and disagree. Specifically identify:
        - Consensus: Points where all or most candidates agree on facts or reasoning.
        - Contradiction: Points where candidates present conflicting facts or draw opposing conclusions from the same facts.
        - Unique Insights: Valuable premises or reasoning steps that are present in only one or a few candidates.
    - Adjudicate Conflicts: Where contradictions exist, you must resolve them using a clear hierarchy of resolution strategies:
        - For Factual Conflicts: Prioritize evidence from more authoritative, recent, or verifiable sources. Discard claims based on demonstrably false information.
        - For Logical Conflicts: Discard arguments that rely on logical fallacies. Prefer reasoning paths that are more direct, valid, and relevant to the question.
        - For Value/Goal Conflicts (where arguments optimize for different outcomes): Acknowledge the different goals. Attempt to find a collaborative "win-win" solution that integrates the valid parts of each argument. If this is not possible, synthesize the final answer by using majority consensus. 
Phase III: Recomposition and Final Argument Construction: 
    - Construct the Synthesized Reasoning Path: This is the most critical part of your output. Do not simply summarize the candidates. Build a new, superior line of reasoning. Your reasoning path should be a logical progression, step-by-step, that leads to the final answer. It should:
        - Start with the most robust premises, integrating the best evidence from all candidates.
        - Use the strongest logical connections, avoiding any fallacies or weak inferences.
        - Address any gaps or weaknesses identified in the candidates' reasoning.
        - Self-contained: Ensure that the reasoning path is self-contained, which should be understandable and convincing on its own, without needing to refer back to the candidates.
    - State the Final Synthesized Answer: Based on your newly constructed reasoning path, state the final, definitive answer to the original question.
    - Perform a Final Self-Critique: Before concluding, review your synthesized answer and reasoning. Ask yourself: "Is this the most logical and best-supported conclusion possible based only on the provided candidate information? Have I fairly represented and adjudicated their points? Is my own reasoning path clear and free of fallacies? Is the final resoning path coherent, logical, and self-contained? Is the final answer is cosistent with the reasoning path and the original question? Does it comprehensively address the question and integrate the best elements of all candidates?" Refine if necessary.

Here are some examples: {examples}

Now, please synthesize the final answer and the reasoning based on the following question and reasoning paths:
Question: {question}
Candidate Answers:
{reasoning_paths}
"""

class SynthesizeFinalAnswerOutput(pydantic.BaseModel):
    final_answer: str = pydantic.Field(
        ...,
        description="The final synthesized answer to the question, derived from the candidate answers."
    )
    final_reasoning_path: str = pydantic.Field(
        ...,
        description="The reasoning path that leads to the final answer, integrating the best elements of all candidate answers."
    )
    confidence: Literal['high', 'medium', 'low'] = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
        description="The confidence level in the final answer, indicating how strongly the synthesis supports it."
    )
    


