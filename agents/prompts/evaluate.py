from enum import Enum
from typing import Union, Literal
import pydantic


EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. Your task is to assess the correctness of a model's generated output, which includes both its reasoning process and its final answer.

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Answer Evaluation:**: First, compare the final conclusion of the predicted answer against the correct answers. Check for semantic equivalence, not just a literal match, a predicted answer is considered correct if it matches any one of the correct answers. If the final answer comprehensively and accurately matches the correct answer, the evaluation is complete. If the final answer is incorrect or incomplete, you must then analyze the full reasoning process. Search for any step or intermediate conclusion where the correct answer is explicitly stated or strongly implied.
3. **Final Decision**: Based on your evaluation, determine the decision using these rules:
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


PATH_AWARE_PROMPT = """You are an expert evaluator tasked with assessing the quality of a single step within a complex reasoning process. Your evaluation must be objective, critical, and strictly adhere to the provided rubric.

### Context:
An agent is attempting to answer a main question by breaking it down into a series of steps. You are provided with the agent's reasoning trace so far, and you must evaluate the quality of the **most recent step**.

**Main Question:**
{main_question}

**Full Reasoning Trace (Prior Steps):**
{reasoning_trace}

---
### Step to Evaluate:

**Sub-Question:**
{sub_question}

**Information Selected for this Step:**
{selected_information}

**Generated Answer for this Step:**
{generated_answer}
---

### Task & Evaluation Rubric:
First, provide a step-by-step analysis based on the four criteria below. Then, assign a score from poor to excellent for each criterion:
1.  **Relevance:**
    *   How relevant was the "Information Selected for this Step" to the "Sub-Question"?
    *   Does it directly address the sub-question, or is it tangential or noisy?
2.  **Sufficiency:**
    *   Did the "Information Selected for this Step" provide enough detail to fully answer the "Sub-Question"?
    *   Was the information comprehensive, or did it leave gaps that needed further explanation?
3.  **Logical Coherence:**
    *   Does the "Generated Answer" follow logically from the "Full Reasoning Trace"?
    *   Does it build upon or contradict previous conclusions in the trace?
4.  **Factuality:**
    *   Is the "Generated Answer" factually correct according to the "Information Selected for this Step"?
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
1. **Step-by-Step Analysis:** For each distinct step or component in the Reasoning Path:
    - Logical Validity: Does the conclusion or assertion of this step logically follow from the preceding steps, given premises, or provided context? Identify any logical fallacies or gaps in inference.
    - Factual Accuracy & Grounding: Are the claims, data, evidence, or premises introduced or utilized in this step factually correct? If context or documents are provided, is the information accurately drawn from and consistent with them? Note any inaccuracies or unsupported claims.
    - Clarity & Precision: Is the language used in this step clear, precise, and unambiguous? Are there any terms or statements that are vague or could lead to misinterpretation? 
    - Relevance: Does this step directly and meaningfully contribute to addressing the Original Question and reaching the final conclusion?
2. **Overall Path Evaluation:** You will then assess the Reasoning Path as a whole, considering the following criteria:
    - Coherence: Does the entire Reasoning Path demonstrate a logical and understandable flow? Do the steps connect smoothly and build upon each other in a cohesive manner? 
    - Completeness & Sufficiency: Does the path include all necessary intermediate steps, information, and considerations required to logically bridge the gap from the Original Question to the final conclusion? Are there any critical omissions? Conversely, are there any redundant or superfluous steps that do not add value
    - Consistency: Are there any internal contradictions or inconsistencies between different parts of the Reasoning Path?
3. **Conclusion Assessment:** Based on whether a Correct Answer is provided or not, you will evaluate the final conclusion of the Reasoning Path:
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
    step_quality: Literal['excellent', 'good', 'fair', 'poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor)$",
        description="The quality of the step in the reasoning process, based on logical validity, factual accuracy, clarity, and relevance."
    )
    step_quality_details: str = pydantic.Field(
        ...,
        description="A detailed explanation of the step quality, including logical validity, factual accuracy, clarity, and relevance."
    )
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
   
        


