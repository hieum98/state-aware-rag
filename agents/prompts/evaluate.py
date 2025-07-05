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
    relevance_score: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
    )
    resoning_for_relevance: str = pydantic.Field(
        ...,
        description="A detailed explanation of the relevance score, including how the selected information relates to the sub-question."
    )
    sufficiency_score: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
    )
    reasoning_for_sufficiency: str = pydantic.Field(
        ...,
        description="A detailed explanation of the sufficiency score, including whether the selected information fully addressed the sub-question."
    )
    coherence_score: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
    )
    reasoning_for_coherence: str = pydantic.Field(
        ...,
        description="A detailed explanation of the coherence score, including how the generated answer relates to the full reasoning trace."
    )
    factuality_score: str = pydantic.Field(
        ...,
        pattern=r"^(poor|fair|good|excellent)$",
    )
    reasoning_for_factuality: str = pydantic.Field(
        ...,
        description="A detailed explanation of the factuality score, including whether the generated answer is supported by the selected information."
    )


OUTCOME_AWARE_PROMPT = """You are an expert evaluator with perfect knowledge of the correct final answer to a complex question. Your task is to assess a hypothetical reasoning path and determine its potential for leading an agent to the correct solution.

### Context:
An agent is trying to answer a main question. You are given the ground-truth answer and a potential future reasoning path that the agent might take.

**Main Question:**
{main_question}

**Ground-Truth Final Answer:**
{ground_truth_answer}

**Reasoning Path to Evaluate:**
{reasoning_path}

### Task & Evaluation Rubric:
First, provide a step-by-step analysis comparing the "Reasoning Path" against the "Ground-Truth Final Answer". Then, assign a single `outcome_score` from dead end to highly promising based on the rubric below.
**Evaluation Criteria:**
*   **Strategic Value:** Does this path move the agent closer to the "Ground-Truth Final Answer"?
*   **Information Sufficiency:** Does the path contain the critical facts, evidence, and intermediate conclusions necessary to derive the final answer?
*   **Path Viability:** How likely is it that an agent, having followed this path, can now correctly and completely answer the "Main Question"? A "highly promising" path is one that makes answering the final question straightforward. A "dead end" path contains errors or irrelevant information that leads away from the correct answer.
"""

class OutcomeAwareOutput(pydantic.BaseModel):
    outcome_score: Literal['dead end', 'low potential', 'moderate potential', 'highly promising'] = pydantic.Field(
        ...,
        pattern=r"^(dead end|low potential|moderate potential|highly promising)$",
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A detailed explanation of the outcome score, including how the reasoning path relates to the ground-truth final answer."
    )



