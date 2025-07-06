import pydantic


SELF_CORRECT_PROMPT = """You are an expert assistant specializing in rigorous answer verification and question answering. For each task, you will receive a question, a proposed answer, and supporting context. Your goal is to systematically verify the answer's correctness and provide a refined response that ensures accuracy, completeness, and logical coherence.

## Instructions:
1. **Question Decomposition:** Parse the question's requirements, scope, and expected answer type.
2. **Context Analysis:** Extract all relevant facts, relationships, and evidence from the provided context.
3. **Answer Evaluation:** Systematically assess the proposed answer to determine if the answer is:
    - **CORRECT:** Accurate, complete, and well-supported
    - **PARTIAL:** Correct but incomplete or lacking detail
    - **INCORRECT:** Contains factual errors or logical flaws
    - **UNSUPPORTED:** Cannot be verified against available context
4. **Response Generation:** If the answer is correct or partially correct, confirm and potentially enrich it. If it is incorrect or unsupported, provide a refined answer

## Examples: 
{examples}

---
**Question:** 
{question}
**Proposed Answer:** 
{answer}
**Context:** 
{context}
"""


class SelfCorrectOutput(pydantic.BaseModel):
    verification_status: str  = pydantic.Field(
        ...,
        pattern=r'^(CORRECT|PARTIAL|INCORRECT|UNSUPPORTED)$',
        description="The verification status of the answer, indicating whether it is correct, partial, incorrect, or unsupported."
    )
    reanswer: str = pydantic.Field(
        ...,
        description="The refined or corrected answer to the question, or just restate the original answer if it is correct."
    )
    reasoning: str = pydantic.Field(
        ...,
        description="Detailed reasoning process, including context analysis, verification steps, and logical deductions."
    )
    confidence: str = pydantic.Field(
        ...,
        pattern=r'^(high|medium|low)$',
        description="Confidence level in the reanswer, one of 'high', 'medium', or 'low'."
    )
