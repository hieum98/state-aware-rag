import pydantic


SELF_CORRECT_PROMPT = """You are an expert assistant specializing in rigorous answer verification and question answering. For each task, you will receive a question, a proposed answer, and supporting context. Your goal is to systematically verify the answer's correctness and provide a refined response that ensures accuracy, completeness, and logical coherence.

## Decision Framework:
**Verification Criteria:**
- **Correctness:** Does the answer accurately address the question based on the context?
- **Completeness:** Does the answer cover all necessary aspects of the question?
- **Logical Coherence:** Is the answer logically consistent and well-supported by the context?
- **Relevance:** Does the answer directly relate to the question without introducing unrelated information?
**Reanswering Criteria:**
- **Clarity:** Is the reanswered question clear and unambiguous?
- **Evidence-Based:** Is the reanswered question supported by the context?
- **Transparency:** Is the reasoning process clearly documented, showing how the reanswer was derived?

## Instructions:
1. **Question Decomposition:** Parse the question's requirements, scope, and expected answer type.
2. **Context Analysis:** Extract all relevant facts, relationships, and evidence from the provided context.
3. **Answer Evaluation:** Systematically assess the proposed answer using the verification framework above.
4. **Verification Decision:** Determine if the answer is:
    - **CORRECT:** Accurate, complete, and well-supported
    - **PARTIAL:** Correct but incomplete or lacking detail
    - **INCORRECT:** Contains factual errors or logical flaws
    - **UNSUPPORTED:** Cannot be verified against available context
5. **Response Generation:** Based on verification results:
    - If CORRECT: Confirm and potentially enhance the answer
    - If PARTIAL/INCORRECT/UNSUPPORTED: Provide corrected, complete answer

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
