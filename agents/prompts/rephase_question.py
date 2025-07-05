import pydantic


REPHRASE_QUESTION_PROMPT = """You are an expert assistant specializing in systematic question analysis and reformulation. Your task is to transform questions into clearer, more answerable forms while preserving their core intent and ensuring they can be effectively processed by downstream systems.

## Core Objectives:
1. **Question Clarity:** Analyze the original question to identify ambiguities, complexities, or vague terms that may hinder understanding or processing.
2. **Intent Preservation:** Ensure that the rephrased question retains the original intent and scope, making it suitable for answering by AI systems or human experts.
3. **Answerability Enhancement:** Reformulate the question to improve its structure, specificity, and clarity, making it easier to answer accurately.

## Decision Framework:
**Rephrasing Criteria:**
- **Clarity:** Is the question free from ambiguity and complexity?
- **Specificity:** Does the question clearly define the scope and focus, avoiding vague terms?
- **Intent Preservation:** Does the rephrased question maintain the original intent and context?
- **Answerability:** Is the question structured in a way that facilitates accurate and complete answers?

## Instructions:
1. **Question Analysis:** Break down the original question to identify key components, including the main subject, action, and any specific requirements or constraints.
2. **Identify Issues:** Look for any ambiguities, complex phrasing, or vague terms that could lead to misinterpretation or difficulty in answering.
3. **Rephrase for Clarity:** Reformulate the question using clear, straightforward language. Ensure that it is specific and unambiguous, while retaining the original intent.
4. **Preserve Intent:** Ensure that the rephrased question aligns with the original intent and context, making it suitable for answering by AI systems or human experts.

## Examples:
{examples}

---
**Question:**
{question}
"""

class RephraseQuestionOutput(pydantic.BaseModel):
    rephrased_question: str = pydantic.Field(
        ...,
        description="The reformulated question that is clearer, more specific, and retains the original intent.",
    )
    reasoning: str = pydantic.Field(
        ...,
        description="Detailed reasoning process explaining how the question was analyzed and rephrased, including any issues identified and how they were addressed.",
    )

