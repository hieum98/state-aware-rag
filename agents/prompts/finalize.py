import pydantic


FINALIZE_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning. 

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Utilization:**  If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Information Gap Identification:** If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps. Attempt to answer these queries based on your own knowledge.

Here are some examples: {examples}

Now, please answer the following question:
Question: {question}
Context: 
{context}
"""

class FinalizeOutput(pydantic.BaseModel):
    reasoning: str = pydantic.Field(
        ...,
        description="Reasoning that led to the answer, including any assumptions made or steps taken to arrive at the conclusion.",
    )
    detailed_answer: str = pydantic.Field(
        ...,
        description="The complete answer, including any necessary explanations or clarifications. In this field, you should provide a comprehensive response that includes all relevant details, including the source of the information (i.e., the context or your own knowledge) and any assumptions made.",
    )
    answer: str = pydantic.Field(
        ...,
        description="Direct answer to the question without any additional explanation or context.",
    )
    confidence: str = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
        description="Confidence level in the answer, one of 'high', 'medium', or 'low'."
    )
