from typing import List
import pydantic

GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.

## Core Principle: No Overlap
The most important rule is: The generated subquestion must NOT be answerable using the provided context. If a logical subquestion can be answered by the context, it is not a true knowledge gap, and you must look for the next piece of missing information.

## Step-by-Step Instructions:
1. **Analyze the Main Question:** Deconstruct the question to identify its core intent (e.g., factual lookup, comparison, causal link), key entities, and the information required for a complete answer.
2. **Map Context to Requirements:* Systematically check if the provided context contains all the facts, entities, and relationships identified in Step 1.
3. **Decision Point: Assess Answerability:**
   - If YES (Context is Sufficient): The main question can be fully and confidently answered. No subquestion is needed.
   - If NO (Context is Insufficient): The context is missing at least one critical piece of information. Proceed to the next steps.
4. **If the Context is Insufficient, Execute the Following:**
   a. Identify the Core Knowledge Gap: Pinpoint the most immediate and crucial piece of missing information. This is the first thing you would need to look up to start solving the main question.
   b. Formulate the Subquestion: Create a clear, self-contained question that precisely targets this single knowledge gap. The subquestion should be:
      * Atomic: Asks for one fact.
      * Relevant: Its answer is essential for answering the main question.
      * Non-Anaphoric: Understandable without reading the main question or context (e.g., avoid pronouns like "he" or "it").
   c. CRITICAL VALIDATION: Before finalizing, you must verify that your formulated subquestion CANNOT be answered by the provided context. If it can be, you have made an error. You must re-evaluate the knowledge gap and formulate a different subquestion that targets information truly missing from the context.

## Examples: 
{examples}

---
**Question:** 
{question}
**Context:** 
{context}
"""

class SubquestionOutput(pydantic.BaseModel):
    answerable_main_question: bool = pydantic.Field(
        ...,
        description="Indicates whether the main question can be answered with the provided context.",
    )
    subquestion: str = pydantic.Field(
        ...,
        description="The generated subquestion that logically follows from the main question if the main question is not answerable; otherwise, an empty string.",
    )
    reasoning: str = pydantic.Field(
        ...,
        description="Reasoning process explaining how you arrived at the decision and subquestion.",
    )
    gap_type: str = pydantic.Field(
        ...,
        pattern=r"^(factual|relational|causal|temporal|logical|null)$",
        description="Type of reasoning gap identified, one of 'factual', 'relational', 'causal', 'temporal', 'logical', or 'null'."
    )
        

ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning. 

## Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Utilization:**  If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Information Gap Identification:** If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps. Attempt to answer these queries based on your own knowledge.

Here are some examples: {examples}

Now, please answer the following question:
Question: {question}
Context: 
{context}
"""

class AnswerOutput(pydantic.BaseModel):
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


