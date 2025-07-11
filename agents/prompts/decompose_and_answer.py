from typing import List
import pydantic

GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.

## Core Principle: The generated subquestion must NOT be answerable using the provided context. If a logical subquestion can be answered by the context, it is not a true knowledge gap, and you must look for the next piece of missing information.

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
3. **Information Gap Identification:** If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps. Attempt to answer these queries based on your own knowledge. If you use your own knowledge to answer the question, you should set the confidence level to 'low'.

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


GENERATE_QUERIES_FOR_RETRIEVER = """"You are a highly advanced Reasoning Engine. Your primary function is to deconstruct a user's Input (a question or statement) into a series of precise, self-contained, and essential search queries. The goal is to generate queries that, when answered, provide all the necessary facts to answer/verify the Input.

## Guiding Principles for Queries
1. The Zero-Synthesis Principle (Most Important): You MUST NOT introduce any new information, entities, or concepts that are not explicitly present in the original Input.
2. Fully Self-Contained: The query MUST BE SELF-CONTAINED, meaning it should be understandable and answerable without needing to refer to the original Input, other queries, or any external context. It should not rely on any anaphoric references or ambiguous terms 
3. Atomic: Each query must ask for one single, indivisible fact. Deconstruct questions containing conjunctions ("and", "or") or multiple attributes into separate queries.
4. Essential & Non-Redundant: Every query must be necessary for the final answer, and must seek a unique piece of information not covered by other queries.

## Instructions:
1. Parse the Input: 
    - If the input is a question: Identify its type (e.g., factual, comparative, causal, temporal), key entities, and the required reasoning steps. 
    - If the input is a statement: Deconstruct it into its core, verifiable claims. Identify the key entities and the asserted relationships between them. Note that, a statement can be a declarative sentence, a claim, or a QA pair. If the input is a QA pair, treat it as a statement with an implied question.
2. Generate Strategic Queries: Formulate a list of search queries to resolve the Input. Each query is a building block to reach the final answer that follows the guiding principles above. Do not try to replace them with the "answer" you think they represent. The goal is to provide the user with all the search components they would need to solve the problem from scratch.
3. Ensure Self-Containment: Each query must be understandable and answerable on its own. Check that no query relies on the original Input, other queries, or any external context. 

## Examples:
{examples}

---
**Input:**
{question}
"""


class QueriesGenerationOutput(pydantic.BaseModel):
    queries: List[str] = pydantic.Field(
        ...,
        description="List of independent search queries generated from the input question or statement. Each query should be atomic, mutually exclusive, essential, and independent.",
    )
    reasoning: str = pydantic.Field(
        ...,
        description="Reasoning process explaining how you arrived at the queries, including the analysis of the input and the identification of knowledge gaps.",
    )

