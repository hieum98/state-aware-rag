from typing import Literal
import pydantic


EXTRACT_PROMPT = """You are an expert assistant specializing in evaluating the relevance of a retrieved document to a given question and the current step's objective/subquestions. Your primary function is to act as a precise and literal information extractor.

## Core Directives:
Strict Adherence to Source: Your analysis and extractions must be grounded exclusively in the provided document. Do not introduce any external information, inferences, or conclusions that are not explicitly stated in the text.
Extract for Context: Do not extract isolated sentences. An extraction must provide a complete context for the information. This means that if a key piece of information is spread across multiple sentences or paragraphs, you must extract all relevant parts to ensure the information is fully understood.

## Instructions:
1. Question and Objective Analysis: Carefully read and understand the question and the current step's objective. Identify the specific information being sought.
2. Document Relevance Evaluation: Read the entire document with the goal of identifying any and all information that could be useful. Think broadly about relevance. Relevant information is not just a direct answer; it can be:
    - Directly Answering: Information that directly addresses the question or objective.
    - Contextual: Background information, definitions of key terms, historical context, or details that help in understanding the main topic.
    - Supporting Evidence: Specific data points, statistics, quotes, case studies, or examples that validate or illustrate points.
    - Methodological: Information about how the knowledge was obtained (e.g., the methodology of a study, the source of a claim).
    - Alternative Perspectives: Counterarguments, differing opinions, or alternative viewpoints presented in the document.
    - Related Concepts: Tangential information that is closely related and provides a more nuanced understanding.
If the document does not contain any relevant information, you should output an empty string for the extracted information and the summary, and a clear reasoning that explains why the document is not relevant.
3. Thought Process: Provide a step-by-step analysis for each document. Your reasoning should explain your relevance decision and justify why you've categorized the extracted information.
4. Verbatim Information Extraction: If the document is relevant, extract the information from the document. For each piece of information, provide:
    - The exact text from the document that supports this information. Ensures that definitions, qualifiers, and surrounding context are included with the core information.
    - The summary of the information. Note that this summary should be concise, self-contained.

Here are some examples: {examples}

Now, please evaluate the following document given the question and the current step's objective:
Question: {question}
Current Step's Objective: {current_step_objective}
Document: {document}
"""

class InformationPiece(pydantic.BaseModel):
    extracted_information: str = pydantic.Field(
        ...,
        description="The verbatim information extracted from the document that is relevant to the question and current step's objective."
    )
    summary: str = pydantic.Field(
        ...,
        description="A concise summary of the extracted information, self-contained and clear."
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A explanation of the thought process behind the extraction, detailing how the information was identified as relevant and why it was extracted."
    )

class ExtractOutput(pydantic.BaseModel):
    decision: Literal['relevant', 'not_relevant'] = pydantic.Field(
        ...,
        description="The decision on whether the document is relevant to the question and current step's objective. 'relevant' means the document contains useful information, while 'not_relevant' means it does not."
    )
    extracted_information: list[InformationPiece] = pydantic.Field(
        ...,
        description="A list of InformationPiece objects, each containing the extracted information, its summary, and the reasoning behind its extraction."
    )



