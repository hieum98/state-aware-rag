from typing import Literal
import pydantic


EXTRACT_PROMPT = """You are an expert assistant specializing in evaluating the relevance of raw data to a given question. Your primary function is to act as a precise and literal information extractor.

## Core Directives:
Strict to Source: Your analysis and extractions must be grounded exclusively in the provided raw data. Do not introduce any external information.
Extract for Context: Do not extract isolated sentences. An extraction must provide a complete context for the information. This means that if a key piece of information is spread across multiple sentences or paragraphs, you must extract all relevant parts to ensure the information is fully understood.

## Instructions:
1. Question Analysis: Carefully read and understand the question. Identify all the key terms, concepts, and entities mentioned in the question. Consider the broader context and implications of the question to ensure a comprehensive understanding.
2. Relevance Evaluation: Read the entire data with the goal of identifying ALL information that could be useful. Think broadly about relevance. Relevant information is not just a direct answer; it can be:
    - Directly Answering: Information that directly addresses the question or objective.
    - Contextual: Background information, definitions of key terms, historical context, or details that help in understanding the main topic.
    - Supporting Evidence: Specific data points, statistics, quotes, case studies, or examples that validate or illustrate points.
    - Methodological: Information about how the knowledge was obtained (e.g., the methodology of a study, the source of a claim).
    - Alternative Perspectives: Counterarguments, differing opinions, or alternative viewpoints presented in the data.
    - Related Concepts: Tangential information that is closely related and provides a more nuanced understanding.
    - Implications: Consequences or implications of the information presented, which may not be directly asked but are crucial for a comprehensive understanding.
    - Enrichment: Additional insights that enhance the understanding
    - Entities: Names of people, organizations, locations, or other entities that are relevant to the entities mentioned in the question or objective. A
3. Verbatim Information Extraction: If the data is relevant, extract the information from the data. For each piece of information, provide:
    - The exact text from the data that supports this information. Ensures that definitions, qualifiers, and surrounding context are included with the core information.

Here are some examples: {examples}

Now, please evaluate the following data given the question:
Question: {question}
Raw Data: 
{document}
"""

class ExtractOutput(pydantic.BaseModel):
    decision: Literal['relevant', 'not_relevant'] = pydantic.Field(
        ...,
        description="The decision on whether the data is relevant to the question."
    )
    information: list[str] = pydantic.Field(
        ...,
        description="A list of extracted information from the data that is relevant to the question"
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A explanation of the thought process behind the extractions, detailing how the information was identified as relevant or not and why it was extracted."
    )



