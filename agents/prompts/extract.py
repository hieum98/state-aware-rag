from typing import Literal
import pydantic


EXTRACT_PROMPT = """You are a meticulous and insightful research analyst. Your primary objective is to build a comprehensive dossier of all information from the provided text that could help a user fully understand and answer their question. You prioritize thoroughness, context, and nuance. You must think step-by-step to ensure no helpful detail, however tangential, is overlooked.

## Instructions: 
- Step 1: Question Deconstruction: First, carefully analyze the user's Question. Identify and list the primary subject, all key entities (people, organizations, concepts), and the specific information or insight the user is seeking. This is your 'search brief'.
- Step 2: Candidate Identification: Next, read the entire Raw Data and identify and quote ALL passages that seem potentially related to the concepts from Step 1. Be liberal and inclusive in this initial pass; we will filter and refine in the next step. If no passages appear even remotely related, state this and proceed to Step 5.
- Step 3: Systematic Relevance Evaluation: Now, for each candidate passage quoted in Step 2, you must perform a systematic evaluation. Iterate through each quote and assess it against the following criteria: Directly Answering, Contextual, Supporting Evidence, Methodological, Alternative Perspectives, Related Concepts, Implications, Enrichment, Entities. For each candidate quote, you must state exactly which criterion (or criteria) it meets and provide a one-sentence justification for your assessment. If a quote meets no criteria, mark it as 'Not Relevant'.
- Step 4: Extraction: Extract ALL relevant information. Ensure the extraction is strictly verbatim and includes full sentences to preserve context. You must also verify that each extraction is self-contained and can stand alone without requiring additional context from the original document or the query to be understood. If an extraction is deemed relevant but it is lacks context or clarity, you should add additional context to ensure clarity and self-containment but make sure that you DO NOT add any information that was not present in the original document and MAKE SURE TO KEEP THE ORIGINAL MEANING INTACT. If no passages were deemed relevant, this section should be left empty.
- Step 5: Final Decision: Based on your analysis in the preceding steps, state your final decision: 'relevant' or 'not_relevant'. A document is only 'not_relevant' if it contains ZERO information that could relate to any entity or concept in the question. If it contains even a single piece of information that could be relevant, it is 'relevant', even if it is not directly answering the question. Refine your extractions against this final decision. If you decide the document is 'relevant', ensure that your extractions are comprehensive and contain all information that could help answer the question. If you decide the document is 'not_relevant', ensure that your extractions are empty.

Here are some examples: {examples}

Now, please evaluate the following data given the question:
Question: {question}
Raw Data: 
{document}
"""

class ExtractOutput(pydantic.BaseModel):
    information: list[str] = pydantic.Field(
        ...,
        description="A list of extracted information from the data that is relevant to the question"
    )
    decision: Literal['relevant', 'not_relevant'] = pydantic.Field(
        ...,
        description="The decision on whether the data is relevant to the question."
    )
    reasoning: str = pydantic.Field(
        ...,
        description="A explanation of the thought process behind the extractions, detailing how the information was identified as relevant or not and why it was extracted."
    )



