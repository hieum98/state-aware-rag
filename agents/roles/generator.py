import time 
from typing import Any, Dict, List, Optional, Union
import pydantic

from agents.llm_agents import LLMAgent
from agents.prompts import (
    decompose_and_answer,
    synthesize,
    finalize,
    self_correct,
    rephase_question,
    )
from agents.utils import extract_info_from_text


class Generator(LLMAgent):
    def __init__(
            self, 
            client_kwargs, 
            generate_kwargs, 
            use_cache = True, 
            cache_dir = './cache/llm_agents',
            verbose: bool = False,
            ):
        super().__init__(client_kwargs, generate_kwargs, use_cache, cache_dir)
        self.verbose = verbose

        # Initialize prompts
        self.generate_subquestion_prompt = decompose_and_answer.GENERATE_SUBQUESTION_PROMPT
        self.generate_subquestion_examples = None

        self.generate_answer_prompt = decompose_and_answer.ANSWER_PROMPT
        self.generate_answer_examples = None

        self.generate_synthesis_prompt = synthesize.SYNTHESIZE_PROMPT
        self.generate_synthesis_examples = None

        self.finalize_prompt = finalize.FINALIZE_PROMPT
        self.finalize_examples = None

        self.self_correct_prompt = self_correct.SELF_CORRECT_PROMPT
        self.self_correct_examples = None

        self.rephase_question_prompt = rephase_question.REPHRASE_QUESTION_PROMPT
        self.rephase_question_examples = None

        self.generate_queries_prompt = decompose_and_answer.GENERATE_QUERIES_FOR_RETRIEVER
        self.generate_queries_examples = None
    
    def generate_answer(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """        Generate answers for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate answers for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.AnswerOutput]: A list of AnswerOutput objects containing the generated answers and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.generate_answer_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_answer_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating answers for questions:", question)
            print("Context:", context)
        kwargs['output_schema'] = decompose_and_answer.AnswerOutput
        return self.role_execute(batch, **kwargs)

    def generate_subquestion(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate subquestions for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate subquestions for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SubquestionOutput]: A list of SubquestionOutput objects containing the generated subquestions and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.generate_subquestion_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_subquestion_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating subquestions for questions:", question)
            print("Context:", context)
        kwargs['output_schema'] = decompose_and_answer.SubquestionOutput
        return self.role_execute(batch, **kwargs)

    def generate_synthesis(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate synthesis for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate synthesis for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SynthesisOutput]: A list of SynthesisOutput objects containing the generated synthesis and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.generate_synthesis_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.generate_synthesis_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating synthesis for questions:", question)
            print("Context:", context)
        kwargs['output_schema'] = synthesize.SynthesizeOutput
        return self.role_execute(batch, **kwargs)

    def finalize(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Generate final answers for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate final answers for.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.FinalizeOutput]: A list of FinalizeOutput objects containing the generated final answers and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1 # Ensure single response for multiple questions
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(context), "If context is provided, it must match the number of questions."
        batch = [
            self.finalize_prompt.format(
                question=q,
                context=c if c else "No context provided.",
                examples=self.finalize_examples or "No examples provided."
            ) for q, c in zip(question, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating final answers for questions:", question)
            print("Context:", context)
        kwargs['output_schema'] = finalize.FinalizeOutput
        return self.role_execute(batch, **kwargs)
    
    def self_correct(
            self,
            question: Union[str, List[str]],
            current_answer: Union[str, List[str]],
            context: Union[str, List[str]] = None,
            **kwargs: Any
    ):
        """Self-correct the current answer for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to self-correct answers for.
            current_answer (Union[str, List[str]]): A single answer or a list of answers corresponding to each question.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.SelfCorrectOutput]: A list of SelfCorrectOutput objects containing the self-corrected answers and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
            assert isinstance(current_answer, str), "Current answer must be a string when question is a string."
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            current_answer = [current_answer]
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1
        if context is None:
            context = ["No context provided."] * len(question)
        assert len(question) == len(current_answer) == len(context), "If context is provided, it must match the number of questions and answers."
        batch = [
            self.self_correct_prompt.format(
                question=q,
                answer=a,
                context=c if c else "No context provided.",
                examples=self.self_correct_examples or "No examples provided."
            ) for q, a, c in zip(question, current_answer, context)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Self-correcting answers for questions:", question)
            print("Current Answers:", current_answer)
            print("Context:", context)
        kwargs['output_schema'] = self_correct.SelfCorrectOutput
        return self.role_execute(batch, **kwargs)

    def rephase_question(
            self,
            question: Union[str, List[str]],
            **kwargs: Any
    ):
        """Rephrase the question for a given question or a batch of questions, optionally using provided context.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to rephrase.
            context (Union[str, List[str]], optional): A single context or a list of contexts corresponding to each question. If None, a default message will be used.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[rephase_question.RephraseQuestionOutput]: A list of RephraseQuestionOutput objects containing the rephrased questions and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
        if len(question) > 1:
            kwargs['n'] = 1
        batch = [
            self.rephase_question_prompt.format(
                question=q,
                examples=self.rephase_question_examples or "No examples provided."
            ) for q in question
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Rephrasing questions:", question)
        kwargs['output_schema'] = rephase_question.RephraseQuestionOutput
        return self.role_execute(batch, **kwargs)

    def generate_queries_for_retriever(
            self,
            question: Union[str, List[str]],
            **kwargs: Any
    ):
        """Generate queries for a given question or a batch of questions to be used in a retriever.
        Args:
            question (Union[str, List[str]]): A single question or a list of questions to generate queries for.
            **kwargs (Any): Additional keyword arguments to pass to the batch generation method, such as temperature, top_p, max_tokens, etc.
        Returns:
            List[decompose_and_answer.QueriesGenerationOutput]: A list of QueriesGenerationOutput objects containing the generated queries and reasoning for each question.
        """
        if isinstance(question, str):
            question = [question]
        if len(question) > 1:
            kwargs['n'] = 1
        batch = [
            self.generate_queries_prompt.format(
                question=q,
                examples=self.generate_queries_examples or "No examples provided."
            ) for q in question
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating queries for questions:", question)
        kwargs['output_schema'] = decompose_and_answer.QueriesGenerationOutput
        return self.role_execute(batch, **kwargs)

if __name__ == "__main__":
    online_model_kwargs = {
        'model_name': 'openai/qwen3-8B', 
        'url': 'http://ip-10-4-228-30:30000/v1', 
        'api_key': 'your_api_key_here',  # Replace with your actual API key
        'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
        'concurrency': 64,
    }
    generate_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 1,  
        'n': 1, 
        'top_p': 0.9,
        'max_tokens': 1024*4,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    generator = Generator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        verbose=True
    )

    question = "The question asks which magazine was started first between 'Arthur's Magazine' and 'First for Women.' By analyzing the provided context: \n\n1. **Arthur's Lady's Home Magazine** was founded in **1852** (Doc2). \n2. **Arthur (magazine)** was founded in **2002** (Doc1). \n3. The context does not mention 'First for Women' explicitly, so its founding date is unknown. A follow-up query would be needed to identify the publication date of 'First for Women' if further clarification is required."
    context = """
    Doc1: Arthur (magazine)"\nArthur (magazine) Arthur magazine was a bi-monthly periodical that was founded in October 2002, by publisher Laris Kreslins and editor Jay Babcock. It received favorable attention from other periodicals such as ""L.A. Weekly"", ""Print"", ""Punk Planet"" and ""Rolling Stone"". ""Arthur"" featured photography and artwork from Spike Jonze, Art Spiegelman, Susannah Breslin, Gary Panter and Godspeed You! Black Emperor. Arthur\'s regular columnists included Byron Coley, Thurston Moore, Daniel Pinchbeck, Paul Cullum, Douglas Rushkoff, and T-Model Ford. ""Arthur"" magazine was particularly drawn to noise music, stoner metal, folk and other types of psychedelia. The first issue of ""Arthur"" featured an interview with
    Doc2: Arthur\'s Lady\'s Home Magazine"\nArthur\'s Lady\'s Home Magazine Arthur\'s Home Magazine (1852-ca.1898) or Ladies\' Home Magazine was an American periodical published in Philadelphia by Timothy Shay Arthur. Editors Arthur and Virginia Francis Townsend selected writing and illustrations intended to appeal to female readers. Among the contributors: Mary Tyler Peabody Mann and Kate Sutherland. In its early years the monthly comprised a selection of articles originally published in Arthur\'s weekly ""Home Gazette."" Its nonfiction stories contained occasional factual inaccuracies for the sake of a good read.
    Doc3: History of women\'s magazines"\nHistory of women\'s magazines This article addresses the history of women\'s magazines. In 1693 the first issue of the first women\'s magazine in Britain, ""The Ladies\' Mercury"", was published. In 1857 the first women\'s magazine in Gujarati, ""Streebodh"", was established by Parsi social activists. In 1892 the first women\'s magazine in Egypt, and indeed in all the Arab countries, ""Al Fatat"", was established by Hind Nawfal. In the period before the American Civil War, ""Godey\'s Lady\'s Book"" was a United States women\'s magazine that was the most widely circulated magazine. Its circulation rose from 70,000 in the 1840s to 150,000
    Doc4: "The Lady\'s Magazine"\nMagazine"" was not the first women\'s magazine. It was conceived by the London bookseller John Coote and the publisher John Wheble, and first appeared in print in August 1770. John Huddlestone Wynne, an early editor of the magazine, also edited several other contemporary publications. ""The Lady\'s Magazine"" dominated the market from its founding to 1830. It claimed a readership base of sixteen thousand, a sum the 18th-century scholar Ros Ballaster considers a success when analysing the country\'s contemporary literacy levels and underdeveloped printing technologies. Its success led to imitations like the ""Lady\'s Monthly Museum"" and the ""New Lady\'s Magazine""
    Doc5: "Men Only"\nMen Only Men Only is a British soft-core pornographic magazine published by Paul Raymond Publications since 1971. However, the title goes back to 1935 when it was founded by C. Arthur Pearson Ltd as a pocket magazine (115×165 mm). It set out its editorial stall in the first issue: \'We don\'t want women readers. We won\'t have women readers...\' It sought \'bright articles on current male topics\'. Humour was at the heart of the title, though from the start it carried fiction, wide-ranging articles and plates of \'art\' nudes. Covers were initially text-only, then carried caricatures of famous people and.
    Doc6: Arthur\'s Lady\'s Home Magazine"\nfashion plates are not quite such extravagant caricatures of rag-baby work as are usually met with in some of the more fancy magazines."" Readers included patrons of the Mercantile Library Association of San Francisco. Arthur\'s Lady\'s Home Magazine Arthur\'s Home Magazine (1852-ca.1898) or Ladies\' Home Magazine was an American periodical published in Philadelphia by Timothy Shay Arthur. Editors Arthur and Virginia Francis Townsend selected writing and illustrations intended to appeal to female readers. Among the contributors: Mary Tyler Peabody Mann and Kate Sutherland.
    Doc7: Ladies\' Magazine"\nLadies\' Magazine The Ladies\' Magazine, an early women\'s magazine, was first published in 1828 in Boston, Massachusetts. Also known as ""Ladies\' Magazine and Literary Gazette"" and later as ""American Ladies Magazine"", it was designed to be American, and named to separate itself from the ""Lady\'s Magazine"" of London. The magazine was founded by Reverend John Lauris Blake, Congregational minister and headmaster of the Cornhill School for Young Ladies, who desired to set a model for American womanhood. It is thought to have been the first magazine to be edited by a woman; from 1828 until 1836, its editor was Sarah
    """
    # reasoning_trace = "In 2018, Alexis Sánchez left Arsenal FC to join Manchester United FC"
    # context = context + " " + reasoning_trace
    # batch = [question] + ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?"] + ["What is the capital of Japan?"]
    results = generator.finalize(question=question, context=context)
    breakpoint()       
        
