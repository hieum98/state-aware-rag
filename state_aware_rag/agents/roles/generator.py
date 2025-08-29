import time 
from typing import Any, Dict, List, Optional, Union
import pydantic

from state_aware_rag.agents.llm_agents import LLMAgent
from state_aware_rag.agents.prompts import (
    decompose_and_answer,
    synthesize,
    finalize,
    self_correct,
    rephase_question,
    )
from state_aware_rag.agents.utils import convert_score_to_confidence
from typing import Tuple


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
        """Generate answers, single or batched, optionally using context."""
        if isinstance(question, str):
            question = [question]
            assert isinstance(context, str) or context is None, "Context must be a string or None when question is a string."
            context = [context if context else "No context provided."]
        if len(question) > 1:
            kwargs['n'] = 1
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
        batch = [[{'role': 'user', 'content': x}] for x in batch]
        if self.verbose:
            print("Generating answers for questions:", question)
            print("Context:", context)
        kwargs['output_schema'] = decompose_and_answer.AnswerOutput
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[decompose_and_answer.AnswerOutput] = []
        for x in items:
            reasoning = x.get('reasoning', "")
            detailed_answer = x.get('detailed_answer', "")
            answer = x.get('answer', "")
            confidence = convert_score_to_confidence(x.get('confidence', None))
            assert answer or detailed_answer, "Either answer or detailed_answer must be provided."
            if not answer and detailed_answer:
                answer = detailed_answer
            if not detailed_answer and answer:
                detailed_answer = answer
            all_results.append(
                decompose_and_answer.AnswerOutput(
                    answer=answer,
                    detailed_answer=detailed_answer,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return all_results
    
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
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[decompose_and_answer.SubquestionOutput] = []
        for i, x in enumerate(items):
            answerable_main_question = x.get('answerable_main_question', False)
            subquestion = x.get('subquestion', "")
            assert subquestion or answerable_main_question, "Either subquestion must be provided or the main question must be answerable."
            if answerable_main_question:
                subquestion = question[i] if len(question) > 1 else question[0]
            reasoning = x.get('reasoning', "")
            gap_type = x.get('gap_type', "null")
            gap_type = gap_type if gap_type else "null"
            all_results.append(
                decompose_and_answer.SubquestionOutput(
                    answerable_main_question=answerable_main_question,
                    subquestion=subquestion,
                    reasoning=reasoning,
                    gap_type=gap_type,
                )
            )
        return all_results

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
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[synthesize.SynthesizeOutput] = []
        for x in items:
            answerable_main_question = x.get('answerable_main_question', False)
            synthesis = x.get('synthesis', "")
            # If the synthesis is empty, set answerable_main_question to False to keep reasoning consistent
            if not synthesis:
                answerable_main_question = False
            reasoning = x.get('reasoning', "")
            confidence = convert_score_to_confidence(x.get('confidence', None))
            assert synthesis, "Synthesis must be provided."
            all_results.append(
                synthesize.SynthesizeOutput(
                    answerable_main_question=answerable_main_question,
                    synthesis=synthesis,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return all_results

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
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[finalize.FinalizeOutput] = []
        for x in items:
            final_answer = x.get('answer', "")
            detailed_final_answer = x.get('detailed_answer', "")
            assert final_answer or detailed_final_answer, "Either final_answer or detailed_final_answer must be provided."
            if not final_answer and detailed_final_answer:
                final_answer = detailed_final_answer # If final_answer is empty, use detailed_final_answer as final_answer
            if not detailed_final_answer and final_answer:
                detailed_final_answer = final_answer # If detailed_final_answer is empty, use final_answer as detailed_final_answer
            reasoning = x.get('reasoning', "")
            confidence = convert_score_to_confidence(x.get('confidence', None))
            all_results.append(
                finalize.FinalizeOutput(
                    answer=final_answer,
                    detailed_answer=detailed_final_answer,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return all_results
    
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
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[self_correct.SelfCorrectOutput] = []
        for x in items:
            reanswer = x.get('reanswer', "")
            reasoning = x.get('reasoning', "")
            verification_status = x.get('verification_status', "UNSUPPORTED")
            confidence = convert_score_to_confidence(x.get('confidence', None))
            assert reasoning or reanswer, "Either reasoning or reanswer must be provided."
            if not reanswer and reasoning:
                reanswer = reasoning # If reanswer is empty, use reasoning as reanswer
            if not reasoning and reanswer:
                reasoning = reanswer # If reasoning is empty, use reanswer as reasoning
            all_results.append(
                self_correct.SelfCorrectOutput(
                    verification_status=verification_status,
                    reanswer=reanswer,
                    reasoning=reasoning,
                    confidence=confidence,
                )
            )
        return all_results

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
        response = self.role_execute(batch, **kwargs)
        items = response if isinstance(response, list) else [response]
        all_results: List[rephase_question.RephraseQuestionOutput] = []
        for x in items:
            rephrased_question = x.get('rephrased_question', "")
            reasoning = x.get('reasoning', "")
            assert rephrased_question or reasoning, "Either rephrased_question or reasoning must be provided."
            if not rephrased_question and reasoning:
                rephrased_question = reasoning # If rephrased_question is empty, use reasoning as rephrased_question
            if not reasoning and rephrased_question:
                reasoning = rephrased_question # If reasoning is empty, use rephrased_question as reasoning
            all_results.append(
                rephase_question.RephraseQuestionOutput(
                    rephrased_question=rephrased_question,
                    reasoning=reasoning,
                )
            )
        return all_results

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
        response = self.role_execute(batch, **kwargs)
        output: List[decompose_and_answer.QueriesGenerationOutput] = []
        items = response if isinstance(response, list) else [response]
        for item in items:
            if 'queries' in item:
                queries = item['queries']
                if isinstance(queries, str):
                    queries = queries.strip().split(',\n')
                    queries = [q.strip().strip('"') for q in queries]
                    # Remove empty queries
                    queries = [q for q in queries if q]
                item['queries'] = queries
                output.append(
                    decompose_and_answer.QueriesGenerationOutput(queries=item['queries'], reasoning=None)
                )
        if not output:
            print("Empty response from the model. Please check!")
            breakpoint()
        return output

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
        
