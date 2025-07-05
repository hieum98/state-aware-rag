import time 
from typing import Any, Dict, List, Optional, Union
import pydantic

from agents.llm_agents import LLMAgent
from agents.prompts import extract
from agents.utils import extract_info_from_text


class Extractor(LLMAgent):
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

        self.extract_prompt = extract.EXTRACT_PROMPT
        self.extract_examples = None
    
    def extract(
            self, 
            question: Union[str, List[str]],
            current_step_objective: Union[str, List[str]],
            document: Union[str, List[str]],
            **kwargs
    ):
        if isinstance(question, str):
            question = [question]
            assert isinstance(current_step_objective, str), "current_step_objective must be a string when question is a string"
            assert isinstance(document, str), "document must be a string when question is a string"
            current_step_objective = [current_step_objective]
            document = [document]
        if len(question) > 1:
            kwargs['n'] = 1
        assert len(question) == len(current_step_objective) == len(document), "question, current_step_objective, and document must have the same length"

        batch = [
            self.extract_prompt.format(
                question=q,
                current_step_objective=co,
                document=d,
                examples=self.extract_examples if self.extract_examples else "No examples provided."
            ) for q, co, d in zip(question, current_step_objective, document)
        ]
        batch = [[{'role': 'user', 'content': x}] for x in batch]  # Format for the client
        if self.verbose:
            print("Generating answers for questions:", question)
            print("Batch size:", len(batch))
            for i, x in enumerate(batch):
                print(f"Batch {i}: {x}")
        kwargs['output_schema'] = extract.ExtractOutput
        responses = self.role_execute(batch, **kwargs)
        extracted_info = []
        for response in responses:
            decision = response['decision']
            try:
                if isinstance(response['extracted_information'], list):
                    info = [f"Information: {x['summary']}\nDetail: {x['reasoning']}" for x in response['extracted_information']]
            except:
                info = [response['extracted_information']]
            extracted_info.append({
                'decision': decision,
                'extracted_information': info,
            })
        return extracted_info           


if __name__ == "__main__":
    online_model_kwargs = {
        'model_name': 'openai/qwen3-8B', 
        'url': 'http://n0998.talapas.uoregon.edu:30000/v1', 
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
        'max_tokens': 1024*16,  # Set to a high value to allow for long responses 
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    generator = Extractor(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        verbose=True
    )

    question = 'In 2018, what Chilean footballer left Arsenal to join the team that The Saints beat in 1976 to win the FA Cup?'
    current_step_objective = 'Which team did The Saints beat in 1976 to win the FA Cup?'

    document = """Southampton Football Club is a professional football club based in Southampton, Hampshire, England. The club competes in the EFL Championship, the second tier of English football. Their home ground since 2001 has been St Mary's Stadium, before which it was based at The Dell. The team play in red and white shirts. They have been nicknamed "The Saints" because of the club's beginnings as a church football team at St Mary's Church. Southampton shares a long-standing South Coast derby rivalry with Portsmouth, in part due to geographic proximity and both cities' respective maritime histories. Founded in 1885, the club joined the Southern League as Southampton St. Mary's in 1894, dropping the St. Mary's from their name three years later. Southampton won the Southern League on six occasions and were beaten FA Cup finalists in 1900 and 1902, before being invited to become founder members of the Football League Third Division in 1920. They won promotion as Third Division South champions in 1921–22, remaining in the Second Division for 31 years until they were relegated in 1953. Crowned Third Division champions under the stewardship of Ted Bates in 1959–60, they were promoted into the First Division at the end of the 1965–66 campaign. They played top-flight football for eight seasons, but won the FA Cup as a Second Division team in 1976 with a 1–0 victory over Manchester United. Manager Lawrie McMenemy then took the club back into the top-flight with promotion in 1977–78. Southampton were beaten finalists in the League Cup in 1979 and finished as runners-up in the First Division in 1983–84, three points behind Liverpool. The club were founder members of the Premier League in 1992 and reached another FA Cup final in 2003. Relegation ended their 27-year stay in the top-flight in 2005, and they were relegated down to the third tier in 2009. Southampton won the Football League Trophy in 2010 and won successive promotion from League One and the EFL Championship in 2010–11 and 2011–12. After an 11-year stint in the top flight, during which they were EFL Cup runners-up in 2017, they were relegated in 2023. The club won the 2024 Championship play-off final and returned to the Premier League at the first attempt, but were relegated back to the Championship in April 2025 with seven games remaining."""
    
    document_2 = """Alexis Alejandro Sánchez Sánchez (born 19 December 1988), is a Chilean footballer who currently plays for Italian club Inter Milan and the Chile national team. In July 2014, Sanchez signed for Arsenal on a long-term contract for an undisclosed fee, reported to be around £35 million. In January 2018, he joined The Red Devils after Henrikh Mkhitaryan swapped clubs with him to Arsenal FC."""

    document_3 = """The 1976 FA Cup Final took place on 1 May 1976 at Wembley Stadium. It was contested between Manchester United and Southampton. Southampton United had finished third in the First Division that season, and were strong favourites, while unfancied Southampton had finished sixth in the Second Division. In one of the biggest shocks in the history of the final, Southampton won 1–0 through an 83rd-minute goal from Bobby Stokes. It was the first time Southampton won a major trophy."""

    document = f"Document 1: {document}\n\nDocument 2: {document_2}\n\nDocument 3: {document_3}"

    results = generator.extract(question=question, current_step_objective=current_step_objective, document=document)
    breakpoint()

