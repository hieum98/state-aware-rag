import copy
import json
import os
import pprint
import shutil
import random
from typing import List, Optional, TextIO, Union
import tqdm
from colorama import Fore, Style
from anytree import RenderTree

from preprocess.utils import simple_preprocess
from planners.MCTS.backbone import MCTS
from planners.MCTS.reasoning_node import *

# Modified from https://vscode.dev/github/zhentingqi/rStar/blob/main/run_src/rstar_utils.py#L60-L120
def print_tree_from_root(root_node: ReasoningNode):
    """
    Print the reasoning tree from the root node.
    """
    for pre, _, node in RenderTree(root_node):
        node_data = node.get_node()
        node_id = node.__str__()
        if node.node_type is NodeType.USER_QUESTION:
            gt = node_data.get('golden_answer', 'N/A')
            user_question = node_data['user_question'].replace('\n', ' ').replace('\r', ' ')
            node_details = f"User: {user_question} - Ground truth: {gt}"
            node_details = f"{Fore.GREEN}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.REPHASED_QUESTION_NODE:
            rephased_question = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            node_details = f"Rephase: {rephased_question}"
            node_details = f"{Fore.YELLOW}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.FINAL_ANSWER:
            final_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Final: {final_answer} - Conf: {confidence}"
            node_details = f"{Fore.BLUE}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SELF_CORRECTED_NODE:
            corrected_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Self_corrected: {corrected_answer} - Conf: {confidence}"
            node_details = f"{Fore.MAGENTA}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SUB_QA_NODE:
            sub_question = node_data['sub_question'].replace('\n', ' ').replace('\r', ' ')
            sub_answer = node_data['sub_answer'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Sub_Q: {sub_question} - Sub_A: {sub_answer} - Conf: {confidence}"
            node_details = f"{Fore.CYAN}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SYNTHESIS_NODE:
            synthesis_reasoning = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Synthesis: {synthesis_reasoning} - Conf: {confidence}"
            node_details = f"{Fore.RED}{node_id}{Style.RESET_ALL} {node_details}"
 
        tree_str = f"{pre}{node_details}"
        print(tree_str)

def find_valid_solution_nodes(node: ReasoningNode) -> list[ReasoningNode]:
    """
    Find all valid solution nodes in the reasoning tree.
    A valid solution node is a node that has no children and is a valid leaf node.
    """
    if not node.children:
        return []

    valid_nodes = []
    for child in node.children:
        if child.is_valid_leaf():
            valid_nodes.append(child)
        else:
            valid_nodes.extend(find_valid_solution_nodes(child))
    
    return valid_nodes


def find_best_solution(root_node: ReasoningNode, verbose: bool = True):
    """
    Find the best solution node in the reasoning tree.
    The best solution node is the one with the highest score.
    """
    solution_nodes = find_valid_solution_nodes(root_node)
    if len(solution_nodes) == 0:
        return None, None
    
    scores = []
    highest_score = float('-inf')
    best_solution = None
    for node in solution_nodes:
        score = node.reward()
        node_data = node.get_node()
        node_data['reward'] = score
        if score >= highest_score:
            highest_score = score
            best_solution = node_data
        scores.append(score)
    return best_solution, scores, solution_nodes


def search(
        # Root node components
        generator: Generator,
        evaluator: Evaluator,
        extractor: Extractor,
        retriever: RetrieverAgent,
        # Question components
        user_question: Optional[str] = None,
        question_id: Optional[str] = None,
        max_depth: int = 15,
        golden_answer: Optional[Union[str, List[str]]] = None,
        # MCTS parameters
        exploration_weight: float = 1.0,
        num_rollouts: int = 16,
        use_golden_answer: bool = False,
        save_tree: bool = False,
        save_dir: str = "mcts_trees",
        top_k: int = 5,
        verbose: bool = False,
):  
    # Initialize the MCTS searcher with the given exploration weight
    mcts_searcher = MCTS(exploration_weight=exploration_weight, verbose=False)

    # Normalize the user question and golden answer if provided
    if user_question is not None:
        user_question = simple_preprocess(user_question)
    if golden_answer is not None:
        if isinstance(golden_answer, str):
            golden_answer = [simple_preprocess(golden_answer)]
        elif isinstance(golden_answer, list):
            golden_answer = [simple_preprocess(ans) for ans in golden_answer]
        else:
            raise ValueError("golden_answer must be a string or a list of strings.")

    # Start the search from the root node
    root_node = ReasoningNode(
        parent=None,
        node_type=NodeType.USER_QUESTION,
        depth=0,
        # Components
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Optional parameters
        max_depth=max_depth,
        golden_answer=golden_answer if use_golden_answer else None,
        user_question=user_question,
        question_id=question_id,
        top_k=top_k,  # Set the top_k for the retriever
        verbose=False,
    )
    for i in range(num_rollouts):
        simulated_node = mcts_searcher.do_rollout(root_node, rollout_id=i)
        if verbose:
            print_tree_from_root(root_node)
        best_solution, scores, solution_nodes = find_best_solution(root_node, verbose=verbose)
        if verbose:
            print("**" * 20)
            print(f"Rollout {i+1}/{num_rollouts}")
            if best_solution is not None:
                print("Best solution found:")
                pprint.pprint(best_solution, indent=4, width=120)
    nodes = []
    solutions = []
    answers = []
    full_answers = []
    reasoning_paths = []
    for _, _, node in RenderTree(root_node):
        nodes.append(node)
        if node.node_type is NodeType.FINAL_ANSWER:
            answer = node.state['node_content']
            detailed_answer = node.state['detailed_answer']
            if (answer is None or answer.strip() == "") and (detailed_answer is None or detailed_answer.strip() == ""):
                continue
            if answer is not None and answer.strip() != "":
                answers.append(answer)
            else:
                answers.append(detailed_answer)
            
            if detailed_answer is not None and detailed_answer.strip() != "":
                full_answers.append(detailed_answer)
            else:
                full_answers.append(answer)
    
            solutions.append(node.get_node())
            reasoning_path = node.get_path()
            reasoning_path, _ = node.get_reasoning_trace(path=reasoning_path)
            if reasoning_path is not None:
                reasoning_paths.append(reasoning_path)
            else:
                # If the reasoning path is None, we only append the node content of the final answer
                reasoning_paths.append(node.state['detailed_answer'])
    # Major voting to find the best solution from the solution nodes of the final tree
    if len(answers) == 0:
        print("No valid solution nodes found in the reasoning tree.")
        final_answer = None
        final_reasoning = None
    try:
        voted_answer, _final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=full_answers)
        final_reasoning = _final_reasoning 
    except Exception as e:
        voted_answer = evaluator.majority_vote(question=user_question, answers=answers)
    final_answer = voted_answer
    # if isinstance(voted_answer, str) and voted_answer not in ['No valid answer generated.'] and  voted_answer.strip() != "":
    #     try:
    #         questions = [user_question] * len(reasoning_paths)
    #         voted_answers = [voted_answer] * len(reasoning_paths)
    #         response = evaluator.evaluate_final_answer(question=questions, correct_answer=voted_answers, predicted_answer=reasoning_paths)
    #         assert len(response) == len(reasoning_paths), "Response length does not match reasoning paths length."
    #         voted_reasonings = [rea for res, rea in zip(response, reasoning_paths) if res['decision']]
    #         voted_reasonings = list(set(voted_reasonings))  # Remove duplicates
    #         random.shuffle(voted_reasonings)  # Shuffle to ensure randomness in selection
    #         total_length = 0
    #         selected_reasonings = []
    #         while total_length < 15000:
    #             # Randomly select reasoning paths until the total length is less than 15000 tokens to avoid exceeding the limit
    #             path = voted_reasonings.pop() if voted_reasonings else None
    #             if path is None:
    #                 break
    #             selected_reasonings.append(path)
    #             total_length += len(path.split())
    #         final_answer, final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=voted_reasonings)
    #     except Exception as e:
    #         try:
    #             final_answer, final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=full_answers)
    #         except Exception as e:
    #             print(f"Error synthesizing final answer: {e}")
    #             final_answer = evaluator.majority_vote(question=user_question, answers=answers)
    #             final_reasoning = None
    # else:
    #     try:
    #         final_answer, final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=full_answers)
    #     except Exception as e:
    #         print(f"Error synthesizing final answer: {e}")
    #         final_answer = evaluator.majority_vote(question=user_question, answers=answers)
    #         final_reasoning = None
    
    if save_tree:
        # check if the save directory does not exist, create it
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/mcts_tree_ {question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = node.get_node()
                node_content['reward'] = mcts_searcher.Q[node]
                node_content['visits'] = mcts_searcher.N[node]
                f.write(json.dumps(node_content) + "\n")
    return final_answer, final_reasoning, reasoning_paths


def clear_agent_cache(generator, extractor, evaluator):
    # Clear the agent cache if it is used
    if generator.use_cache:
        cache_dir = generator.cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)
    if extractor.use_cache:
        cache_dir = extractor.cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)
    if evaluator.use_cache:
        cache_dir = evaluator.cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    # Example usage
    online_model_kwargs = {
        'model_name': 'openai/qwen3-8B', 
        'url': 'http://ip-10-4-241-174:30000/v1', 
        'api_key': 'your_api_key_here',  # Replace with your actual API key
        'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
        'concurrency': 64,
    }
    api_model_kwargs = {
        # 'model_name': 'bedrock/us.anthropic.claude-opus-4-20250514-v1:0',
        'model_name': 'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        # 'model_name': 'bedrock/us.deepseek.r1-v1:0',  # Use DeepSeek R1 model
        'url': None,  # Use default URL for the model
        'api_key': None,  # Set your API key if required
        'aws_profile_name': 'hieu', # 'aws_profile_name': 'hieu',  # Set your AWS profile name if using AWS Bedrock
        # 'model_name': 'openai/qwen3-8B', 
        # 'url': 'http://ip-10-4-226-205:30000/v1', 
        # 'api_key': 'your_api_key_here',  # Replace with your actual API key
        # 'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
        'concurrency': 64,
    }
    generate_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 1,  
        'n': 3, 
        'top_p': 0.9,
        'max_tokens': 1024*8,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    generator = Generator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        # verbose=True,
        use_cache=True,
        cache_dir="mcts_cache/generator_cache",
    )
    eval_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 0.1,  
        'n': 5, 
        'top_p': 0.9,
        'max_tokens': 1024*8,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    evaluator = Evaluator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=eval_kwargs, 
        # verbose=True,
        use_cache=True, 
        cache_dir="mcts_cache/evaluator_cache",
    )
    extract_kwargs = {
        # For creative tasks (creative writing) set it ~ 1, 
        # For logical or factual tasks (summarization, coding, analysis) set it ~ 0
        # For general conversation set it ~ 0.7
        'temperature': 0.1,  
        'n': 1, 
        'top_p': 0.9,
        'max_tokens': 1024*8,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    extractor = Extractor(
        # client_kwargs=online_model_kwargs, 
        client_kwargs=api_model_kwargs,
        generate_kwargs=extract_kwargs, 
        # verbose=True,
        use_cache=True,
        cache_dir="mcts_cache/extractor_cache",
    )

    retriever_online_kwargs = {
        "url": "http://ip-10-4-225-181:5000/search",
        "retrieval_topk": 64,
    }
    retriever = RetrieverAgent(online_kwargs=retriever_online_kwargs)

    question = "Where was the director of film Breakup Buddies born?"

    final_answer, final_reasoning, reasoning_paths = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id="example_question_1",
        max_depth=6,
        golden_answer="Taiyuan",
        # MCTS parameters
        num_rollouts=10,
        use_golden_answer=True,
        save_tree=True,
        save_dir="mcts_data",
        verbose=True,
    )
    breakpoint()  # Debugging point to inspect the final answer and solution
