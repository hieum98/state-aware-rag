import copy
import json
import os
import shutil
from typing import List, Optional, TextIO, Union
import tqdm
from colorama import Fore, Style

from planners.MCTS.backbone import MCTS
from planners.MCTS.reasoning_node import *

# Modified from https://vscode.dev/github/zhentingqi/rStar/blob/main/run_src/rstar_utils.py#L60-L120
def print_tree_from_root(
        mcts_searcher: MCTS, 
        rollout_id: int, 
        root_node: ReasoningNode, 
        chosen_node: Optional[ReasoningNode] = None, 
        file: Optional[TextIO] = None
):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(
            parent_node: ReasoningNode, 
            node: ReasoningNode, 
            rollout_id: int,
            file: Optional[TextIO] = None, 
            ):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "

        # uct_value = "UCT: " + str(
        #     round(mcts_searcher.uct(parent_node=parent_node, child_node=node), 2)
        # )
        # attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_terminal() else ""

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_terminal():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        node_data = copy.deepcopy(node.get_node())
        if node.node_type is NodeType.USER_QUESTION:
            gt = node_data.get('golden_answer', 'N/A')
            user_question = node_data['user_question'].replace('\n', ' ').replace('\r', ' ')
            node_details += f"User: {user_question}" + "\n" + space + " " * len(node_info) + f"Ground truth: {gt}"
        elif node.node_type is NodeType.REPHASED_QUESTION_NODE:
            rephased_question = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            node_details += f"Rephase: {rephased_question}"
        elif node.node_type is NodeType.FINAL_ANSWER:
            final_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details += f"Final: {final_answer} - Conf: {confidence}"
        elif node.node_type is NodeType.SELF_CORRECTED_NODE:
            corrected_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details += f"Self_corrected: {corrected_answer} - Conf: {confidence}"
        elif node.node_type is NodeType.SUB_QA_NODE:
            sub_question = node_data['sub_question'].replace('\n', ' ').replace('\r', ' ')
            sub_answer = node_data['sub_answer'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details += f"Sub_Q: {sub_question} - Sub_A: {sub_answer} - Conf: {confidence}"
        elif node.node_type is NodeType.SYNTHESIS_NODE:
            synthesis_reasoning = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details += f"Synthesis: {synthesis_reasoning} - Conf: {confidence}"

        to_print += dash + node_details

        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)

        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)


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
        verbose: bool = False,
):  
    # Initialize the MCTS searcher with the given exploration weight
    mcts_searcher = MCTS(exploration_weight=exploration_weight, verbose=verbose)

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
        verbose=False,
    )
    for i in range(num_rollouts):
        simulated_node = mcts_searcher.do_rollout(root_node)
        print_tree_from_root(
            mcts_searcher=mcts_searcher, 
            rollout_id=i, 
            root_node=root_node,
        )
        breakpoint()
        best_solution, scores, solution_nodes = find_best_solution(root_node, verbose=verbose)
        if verbose:
            print("**" * 20)
            print(f"Rollout {i+1}/{num_rollouts}")
            if best_solution is not None:
                print("Best solution found:")
                pprint.pprint(best_solution, indent=4, width=120)
    
    solutions = []
    answers = []
    for node in solution_nodes:
        answers.append(node.state['node_content'])
        solutions.append(node.get_node())
    # Major voting to find the best solution from the solution nodes of the final tree
    if len(answers) == 0:
        print("No valid solution nodes found in the reasoning tree.")
        final_answer = None
    final_answer = evaluator.majority_vote(question=user_question, answers=answers)
    
    if save_tree:
        # Save all nodes of the tree
        nodes = []
        queue = [root_node]
        while queue:
            current_node = queue.pop(0)
            nodes.append(current_node)
            for child in current_node.children:
                queue.append(child)
        # check if the save directory does not exist, create it
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/mcts_tree_ {question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = node.get_node()
                node_content['reward'] = mcts_searcher.Q[node]
                node_content['visits'] = mcts_searcher.N[node]
                f.write(json.dumps(node_content) + "\n")
    return final_answer, solutions


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
        'url': 'http://n0998:30000/v1', 
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
        # 'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
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
        'n': 1, 
        'top_p': 0.9,
        'max_tokens': 1024*4,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        # 'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
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
        'max_tokens': 1024*4,  # Set to a high value to allow for long responses
        # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
        # For greedy decoding set it to 1
        'top_k': 20,
        'tensor_parallel_size': 1,
        # 'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
    }
    extractor = Extractor(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=extract_kwargs, 
        # verbose=True,
        use_cache=True,
        cache_dir="mcts_cache/extractor_cache",
    )

    retriever_online_kwargs = {
        "url": "http://n0998:5000/search",
        "retrieval_topk": 64,
        "query_instruction": "query: ",
    }
    retriever = RetrieverAgent(online_kwargs=retriever_online_kwargs)

    question = "Where was the performer of song (Last Night) I Heard You Crying In Your Sleep born?"

    final_answer, solution = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id="example_question_1",
        max_depth=7,
        golden_answer=None,
        # MCTS parameters
        num_rollouts=16,
        use_golden_answer=False,
        save_tree=True,
        save_dir="mcts_data",
        verbose=True,
    )
