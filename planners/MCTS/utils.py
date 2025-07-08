import copy
import json
import os
import shutil
from typing import List, Optional, Union
import tqdm
from planners.MCTS.backbone import MCTS
from planners.MCTS.reasoning_node import *



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
    for node in tqdm.tqdm(solution_nodes, desc="Evaluating solution nodes"):
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
        verbose: bool = True,
):  
    # Initialize the MCTS searcher with the given exploration weight
    mcts_searcher = MCTS(exploration_weight=exploration_weight, verbose=False)

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
        verbose=verbose,
    )
    solutions = []
    for i in tqdm.tqdm(range(num_rollouts), desc="MCTS Rollouts"):
        mcts_searcher.do_rollout(root_node)
        best_solution, scores, solution_nodes = find_best_solution(root_node, verbose=verbose)
        best_solution['rollout_index'] = i  # Add the rollout index to the solution
        solutions.append(best_solution)
        if verbose:
            print("**" * 20)
            print(f"Rollout {i+1}/{num_rollouts}")
            i = 0
            for node, score in zip(solution_nodes, scores):
                print(f"Solution {i+1} with score {score}:")
                node.print_node()
                i += 1
            if best_solution is not None:
                print("Best solution found:")
                pprint.pprint(best_solution, indent=4, width=120)
    
    if save_tree:
        # Save all nodes of the tree from the root node with BFS traversal
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
    return solutions


if __name__ == "__main__":
    # Example usage
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
    evaluator = Evaluator(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        # verbose=True,
        use_cache=True, 
        cache_dir="mcts_cache/evaluator_cache",
    )
    extractor = Extractor(
        client_kwargs=online_model_kwargs, 
        generate_kwargs=generate_kwargs, 
        # verbose=True,
        use_cache=True,
        cache_dir="mcts_cache/extractor_cache",
    )

    retriever_online_kwargs = {
        "url": "http://n0998.talapas.uoregon.edu:5000/search",
        "retrieval_topk": 64,
        "query_instruction": "query: ",
    }
    retriever = RetrieverAgent(online_kwargs=retriever_online_kwargs)

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    solution = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id="example_question_1",
        max_depth=15,
        golden_answer=["Arthur's Magazine"],
        # MCTS parameters
        num_rollouts=16,
        use_golden_answer=True,
        save_tree=True,
        save_dir="mcts_data",
        verbose=True,
    )
    breakpoint()
