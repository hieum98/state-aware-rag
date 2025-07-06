import copy
import json
import os
import shutil
from typing import List, Optional, Union
import tqdm
from planners.MCTS.backbone import MCTS
from planners.MCTS.reasoning_node import NodeType, ReasoningNode
from planners.generator import Generator
from planners.retriever import Retriever


def find_valid_solution_nodes(node: ReasoningNode) -> list[ReasoningNode]:
    """
    Find all valid solution nodes in the reasoning tree.
    A valid solution node is a node that has no children and is not a leaf node.
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
        if node.node_type == NodeType.DIRECT_ANSWER:
            answer = node.node_content["direct_answer"]
        elif node.node_type == NodeType.SUBQUESTION:
            answer = node.node_content["subanswer"]
        reasoning_trace = node.get_reasoning_trace()
        user_question = node.node_config['user_question']
        score = node.reward()
        if score >= highest_score:
            highest_score = score
            best_solution = {
                'question': user_question,
                'answer': answer,
                'score': score,
                'reasoning_trace': reasoning_trace,
            }
        scores.append({
            'question': user_question,
            'answer': answer,
            'score': score,
            'reasoning_trace': reasoning_trace,
        })
    assert len(scores) == len(solution_nodes), "Scores length does not match solution nodes length"
    return best_solution, scores
        
def search(
        generator: Generator,
        retriever: Retriever,
        question: str,
        question_id: str,
        golden_answer: Optional[Union[str, List[str]]] = None,
        max_depth: int = 5,
        refine_retrieved_docs: bool = True,
        # MCTS parameters
        exploration_weight: float = 1.0,
        num_rollouts: int = 16,
        use_golden_answer: bool = False,
        save_tree: bool = False,
        save_dir: str = "mcts_trees",
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
        retriever=retriever,
        question=question,
        # Optional parameters
        refine_retrieved_docs=refine_retrieved_docs,
        max_depth=max_depth,
        golden_answer=golden_answer,
    )
    solutions = []
    for i in tqdm.tqdm(range(num_rollouts), desc="MCTS Rollouts"):
        mcts_searcher.do_rollout(root_node)
        best_solution, solution_scores  = find_best_solution(root_node)
        if best_solution is not None:
            best_solution['rollout_index'] = i  # Add the rollout index to the solution
            solutions.append(best_solution)
            print(f"Rollout {i+1}/{num_rollouts}:")
            print(f"  Question: {best_solution['question']}")
            print(f"  Answer: {best_solution['answer']}")
            print(f"  Score: {best_solution['score']}")
            print(f"  Reasoning Trace: \n{best_solution['reasoning_trace']}")
    
    if save_tree:
        # Save all nodes of the tree from the root node with BFS traversal
        nodes = []
        queue = [root_node]
        while queue:
            current_node = queue.pop(0)
            nodes.append(current_node)
            for child in current_node.children:
                queue.append(child)
        root_node.node_content['goden_answer'] = golden_answer
        # check if the save directory does not exist, create it
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/mcts_tree_ {question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = copy.deepcopy(node.node_content)
                node_content['node_type'] = node.node_type.value
                node_content['depth'] = node.depth
                node_content['parent'] = hash(node.parent) if node.parent is not None else "ROOT"
                node_content['reward'] = mcts_searcher.Q[node]
                node_content['visits'] = mcts_searcher.N[node]
                node_content['hash'] = hash(node)
                f.write(json.dumps(node_content) + "\n")
    # Clear the agent cache if it is used
    if generator.llm_agent.use_cache:
        cache_dir = generator.llm_agent.cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)
    return solutions


if __name__ == "__main__":
    # Example usage
    generator_online_model_kwargs = {
        'model_name': 'qwen3-32b',
        'url': 'http://n0999.talapas.uoregon.edu:30000/v1',
        'api_key': 'None',
        'concurrency': 64,
    }
    generate_kwargs = {
        'temperature': 0.6,
        'n': 1, # should be odd number ás it is used for majority voting
        'top_p': 0.95,
        'max_tokens': 8192,
        'top_k': 20,
        'repetition_penalty': 1.1,
        'logprobs': 1,
        'tensor_parallel_size': 1,
    }
    generator = Generator(online_model_kwargs=generator_online_model_kwargs, generate_kwargs=generate_kwargs, verbose=False, use_cache=True, cache_dir="cache/llm_agents")

    retriever_online_kwargs = {
        "url": "http://n0999.talapas.uoregon.edu:5000/search",
        "retrieval_topk": 5,
        "query_instruction": "query: ",
    }
    retriever = Retriever(online_kwargs=retriever_online_kwargs)

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    solution = search(
        generator=generator,
        retriever=retriever,
        question=question,
        question_id="example_question",
        golden_answer=["No"],
        max_depth=15,
        refine_retrieved_docs=True,
        # MCTS parameters
        exploration_weight= 1.0,
        num_rollouts=5,
        use_golden_answer=True,
        save_tree=True,
        save_dir="mcts_trees",
    )
    breakpoint()
