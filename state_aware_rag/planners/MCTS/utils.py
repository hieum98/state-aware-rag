from collections import defaultdict
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

from state_aware_rag.preprocess.utils import simple_preprocess
from state_aware_rag.agents.utils import format_memory
from state_aware_rag.planners.MCTS.backbone import MCTS
from state_aware_rag.planners.reasoning_node import *

# Modified from https://vscode.dev/github/zhentingqi/rStar/blob/main/run_src/rstar_utils.py#L60-L120
def print_tree_from_root(root_node: ReasoningNode):
    for pre, _, node in RenderTree(root_node):
        node: ReasoningNode
        node_data = node.get_node()
        node_id = f"{node_data['rollout_id']}-{node_data['hash']}-{node_data['node_type']}"
        memory = node_data['memory']
        memory = [f'M. {m}' for m in memory if m]  # Filter out empty memory entries
        memory_str = ' | '.join(memory) if memory else 'No memory'
        if node.node_type is NodeType.USER_QUESTION:
            gt = node_data.get('golden_answer', 'N/A')
            user_question = node_data['user_question'].replace('\n', ' ').replace('\r', ' ')
            node_details = f"User: {user_question} - Ground truth: {gt}"
            node_details = f"{Fore.GREEN}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.REPHASED_QUESTION_NODE:
            rephased_question = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            node_details = f"Rephase: {rephased_question} - Memory: {memory_str}"
            node_details = f"{Fore.YELLOW}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.FINAL_ANSWER:
            final_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Final: {final_answer} - Memory: {memory_str} - Conf: {confidence}"
            node_details = f"{Fore.BLUE}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SELF_CORRECTED_NODE:
            corrected_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Self_corrected: {corrected_answer} - Memory: {memory_str} - Conf: {confidence}"
            node_details = f"{Fore.MAGENTA}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SUB_QA_NODE:
            sub_question = node_data['sub_question'].replace('\n', ' ').replace('\r', ' ')
            sub_answer = node_data['sub_answer'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Sub_Q: {sub_question} - Sub_A: {sub_answer} - Memory: {memory_str} - Conf: {confidence}"
            node_details = f"{Fore.CYAN}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SYNTHESIS_NODE:
            synthesis_reasoning = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Synthesis: {synthesis_reasoning} - Memory: {memory_str} - Conf: {confidence}"
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


def get_tree_from_file(
        save_path: str,
        generator: GeneratorAgent,
        evaluator: EvaluatorAgent,
        extractor: ExtractorAgent,
        retriever: RetrievalAgent,
        user_question: Optional[str] = None,
        question_id: Optional[str] = None,
        max_depth: int = 15,
        golden_answer: Optional[Union[str, List[str]]] = None,
        use_golden_answer: bool = False,
        top_k: int = 5,
        is_cot: bool = False,
        verbose: bool = False,
        **kwargs
    ):
    # Start the search from the root node
    root_node = ReasoningNode(
        parent=None,
        node_type=NodeType.USER_QUESTION,
        is_cot=is_cot,
        # Components
        generator=generator,
        retriever=retriever,
        extractor=extractor,
        evaluator=evaluator,
        # Optional parameters
        max_depth=max_depth,
        golden_answer=golden_answer if use_golden_answer else None,
        user_question=user_question,
        question_id=question_id,
        top_k=top_k,  
    )
    all_nodes = {}
    with open(save_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            node_data = json.loads(line.strip())
            node_config = {
                "max_depth": max_depth,
                "golden_answer": golden_answer,
                "user_question": user_question,
                "question_id": question_id,
                "top_k": top_k,
                "is_cot": is_cot,
                # Node agents
                "generator": generator,
                "retriever": retriever,
                "evaluator": evaluator,
                "extractor": extractor,
            }
            node_type = NodeType(node_data['node_type'])
            node_content = {}
            if node_type is NodeType.USER_QUESTION:
                node_content = {}
            elif node_type is NodeType.FINAL_ANSWER:
                node_content = {
                    'answer': node_data['node_content'],
                    'reasoning': node_data['detailed_answer']
                }
            elif node_type is NodeType.SUB_QA_NODE:
                node_content = {
                    'question': node_data['sub_question'],
                    'answer': node_data['sub_answer'],
                }
            elif node_type is NodeType.REPHASED_QUESTION_NODE:
                node_content = {
                    'question': node_data['node_content'],
                }
            elif node_type is NodeType.SELF_CORRECTED_NODE:
                node_content = {
                    'question': node_data['sub_question'],
                    'answer': node_data['sub_answer'],
                    'reasoning': node_data['reasoning'],
                }
            elif node_type is NodeType.SYNTHESIS_NODE:
                node_content = {
                    'reasoning': node_data['node_content'],
                }
            else:
                raise ValueError(f"Unknown node type: {node_type}")                
            node = ReasoningNode(
                parent=None, # Temporarily set to None, will be updated later
                node_type=NodeType(node_data['node_type']),
                confidence=node_data['confidence'],
                memory=node_data['memory'],
                **node_config,
                **node_content
            )
            node.set_rollout_id(node_data['rollout_id'])
            node_parent = node_data['parent']
            node_hash = node_data['hash']
            if node_parent is None:
                node = root_node
            all_nodes[node_hash] = [node, node_parent]
    # Rebuild the tree from the saved nodes
    for node_hash, (node, parent_hash) in all_nodes.items():
        if parent_hash is None:
            # If the parent is None, it means this is the root node
            continue
        parent_node = all_nodes[parent_hash][0]
        node.parent = parent_node
    if verbose:
        print(f"Loaded MCTS tree from {save_path} with {len(all_nodes)} nodes.")
        print_tree_from_root(root_node)
    return root_node


def search_with_mcts(
        # Root node components
        generator: GeneratorAgent,
        evaluator: EvaluatorAgent,
        extractor: ExtractorAgent,
        retriever: RetrievalAgent,
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
        **kwargs
    ):  
    # Initialize the MCTS searcher with the given exploration weight
    mcts_searcher = MCTS(exploration_weight=exploration_weight)

    # Start the search from the root node
    root_node = ReasoningNode(
        parent=None,
        node_type=NodeType.USER_QUESTION,
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
        is_cot=False,
    )
    for i in range(num_rollouts):
        simulated_node = mcts_searcher.do_rollout(root_node, rollout_id=i)
        if verbose:
            print(f"Rollout {i+1}/{num_rollouts} completed")
            print_tree_from_root(root_node)
    
    if save_tree:
        # check if the save directory does not exist, create it
        nodes = []
        for _, _, node in RenderTree(root_node):
            nodes.append(node)
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/mcts_tree_{question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = node.get_node()
                node_content['reward'] = mcts_searcher.Q[node]
                node_content['visits'] = mcts_searcher.N[node]
                f.write(json.dumps(node_content) + "\n")
    return root_node


def search(
       # Root node components
        generator: GeneratorAgent,
        evaluator: EvaluatorAgent,
        extractor: ExtractorAgent,
        retriever: RetrievalAgent,
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
        **kwargs
    ):

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
        
    save_path = f"{save_dir}/mcts_tree_{question_id}.jsonl" if save_tree else None
    if save_path and os.path.exists(save_path) and save_tree:
        # If the file already exists, load the tree from the file
        logger.info(f"Loading existing MCTS tree from {save_path}")
        root_node = get_tree_from_file(
            save_path=save_path,
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            user_question=user_question,
            question_id=question_id,
            is_cot=False,
            max_depth=max_depth,
            golden_answer=golden_answer,
            use_golden_answer=use_golden_answer,
            top_k=top_k,
            verbose=verbose
        )
    else:
        # If the file does not exist, perform a new search
        logger.info(f"Performing new MCTS search for question: {user_question} with ID: {question_id}")
        root_node = search_with_mcts(
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            user_question=user_question,
            question_id=question_id,
            max_depth=max_depth,
            golden_answer=golden_answer,
            exploration_weight=exploration_weight,
            num_rollouts=num_rollouts,
            use_golden_answer=use_golden_answer,
            save_tree=save_tree,
            save_dir=save_dir,
            top_k=top_k,
            verbose=verbose
        )

    nodes = []
    answers = []
    full_answers = []
    all_memories = []
    for _, _, node in RenderTree(root_node):
        node: ReasoningNode
        nodes.append(node)
        if node.node_type is NodeType.FINAL_ANSWER:
            answer = node.state['final_answer']
            detailed_answer = node.state['detailed_answer']
            memory = node.memory
            memory_str = format_memory(memory)
            if memory_str:
                all_memories.append(memory_str)
            if answer and answer.strip():
                answers.append(answer)
            if detailed_answer and detailed_answer.strip():
                full_answers.append(detailed_answer)
    if len(answers) == 0:
        leaf_nodes = root_node.leaves
        for leaf in leaf_nodes:
            answer = format_memory(leaf.memory)
            if answer and answer.strip():
                answers.append(answer)
                full_answers.append(answer)
                all_memories.append(answer)

    # Major voting to find the best solution from the solution nodes of the final tree
    if len(answers) == 0:
        print("No valid solution nodes found in the reasoning tree.")
        final_answer = None
        final_reasoning = None
    try:
        full_answers = [fa for fa in full_answers if fa and fa.strip()]
        if len(full_answers) == 0:
            agent_input = {
                'evaluate_fn': 'synthesize_final_answer',
                'question': user_question,
                'answers': answers,
            }
        else:
            agent_input = {
                'evaluate_fn': 'synthesize_final_answer',
                'question': user_question,
                'answers': full_answers,
            }
        instance_id, _ = asyncio.run(evaluator.create())
        response, _, _ = asyncio.run(evaluator.execute(instance_id, agent_input))
        final_answer = response['final_answer']
        final_reasoning = response['final_reasoning']
    except Exception as e:
        logger.error(f"Synthesis failed with error: {e}. Falling back to majority vote.")
        agent_input = {
            'evaluate_fn': 'majority_vote',
            'question': user_question,
            'answer_lists': answers,
        }
        instance_id, _ = asyncio.run(evaluator.create())
        final_answer, _, _ = asyncio.run(evaluator.execute(instance_id, agent_input))
        final_reasoning = final_answer  # Fallback to the final answer as reasoning if synthesis fails
    return final_answer, final_reasoning


def clear_agent_cache(generator: GeneratorAgent, extractor: ExtractorAgent, evaluator: EvaluatorAgent):
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


