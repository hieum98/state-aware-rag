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


def get_tree_from_file(
        save_path: str,
        generator: Generator,
        evaluator: Evaluator,
        extractor: Extractor,
        retriever: RetrieverAgent,
        user_question: Optional[str] = None,
        question_id: Optional[str] = None,
        max_depth: int = 15,
        golden_answer: Optional[Union[str, List[str]]] = None,
        use_golden_answer: bool = False,
        top_k: int = 5,
        verbose: bool = False,
        **kwargs
    ):
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
    all_nodes = {}
    with open(save_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            node_data = json.loads(line.strip())
            node_config = {
                "verbose": verbose,  # Whether to print verbose output
                "max_depth": max_depth,  # Maximum depth of the reasoning tree
                "golden_answer": golden_answer,  # The golden answer for the user question, if available
                "user_question": user_question,  # The main user question for USER_QUESTION nodes
                "question_id": question_id,  # The ID of the question, if available
                "generator": generator,  # The generator component for the node
                "evaluator": evaluator,  # The evaluator component for the node
                "extractor": extractor,  # The extractor component for the node
                "retriever": retriever,  # The retriever component for the node
                "top_k": top_k,  # The number of top-k results to retrieve from the retriever
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
                    'question': node_data['node_content'],
                    'answer': ""
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
                depth=node_data['depth'],
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
        **kwargs
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
        question_id=question_id,
        top_k=top_k,  # Set the top_k for the retriever
        verbose=False,
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
    if os.path.exists(save_path) and save_tree:
        # If the file already exists, load the tree from the file
        if verbose:
            print(f"Loading existing MCTS tree from {save_path}")
        root_node = get_tree_from_file(
            save_path=save_path,
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            user_question=user_question,
            question_id=question_id,
            max_depth=max_depth,
            golden_answer=golden_answer,
            use_golden_answer=use_golden_answer,
            top_k=top_k,
            verbose=verbose
        )
    else:
        # If the file does not exist, perform a new search
        if verbose:
            print(f"Performing new MCTS search for question: {user_question} with ID: {question_id}")
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
    solutions = []
    answers = []
    full_answers = []
    reasoning_paths = []
    for _, _, node in RenderTree(root_node):
        nodes.append(node)
        if node.node_type is NodeType.FINAL_ANSWER:
            answer = node.state['node_content']
            detailed_answer = node.state['detailed_answer']
            answers.append(answer)
            full_answers.append(detailed_answer)
    
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
        total_length = sum([len(full_answer.split()) for full_answer in full_answers])
        if total_length < 15000: # Prevent exceeding the token limit
            final_answer, final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=full_answers)
        else:
            l = 0
            selected_answers = []
            random.shuffle(full_answers)  # Shuffle the full answers to ensure randomness in selection
            for full_answer in full_answers:
                if l < 10000:
                    selected_answers.append(full_answer)
                    l += len(full_answer.split())
                else:
                    break
            final_answer, final_reasoning = evaluator.synthesize_final_answer(question=user_question, reasoning_paths=selected_answers)
    except Exception as e:
        final_answer = evaluator.majority_vote(question=user_question, answers=answers)
        final_reasoning = final_answer  # Fallback to the final answer as reasoning if synthesis fails
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
        'url': 'http://ip-10-4-226-205:30000/v1', 
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
        'max_tokens': 1024*4,  # Set to a high value to allow for long responses
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
        'max_tokens': 1024*4,  # Set to a high value to allow for long responses
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
        "url": "http://ip-10-4-226-205:5000/search",
        "retrieval_topk": 64,
    }
    retriever = RetrieverAgent(online_kwargs=retriever_online_kwargs)

    question = " Which magazine was started first Arthur's Magazine or First for Women?"

    final_answer, final_reasoning, reasoning_paths = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id="example_question_1",
        max_depth=3,
        golden_answer="Arthur's Magazine",
        # MCTS parameters
        num_rollouts=3,
        use_golden_answer=True,
        save_tree=True,
        save_dir="mcts_data",
        verbose=True,
    )
    breakpoint()  # Debugging point to inspect the final answer and solution
