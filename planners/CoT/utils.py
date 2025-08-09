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
from planners.CoT.backbone import do_rollout
from planners.CoT.reasoning_node import *

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
        elif node.node_type is NodeType.FINAL_ANSWER:
            final_answer = node_data['node_content'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Final: {final_answer} - Conf: {confidence}"
            node_details = f"{Fore.BLUE}{node_id}{Style.RESET_ALL} {node_details}"
        elif node.node_type is NodeType.SUB_QA_NODE:
            sub_question = node_data['sub_question'].replace('\n', ' ').replace('\r', ' ')
            sub_answer = node_data['sub_answer'].replace('\n', ' ').replace('\r', ' ')
            confidence = node_data['confidence']
            node_details = f"Sub_Q: {sub_question} - Sub_A: {sub_answer} - Conf: {confidence}"
            node_details = f"{Fore.CYAN}{node_id}{Style.RESET_ALL} {node_details}"
 
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
        # CoT parameters
        num_rollouts: int = 1,
        save_tree: bool = False,
        save_dir: str = "cot_trees",
        top_k: int = 5,
        verbose: bool = False,
        memory: Optional[List[str]] = None,
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

    # Start the search from the root node
    nodes = []
    solutions = []
    answers = []
    full_answers = []
    reasoning_paths = []
    for i in range(num_rollouts):
        root_node =  ReasoningNode(
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
            golden_answer=golden_answer,
            user_question=user_question,
            question_id=question_id,
            top_k=top_k,  # Set the top_k for the retriever
            verbose=False,
        )
        if memory is not None:
            root_node.set_memory(memory)
        if root_node.generator.client.random_seed is not None:
            root_node.generator.client.random_seed = root_node.generator.client.random_seed + i
        else:
            root_node.generator.client.random_seed = i
        if root_node.extractor.client.random_seed is not None:
            root_node.extractor.client.random_seed = root_node.extractor.client.random_seed + i
        else:
            root_node.extractor.client.random_seed = i
        if root_node.evaluator.client.random_seed is not None:
            root_node.evaluator.client.random_seed = root_node.evaluator.client.random_seed + i
        else:
            root_node.evaluator.client.random_seed = i
            
        cot = do_rollout(node=root_node)
        answer_node = cot[-1] if cot[-1].is_valid_leaf() else None
        if verbose:
            print_tree_from_root(root_node)
        if verbose:
            print("**" * 20)
            print(f"Rollout {i+1}/{num_rollouts}")
            if answer_node is not None:
                print("Best solution found:")
                pprint.pprint(answer_node, indent=4, width=120)

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
    
    if save_tree:
        # check if the save directory does not exist, create it
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/cot_tree_{question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = node.get_node()
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
        'url': 'http://ip-10-4-225-181:30000/v1', 
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
        cache_dir="cot_cache/generator_cache",
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
        cache_dir="cot_cache/evaluator_cache",
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
        cache_dir="cot_cache/extractor_cache",
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
        # CoT parameters
        num_rollouts=2,
        save_tree=True,
        save_dir="cot_data",
        verbose=True,
    )
    breakpoint()  # Debugging point to inspect the final answer and solution
