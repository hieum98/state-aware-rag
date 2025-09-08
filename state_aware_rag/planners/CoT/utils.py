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
from state_aware_rag.planners.CoT.backbone import do_rollout
from state_aware_rag.planners.reasoning_node import *
from state_aware_rag.planners.MCTS.utils import print_tree_from_root, get_tree_from_file


def search_with_cot(
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
        # CoT parameters
        num_rollouts: int = 1,
        save_tree: bool = False,
        save_dir: str = "cot_trees",
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

    # Start the search from the root node
    nodes = []
    all_answer_nodes = []
    solutions = []
    answers = []
    full_answers = []
    reasoning_paths = []
    for i in range(num_rollouts):
        random_seed = generator.generation_config.get('random_seed', 0)
        generator.generation_config['random_seed'] = random_seed + i if random_seed is not None else i
        generator.use_cache = False  # Disable cache during multiple rollouts to ensure diversity
        extractor.generation_config['random_seed'] = random_seed + i if random_seed is not None else i
        extractor.use_cache = False
        evaluator.generation_config['random_seed'] = random_seed + i if random_seed is not None else i
        evaluator.use_cache = False
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
            golden_answer=golden_answer,
            user_question=user_question,
            question_id=question_id,
            top_k=top_k,  # Set the top_k for the retriever
            is_cot=True,
        )
            
        cot = do_rollout(node=root_node)
        answer_node = root_node.leaves
        answer_node = [node for node in answer_node if node.node_type != NodeType.USER_QUESTION]
        if len(answer_node) == 0:
            print(f"No answer node found in rollout {i}.")
            continue
        if verbose:
            print_tree_from_root(root_node)

        for _, _, node in RenderTree(root_node):
            node: ReasoningNode
            node.set_rollout_id(i)
            nodes.append(node)
            if node in answer_node:
                reasoning_path = node.get_path()
                reasoning_path, _ = node.get_reasoning_trace(path=reasoning_path)
                if node.node_type is NodeType.FINAL_ANSWER:
                    answer = node.state['node_content']
                    detailed_answer = node.state['detailed_answer']
                else:
                    answer = reasoning_path
                    detailed_answer = reasoning_path
                if answer and answer.strip():
                    answers.append(answer)
                if detailed_answer and detailed_answer.strip():
                    full_answers.append(detailed_answer)            
                if reasoning_path and reasoning_path.strip():
                    reasoning_paths.append(reasoning_path)
                else:
                    # If the reasoning path is None, we only append the node content of the final answer
                    reasoning_paths.append(detailed_answer)
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
    
    if save_tree:
        # check if the save directory does not exist, create it
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/cot_tree_{question_id}.jsonl", 'w') as f:
            for node in nodes:
                node_content = node.get_node()
                f.write(json.dumps(node_content) + "\n")
    return final_answer, final_reasoning, reasoning_paths


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
        # CoT parameters
        num_rollouts: int = 1,
        save_tree: bool = False,
        save_dir: str = "cot_trees",
        top_k: int = 5,
        verbose: bool = False,
        **kwargs
):
    save_path = f"{save_dir}/cot_tree_{question_id}.jsonl"
    if save_tree and os.path.exists(save_path):
        root_node = get_tree_from_file(
            save_path=save_path,
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            user_question=user_question,
            question_id=question_id,
            is_cot=True,
            max_depth=max_depth,
            golden_answer=golden_answer,
            use_golden_answer=True,
            top_k=top_k,
            verbose=verbose
        )
        nodes = []
        answers = []
        full_answers = []
        leaf_nodes = root_node.leaves
        leaf_nodes = [node for node in leaf_nodes if node.node_type != NodeType.USER_QUESTION]
        if len(leaf_nodes) == 0:
            # If no leaf nodes found, fallback to normal search
            final_answer, final_reasoning, reasoning_paths = search_with_cot(
                generator=generator,
                evaluator=evaluator,
                extractor=extractor,
                retriever=retriever,
                user_question=user_question,
                question_id=question_id,
                max_depth=max_depth,
                golden_answer=golden_answer,
                num_rollouts=num_rollouts,
                save_tree=save_tree,
                save_dir=save_dir,
                top_k=top_k,
                verbose=verbose,
            )

        for _, _, node in RenderTree(root_node):
            node: ReasoningNode
            nodes.append(node)
            if node in leaf_nodes:
                reasoning_path, _ = node.get_reasoning_trace()
                if node.node_type is NodeType.FINAL_ANSWER:
                    answer = node.state['node_content']
                    detailed_answer = node.state['detailed_answer']
                else:
                    answer = reasoning_path
                    detailed_answer = reasoning_path
                answers.append(answer)
                full_answers.append(detailed_answer)
            
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
    else:
        final_answer, final_reasoning, reasoning_paths = search_with_cot(
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            user_question=user_question,
            question_id=question_id,
            max_depth=max_depth,
            golden_answer=golden_answer,
            num_rollouts=num_rollouts,
            save_tree=save_tree,
            save_dir=save_dir,
            top_k=top_k,
            verbose=verbose,
        )
    return final_answer, final_reasoning


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


