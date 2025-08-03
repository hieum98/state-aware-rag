import os
import re
import json
from datasets import Dataset
import tqdm


def parse_full_reasoning(full_reasoning):
    reasoning = []
    for node in full_reasoning:
        if node.get("node_type") not in ['REPHASE_QUESTION', 'FINAL_ANSWER', 'USER_QUESTION']:
            reasoning.append(node.get("node_content", ""))
    reasoning = [f"Step {i+1}: {step}\n" for i, step in enumerate(reasoning)]
    if full_reasoning[-1].get("node_type") == "FINAL_ANSWER":
        final_answer = f"{full_reasoning[-1].get('detailed_answer', '')}\nFinal answer: {full_reasoning[-1].get('node_content', '')}"
        reasoning.append(final_answer)
    return '\n'.join(reasoning)


def process_tree_logs(tree_logs_path):
    all_answers = []
    user_question = ""
    with open(tree_logs_path, 'r') as f:
        for line in f:
            try:
                data_dict = json.loads(line)
                if data_dict.get("node_type") == "FINAL_ANSWER":
                    if user_question == "":
                        user_question = data_dict.get("user_question", "")
                    answer = data_dict.get("node_content", "")
                    detailed_answer = data_dict.get("detailed_answer", "")
                    reasoning_path = data_dict.get("reasoning_path", "")
                    full_reasoning = data_dict.get("full_reasoning_path", [])
                    reasoning = parse_full_reasoning(full_reasoning)
                    # TODO: Add rollout_id if available
                    all_answers.append({
                        'answer': answer,
                        'detailed_answer': detailed_answer,
                        'reasoning_path': reasoning_path,
                        'full_reasoning': reasoning
                    })
            except:
                print(f"Error decoding JSON in file {tree_logs_path}: {line}")
                continue
    return user_question, all_answers


def generate_final_answer(example, evaluator):
    question = example['question']
    answers = example['answers']
    final_answer = None
    final_reasoning = None
    if len(answers) > 0:
        reasoning_paths = [(answer['full_reasoning'], answer['rollout_id']) for answer in answers]
        detailed_answers = [answer['detailed_answer'] for answer in answers if answer['detailed_answer']]
        try:
            final_answer, final_reasoning = evaluator.synthesize_final_answer(question=question, reasoning_paths=reasoning_paths)
        except:
            print(f"Error synthesizing final answer for question: {question}")
            print(f"Reasoning paths: {reasoning_paths}")
            print(f"Number of candidates: {len(reasoning_paths)}")
            final_answer, final_reasoning = evaluator.synthesize_final_answer(question=question, reasoning_paths=detailed_answers)
            
    return {
        'id': example['id'],
        'question': question,
        'pred': final_answer,
        'detailed_answer': final_reasoning,
        'all_candidates_answers': reasoning_paths
    }


def get_the_golden_answer(example, origin_dataset):
    # Find the item in the original dataset with the same id
    item = origin_dataset.filter(lambda x: x['id'] == example['id'])
    assert len(item) >= 1, f"Item with id {example['id']} not found in the original dataset"
    return {
        'golden_answers': item[0]['golden_answers'],
    }

