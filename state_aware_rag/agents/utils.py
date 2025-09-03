import re
from typing import List, Union
from state_aware_rag.agents.prompts import extract


def extract_info_from_text(text, keys: List[str], value_type: List[str]=None):
    if value_type is None:
        value_type = ['str'] * len(keys)
    assert len(keys) == len(value_type), "keys and value_type must have the same length"
    extracted_info = {}
    for key, vtype in zip(keys, value_type):
        if vtype in ['str', 'Literal']:
            # When the value is a string, we can use regex to extract the value in the format of "key": "value"
            pattern = rf'"{key}":\s*"([^"]*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted_info[key] = match.group(1)
            else:
                extracted_info[key] = "" # default to empty string if not found
        elif vtype == 'bool':
            # When the value is a boolean, we can use regex to extract the value in the format of "key": true/false
            pattern = rf'"{key}":\s*(true|false)'
            match = re.search(pattern, text)
            if match:
                extracted_info[key] = match.group(1) == 'true'
            else:
                extracted_info[key] = False # default to False if not found
        elif vtype in ['int', 'float']:
            # When the value is a number, we can use regex to extract the value in the format of "key": number
            pattern = rf'"{key}":\s*([-+]?\d*\.?\d+)'
            match = re.search(pattern, text)
            if match:
                if vtype == 'int':
                    extracted_info[key] = int(match.group(1))
                else:
                    extracted_info[key] = float(match.group(1))
            else:
                extracted_info[key] = 0 # default to 0 if not found
        elif vtype in ['List', 'list']:
            # When the value is a list, we can use regex to extract the value in the format of "key": [value1, value2, ...]
            pattern = rf'"{key}":\s*\[(.*?)\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Split the values by comma and strip whitespace
                info = match.group(1)
                info =  info.strip()
                # split by ",\n"
                list_items = re.split(r',\s*\n', info)
                # strip each item
                list_items = [item.strip().strip('"') for item in list_items if item.strip()]
                extracted_info[key] = list_items
            else:
                # match the case where it does not have ] in the end
                pattern = rf'"{key}":\s*\[(.*)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    # Split the values by comma and strip whitespace
                    # values = [v.strip().strip('"') for v in match.group(1).split(',')]
                    info = match.group(1)
                    info =  info.strip()
                    # split by ",\n"
                    list_items = re.split(r',\s*\n', info)
                    # strip each item
                    list_items = [item.strip().strip('"') for item in list_items if item.strip()]
                    extracted_info[key] = list_items
                else:
                    extracted_info[key] = []
        else:
            raise ValueError(f"Unsupported value type: {vtype}. Supported types are: str, bool, int, float, list.")
    return extracted_info


def convert_confidence_to_score(confidence: str) -> float:
    confidence = confidence.lower() if isinstance(confidence, str) else 'low'
    if confidence == 'high':
        return 1.0
    elif confidence == 'medium':
        return 0.5
    else:
        return 0.1


def convert_score_to_confidence(score: float) -> str:
    if not isinstance(score, (int, float)):
        return 'low'
    if score >= 0.75:
        return 'high'
    elif score >= 0.25:
        return 'medium'
    else:
        return 'low'
    

def format_reasoning_trace(trace: List[str]) -> str:
    formatted_trace = ""
    try:
        i = 1
        for step in trace:
            if step.strip():
                formatted_trace += f"Step {i}: {step.strip()}\n"
                i += 1
        return formatted_trace.strip()
    except Exception as e:
        print(f"Error in formatting reasoning trace: {e}")
        return ""


def format_memory(memory: List[str]) -> str:
    formatted_memory = ""
    try:
        if isinstance(memory, str):
            return memory.strip()
        for mem in memory:
            if mem and mem.strip():
                formatted_memory += f"- {mem.strip()}\n"
        return formatted_memory.strip()
    except Exception as e:
        print(f"Error in formatting memory: {e}")
        return ""


def format_context(memory: str = None, reasoning_trace: str = None, explored_data: str = None):
    context = ""
    reasoning_trace = reasoning_trace.strip()
    if memory:
        context += f"\t**Memory knowledge**\n{memory}\n----------\n"
    if explored_data:
        context += f"\t**Information from external KB**\n{explored_data}\n----------\n"
    if reasoning_trace:
        context += f"\t**Reasoning trace**\n{reasoning_trace}"
    return context


def format_reflection_context(current_memory: str = None, intermediate_conclusions: str = None, explored_data: str = None):
    context = ""
    current_memory = current_memory.strip() if current_memory else ""
    intermediate_conclusions = intermediate_conclusions.strip() if intermediate_conclusions else ""
    explored_data = explored_data.strip() if explored_data else ""
    if current_memory:
        context += f"\t**Current memory knowledge**\n{current_memory}\n----------\n"
    if intermediate_conclusions:
        context += f"\t**Intermediate conclusions**\n{intermediate_conclusions}"
    if explored_data:
        context += f"\t**Information from external KB**\n{explored_data}\n----------\n"
    if context.strip() == "":
        print("[WARNING] Reflection context is empty!")
    return context


def format_extractor_messages(question: Union[str, List[str]], context: Union[str, List[str]]):
    if isinstance(question, str):
        question = [question]
    if isinstance(context, str):
        context = [context] 
    assert len(question) == len(context), "Number of questions and contexts must be the same"
    batch = [
        extract.EXTRACT_PROMPT.format(
            question=q,
            document=d,
            examples="No examples provided."
            ) for q, d in zip(question, context)
    ]
    batch = [[{'role': 'user', 'content': x}] for x in batch]
    if len(batch) == 1:
        return batch[0]
    return batch


if __name__ == "__main__":
    subquestion = "What is the capital of France?\nParis is the capital of France.\nFrance is a country in Europe."
    text = '{\n\n"answerable_main_question": false,\n"subquestion": ' + f'"{subquestion}"' +\
    ',\n"evidence": [\n    "Paris is the capital of France.",\n    "France is a country in Europe."\n  ]\n\n}'
    print(text)
    info = extract_info_from_text(text, 
                                  ['answerable_main_question', 'subquestion', 'evidence'], 
                                  ['bool', 'str', 'list'])
    print(info)
