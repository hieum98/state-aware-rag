import re
from typing import List


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
                extracted_info[key] = None
        elif vtype == 'bool':
            # When the value is a boolean, we can use regex to extract the value in the format of "key": true/false
            pattern = rf'"{key}":\s*(true|false)'
            match = re.search(pattern, text)
            if match:
                extracted_info[key] = match.group(1) == 'true'
            else:
                extracted_info[key] = None
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
                extracted_info[key] = None
        elif vtype in ['List', 'list']:
            # When the value is a list, we can use regex to extract the value in the format of "key": [value1, value2, ...]
            pattern = rf'"{key}":\s*\[(.*?)\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Split the values by comma and strip whitespace
                extracted_info[key] = match.group(1)
            else:
                # match the case where it does not have ] in the end
                pattern = rf'"{key}":\s*\[(.*)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    # Split the values by comma and strip whitespace
                    # values = [v.strip().strip('"') for v in match.group(1).split(',')]
                    extracted_info[key] = match.group(1)
                else:
                    extracted_info[key] = []
        else:
            raise ValueError(f"Unsupported value type: {vtype}. Supported types are: str, bool, int, float, list.")
    return extracted_info


if __name__ == "__main__":
    subquestion = "What is the capital of France?\nParis is the capital of France.\nFrance is a country in Europe."
    text = '{\n\n"answerable_main_question": false,\n"subquestion": ' + f'"{subquestion}"' +\
    ',\n"evidence": [\n    "Paris is the capital of France.",\n    "France is a country in Europe."\n  ]\n\n}'
    print(text)
    info = extract_info_from_text(text, 
                                  ['answerable_main_question', 'subquestion', 'evidence'], 
                                  ['bool', 'str', 'list'])
    print(info)
