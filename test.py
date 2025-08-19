from collections import Counter
import os
import datasets
import tqdm
from agents.prompts.extract import EXTRACT_PROMPT, ExtractOutput
from preprocess.utils import process_extractor_data_from_cache
from preprocess.minhash import dedup

def create_full_content(example):
    content = example['content']
    if isinstance(content, list):
        content = "\n".join(content)
    question = example['question']
    return {
        'full_content': f"{question}\n{content}",
    }

def convert_to_conversational(example):
    question = example['question']
    content = example['content']
    content = "\n".join(content) if isinstance(content, list) else content
    input_content = EXTRACT_PROMPT.format(
        question=question,
        document=content,
        examples="No examples provided."
    )
    output = example['output']
    # Convert output dict to ExtractOutput
    output = ExtractOutput.model_validate(output)
    output = output.model_dump_json(indent=2)
    output_reasoning = example['reasoning']
    output_content = f"<think>{output_reasoning}</think>{output}" # Hardcoded for now, which only works for Qwen3
    messages = [
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': output_content}
    ]
    return {'messages': messages}

datapath = "/fsx/ubuntu/users/hieuman/state-aware-rag/mcts_cache/train-small-cache/extractor/bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
files = [f for f in os.listdir(datapath) if f.endswith('.json')]
# files = files[:100]
all_data = []
for file in tqdm.tqdm(files, desc="Processing files"):
    full_path = os.path.join(datapath, file)
    file_data = process_extractor_data_from_cache(full_path)
    all_data.extend(file_data)

dataset = datasets.Dataset.from_list(all_data)
dataset = dataset.map(
    lambda x: {'content': [c.strip() for c in x['content'] if isinstance(c, str) and c.strip()]},
    num_proc=32
)
dataset = dataset.filter(
    lambda x: len(x['content']) > 0,
    num_proc=32
)
# Deduplicate the dataset
dataset = dataset.map(
    create_full_content,
    num_proc=64
)
dataset = dedup(
    column='full_content',
    data_path=None,
    num_proc=32,
    ds=dataset,
    batch_size=1000,
    idx_column=None, 
    ngram=5,
    min_length=5,
    num_perm=250,
    threshold=0.7,
)

dataset.save_to_disk('/fsx/ubuntu/users/hieuman/state-aware-rag/data/extractor_dataset-v2')

# Print out the statistics of the dataset
print(f"Number of examples in the dataset: {len(dataset)}")
# Number of each type of example
types = dataset['type']
type_counts = Counter(types)
print("Type counts:")
for type_name, count in type_counts.items():
    print(f"{type_name}: {count}")

### Convert to conversational format
dataset = datasets.load_from_disk('/fsx/ubuntu/users/hieuman/state-aware-rag/data/extractor_dataset-v2')
dataset = dataset.map(
    convert_to_conversational,
    num_proc=32,
    remove_columns=dataset.column_names
)
# Save the final dataset
dataset.save_to_disk('/fsx/ubuntu/users/hieuman/state-aware-rag/data/SFT-data-v2')