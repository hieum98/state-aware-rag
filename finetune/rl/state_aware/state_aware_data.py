import argparse
import os
import re

import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="RUC-AIBOX/0.8k-data-SimpleDeepSearcher")
    parser.add_argument("--local_dir", default="data/state_aware_rag-rl")

    args = parser.parse_args()

    name = args.data_name.replace("/", "-")
    try:
        dataset = datasets.load_dataset(args.data_name, split="train")
    except Exception as e:
        dataset = datasets.load_from_disk(args.data_name)

    
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
            data = {
                "data_source": name,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ], # We do not use this prompt for training but add it for compatibility with verl
                "ability": "search",
                "reward_model": {"style": "state_aware", "ground_truth": answer_raw},
                # Fields used by StageAwareLoop
                "raw_prompt": question_raw,
                "correct_answer": answer_raw,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
                # Route to our StageAwareLoop via agent_name
                "agent_name": "state_aware",
            }
            return data

        return process_fn
    
    # Randomly select 50 samples for testing
    test_dataset = dataset.shuffle(seed=42).select(range(8))
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True, remove_columns=dataset.column_names)
    dataset = dataset.map(make_map_fn("train"), with_indices=True, remove_columns=dataset.column_names)

    # filter out empty questions/answers
    def filter_empty(example):
        question = example.get("raw_prompt", "").strip()
        answer = example.get("correct_answer", "").strip()
        return question != "" and answer != ""
    
    dataset = dataset.filter(filter_empty)
    test_dataset = test_dataset.filter(filter_empty)
    
    print(f"Train dataset size: {len(dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Make sure the dataset directory exists
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet format
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    dataset.to_parquet(train_path)
    print(f"Dataset saved to {train_path}")
    test_dataset.to_parquet(test_path)
    print(f"Test dataset saved to {test_path}")

    