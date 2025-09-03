import os
from typing import Optional

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

try:
    # Preferred path in this repo
    from verl.utils.dataset.rl_dataset import RLHFDataset as BaseRLHFDataset  # type: ignore
except Exception:  # pragma: no cover - compatibility import
    from verl.utils.dataset.rlhf_dataset import RLHFDataset as BaseRLHFDataset  # type: ignore


class StateAwareDataset(BaseRLHFDataset):
    """
    Extend RLHFDataset to provide fields for StageAwareLoop:
    - question: str
    - correct_answer: str
    - agent_name: "state_aware"

    Expected parquet schema should include at least:
    - prompt: chat messages (list of {role, content}) or a string prompt
    - question: the user question (string)
    - correct_answer: ground truth answer (string)
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

    def __getitem__(self, idx):
        row = super().__getitem__(idx)

        # Load original row to get extra fields "question" and "correct_answer" if present
        # HuggingFace dataset is stored in self.dataframe
        raw_row = self.dataframe[idx]
        question = raw_row.get("question")
        if question is None and isinstance(row.get("raw_prompt"), list):
            # Best-effort: use last user content in raw_prompt
            for msg in reversed(row["raw_prompt"]):
                if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    question = msg["content"].strip()
                    break
        row["question"] = question

        correct_answer = raw_row.get("correct_answer")
        row["correct_answer"] = correct_answer

        # Tell AgentLoopWorker to instantiate StageAwareLoop
        row["agent_name"] = "state_aware"
        return row
