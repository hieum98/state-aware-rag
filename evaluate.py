from typing import List
import datasets
import os
import argparse
from pprint import pprint
import logging
import warnings
import hydra
from hydra import utils as hy_utils
from omegaconf import OmegaConf
from inference import _compute_results_dir, _maybe_load_agent_cfg

from state_aware_rag.utils.metrics import (
    ExactMatch, 
    Sub_ExactMatch, 
    F1_Score, 
    Retrieval_Recall,  
    LLMJudge
    )


def _set_logging_warning():
    """Reduce logging noise during evaluation by setting WARNING level globally.
    Apply to common noisy libraries and silence warnings.
    Call this both at import-time and inside main() to override Hydra's logging.
    """
    # Silence Python warnings
    warnings.filterwarnings("ignore")

    # Force root to WARNING and reset existing handlers if any were set
    try:
        logging.basicConfig(level=logging.WARNING, force=True)
    except TypeError:
        # Older Python fallback (force not supported)
        logging.getLogger().setLevel(logging.WARNING)

    # Quiet common libraries
    for name in [
        "hydra",
        "omegaconf",
        "urllib3",
        "httpx",
        "httpcore",
        "openai",
        "litellm",
        "datasets",
        "transformers",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Datasets specific verbosity helper (safe if unavailable)
    try:
        from datasets.utils.logging import set_verbosity_warning as _ds_warn
        _ds_warn()
    except Exception:
        pass

    # Transformers specific verbosity helper (safe if unavailable)
    try:
        from transformers.utils.logging import set_verbosity_warning as _tf_warn
        _tf_warn()
    except Exception:
        pass

    os.environ["LOGGING_LEVEL"] = "WARN"


# Apply at import time (may be overridden by Hydra; we'll set again in main)
_set_logging_warning()


def compute_metrics(data: datasets.Dataset, metrics: List[str]=['all'], dataset_name: str = "default", **kwargs):
    if 'all' in metrics:
        metrics = ['sub_em', 'llm_judge']
    results = {}
    metric_config = {
        'dataset_name': dataset_name,
    }
    assert 'golden_answers' in data.features, "Dataset must contain 'golden_answers' field."
    for metric in metrics:
        if metric == 'f1':
            assert 'pred' in data.features, "Dataset must contain 'pred' field."
            f1_metric = F1_Score(config=metric_config)
            f1_score, f1_detail = f1_metric.calculate_metric(data)
            data = data.add_column('f1_detail', f1_detail)
            results['f1'] = f1_score
            print(f"F1 Score: {f1_score}")
        elif metric == 'em':
            assert 'pred' in data.features, "Dataset must contain 'pred' field."
            em_metric = ExactMatch(config=metric_config)
            em_score, em_detail = em_metric.calculate_metric(data)
            data = data.add_column('em_detail', em_detail)
            results['em'] = em_score
            print(f"Exact Match Score: {em_score}")
        elif metric == 'sub_em':
            assert 'pred' in data.features, "Dataset must contain 'pred' field."
            sub_em_metric = Sub_ExactMatch(config=metric_config)
            sub_em_score, sub_em_detail = sub_em_metric.calculate_metric(data)
            data = data.add_column('sub_em_detail', sub_em_detail)
            results['sub_em'] = sub_em_score
            print(f"Sub Exact Match Score: {sub_em_score}")
        elif metric == 'llm_judge':
            assert 'pred' in data.features and 'golden_answers' in data.features, "Dataset must contain 'pred' and 'golden_answers' fields."
            metric_config['judge_setting'] = kwargs.get('llm_setting', None)
            llm_judge_metric = LLMJudge(config=metric_config)
            llm_judge_score, llm_judge_detail = llm_judge_metric.calculate_metric(data)
            data = data.add_column('llm_judge_detail', llm_judge_detail)
            results['llm_judge'] = llm_judge_score
            print(f"LLM Judge Score: {llm_judge_score}")
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
    return results, data

@hydra.main(config_path="configs/infer", config_name="base", version_base=None)
def main(cfg):
    # Re-apply after Hydra config to ensure WARNING-only logs
    _set_logging_warning()
    print("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    data_path = cfg.to_eval_path
    if data_path is None:
        data_path = os.path.join(hy_utils.get_original_cwd(), _compute_results_dir(cfg))
    # Load the dataset
    dataset = datasets.load_from_disk(data_path)

    metric_model = _maybe_load_agent_cfg(cfg.agents.evaluator_metric)
    # Compute metrics
    results, updated_dataset = compute_metrics(
        data=dataset,
        metrics=cfg.data.metrics,
        dataset_name=cfg.data.name,
        llm_setting=metric_model,
    )

    # Print results
    pprint(results)
    # Save the updated dataset
    result_path = data_path + "_with_scores"
    updated_dataset.save_to_disk(data_path + "_with_scores")
    print(f"Updated dataset saved to '{data_path}_with_scores'.")
    # Save results to a file
    path = os.path.join(result_path, 'evaluation_results.json')
    with open(path, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {path}.")


if __name__ == "__main__":
    main()

