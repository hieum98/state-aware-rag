from typing import List
import datasets
from utils.metrics import (
    ExactMatch, 
    Sub_ExactMatch, 
    F1_Score, 
    Retrieval_Recall,  
    LLMJudge
    )


def compute_metrics(data: datasets.Dataset, metrics: List[str]=['all'], dataset_name: str = "default", **kwargs):
    if 'all' in metrics:
        metrics = ['f1', 'em', 'sub_em', 'retrieval_recall', 'llm_judge']
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
        elif metric == 'em':
            assert 'pred' in data.features, "Dataset must contain 'pred' field."
            em_metric = ExactMatch(config=metric_config)
            em_score, em_detail = em_metric.calculate_metric(data)
            data = data.add_column('em_detail', em_detail)
            results['em'] = em_score
        elif metric == 'sub_em':
            assert 'pred' in data.features, "Dataset must contain 'pred' field."
            sub_em_metric = Sub_ExactMatch(config=metric_config)
            sub_em_score, sub_em_detail = sub_em_metric.calculate_metric(data)
            data = data.add_column('sub_em_detail', sub_em_detail)
            results['sub_em'] = sub_em_score
        elif metric == 'retrieval_recall':
            assert 'retrieval_result' in data.features, "Dataset must contain 'retrieval_result' field."
            retrieval_recall_topk = kwargs.get('retrieval_recall_topk', 5)
            metric_config['metric_setting'] = {
                'retrieval_recall_topk': retrieval_recall_topk
            }
            retrieval_recall_metric = Retrieval_Recall(config=metric_config)
            retrieval_recall_score, retrieval_recall_detail = retrieval_recall_metric.calculate_metric(data)
            data = data.add_column(f"retrieval_recall_detail_{retrieval_recall_topk}", retrieval_recall_detail)
            results['retrieval_recall'] = retrieval_recall_score
        elif metric == 'llm_judge':
            assert 'pred' in data.features and 'golden_answers' in data.features, "Dataset must contain 'pred' and 'golden_answers' fields."
            default_llm_setting = {
                'model_name': 'openai/judge', 
                'url': 'http://0.0.0.0:30000/v1', 
                'api_key': 'your_api_key_here',  # Replace with your actual API key
                'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
                'concurrency': 64,
            }
            default_generate_setting = {
                'temperature': 0.1,  
                'n': 1, 
                'top_p': 0.9,
                'max_tokens': 4096,  
                # Want more varied responses (alongside high temperature) set top_k to 50 - 100 
                # For greedy decoding set it to 1
                'top_k': 20,
                'tensor_parallel_size': 1,
                'reasoning_effort': 'medium',  # Set to 'high'/'medium'/'low' for using thinking capabilities
            }
            metric_config['judge_setting'] = {
                'llm_setting' : kwargs.get('llm_setting', default_llm_setting),
                'generate_setting': kwargs.get('generate_setting', default_generate_setting),
            }
            llm_judge_metric = LLMJudge(config=metric_config)
            llm_judge_score, llm_judge_detail = llm_judge_metric.calculate_metric(data)
            data = data.add_column('llm_judge_detail', llm_judge_detail)
            results['llm_judge'] = llm_judge_score
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
    return results, data


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Compute evaluation metrics for a dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--metrics", type=str, nargs='+', default=['all'], help="List of metrics to compute: 'f1', 'em', 'sub_em', 'retrieval_recall', 'llm_judge', or 'all' for all metrics.")
    parser.add_argument("--dataset_name", type=str, default="default", help="Name of the dataset.")
    parser.add_argument("--retrieval_recall_topk", type=int, default=5, help="Top-k for retrieval recall metric.")
    parser.add_argument("--llm_judge_model_name", type=str, default='Qwen/Qwen3-8B', help="Model name for LLM judge metric.")
    parser.add_argument("--api_url", type=str, default=None, help="API URL for LLM judge metric.")

    args = parser.parse_args()

    # Load the dataset
    dataset = datasets.load_from_disk(args.dataset_path)

    llm_setting = {
            'model_name': f'openai/{args.llm_judge_model_name}', 
            'url': args.api_url if args.api_url else 'http://0.0.0.0:30000/v1', 
            'api_key': 'your_api_key_here',  # Replace with your actual API key
            'client_type': 'openai',  # Use 'litellm' for LiteLLMClient or 'openai' for OpenAIClient
            'concurrency': 64,
        }
    # Compute metrics
    results, updated_dataset = compute_metrics(
        data=dataset,
        metrics=args.metrics,
        dataset_name=args.dataset_name,
        retrieval_recall_topk=args.retrieval_recall_topk,
        llm_setting=llm_setting,
    )

    # Print results
    pprint(results)
    # Save results to a file
    with open('evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=4)
    print("Evaluation results saved to 'evaluation_results.json'.")

    # Save the updated dataset
    updated_dataset.save_to_disk(args.dataset_path + "_with_scores")
    print(f"Updated dataset saved to '{args.dataset_path}_with_scores'.")

