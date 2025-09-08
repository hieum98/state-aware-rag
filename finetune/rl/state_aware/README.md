## State-Aware RL with Verl PPO

This recipe trains a state-aware multi-hop agent with PPO using Verl. It plugs a custom agent loop, `StageAwareLoop`, into Verl’s rollout stack so each input can yield multiple training samples (extract, reflect, update-memory, etc.), with outcome- and path-aware rewards.

### What it does

- Iteratively: generate a subquestion → retrieve docs → extract structured info → answer/reflect → update memory.
- Produces multiple AgentLoopOutput items per input for PPO (flattened/batched by Verl).
- Rewards combine final-answer accuracy, step-level judgments, and structure validity of extractions.

## Requirements

- Python environment with this repo installed and Verl available from `finetune/rl/verl`.
- GPUs (multi-GPU recommended; the provided script targets 8x A100/80GB nodes with vLLM rollout).
- Access to your LLM weights (the script defaults to `models/qwen3-4b-sft-v1`).
- Retriever/Generator/Evaluator configs under `configs/*.yaml` filled out for your endpoints/keys.

## Data preparation

Option A — from a Hugging Face dataset

- Use the helper to build Parquet files consumed by Verl:

```bash
python -m finetune.rl.verl.recipe.state_aware.state_aware_data \
	--data_name RUC-AIBOX/0.8k-data-SimpleDeepSearcher \
	--local_dir data/state-aware-rl
```

This produces `data/state-aware-rl/train.parquet` and `data/state-aware-rl/test.parquet`.

Option B — bring your own Parquet

- Each row should provide at minimum:
	- raw_prompt: string (the question). Alternatively, a chat-style `prompt` is accepted; the loop will infer the last user message as the question.
	- correct_answer: string (ground-truth answer).
	- agent_name: "state_aware" (routes samples to this loop).
- Recommended for Verl compatibility:
	- prompt: list[chat messages] (not used for training, but kept for schema compatibility).
	- reward_model: {"style": "state_aware", "ground_truth": "ANSWER"}.

## Configure the agent loop

Edit `finetune/rl/verl/recipe/state_aware/config/state_aware.yaml`:

- retrieval_config_path: points to `configs/retriever.yaml`
- generator_config_path: points to `configs/generator.yaml`
- evaluator_config_path: points to `configs/evaluator.yaml`
- max_iterations: number of reasoning/retrieval rounds per sample

These YAMLs hold model URLs, API keys, and per-agent settings. Make sure they’re valid for your environment.

## Train

The provided SLURM script runs PPO with vLLM async rollout and the state-aware loop.

```bash
bash finetune/rl/verl/recipe/state_aware/train.sh
```

Notes

- Script expects multi-GPU (>=2). It configures TP for inference and sequence parallel for training.
- Defaults read Parquet from `data/state-aware-rl/{train,test}.parquet` and model from `models/qwen3-4b-sft-v1`.
- You can override two common knobs without editing the script:
	- EXPERIMENT_NAME: run name used for logging/checkpoints.
	- ACTOR_LR: actor learning rate.

Example

```bash
EXPERIMENT_NAME=Qwen3-4b-rl ACTOR_LR=1e-6 bash finetune/rl/verl/recipe/state_aware/train.sh
```

Outputs

- Checkpoints and logs under `checkpoint/<EXPERIMENT_NAME>/`.
- Optional Weights & Biases logging if `WANDB_API_KEY` is set in your environment.

## How rewards are computed (overview)

- Outcome-aware: judge the final answer vs `correct_answer`.
- Path-aware: judge step answers and the reasoning path.
- Structure bonus: valid extractor outputs get full reward; invalid ones receive a down-weighted reward.

## Troubleshooting

- No retrieval results: verify `configs/retriever.yaml` endpoint/keys and the query template.
- Tokenizer/model mismatch: for custom models (e.g., Qwen), ensure `trust_remote_code=True` is honored in your environment.
- Overlong samples: the script enforces `max_prompt_length` and `max_response_length` and filters/truncates as configured.
- Batching shape issues: the loop pads its per-input outputs to a multiple of 8 for stable batching; ensure your batch sizes align with GPU memory.

## Quick test

Run the unit test for the loop (requires test deps and a minimal config):

```bash
pytest -q finetune/rl/verl/recipe/state_aware/test_state_aware_loop.py
```

## File map

- stage_aware_loop.py: the agent loop plugged into Verl (registered as `state_aware`).
- config/state_aware.yaml: loop registration and pointers to agent configs.
- state_aware_data.py: utility to build Parquet files from HF datasets.
- train.sh: SLURM-compatible launcher configuring PPO, rollout, and logging.
