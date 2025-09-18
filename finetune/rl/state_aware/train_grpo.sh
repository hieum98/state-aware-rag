#!/bin/bash -l

#SBATCH -o SLURM_Logs/%j/%x.log
#SBATCH -e SLURM_Logs/%j/%x.log
#SBATCH -D ./
#SBATCH -J hm-rl-qwen3-4b

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --partition="ml-p4d-24xlarge-us-west-2d"
#SBATCH --gres=gpu:8


set -xeuo pipefail

# ================= Env Setup =================
export HF_HOME=/fsx/ubuntu/users/hieuman/hf_data
export HF_DATASETS_CACHE=/fsx/ubuntu/users/hieuman/hf_data/datasets
source /fsx/ubuntu/users/hieuman/.bashrc
conda activate rl-agent

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ================= cluster topology =================
export GPUS_PER_NODE=8  # GPUs on this node, default to 8
export NNODES=1
export RAY_NUM_NODES=$NNODES
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Force to stop all ray processes to avoid port conflicts
ray stop --force

# Require at least 2 GPUs
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
if [ "$TOTAL_GPUS" -lt 2 ]; then
  echo "Error: at least 2 GPUs are required, detected $TOTAL_GPUS." >&2
  exit 1
fi

echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}
PROJECT_ROOT_PATH=/fsx/ubuntu/users/hieuman/state-aware-rag

model_path=/fsx/ubuntu/users/hieuman/state-aware-rag/models/qwen3-4b-sft-v1

# Use the default output directory produced by create_dataset.py
train_files=$PROJECT_ROOT_PATH/data/state-aware-rl/train.parquet
test_files=$PROJECT_ROOT_PATH/data/state-aware-rl/test.parquet

# Agent config
agent_loop_config_path=/fsx/ubuntu/users/hieuman/state-aware-rag/finetune/rl/verl/recipe/state_aware/config/state_aware.yaml

# Args
# Read experiment name and learning rate with fallbacks
# Priority: ENV (EXPERIMENT_NAME/ACTOR_LR) > positional args ($1/$2) > interactive prompt (if TTY) > defaults
experiment_name="${EXPERIMENT_NAME:-${1:-}}"
actor_lr="${ACTOR_LR:-${2:-}}"

if [ -t 0 ]; then
  # Interactive shell: prompt for missing values
  if [ -z "${experiment_name}" ]; then
    read -r -p "Enter experiment name [Qwen3-4b-rl]: " experiment_name
    experiment_name=${experiment_name:-Qwen3-4b-rl}
  fi
  if [ -z "${actor_lr}" ]; then
    read -r -p "Enter actor learning rate [1e-6]: " actor_lr
    actor_lr=${actor_lr:-1e-6}
  fi
else
  # Non-interactive (e.g., SLURM): use safe defaults if still unset
  experiment_name=${experiment_name:-Qwen3-4b-rl}
  actor_lr=${actor_lr:-1e-6}
fi
echo "Experiment name: $experiment_name"
echo "Actor LR: $actor_lr"

# =================== wandb ===================
mkdir -p /fsx/ubuntu/users/hieuman/.wandb
export WANDB_CONFIG_DIR=/fsx/ubuntu/users/hieuman/.wandb
export WANDB_API_KEY=""
wandb login
project_name=state-aware-rl
default_local_dir="$PROJECT_ROOT_PATH/checkpoint/$experiment_name"

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=True # Tune
kl_coef=0.001 # Tune
use_kl_loss=True # Tune
kl_loss_coef=0.001 # Tune

clip_ratio_low=0.2
clip_ratio_high=0.28

# max_turns=8
max_prompt_length=2048
max_response_length=4096
# actor_lr is set above (supports ENV/args/prompt/default)

train_batch_size=64
ppo_mini_batch_size=32
n_resp_per_prompt=4
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

infer_tp=2  # vLLM tensor parallel size
train_sp=4  # Ulysses sequence parallel size for actor
offload=False  # Whether to use FSDP offloading

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

cd /fsx/ubuntu/users/hieuman/state-aware-rag/finetune/rl/verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=true \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    critic.model.path="$model_path" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=$offload \
    critic.model.fsdp_config.optimizer_offload=$offload \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.balance_batch=true \
    trainer.val_before_train=false \
    trainer.log_val_generations=50 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=20 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=-1 \
    trainer.total_epochs=5

