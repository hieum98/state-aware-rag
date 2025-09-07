#!/usr/bin/env bash
#SBATCH -e SLURM_eval/%j/%x_eval.log
#SBATCH -o SLURM_eval/%j/%x_eval.log
#SBATCH -D ./
#SBATCH -J hm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G

set -euo pipefail
source ~/users/hieuman/.bashrc
conda activate rl-agent
cd ~/users/hieuman/state-aware-rag

# -------- Args --------
print_usage() {
	cat <<EOF
Usage: $0 \
	--mode <mcts|cot> \
	--data-name <dataset> \
	[--gen-model <MODEL>] [--gen-url <URL>] \
	[--eval-model <MODEL>] [--eval-url <URL>] \
	[--ext-model <MODEL>]  [--ext-url <URL>]  \
	[--ret-url <URL>]      

Notes:
- Unspecified options fall back to values in configs/infer/base.yaml
- retriever takes only --ret-url (no model)
EOF
}

MODE=""
DATA_NAME=""
GEN_MODEL=""; GEN_URL=""
EVAL_MODEL=""; EVAL_URL=""
EXT_MODEL=""; EXT_URL=""
RET_URL=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--mode) MODE="$2"; shift 2 ;;
		--data-name) DATA_NAME="$2"; shift 2 ;;
		--gen-model) GEN_MODEL="$2"; shift 2 ;;
		--gen-url) GEN_URL="$2"; shift 2 ;;
		--eval-model) EVAL_MODEL="$2"; shift 2 ;;
		--eval-url) EVAL_URL="$2"; shift 2 ;;
		--ext-model) EXT_MODEL="$2"; shift 2 ;;
		--ext-url) EXT_URL="$2"; shift 2 ;;
		--ret-url) RET_URL="$2"; shift 2 ;;
		-h|--help)
			print_usage; exit 0 ;;
		*)
			echo "Unknown argument: $1" >&2
			print_usage; exit 1 ;;
	esac
done

# Launch directly with inline Hydra overrides (include only if set)
set -x
python -m inference \
	${MODE:+mode="$MODE"} \
	${DATA_NAME:+data.name="$DATA_NAME"} \
	${GEN_MODEL:+agents\.generator\.client_kwargs\.model_name="$GEN_MODEL"} \
	${GEN_URL:+agents\.generator\.client_kwargs\.url="$GEN_URL"} \
	${EVAL_MODEL:+agents\.evaluator\.client_kwargs\.model_name="$EVAL_MODEL"} \
	${EVAL_URL:+agents\.evaluator\.client_kwargs\.url="$EVAL_URL"} \
	${EXT_MODEL:+agents\.extractor\.client_kwargs\.model_name="$EXT_MODEL"} \
	${EXT_URL:+agents\.extractor\.client_kwargs\.url="$EXT_URL"} \
	${RET_URL:+agents\.retriever\.online_retrieval_config\.url="$RET_URL"}
set +x





