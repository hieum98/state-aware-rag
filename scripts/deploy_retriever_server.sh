#!/bin/bash -l

# SLURM job to deploy a retrieval stack: encoder (embeddings) server + retriever API
#
# Behavior (opinionated):
# - Always launches Qwen/Qwen3-Embedding-4B via vLLM.
# - Always launches retriever with config: servers/configs/retriever-Qwen3-4B-wiki-23.yaml
# - No required inputs; sensible defaults. You may optionally override GPU list or ports via env vars.
#
# Optional env vars (with defaults):
#   - ENCODER_VISIBLE_GPUS=0,1                # GPU IDs to use for vLLM
#   - ENCODER_PORT=8000                       # vLLM HTTP port (config expects default 8000)
#   - ENCODER_API_KEY=EMPTY                   # OpenAI-style API key for vLLM
#   - RETRIEVER_PORT=5000                     # FastAPI retriever port
#
# Outputs:
#   - SLURM_Logs/<job_id>/vllm_encoder.log         # vLLM server logs
#   - SLURM_Logs/<job_id>/retriever_server.log     # retriever server logs

#SBATCH -o SLURM_Logs/%j/%x.log
#SBATCH -e SLURM_Logs/%j/%x.log
#SBATCH -D ./
#SBATCH -J hm-retrieval

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ml-p4d-24xlarge-us-west-2d
#SBATCH --gres=gpu:2

set -euo pipefail

echo "[INFO] SLURM job ID: $SLURM_JOB_ID"
echo "[INFO] Node list: $SLURM_NODELIST"

HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
LOG_DIR="SLURM_Logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# -------- Parameters --------
ENCODER_MODEL="Qwen/Qwen3-Embedding-4B"
ENCODER_VISIBLE_GPUS=${ENCODER_VISIBLE_GPUS:-"0,1"}
ENCODER_PORT=${ENCODER_PORT:-8000}

RETRIEVER_PORT=${RETRIEVER_PORT:-5000}

echo "[INFO] Encoder model:        ${ENCODER_MODEL}"
echo "[INFO] Encoder GPUs:         ${ENCODER_VISIBLE_GPUS}"
echo "[INFO] Encoder port:         ${ENCODER_PORT}"
echo "[INFO] Retriever port:       ${RETRIEVER_PORT}"

# Compute DP from number of visible GPUs; keep TP=1 for embedding models
GPU_COUNT=$(python - <<'PY'
import os
g=os.environ.get('ENCODER_VISIBLE_GPUS','')
print(len([x for x in g.split(',') if x.strip()!='']))
PY
)
if [[ "$GPU_COUNT" -lt 1 ]]; then GPU_COUNT=1; fi
TP=1
DP=$GPU_COUNT

cleanup() {
  echo "[INFO] Cleaning up background processes..."
  pkill -9 -f "vllm serve" || true
  pkill -9 -f "state_aware_rag.servers.retriever" || true
  pkill -9 uvicorn || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "[INFO] Activating conda environment for servers"
source /fsx/ubuntu/users/hieuman/miniconda/bin/activate server || true
source ~/.bashrc || true

### Start encoder (vLLM) server ###
echo "[INFO] Launching embeddings server on ${HEAD_NODE} GPUs ${ENCODER_VISIBLE_GPUS}"
CUDA_VISIBLE_DEVICES=${ENCODER_VISIBLE_GPUS} \
vllm serve "${ENCODER_MODEL}" \
  --host 0.0.0.0 \
  --port ${ENCODER_PORT} \
  -tp ${TP} -dp ${DP} \
  > "${LOG_DIR}/vllm_encoder.log" 2>&1 &

echo "[INFO] Waiting for embeddings server to become ready..."
ENCODER_TIMEOUT=900
for ((i=0; i<ENCODER_TIMEOUT; i++)); do
  if grep -q "Application startup complete" "${LOG_DIR}/vllm_encoder.log"; then
    echo "[INFO] Embeddings server is up."
    break
  fi
  sleep 1
done
if [[ $i -eq $ENCODER_TIMEOUT ]]; then
  echo "[ERROR] Embeddings server failed to start within ${ENCODER_TIMEOUT}s"
  exit 1
fi

ENCODER_BASE_URL="http://${HEAD_NODE}:${ENCODER_PORT}/v1"
echo "[INFO] Encoder base URL: ${ENCODER_BASE_URL}"

### Start retriever server ###
echo "[INFO] Launching retriever server on port ${RETRIEVER_PORT}"
python -m servers.retriever \
  --port ${RETRIEVER_PORT} \
  --config servers/configs/retriever-Qwen3-4B-wiki-23.yaml \
  > "${LOG_DIR}/retriever_server.log" 2>&1 &

echo "[INFO] Waiting for retriever server to become ready..."
RETR_TIMEOUT=300
for ((i=0; i<RETR_TIMEOUT; i++)); do
  if grep -q "Uvicorn running on" "${LOG_DIR}/retriever_server.log"; then
    echo "[INFO] Retriever server is up."
    break
  fi
  sleep 1
done
if [[ $i -eq $RETR_TIMEOUT ]]; then
  echo "[ERROR] Retriever server failed to start within ${RETR_TIMEOUT}s"
  exit 1
fi

RETR_ADDR="http://${HEAD_NODE}:${RETRIEVER_PORT}"
echo "[INFO] Retriever URL: ${RETR_ADDR}"

echo "[INFO] Services are running. Tailing logs... (job will persist until timeout/cancel)"
echo "[INFO] - Encoder logs:   ${LOG_DIR}/vllm_encoder.log"
echo "[INFO] - Retriever logs: ${LOG_DIR}/retriever_server.log"

# Keep the job alive
wait
