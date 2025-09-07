#!/bin/bash -l

#SBATCH -o SLURM_Logs/%j/%x_master.log
#SBATCH -e SLURM_Logs/%j/%x_master.log
#SBATCH -D ./
#SBATCH -J hm-llm

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --partition="ml-p4d-24xlarge-us-west-2d"
#SBATCH --gres=gpu:8

set -euo pipefail

# -------- Activate environment --------
source ~/users/hieuman/.bashrc
conda activate sglang_server

# -------- Args --------
print_usage() {
  cat <<EOF
Usage: $0 <model-name> [--port 30000] [--tp 1] [--parser qwen3|deepseek-r1]

Examples:
  $0 Qwen/Qwen3-8B
  $0 Qwen/Qwen3-8B --port 30000 --tp 1
  $0 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --parser deepseek-r1
EOF
}

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Model name is required."
  print_usage
  exit 1
fi

MODEL_NAME="$1"; shift || true
PORT=30000
TP=1
PARSER="deepseek-r1"  # Default parser

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:-}"; shift 2 ;;
    --tp) TP="${2:-}"; shift 2 ;;
    --parser) PARSER="${2:-}"; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

# Determine GPU count (honor CUDA_VISIBLE_DEVICES if set), then set DP = n_gpu // tp (min 1)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _garr <<< "$CUDA_VISIBLE_DEVICES"
  NGPU=${#_garr[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
  NGPU=${NGPU:-0}
else
  NGPU=0
fi

# Validate TP and compute DP
if ! [[ "$TP" =~ ^[0-9]+$ ]]; then
  echo "[WARN] Invalid --tp '$TP', defaulting to 1"
  TP=1
fi
(( TP < 1 )) && TP=1
DP=$(( NGPU / TP ))
(( DP < 1 )) && DP=1

# -------- Logging / address --------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_LOG_DIR="SLURM_Logs/${SLURM_JOB_ID}"
else
  JOB_LOG_DIR="SLURM_Logs/standalone_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$JOB_LOG_DIR"
LOG_FILE="$JOB_LOG_DIR/sglang_${PORT}.log"

# Determine server address, using slurm node hostname if in slurm job
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_NODELIST:-}" ]]; then
  HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
else
  HOSTNAME=localhost
fi
SERVER_ADDRESS="http://${HOSTNAME}:${PORT}/v1"

echo "[INFO] Launch settings:" | tee -a "$LOG_FILE"
echo "  model  : $MODEL_NAME" | tee -a "$LOG_FILE"
echo "  parser : $PARSER" | tee -a "$LOG_FILE"
echo "  host   : 0.0.0.0:$PORT (public: $SERVER_ADDRESS)" | tee -a "$LOG_FILE"
echo "  gpus   : ${NGPU:-0}" | tee -a "$LOG_FILE"
echo "  dp/tp  : $DP/$TP" | tee -a "$LOG_FILE"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

# Free the port if it's occupied
if command -v fuser &>/dev/null; then fuser -k ${PORT}/tcp || true; fi

# -------- Launch server --------
set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} python -m sglang.launch_server \
  --host 0.0.0.0 \
  --model-path "$MODEL_NAME" \
  --reasoning-parser "$PARSER" \
  --mem-fraction-static 0.8 \
  --dtype bfloat16 \
  --port "$PORT" \
  --dp "$DP" \
  --tp "$TP" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
set -e

echo "[INFO] LLM server PID: $SERVER_PID" | tee -a "$LOG_FILE"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[INFO] Stopping server PID $SERVER_PID" | tee -a "$LOG_FILE"
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..." | tee -a "$LOG_FILE"
READY=0
for i in {1..300}; do
  if grep -q "Uvicorn running on" "$LOG_FILE"; then READY=1; break; fi
  sleep 2
done

if [[ "$READY" -eq 1 ]]; then
  echo "[INFO] Server ready: $SERVER_ADDRESS" | tee -a "$LOG_FILE"
else
  echo "[ERROR] Server did not start in time. See logs: $LOG_FILE" | tee -a "$LOG_FILE"
  exit 1
fi

# Keep process alive when submitted via sbatch; in interactive shell, just tail the log
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  wait "$SERVER_PID"
else
  echo "[INFO] Tailing logs. Press Ctrl+C to stop."; tail -f "$LOG_FILE"
fi


