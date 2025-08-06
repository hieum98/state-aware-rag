export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCHELASTIC_ERROR_FILE=train_logs/torcherror.log

axolotl train finetune/Qwen3-SFT.yaml

