import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Hieuman/Extractor-Qwen3-4B-SFT-v1"
output_dir = "/fsx/ubuntu/users/hieuman/state-aware-rag/models/qwen3-4b-sft-v1"

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

# Patch config with custom attribute before saving
model.config.attn_implementation = "eager"

model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)