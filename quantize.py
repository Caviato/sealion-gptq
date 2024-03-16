from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import torch

import time
import sys
import random

from datasets import load_dataset

if torch.cuda.is_available():
    print("CUDA available, continuing...")
else:
    print("This script is only meant to be used with CUDA, enable CUDA and re-run it")
    sys.exit(0)

base_model_path = "path/to/base"
quant_mode_path = "path/to/quant" 

seed = 0
random.seed(seed)

quantize_dataset = []
n_samples = 128
seqlen = 2048
chunk_size = 100

tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        )

quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128
        )

model = AutoGPTQForCausalLM.from_pretrained(
        base_model_path,
        quantize_config,
        trust_remote_code=True,
        )

paths = []
data = load_dataset("json", data_files=paths, cache_dir=scratch_dir, split="train")

for _ in range(n_samples):
    i = random.randint(0, data.num_rows - chunk_size - 1)
    chunk = "".join(data["text"][i:i+chunk_size])
    token_data = tokenizer(chunk, return_tensors="pt")
    inp = token_data.input_ids[:, :seqlen]
    attention_mask = torch.ones_like(inp)
    quantize_dataset.append({"input_ids": inp, "attention_mask": attention_mask})

print("Starting quantization...")
model.to("cuda:0")

start = time.perf_counter()
model.quantize(quantize_dataset, batch_size=10)
end = time.perf_counter()
print(f"Quantization time (s): {end - start}")

print("Saving model...")
model.save_quantized(
        quant_model_path,
        use_safetensors=True
        )
