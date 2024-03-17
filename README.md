# Purpose 

This repository provides a guide and a collection of scripts to help with the quantization and inference of the SEA-LION 7B instruct model developed by AI Singapore. The goal is to further democratise access to SEA-LION by allowing it to run on commercial hardware, due to quantization.

The 4-bit, 128 group size quantized model can be found [here]().

# Quantization 

The main work is done by the [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) library. As of March 2024 however, the library does not directly support GPTQ quantization for the MPT architecture which SEA-LION is based on. Instead, a specific fork of the library is used, created by @LaaZa. For convenience, we have forked Laaza's patch, which can be found [here](https://github.com/Caviato/AutoGPTQ). 

The class `AutoGPTQForCausalLM` is very similar to `AutoModelForCausalLM` by HuggingFace's Transformers library, except that you have to pass in a quantization config.  

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

tokenizer = AutoTokenizer.from_pretrained( # will be loaded to GPU
        "path/to/model/dir",
        trust_remote_code=True,
        device_map = "cuda"
        ) 

quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128
        )

model = AutoGPTQForCausalLM.from_pretrained( # will be loaded to CPU by default
        "path/to/model/dir",
        quantize_config,
        trust_remote_code=True,
        )
model.to("cuda:0") # load model to GPU 
```

The GPTQ algorithm requires some input data. Due to the multilingual nature of SEA-LION, we used data from each language available in SEA-LION. 
```python
from datasets import load_dataset
import random

seed = 0
random.seed(seed)

paths = []
data = load_dataset("json", data_files=paths, split="train")

quantize_dataset = []
n_samples = 128 # from paper 
seqlen = 2048 
chunk_size = 100 # arbitrary value, to make sure there is enough data to reach a sequence length of 2048 
for _ in range(n_samples):
    i = random.randint(0, data.num_rows - chunk_size - 1)
    chunk = "".join(data["text"][i:i+chunk_size])
    token_data = tokenizer(chunk, return_tensors="pt")
    inp = token_data.input_ids[:, :seqlen]
    attention_mask = torch.ones_like(inp)
    quantize_dataset.append({"input_ids": inp, "attention_mask": attention_mask})
```

Finally, we can quantize and save our model. Keep in mind that the other model files will need to be copied over to this new directory in order to run inference. 

```python
model.quantize(quantize_dataset, batch_size=10)

model.save_quantized(
        "path/to/output/dir",
        use_safetensors=True
        )
```

# Inference 

Create your tokenizer, quantization config and model:
```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

tokenizer = AutoTokenizer.from_pretrained( # will be loaded to GPU
        "path/to/model/dir",
        trust_remote_code=True,
        device_map = "cuda",
        )

quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128
        )

model = AutoGPTQForCausalLM.from_quantized( # will be loaded to GPU
        "path/to/model/dir",
        device = "cuda:0",
        quantize_config = quantize_config,
        torch_dtype=torch.float16,
        trust_remote_code = True
        )

generation_kwargs = {
        "do_sample": False,  # set to true if temperature is not 0
        "temperature": None,
        "max_new_tokens": 256,
        "top_k": 50,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "eos_token_id": tokenizer.eos.token_id
        }
```

Create your prompt: 
```python 
prompt_template = "### USER:\n{human_prompt}\n\n### RESPONSE:\n"
prompt_in = """Apa sentimen dari kalimat berikut ini?
Kalimat: Buku ini sangat membosankan.
Jawaban: """

full_prompt = prompt_template.format(human_prompt=prompt_in)
```

Tokenize your prompt and pass it into your model to generate your response! 
```python
tokens = tokenizer(full_prompt, return_tensors="pt")

input_ids = tokens["input_ids"].to("cuda:0") # move tokenized input to GPU 

# Remove unneeded kwargs
if generation_kwargs["do_sample"] == False:
    generation_kwargs.pop("temperature")
    generation_kwargs.pop("top_k")
    generation_kwargs.pop("top_p")

output = model.generate(
		input_ids = input_ids,
		**generation_kwargs
		)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# Benchmark 

| Model                                        | ARC | HellaSwag | MMLU | TruthfulQA | Average |
| -------------------------------------------- | --- | --------- | ---- | ---------- | ------- |
| SEA-LION 7B Instruct (Base)                  |   40.78  |     68.20     |   27.12   |    36.29     |    43.10     |
| SEA-LION 7B Instruct (4-Bit, 128 group size) |   39.93  |     67.32     |   27.11   |    36.32     |    42.67     |

Although the evaluations were run with the same n-shot values as Hugging Face's LLM Leaderboard, the evaluations were run using version 0.4.1 of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.1) by EleutherAI.

| Tasks | n-shots |
|-------|---------|
| ARC (arc_challenge) | 25 |
| HellaSwag (hellaswag) | 10 |
| MMLU (mmlu) | 5 |
| TruthfulQA (truthfulqa_mc2) | 0 |

# Work In Progress (WIP)

- [ ] Inference time comparisons on A100 
- [ ] Inference time of quantized model on GTX1070 (8GB)
- [ ] Inference time of quantized model on RTX4080 (8GB)

# Acknowledgements 

Thank you to the AI Singapore team for their guidance and resources, with special thanks to: 
- Ng Boon Cheong Raymond
- Teng Walter
- Siow Bryan
