from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import sys

if torch.cuda.is_available():
    print("CUDA available")
else:
    print("This script is only meant to be used with CUDA, enable CUDA and re-run it")
    sys.exit(0)

model_path = "path/to/model"


tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map = "cuda",
        )

quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        )

model = AutoGPTQForCausalLM.from_quantized(
        model_path,
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
        "eos_token_id": tokenizer.eos_token_id
        }

prompt_template = "### USER:\n{human_prompt}\n\n### RESPONSE:\n"
prompt_in = """Apa sentimen dari kalimat berikut ini?
Kalimat: Buku ini sangat membosankan.
Jawaban: """

full_prompt = prompt_template.format(human_prompt=prompt_in)

tokens = tokenizer(full_prompt, return_tensors="pt")

input_ids = tokens["input_ids"].to("cuda:0")

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
