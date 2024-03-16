model_loc=
model_base_name=sea-lion-7b-instruct
model_quant_name=
safetensor_quant_name=

output_path=
result_base_name=
result_quant_name=

model_base_path="${model_loc}${model_base_name}"
model_quant_path="${model_loc}${model_quant_name}"

task_name="arc_challenge"
n_shot=25

lm_eval --model hf \
        --model_args pretrained=${model_base_path},trust_remote_code=True \
        --tasks ${task_name} \
        --num_fewshot ${n_shot} \
        --output_path "${output_path}${result_base_name}"

lm_eval --model hf \
        --model_args pretrained=${model_quant_path},autogptq=${safetensor_quant_name},gptq_use_triton=True,trust_remote_code=True \
        --tasks ${task_name} \
        --num_fewshot ${n_shot} \
        --output_path "${output_path}${result_quant_name}"

