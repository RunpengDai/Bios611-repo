import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from finetune import generate_and_tokenize_prompt, generate_prompt, tokenize
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


base_model='/overflow/htzhu/runpeng/pretrained_models/meta_Llama_7Bhf' 
lora_weights='meta_on_inputsFalse_r8_module2/checkpoint-200'


tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        use_safetensors= False
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
    )


def evaluate(
    datapoint,
    num_beams=1,
    max_new_tokens=20,
    **kwargs,
):
    prompt = generate_prompt(datapoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        num_beams=num_beams,
        **kwargs,
    )

    # Without streaming
    print("Without streaming:")
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(output.encode('unicode_escape'))

datapoint = [{"instruction": "Detect the sentiment of the tweet.", "input": "@CryptoShillNye HEY FUCK YOU, TRX IS NEXT BITCOIN BITCH.", "output": ""}, {"instruction": "Detect the sentiment of the tweet.", "input": "@NickSzabo4 Monopoly: Bitcoin Edition would be the least fun board game ever", "output": ""}, {"instruction": "Detect the sentiment of the tweet.", "input": "Earn bitcoin on a daily basis!1. Follow @slidecoin 2. Complete instructions in pinned tweet", "output": ""}, {"instruction": "Detect the sentiment of the tweet.", "input": "ICE Agency Charges Payza and Two Canadian Citizens With Bitcoin Money Laundering #ico #cryptocurrency #token", "output": ""}, {"instruction": "Detect the sentiment of the tweet.", "input": "Anybody that knows how to use bitcoin?", "output": ""}, {"instruction": "Detect the sentiment of the tweet.", "input": "@CryptoCobain I want to be a big man can u plzzz give me free bitcoin?", "output": ""}]

for data in datapoint:
    evaluate(data, num_beams=1, max_new_tokens=10)