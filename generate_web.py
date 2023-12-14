import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from config import *
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
    torch_type = torch.float16
else:
    device = "cpu"
    torch_type = torch.float32



def main(
    load_8bit: bool = False,
    lora_weights = None,
    model_name = "meta7b",
    text = "Geno-Pheno llama",
    online = False
):
    base_model = DIR.base_model[model_name] if not online else DIR.online_model[model_name]
    if lora_weights is None:
        text += " with base model {}".format(model_name)
    else:
        text += (" with LORA model " +lora_weights)
    print(text)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch_type,
        device_map="auto"
    )
    if lora_weights is not None:
        text+= " with base model"
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch_type,
        )

    prompter = Prompter(PROMPT.tamplate_name)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # # unwind broken decapoda-research config
    if model_name == "decapoda":
        tokenizer.padding_side = "left" 
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
    elif model_name == "meta7b":
        tokenizer.pad_token_id = tokenizer.eos_token_id

    

    if not load_8bit and device == "cuda":
        model.half()  # seems to fix bugs for some users.
    model.eval()


    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        num_beams=4,
        max_new_tokens=256,
        stopping = True
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            do_sample=False, num_beams=1, max_length = 256
        )
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(inputs = input_ids,generation_config=generation_config)
        print(generation_output)
        s = generation_output[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


    instruction = gr.components.Textbox(lines=2, label="Instruction",placeholder="Tell me about alpacas.",)
    input = gr.components.Textbox(lines=2, label="Input", placeholder="none")
    temperature = gr.components.Slider(minimum=0.0, maximum=1.0, label="Temperature")
    stopping = gr.components.Checkbox(label="Stopping")
    num_beams = gr.components.Number(label="Number of Beams")


    iface = gr.Interface(fn= evaluate, inputs=[instruction, input, temperature, num_beams, stopping], outputs="text", title="Toy Gentype Phenotype llama",
    description= text)
    iface.queue().launch(share=True)

if __name__ == "__main__":
    fire.Fire(main)
