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
else:
    device = "cpu"



def main(
    load_8bit: bool = False,
    lora_weights: str = "meta_on_inputsFalse_r8_module2/checkpoint-200",
    model_name = "meta"
):
    base_model = DIR.base_model[model_name]
    print(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    prompter = Prompter(PROMPT.tamplate_name)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # # unwind broken decapoda-research config
    if model_name == "decapoda":
        tokenizer.padding_side = "left" 
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
    elif model_name == "meta":
        tokenizer.pad_token_id = tokenizer.eos_token_id

    

    if not load_8bit:
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


    iface = gr.Interface(fn= evaluate, inputs=[instruction, input, temperature, num_beams, stopping], outputs="text")
    iface.queue().launch(share=True)

if __name__ == "__main__":
    fire.Fire(main)
