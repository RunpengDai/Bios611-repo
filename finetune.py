from config import *
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
from utils.prompter import Prompter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import tensorboard

import fire
import torch
from datasets import load_dataset
import pandas as pd
 
from pylab import rcParams
 
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 




def train(model_name = "meta"):
    base_model = DIR.base_model[model_name]
    output_dir = "./"+model_name+"_on_inputs{}_r{}_module{}".format(HYPER.train_on_inputs, LORA.r, len(LORA.modules))


    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"HYPER.cutoff_len: {HYPER.cutoff_len}\n"
            f"LORA.r: {LORA.r}\n"
            f"lora,modules: {LORA.modules}\n"
            f"HYPER.train_on_inputs: {HYPER.train_on_inputs}\n"
            f"prompt template: {PROMPT.tamplate_name}\n"
            f"base model: {model_name}\n"
        )
    ### load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit= True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    #print("model loaded")
    model = prepare_model_for_int8_training(model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(model.config)
    print(tokenizer)
    if model_name == "decapoda":
        tokenizer.padding_side = "left" 
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
    elif model_name == "meta":
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ### load data and tokenize the data
    prompter = Prompter(PROMPT.tamplate_name)
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=HYPER.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < HYPER.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not HYPER.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=HYPER.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if HYPER.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    data = load_dataset("json", data_files= DIR.json_data)
    if HYPER.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=HYPER.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    print(data["train"])



    
    config = LoraConfig(
        r=LORA.r,
        lora_alpha=LORA.alpha,
        target_modules=LORA.modules,
        lora_dropout=LORA.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=1,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_safetensors=False,
    save_strategy="steps",
    eval_steps=100 if HYPER.val_set_size > 0 else None,
    save_steps=10,
    output_dir=output_dir,
    save_total_limit=3,
    report_to="tensorboard")

    trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
    model.config.use_cache = False


    os.system('nvidia-smi')
    with torch.autocast("cuda"): 
        trainer.train()
    model.save_pretrained(output_dir, safe_serialization = False)


if __name__ == "__main__":
    fire.Fire(train)