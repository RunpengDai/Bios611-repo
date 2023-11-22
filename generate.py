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


base_model='/overflow/htzhu/runpeng/pretrained_models/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348' 
lora_weights='/nas/longleaf/home/runpeng/LLM/my/experiments/checkpoint-250'


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
    )
# elif device == "mps":
#     model = LlamaForCausalLM.from_pretrained(
#         base_model,
#         device_map={"": device},
#         torch_dtype=torch.float16,
#     )
#     model = PeftModel.from_pretrained(
#         model,
#         lora_weights,
#         device_map={"": device},
#         torch_dtype=torch.float16,
#     )
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
    prompt = generate_prompt(datapoint, label = False)
    print(prompt)
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
    print(output)

datapoint = {"instruction": "Given the discription of a brain region and the features of a mutation, are these two related?", "input": "The discription of the brain region is:   The right lateral ventricle is a structure located within the brain  one part of a system of four communicating cavities that are continuous with the central canal of the spinal cord. The right lateral ventricle is situated in the right hemisphere of the brain  extending from the frontal lobe to the occipital lobe  and in depth into the temporal lobe. It is divided into an anterior (frontal)  a posterior (occipital)  and an inferior (temporal) horn. It communicates with the third ventricle through the interventricular foramen (foramen of Monro). The right lateral ventricle plays a crucial role in the circulation of cerebrospinal fluid that cushions the brain and spinal cord  as well as in the clearance of waste products. Any disruption or blockage to this can lead to a condition known as hydrocephalus  where an excessive amount of cerebrospinal fluid builds up in the brain. This can cause increased intracranial pressure  which can lead to a host of complications. The volume of the lateral ventricles can also be increased in several neurological conditions  such as schizophrenia and Alzheimer's disease. However  the specific clinical functions of the right lateral ventricle alone are not well elucidated. The formation and function of the right lateral ventricle  like all structures of the brain  are regulated by various genes acting in a complex interplay. Several genes are known to be involved in the development of the ventricular system  including those that code for transcription factors and signaling molecules that guide neuronal proliferation  migration  and differentiation. Among these genes are FOXJ1  which helps lining cells to produce cerebrospinal fluid  and SHH  which signals the development of structures around the ventricles. Genetic mutations or alterations can potentially lead to malformations of the ventricles and associated neurological conditions.\nThe discription of the mutation is: Mutation A-G happens on position 44021717 of chromosome 17. This mutation region belongs to a noncoding gene and its Genecode Comprehensive Caregory is intronic. The LINSIGHT score of this mutation is 6.08152755438231, where a higher score indicates more functionality. The GC content of this mutation is 0.523, and the CpG content is 0.053, which measures the percent of GC or CpG in the neighboring region of that mutation. The priPhCons score of this mutation is 0.335, which measures the conservation of the mutation region across 100 vertebrate species. The mamPhyloP score of this mutation is 0.223, which measures the conservation of the mutation region across 35 mammalian species. The verPhyloP score of this mutation is 0.223, which measures the conservation of the mutation region across 5 primate species. The DNase score of this mutation is 0.56, which measures the accessibility of the mutation region. The CADD RawScore of this mutation is 0.74317, which measures the deleteriousness of the mutation. The Nucleotide Diversity of this mutation is 2.9, which measures the diversity of the mutation region. "}

evaluate(datapoint, num_beams=1, max_new_tokens=128)