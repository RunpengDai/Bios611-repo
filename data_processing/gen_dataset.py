import sys
import json
import os
import random
import numpy as np
import glob
import re
import pandas as pd
from matplotlib import pyplot as plt 
from tqdm import tqdm
from multiprocessing.dummy import Pool as pool
sys.path.append("..")
from config import *

with open('../mid_data/label_data.json', 'r') as f:
    label_data = json.load(f)

GPT_discription = {}
pheno_name = {}
with open('../source_data/GPT/volume_gwas_gpt.csv', 'r') as file:
    lines = file.readlines()
for line in lines:
    parts = line.split(",")
    GPT_discription[parts[0]] = " ".join(parts[2:])
    pheno_name[parts[0]] = parts[1]


def gen_SNPdiscription():
    if not os.path.exists("../mid_data/SNP_description.csv"):
        SNPfiles = glob.glob("../source_data/SNP/*.csv")
        df_list = []
        for file in SNPfiles:
            df = pd.read_csv(file)
            df = df[SNP.features]
            df_list.append(df)
        
        conbined_df = pd.concat(df_list)

        conbined_df.to_csv("../mid_data/SNP_description.csv", index=False)
    else:
        conbined_df = pd.read_csv("../mid_data/SNP_description.csv")
    uniqueSNP = conbined_df["Variant (VCF)"].tolist()
    for key in label_data.keys():
        label_data[key] = list(set(label_data[key]) & set(uniqueSNP))
    phenos, counts = [], []
    for key in label_data.keys():
        phenos.append(int(key[5:]))
        counts.append(len(label_data[key]))
    plt.bar(phenos, counts)
    plt.savefig("../img/pre-phenos.png")
    plt.clf()
    return conbined_df

def chosephenos():
    phenos = []
    for key in label_data.keys():
        phenos.append(key) if len(label_data[key])>5 and len(label_data[key])<700 else None
    counts = [len(label_data[key]) for key in phenos]
    plt.figure(figsize=(10, 6))
    plt.bar([pheno[5:]for pheno in phenos], counts)
    plt.xticks(rotation=90)
    plt.savefig("../img/post-phenos.png")
    return phenos

def process_GPT_result(pheno):
    """ Given the GPT definition of each brain region, take out the three parts and conbine to a passage.
    """
    raw_discription = GPT_discription[pheno]
    parts = raw_discription.split("**")
    discription = " "
    for part in parts:
        part = " ".join(part.split(":")[1:])
        discription += part
    return discription

def process_SNP_result(SNP, SNPs):
    """ Given a SNP, write its feature into a passage.
    """
    info = SNPs[SNPs["Variant (VCF)"] == SNP].to_numpy()[0]
    mutation = info[0].split("-")[-2] + "-" + info[0].split("-")[-1]
    part1 = "The genetic mutation {} happens on position {} of chromosome {}. This mutation region belongs to a {} gene and its Genecode Comprehensive Caregory is {}. ".format(mutation, info[2], info[1], info[-1], info[3])
    part2 = "The LINSIGHT score of this mutation is {}, where a higher score indicates more functionality. The GC content of this mutation is {}, and the CpG content is {}, which measures the percent of GC or CpG in the neighboring region of that mutation. ".format(info[5], info[6], info[7])
    part3 = "The priPhCons score of this mutation is {}, which measures the conservation of the mutation region across 100 vertebrate species. The mamPhyloP score of this mutation is {}, which measures the conservation of the mutation region across 35 mammalian species. The verPhyloP score of this mutation is {}, which measures the conservation of the mutation region across 5 primate species. ".format(info[8], info[9], info[10])
    part4 = "The DNase score of this mutation is {}, which measures the accessibility of the mutation region. The CADD RawScore of this mutation is {}, which measures the deleteriousness of the mutation. The Nucleotide Diversity of this mutation is {}, which measures the diversity of the mutation region. ".format(info[11], info[12], info[13])
    passage = part1 + part2 + part3 
    return passage

def gen_dataset(phenos, SNPs):
    dataset = []
    all_SNPs = SNPs["Variant (VCF)"].tolist()
    print("Total SNP number:", len(all_SNPs))
    index = 0
    if not os.path.exists("../mid_data/dataset.json"):
        for pheno in tqdm(phenos):
            pheno_discription = process_GPT_result(pheno)
            related_SNPs = label_data[pheno]
            unrelated_SNPs = random.sample(list(set(all_SNPs) - set(related_SNPs)), len(related_SNPs))
            for SNP in related_SNPs:
                SNP_discription = process_SNP_result(SNP, SNPs)
                dataset.append({
                    "instruction": PROMPT.instruction,
                    "input": re.sub(' +', ' ', pheno_discription + SNP_discription),
                    "output": "Yes they are related.",
                })

            for SNP in unrelated_SNPs:
                SNP_discription = process_SNP_result(SNP, SNPs)
                dataset.append({
                    "instruction": PROMPT.instruction,
                    "input": re.sub(' +', ' ', pheno_discription + SNP_discription),
                    "output": "No they are unrelated.",
                })
            index += 1
        random.shuffle(dataset)
        print("Dataset size:", len(dataset))
        with open("../mid_data/dataset.json", "w") as f:
            json.dump(dataset, f)
    else:
        print("Dataset already exists.")
        with open('../mid_data/dataset.json', 'r') as f:
            dataset = json.load(f)
        print("Dataset size:", len(dataset))

def main():
    # generate the SNP description file
    SNP_discription = gen_SNPdiscription()
    print(SNP_discription.head())
    # select phenoes with enough related SNPs
    phenos = chosephenos()

    # generate the dataset
    gen_dataset(phenos, SNP_discription)


if __name__ == "__main__":
    main()