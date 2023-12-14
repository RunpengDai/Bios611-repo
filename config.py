from easydict import EasyDict as edict

DIR = edict()
DIR.volume_gwas = '/proj/htzhu/UKB_GWAS/results/ukb_phase1and2_roi/'
DIR.volume_gwas_name = 'ukb_roi_volume_may12_2019_phase1and2_{}_allchr_withA2.txt'

DIR.base_model = {"meta7b":"/overflow/htzhu/runpeng/pretrained_models/Llama-2-7b-hf", "decapoda":"/overflow/htzhu/runpeng/pretrained_models/decapoda-research-llama-7b-hf"}
DIR.online_model = {"meta7b":"meta-llama/Llama-2-7b-hf", "decapoda":"linhvu/decapoda-research-llama-7b-hf"}

DIR.json_data = "mid_data/dataset.json"
DIR.output = "./weights"


HYPER = edict()
HYPER.cutoff_len = 5000
HYPER.train_on_inputs = True
HYPER.add_eos_token = True
HYPER.val_set_size = 400


GPT = edict()
GPT.key = "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"
GPT.model = "gpt-4"

GWAS = edict()
GWAS.p = 5e-9

SNP = edict()
SNP.features = ["Variant (VCF)", "Chromosome", "Position", "Genecode Comprehensive Category", "RefSeq Info", "LINSIGHT", "GC", "CpG", "priPhCons", "mamPhyloP", "verPhyloP", "DNase", "CADD RawScore", "Nucleotide Diversity", "Funseq Description"]



PROMPT = edict()
PROMPT.instruction = "Given the discription of a brain region and some features of a genetic mutation, are these two related?"
PROMPT.tamplate_name = "alpaca"

LORA = edict()
LORA.r = 8
LORA.alpha = 16
LORA.dropout= 0.05
LORA.modules = [
    "q_proj",
    "v_proj"
]