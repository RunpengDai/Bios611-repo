from easydict import EasyDict as edict

DIR = edict()
DIR.volume_gwas = '/proj/htzhu/UKB_GWAS/results/ukb_phase1and2_roi/'
DIR.volume_gwas_name = 'ukb_roi_volume_may12_2019_phase1and2_{}_allchr_withA2.txt'

DIR.base_model = {"meta":"/overflow/htzhu/runpeng/pretrained_models/meta_Llama_7Bhf", "decapoda":"/overflow/htzhu/runpeng/pretrained_models/decapoda_Llama_7Bhf"}
DIR.json_data = "/nas/longleaf/home/runpeng/LLM/tweet/alpaca-bitcoin-sentiment-dataset.json"
DIR.output = "./weights"


HYPER = edict()
HYPER.cutoff_len = 516
HYPER.train_on_inputs = False
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
PROMPT.instruction = "Given the discription of a brain region and the features of a mutation, are these two related?"
PROMPT.tamplate_name = "alpaca"

LORA = edict()
LORA.r = 8
LORA.alpha = 16
LORA.dropout= 0.05
LORA.modules = [
    "q_proj",
    "v_proj"
]