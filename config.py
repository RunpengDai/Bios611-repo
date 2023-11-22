from easydict import EasyDict as edict

DIR = edict()
DIR.volume_gwas = '/proj/htzhu/UKB_GWAS/results/ukb_phase1and2_roi/'
DIR.volume_gwas_name = 'ukb_roi_volume_may12_2019_phase1and2_{}_allchr_withA2.txt'
DIR.pretrained_model = '/overflow/htzhu/runpeng/pretrained_models/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348'
DIR.json_data = "mid_data/dataset.json"
DIR.output = "./weights"


HYPER = edict()
HYPER.cutoff_len = 5000




GPT = edict()
GPT.key = "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"
GPT.model = "gpt-4"

GWAS = edict()
GWAS.p = 5e-9

SNP = edict()
SNP.features = ["Variant (VCF)", "Chromosome", "Position", "Genecode Comprehensive Category", "RefSeq Info", "LINSIGHT", "GC", "CpG", "priPhCons", "mamPhyloP", "verPhyloP", "DNase", "CADD RawScore", "Nucleotide Diversity", "Funseq Description"]



PROMPT = edict()
PROMPT.instruction = "Given the discription of a brain region and the features of a mutation, are these two related?"