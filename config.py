from easydict import EasyDict as edict

DIR = edict()
DIR.volume_gwas = '/proj/htzhu/UKB_GWAS/results/ukb_phase1and2_roi/'
DIR.volume_gwas_name = 'ukb_roi_volume_may12_2019_phase1and2_{}_allchr_withA2.txt'


GPT = edict()
GPT.key = "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"
GPT.model = "gpt-4"

GWAS = edict()
GWAS.p = 5e-8


