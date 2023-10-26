import sys
import os
import numpy as np
import pandas as pd

sys.path.append("..")

from config import *

def finish_dir(name):
    return DIR.volume_gwas + '/' + DIR.volume_gwas_name.format(name)

df = pd.read_excel('../source_data/GPT/volume_gwas.xlsx')
df.iloc[:, 0] = df.iloc[:, 0].str.replace('.', ' ', regex=False)

dir = DIR.volume_gwas + '/' + DIR.volume_gwas_name
df["dir"] = df['ID'].apply(lambda x: dir.format(x))

df.to_csv('../mid_data/volume_gwas_mapping.csv', index=False, header = False)