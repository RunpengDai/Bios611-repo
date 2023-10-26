import sys
import os 
import glob
import json
import numpy as np
import pandas as pd
import concurrent.futures 
from multiprocessing.dummy import Pool as pool
sys.path.append("..")
from config import *


# get dir and pheno number of all the files in directory
file_pattern = os.path.join(DIR.volume_gwas, '*A2.txt')
file_list = glob.glob(file_pattern)
pheno_list = [file.split("_")[-3] for file in file_list]
dirs = np.array([file_list,pheno_list]).T
result = {}

def parse_single(dir):
    print("processing:", dir[1])
    data = []
    with open(dir[0], 'r') as file:
        lines = file.readlines()
    #print(lines[1:])
    for line in lines[1:]:
        line = line.split()
        #print(line)
        if float(line[-2]) > GWAS.p:
            continue
        else:
            #print(line)
            data.append([line[1], line[3], line[-2], line[-1]]) 
    result[dir[1]] = data



with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(parse_single, dirs)

with open('label_data.json', 'w') as fp:
    json.dump(result, fp)