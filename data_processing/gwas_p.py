import sys
import os 
import glob
import json
import numpy as np
import concurrent.futures 
from multiprocessing.dummy import Pool as pool
sys.path.append("..")
from config import *


result = {}


def parse_single(dir):
    print("processing:", dir[1])
    data = []
    with open(dir[0], 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        line = line.split()
        if float(line[-2]) > GWAS.p or len(line[3]) > 1 or len(line[-1]) > 1:
            continue
        else:
            data.append(line[0]+"-"+line[2] + "-" +line[3] + "-" + line[-1]) 
    result[dir[1]] = data

def main():
     # get dir and pheno number of all the files in directory
    file_pattern = os.path.join(DIR.volume_gwas, '*A2.txt')
    file_list = glob.glob(file_pattern)
    pheno_list = [file.split("_")[-3] for file in file_list]
    dirs = np.array([file_list,pheno_list]).T

    with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(parse_single, dirs)

    with open('../mid_data/label_data.json', 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()