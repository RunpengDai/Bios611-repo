import numpy as np
import json

def parse_single(value):
    snp = np.array(value)
    if snp.shape[0] == 0:
        return []
    else:
        return np.unique(snp).tolist()


def main():
    with open('../mid_data/label_data.json', 'r') as f:
        data = json.load(f)
    unique_snp = []
    for _, value in data.items():
        unique_snp += parse_single(value)
    unique_snp = np.unique(unique_snp)
    num_snp = len(unique_snp)
    batch = 1000
    print("Collected {} different SNPs".format(num_snp))
    sep = num_snp // batch
    for i in range(sep+1):
        with open('../mid_data/SNP{}.txt'.format(i), 'w') as f:
            for j in range(batch):
                idx = i * batch + j
                f.write("%s\n" % unique_snp[idx]) if idx < num_snp else None


if __name__ == "__main__":
    main()