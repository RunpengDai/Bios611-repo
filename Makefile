PHONEY: clean
PHONEY: middata

clean:
	find mid_data/ ! -name label_data.json -type f -exec rm -f {} +
	rm -rf img/*

mid_data: mid_data/volume_gwas_mapping.csv mid_data/label_data.json mid_data/SNP*.txt 
dataset_files: img/pre-phenos.png img/post-phenos.png mid_data/dataset.json


mid_data/volume_gwas_mapping.csv: source_data/GPT/volume_gwas.xlsx 
	cd data_processing && python gwasname.py && cd ..

mid_data/label_data.json: 
	cd data_processing && python gwas_p.py && cd ..

mid_data/SNP*.txt: mid_data/label_data.json
	cd data_processing && python snp_extracting.py && cd ..

img/pre-phenos.png img/post-phenos.png mid_data/dataset.json: mid_data/label_data.json mid_data/SNP*.txt
	cd data_processing && python gen_dataset.py && cd ..


source_data/GPT/volume_gwas_gpt.csv: mid_data/volume_gwas_mapping.csv config.py
	cd data_processing && python ask_gpt.py && cd ..
