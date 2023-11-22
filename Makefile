PHONEY: clean
PHONEY: middata
DATASET_FILES = img/pre-phenos.png img/post-phenos.png mid_data/dataset.json

clean:
	rm -rf mid_data/*

mid_data: mid_data/volume_gwas_mapping.csv mid_data/label_data.json mid_data/SNP*.txt img/pre-phenos.png

mid_data/volume_gwas_mapping.csv: source_data/GPT/volume_gwas.xlsx 
	cd data_processing && python gwasname.py && cd ..

mid_data/label_data.json: 
	cd data_processing && python gwas_p.py && cd ..

mid_data/SNP*.txt: mid_data/label_data.json
	cd data_processing && python snp_extracting.py && cd ..

$(DATASET_FILES): mid_data/label_data.json mid_data/SNP*.txt
	cd data_processing && python gen_dataset.py && cd ..


source_data/GPT/volume_gwas_gpt.csv: mid_data/volume_gwas_mapping.csv config.py
	cd data_processing && python ask_gpt.py && cd ..
