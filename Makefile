PHONEY: clean
PHONEY: middata

clean:
	rm -rf mid_data/*

mid_data: mid_data/volume_gwas_mapping.csv mid_data/label_data.json

mid_data/volume_gwas_mapping.csv: source_data/GPT/volume_gwas.xlsx config.py
	cd data_processing && python gwasname.py && cd ..

mid_data/label_data.json: config.py
	cd data_processing && python gwas_p.py && cd ..

source_data/GPT/volume_gwas_gpt.csv: mid_data/volume_gwas_mapping.csv config.py
	cd data_processing && python ask_gpt.py && cd ..
