from openai_multi_client import OpenAIMultiClient
import sys
import numpy as np
import openai
sys.path.append("..")
from config import *
openai.api_key = GPT.key

# load the name data
name_data = np.loadtxt('../mid_data/volume_gwas_mapping.csv', dtype=str, delimiter=',')
regions, ids = name_data[:, 0], name_data[:, 1]

# ask openai
api = OpenAIMultiClient(endpoint="chats", data_template={"model": GPT.model})

def make_requests():
    for idx, region in enumerate(regions):
        print(idx, region)
        api.request(data={
            "messages": [{"role": "user", "content": f"Can you tell me the Anatomy, Clinical function and genetic information of {region}? Please organize the answer with less than 300 words without any summary or conclusion. Only include the description items, organize the answer in three paragraphs, starting with Anatomy:, Clinical function:, genetic information:."}]
        }, metadata={'region': region, 'id': ids[idx]})


api.run_request_function(make_requests)

# processing the results
print("processing the results")
final_data = []
for result in api:
    line = result.response['choices'][0]['message']['content'].replace('\n','*')
    final_data.append([result.metadata['id'], result.metadata['region'], line])

print(final_data[-1])
final_data = np.array(final_data)
np.savetxt('../source_data/GPT/volume_gwas_gpt.csv', final_data, delimiter=',', fmt='%s')