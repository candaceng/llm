import argparse
import gzip
import json
from functools import reduce
import numpy as np
from transformers import AutoTokenizer
import random
import torch
import pickle

parse = argparse.ArgumentParser()
parse.add_argument('--nsamples', type=int, default=1000, help='number of samples')
parse.add_argument('--ntokens', type=int, default=5, help='number of tokens in prompt')
parse.add_argument('--output_file', type=str, default='output.pkl', help='output file name')
args = parse.parse_args()

tokenizer = AutoTokenizer.from_pretrained('./tokenizer_files')

# Open the gzipped file and read all content into the list
all_data = []
with gzip.open('c4-train.00000-of-01024.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        all_data.append(data)
num_docs = len(all_data)

np.random.seed(1023)
indices = result = reduce(lambda acc, x: acc + [np.random.randint(0, num_docs)], range(args.nsamples), [])

# Sample prompts
prompts_and_labels = []
for idx in indices:
    text = tokenizer.encode(all_data[idx]['text'])

    start_index = random.randint(0, len(text) - args.ntokens - 1)
    prompt = tokenizer.decode(text[start_index:start_index + args.ntokens])
    label = torch.tensor(text[start_index + args.ntokens], dtype=torch.long)

    prompts_and_labels.append((prompt, label))

with open(args.output_file, "wb") as fp:  
    pickle.dump(prompts_and_labels, fp)


# python generate_prompts.py --nsamples 10000 --ntokens 5 --output_file 'test.pkl'
        
