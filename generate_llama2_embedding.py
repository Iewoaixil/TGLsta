import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import fire
from llama import Llama
from typing import List
import numpy as np
import torch
import json
import csv

def main(
    ckpt_dir: str = 'llama-2-7b/',
    tokenizer_path: str = 'tokenizer.model',
    max_seq_len: int = 2000,
    max_batch_size: int = 2,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    texts = []
    data_name = 'cora'
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            texts.append(line[2])
            
    logits_list = []
    for text in texts:
        result = generator.generate_logits(text)
        result = result[0]
        result = torch.mean(result, dim=0)
        result = result.cpu()
        logits_list.append(result)
    logits = torch.stack(logits_list)
    logits = logits.numpy()
    np.save(f'llama2_{data_name}_logits.npy', logits)



if __name__ == "__main__":
    fire.Fire(main)
