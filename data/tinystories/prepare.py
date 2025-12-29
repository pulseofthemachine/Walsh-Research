"""
TinyStories Dataset Preparation for Walsh

Downloads the TinyStories dataset from HuggingFace and tokenizes using GPT-2 tokenizer.
This gives us token-level (subword) training instead of character-level.

TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
"""

import os
import numpy as np
import pickle
from datasets import load_dataset
from transformers import GPT2Tokenizer

DATA_DIR = os.path.dirname(__file__)

def prepare():
    # Load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load TinyStories from HuggingFace
    print("Loading TinyStories dataset from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Get validation split (last 5% of stories)
    n = len(dataset)
    val_start = int(n * 0.95)
    
    print(f"Total stories: {n:,}")
    print(f"Train stories: {val_start:,}")
    print(f"Val stories: {n - val_start:,}")
    
    # Tokenize and collect
    print("Tokenizing train split...")
    train_ids = []
    for i in range(val_start):
        text = dataset[i]["text"]
        ids = tokenizer.encode(text)
        train_ids.extend(ids)
        if i > 0 and i % 100000 == 0:
            print(f"  {i:,}/{val_start:,} stories tokenized...")
    
    print("Tokenizing val split...")
    val_ids = []
    for i in range(val_start, n):
        text = dataset[i]["text"]
        ids = tokenizer.encode(text)
        val_ids.extend(ids)
    
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    # Save to binary files
    train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
    val_ids.tofile(os.path.join(DATA_DIR, 'val.bin'))
    
    # Save meta info (GPT-2 vocab)
    meta = {
        'vocab_size': 50257,  # GPT-2 vocab size
        'tokenizer': 'gpt2',  # Indicate which tokenizer to use
    }
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Done!")

if __name__ == "__main__":
    prepare()
