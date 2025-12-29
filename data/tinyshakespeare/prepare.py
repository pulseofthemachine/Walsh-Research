"""
TinyShakespeare Dataset Preparation for Walsh

Downloads the TinyShakespeare dataset and prepares train/val splits.
"""

import os
import requests
import numpy as np
import pickle

# Download the dataset
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(__file__)

def prepare():
    input_file = os.path.join(DATA_DIR, 'input.txt')
    
    # Download if not exists
    if not os.path.exists(input_file):
        print(f"Downloading TinyShakespeare...")
        response = requests.get(DATA_URL)
        with open(input_file, 'w') as f:
            f.write(response.text)
    
    with open(input_file, 'r') as f:
        data = f.read()
    print(f"Dataset size: {len(data):,} characters")
    
    # Character-level tokenization
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    # Encode full dataset
    data_ids = np.array(encode(data), dtype=np.uint16)
    
    # Train/Val split (90/10)
    n = len(data_ids)
    train_ids = data_ids[:int(n*0.9)]
    val_ids = data_ids[int(n*0.9):]
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    # Save to binary files
    train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
    val_ids.tofile(os.path.join(DATA_DIR, 'val.bin'))
    
    # Save meta info
    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Done!")

if __name__ == "__main__":
    prepare()
