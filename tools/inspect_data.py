"""
SpinNet Data Biopsy
-------------------
Decodes random chunks of the training file to check for
Topic Diversity vs. Monotony.
"""
import os
import numpy as np
import tiktoken

# CONFIG
data_dir = 'data/fineweb'
num_samples = 5
sample_len = 100

# LOAD
bin_path = os.path.join(data_dir, 'train.bin')
print(f"Reading from {bin_path}...")

data = np.memmap(bin_path, dtype=np.uint16, mode='r')
print(f"Total Tokens: {len(data):,}")

enc = tiktoken.get_encoding("gpt2")

print("-" * 60)
for i in range(num_samples):
    # Pick a random spot in the file
    idx = np.random.randint(0, len(data) - sample_len)
    chunk = data[idx:idx+sample_len].astype(np.int64)
    
    text = enc.decode(chunk)
    
    print(f"SAMPLE {i+1} (Offset {idx}):")
    print(text.replace('\n', '\\n')) # Escape newlines for readability
    print("-" * 60)
