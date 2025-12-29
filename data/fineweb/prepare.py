"""
Walsh Data Engineer: The Chinchilla Slice (Fixed)
-------------------------------------------
Dataset: FineWeb-Edu (High Quality Reasoning)
Target: 5 Billion Tokens (~10GB on disk)
Method: Streaming & Sharding (RAM Safe)
"""
import os
import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET_NAME = "sample-10BT" 

TARGET_TOKENS = 5_000_000_000 # 5 Billion Tokens
CHUNK_SIZE = 100_000_000      # Flush to disk every 100M tokens

# Output paths
os.makedirs('data/fineweb', exist_ok=True)
TRAIN_BIN = 'data/fineweb/train.bin'
VAL_BIN = 'data/fineweb/val.bin'

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
eos_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

print(f">>> PREPARING CHINCHILLA SLICE ({TARGET_TOKENS/1e9}B Tokens) <<<")
print(f"Source: {DATASET_NAME} ({SUBSET_NAME})")

dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="train", streaming=True)

# -----------------------------------------------------------------------------
# PROCESSING LOOP
# -----------------------------------------------------------------------------
token_count = 0
buffer = []
pbar = tqdm.tqdm(total=TARGET_TOKENS, unit="tok", desc="Processing")

# Create/Clear files
open(TRAIN_BIN, 'wb').close()
open(VAL_BIN, 'wb').close()

def flush_buffer(buf, is_val=False):
    """Writes the current buffer to disk as uint16."""
    if not buf: return
    arr = np.array(buf, dtype=np.uint16)
    filename = VAL_BIN if is_val else TRAIN_BIN
    with open(filename, 'ab') as f:
        f.write(arr.tobytes())

for entry in dataset:
    text = entry['text']
    
    # --- BUG FIX: Allow special tokens inside the text ---
    try:
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        tokens.append(eos_id) 
        
        buffer.extend(tokens)
        token_count += len(tokens)
        pbar.update(len(tokens))

        # Flush when buffer gets big 
        if len(buffer) >= CHUNK_SIZE:
            split_idx = int(len(buffer) * 0.995) 
            
            flush_buffer(buffer[:split_idx], is_val=False)
            flush_buffer(buffer[split_idx:], is_val=True)
            
            buffer = [] 

        if token_count >= TARGET_TOKENS:
            break
            
    except Exception as e:
        print(f"Skipping corrupt entry: {e}")
        continue

# Final Flush
if buffer:
    split_idx = int(len(buffer) * 0.995)
    flush_buffer(buffer[:split_idx], is_val=False)
    flush_buffer(buffer[split_idx:], is_val=True)

pbar.close()
print("-" * 50)
print(f"Complete.")
print(f"Total Tokens: {token_count/1e9:.2f} Billion")
print(f"Train Path:   {TRAIN_BIN} ({os.path.getsize(TRAIN_BIN)/1e9:.2f} GB)")
print("Ready for Walsh.")
