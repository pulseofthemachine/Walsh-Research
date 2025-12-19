"""
SpinNet TinyShakespeare Text Generator
--------------------------------------
Generates Shakespeare-style text using the trained character-level model.
"""
import os
import pickle
import torch
from src.model import SpinNetConfig, SpinNet

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ckpt_path = 'experiments/out-tinyshakespeare/ckpt.pt'
meta_path = 'data/tinyshakespeare/meta.pkl'
start_prompt = "To be or not to be"
num_samples = 3
max_new_tokens = 500
temperature = 0.8
top_k = None  # Character-level doesn't need top_k usually

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# LOAD TOKENIZER (Character-level)
# -----------------------------------------------------------------------------
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
print(f"Loading SpinNet from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = SpinNetConfig(**checkpoint['model_args'])
model = SpinNet(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# -----------------------------------------------------------------------------
# GENERATE
# -----------------------------------------------------------------------------
start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print("=" * 60)
print(f"PROMPT: {start_prompt!r}")
print("=" * 60)

with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated = decode(y[0].tolist())
            print(f"\n--- SAMPLE {k+1} ---")
            print(generated)
            print()
