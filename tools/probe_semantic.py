"""
SpinNet Semantic Probe
----------------------
Measures the geometric distance (Cosine Similarity) between concept vectors.
Verifies if the model is grouping related ideas in the Octonion space.
"""
import sys
import os
import torch
import torch.nn.functional as F
import tiktoken

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.model import SpinNetConfig, SpinNet

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ckpt_path = os.path.join(parent_dir, 'experiments', 'scholar_pilot', 'ckpt.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pairs to test
pairs = [
    # Geography
    ("Paris", "France"),
    ("London", "England"),
    
    # Science
    ("physics", "energy"),
    ("code", "python"),
    ("star", "galaxy"),
    
    # Grammar
    ("is", "are"),
    ("he", "she"),
    
    # Controls (Should be far apart)
    ("Paris", "banana"),
    ("physics", "puppy"),
    ("code", "cloud") 
]

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------
if not os.path.exists(ckpt_path):
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    sys.exit(1)

print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = SpinNetConfig(**checkpoint['model_args'])
model = SpinNet(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print(f"{'WORD A':<15} | {'WORD B':<15} | {'SIM':<8} | {'STATUS'}")
print("-" * 60)

def get_vec(word):
    # Take first token embedding
    idx = enc.encode(word)[0]
    return model.tok_embeddings.weight[idx].detach()

for w1, w2 in pairs:
    v1 = get_vec(w1)
    v2 = get_vec(w2)
    
    # Cosine Similarity
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    
    if sim > 0.3: status = "LINKED"
    elif sim > 0.2: status = "WEAK"
    else: status = "DISTANT"
    
    # Control Check
    if w2 in ["banana", "puppy"] and sim < 0.2:
        status = "PASS"
    elif w2 in ["banana", "puppy"] and sim >= 0.2:
        status = "FAIL (Confused)"

    print(f"{w1:<15} | {w2:<15} | {sim:.4f}   | {status}")

print("-" * 60)