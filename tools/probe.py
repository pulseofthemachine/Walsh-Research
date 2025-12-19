"""
SpinNet Deep Probe v2.0: The MRI Scan
---------------------------------------------
1. Embedding Stability
2. Octonion Orthogonality
3. PER-LAYER Ghost Weight Analysis (The Critical Update)
"""
import sys
import os
import torch
import numpy as np

# --- PATH SETUP ---
# Adjust this to point to your project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.model import SpinNetConfig, SpinNet

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# CHECKPOINT: Update this to point to the active run
experiment_dir = 'out-crypto'    # The new 124M run

ckpt_path = os.path.join(parent_dir, 'experiments', experiment_dir, 'ckpt.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# SYSTEM BOOT
# -----------------------------------------------------------------------------
print(f"\n[SYSTEM] Initializing Probe on: {device.upper()}")
if not os.path.exists(ckpt_path):
    print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
    print("         Did the run save the first checkpoint yet?")
    sys.exit(1)

print(f"[SYSTEM] Loading Checkpoint: {experiment_dir}...")
try:
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = SpinNetConfig(**checkpoint['model_args'])
    model = SpinNet(gptconf)
    
    # Unwanted prefix handling (if compiled)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[SYSTEM] Model Loaded. Depth: {gptconf.n_layer} Layers. Width: {gptconf.n_embd}.")
except Exception as e:
    print(f"[FATAL] Model Load Failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# SCAN 1: THE SHELL (Embeddings)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print(">>> SCAN 1: EMBEDDING HEALTH (THE SHELL) <<<")
print("="*60)

embeddings = model.tok_embeddings.weight.detach()
norms = torch.norm(embeddings, p=2, dim=1)

print(f"Stats -> Mean: {norms.mean().item():.3f} | Std: {norms.std().item():.3f}")
print(f"Range -> Min:  {norms.min().item():.3f} | Max: {norms.max().item():.3f}")

if norms.max().item() > 20.0:
    print("[WARN]  High Energy Embeddings. Potential instability.")
else:
    print("[PASS]  Embeddings are stable.")

# -----------------------------------------------------------------------------
# SCAN 2: THE BRAIN (Per-Layer Sparsity)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print(f">>> SCAN 2: LAYER-WISE SPARSITY (THE PULSE) <<<")
print("Threshold: +/- 0.5 (Standard Ternary)")
print("="*60)
print(f"{'Lyr':<4} | {'Mean':<7} | {'Std':<7} | {'Sparsity':<9} | {'Status'}")
print("-" * 60)

global_active = 0
global_total = 0
death_zone_threshold = 0.5

for i, layer in enumerate(model.layers):
    # We aggregate all weights in the layer (Attn + MLP)
    layer_params = []
    for name, param in layer.named_parameters():
        if 'weight' in name and param.dim() > 1:
            layer_params.append(param.detach().flatten())
    
    if not layer_params:
        print(f"{i:<4} | [NO WEIGHTS FOUND]")
        continue

    # Flatten entire layer into one massive vector
    w_flat = torch.cat(layer_params)
    
    # Calculate Vital Signs
    mean = w_flat.mean().item()
    std = w_flat.std().item()
    total = w_flat.numel()
    
    # The Pruning Logic
    zeros = ((w_flat > -death_zone_threshold) & (w_flat < death_zone_threshold)).sum().item()
    sparsity = (zeros / total) * 100
    
    global_active += (total - zeros)
    global_total += total
    
    # Diagnosis
    status = "OK"
    if sparsity > 99.5: status = "DEAD (!!)" # Signal blockage
    elif sparsity < 50.0: status = "DENSE"   # Failing to prune
    elif sparsity > 95.0: status = "ANEMIC"  # Too sparse?
    elif std < 0.001:     status = "FROZEN"  # No gradient movement
    
    print(f"{i:<4} | {mean:+.4f} | {std:.4f}  | {sparsity:.2f}%    | {status}")

print("-" * 60)
global_sparsity = (1 - (global_active / global_total)) * 100
print(f"GLOBAL BRAIN SPARSITY: {global_sparsity:.2f}%")

# -----------------------------------------------------------------------------
# SCAN 3: GEOMETRIC INTEGRITY (Orthogonality Check)
# -----------------------------------------------------------------------------
# We check the middle layer (Layer 12) to see if geometry holds deep in the stack
target_layer = len(model.layers) // 2
print("\n" + "="*60)
print(f">>> SCAN 3: OCTONION GEOMETRY (LAYER {target_layer}) <<<")
print("="*60)

weights = model.layers[target_layer].attention.wq.weight.detach()
# Assuming standard spinnet layout [8, Out, In] or similar
# If shape is [Out, In], we need to infer octonion chunks
if len(weights.shape) == 3:
    # It's already in octonion format [8, A, B]
    n_groups = weights.shape[2]
    w_flat = weights.permute(2, 0, 1).reshape(n_groups, -1).float()
else:
    # It's flattened [Out, In]. 
    # NOTE: This depends on your OctonionLinear implementation. 
    # If it's stored flat, we skip this scan or need reshaping logic.
    print("[INFO] Weights are flat. Skipping deep orthogonality scan to avoid shape errors.")
    w_flat = None

if w_flat is not None:
    # Normalize and Correlate
    w_norm = w_flat / (w_flat.norm(dim=1, keepdim=True) + 1e-8)
    corr_matrix = torch.mm(w_norm, w_norm.t())
    
    # Mask diagonal
    mask = torch.eye(corr_matrix.shape[0], device=device).bool()
    corr_matrix.masked_fill_(mask, 0)
    
    max_corr = corr_matrix.max().item()
    mean_abs_corr = corr_matrix.abs().mean().item()
    
    print(f"Max Cross-Correlation:  {max_corr:.4f}")
    print(f"Mean Abs Correlation:   {mean_abs_corr:.4f}")
    
    if mean_abs_corr < 0.1:
        print("[PASS] Geometry is perfectly Orthogonal.")
    else:
        print("[WARN] Feature leakage detected.")

print("\n[SYSTEM] DIAGNOSTIC COMPLETE.\n")
