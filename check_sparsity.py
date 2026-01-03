import torch
import numpy as np
import os

def check_sparsity(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']
    
    total_ternary_elements = 0
    total_zeros = 0
    
    # We look for weights that are likely part of a Ternary Layer
    # Based on the architecture, these are the .weight parameters in attention and mlp
    for name, param in state_dict.items():
        if '.weight' in name and any(x in name for x in ['attention', 'feed_forward', 'context_mlp', 'router']):
            # Skip normalization weights which are usually 1D or names like 'norm'
            if 'norm' in name or param.ndim < 2:
                continue
                
            w = param.float()
            gamma = w.abs().mean() + 1e-8
            
            # Binary mask of zeros after quantization
            zeros_mask = (torch.round(w / gamma).clamp(-1, 1) == 0)
            
            n_elems = param.numel()
            n_zeros = zeros_mask.sum().item()
            
            sparsity = n_zeros / n_elems
            print(f"Layer {name:50} | Sparsity: {sparsity:6.2%} | Shape: {list(param.shape)}")
            
            total_ternary_elements += n_elems
            total_zeros += n_zeros
            
    if total_ternary_elements > 0:
        overall_sparsity = total_zeros / total_ternary_elements
        print("-" * 80)
        print(f"OVERALL MODEL SPARSITY (Ternary Zeros): {overall_sparsity:.2%}")
        print(f"Total Ternary Parameters: {total_ternary_elements:,}")
        print(f"Total Zero Parameters   : {total_zeros:,}")
    else:
        print("No ternary layers found.")

if __name__ == "__main__":
    ckpt_path = 'experiments/out-wikipedia-512-levels-4-5/ckpt.pt'
    if os.path.exists(ckpt_path):
        check_sparsity(ckpt_path)
    else:
        print(f"Checkpoint {ckpt_path} not found.")
