"""
Octonion Dimension Analyzer
---------------------------
Analyzes trained SpinNet models to understand:
1. Sparsity per octonion dimension (which W[0..7] are most used)
2. Weight distribution per dimension
3. Activation patterns for different word types

Usage:
    python tools/analyze_octonion.py --ckpt experiments/out-tinystories-octonion/ckpt.pt
"""

import argparse
import torch
import numpy as np
from collections import defaultdict
import tiktoken

def load_model(ckpt_path, device='cuda'):
    """Load a SpinNet checkpoint."""
    import sys
    import os
    # Add parent directory to path for src imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import SpinNetConfig, SpinNet
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = SpinNetConfig(**checkpoint['model_args'])
    model = SpinNet(config)
    
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[11:]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config

def analyze_weight_sparsity(model):
    """Analyze sparsity of each octonion dimension across all layers."""
    print("\n" + "="*60)
    print("OCTONION WEIGHT SPARSITY ANALYSIS")
    print("="*60)
    
    # Collect all OctonionTernaryLinear weights
    octonion_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.dim() == 3 and module.weight.shape[0] == 8:
            octonion_layers.append((name, module.weight.data))
    
    print(f"\nFound {len(octonion_layers)} octonion layers")
    
    # Per-dimension sparsity across all layers
    dim_sparsity = defaultdict(list)
    dim_pos_ratio = defaultdict(list)
    
    for name, weight in octonion_layers:
        # weight shape: [8, out_o, in_o]
        for i in range(8):
            w = weight[i].float()
            # Quantize to ternary for analysis
            w_q = torch.round(w).clamp(-1, 1)
            
            sparsity = (w_q == 0).float().mean().item()
            pos_ratio = (w_q == 1).float().sum().item() / (w_q != 0).float().sum().item() if (w_q != 0).any() else 0.5
            
            dim_sparsity[i].append(sparsity)
            dim_pos_ratio[i].append(pos_ratio)
    
    # Print summary
    print("\n{:^5} | {:^12} | {:^12} | {:^12}".format(
        "Dim", "Avg Sparsity", "Std", "+1 Ratio"
    ))
    print("-"*50)
    
    dim_names = ['e₀ (real)', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇']
    
    for i in range(8):
        avg_sparsity = np.mean(dim_sparsity[i])
        std_sparsity = np.std(dim_sparsity[i])
        avg_pos = np.mean(dim_pos_ratio[i])
        print(f"{dim_names[i]:^9} | {avg_sparsity:^12.1%} | {std_sparsity:^12.2%} | {avg_pos:^12.1%}")
    
    return dim_sparsity

def analyze_head_mixer(model):
    """Analyze the OctonionHeadMixer weights if present."""
    print("\n" + "="*60)
    print("OCTONION HEAD MIXER ANALYSIS")
    print("="*60)
    
    mixers = []
    for name, module in model.named_modules():
        if type(module).__name__ == 'OctonionHeadMixer':
            mixers.append((name, module))
    
    if not mixers:
        print("No OctonionHeadMixer found (octonion_attention=False?)")
        return
    
    print(f"\nFound {len(mixers)} head mixers")
    
    for name, mixer in mixers:
        print(f"\n{name}:")
        W = mixer.W.data  # [8, head_dim, head_dim]
        beta = mixer.beta.data
        
        print(f"  Beta (scale): mean={beta.mean():.4f}, std={beta.std():.4f}")
        
        # Frobenius norm per dimension
        norms = [torch.norm(W[i]).item() for i in range(8)]
        print(f"  W norms: {['%.2f' % n for n in norms]}")
        
        # Which dimensions have largest weights?
        sorted_dims = sorted(range(8), key=lambda i: norms[i], reverse=True)
        print(f"  Most active dims: {sorted_dims[:3]}")

def analyze_activations(model, device='cuda'):
    """Probe which octonion dimensions activate for different word types."""
    print("\n" + "="*60)
    print("ACTIVATION ANALYSIS BY WORD TYPE")
    print("="*60)
    
    enc = tiktoken.get_encoding('gpt2')
    
    # Test prompts with clear word types
    test_cases = {
        'nouns': ['The cat', 'The dog', 'A house', 'The tree', 'The boy', 'A girl'],
        'verbs': ['She runs', 'He jumps', 'They walk', 'It falls', 'We play', 'I think'],
        'adjectives': ['very big', 'so small', 'too fast', 'quite slow', 'really happy'],
        'pronouns': ['He said', 'She saw', 'They went', 'We have', 'I am', 'It was'],
        'prepositions': ['in the', 'on the', 'at the', 'to the', 'from the', 'with a'],
        'numbers': ['one day', 'two cats', 'three times', 'first time', 'second one'],
        'emotions': ['was happy', 'felt sad', 'got angry', 'became scared', 'very excited'],
        'actions': ['ran away', 'jumped up', 'fell down', 'picked up', 'put down'],
        'story_start': ['Once upon', 'One day', 'There was', 'Long ago', 'A little'],
        'dialogue': ['he said', 'she asked', 'they replied', 'mom said', 'friend asked'],
    }
    
    # Hook to capture activations after first layer
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hook on first transformer layer output
    if hasattr(model, 'layers') and len(model.layers) > 0:
        model.layers[0].register_forward_hook(hook_fn('layer0'))
    
    dim_activations = {cat: [[] for _ in range(8)] for cat in test_cases}
    
    with torch.no_grad():
        for category, phrases in test_cases.items():
            for phrase in phrases:
                tokens = enc.encode(phrase)
                x = torch.tensor(tokens, device=device)[None, ...]
                
                # Forward pass
                try:
                    _ = model(x)
                except:
                    continue
                
                if 'layer0' in activations:
                    act = activations['layer0']  # [1, seq_len, n_embd]
                    # Split into 8 octonion parts
                    n_embd = act.shape[-1]
                    parts = act.split(n_embd // 8, dim=-1)
                    
                    # Record mean activation magnitude per part
                    for i, part in enumerate(parts):
                        dim_activations[category][i].append(
                            part.abs().mean().item()
                        )
    
    # Print results
    print("\n{:^12} | " + " | ".join([f"e{i:^4}" for i in range(8)]))
    print("-"*70)
    
    for category in test_cases:
        means = [np.mean(dim_activations[category][i]) if dim_activations[category][i] else 0 
                 for i in range(8)]
        row = f"{category:^12} | " + " | ".join([f"{m:^6.3f}" for m in means])
        print(row)
    
    # Highlight which dimensions are most active for each category
    print("\nMost active dimensions:")
    for category in test_cases:
        means = [np.mean(dim_activations[category][i]) if dim_activations[category][i] else 0 
                 for i in range(8)]
        top3 = sorted(range(8), key=lambda i: means[i], reverse=True)[:3]
        print(f"  {category}: e{top3[0]}, e{top3[1]}, e{top3[2]}")

def main():
    parser = argparse.ArgumentParser(description='Analyze octonion dimensions in SpinNet')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    print(f"Loading model from {args.ckpt}...")
    model, config = load_model(args.ckpt, args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    print(f"Octonion attention: {getattr(config, 'octonion_attention', False)}")
    
    # Run analyses
    analyze_weight_sparsity(model)
    analyze_head_mixer(model)
    analyze_activations(model, args.device)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
