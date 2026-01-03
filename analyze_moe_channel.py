#!/usr/bin/env python3
"""
Analyze MoE Expert Routing and Channel Modulation in Walsh Models
-----------------------------------------------------------------
Visualizes:
1. Expert routing patterns per word/sentence category
2. Channel modulation scales per context
3. Expert specialization (function vs content words)
4. Combined MoE + ChannelMod analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import tiktoken
from pathlib import Path

# Test words for analysis
WORD_CATEGORIES = {
    "Function Words": ["the", "a", "an", "in", "on", "at", "to", "for", "of", "with"],
    "Content Nouns": ["scientist", "particle", "theory", "mountain", "computer", "ocean"],
    "Action Verbs": ["run", "jump", "discover", "create", "build", "write"],
    "Adjectives": ["big", "small", "fast", "slow", "bright", "dark"],
    "Numbers": ["one", "two", "three", "hundred", "million"],
    "Technical": ["algorithm", "quantum", "electron", "database", "protein"],
}

SENTENCE_TYPES = {
    "Scientific": [
        "The electron orbits the nucleus.",
        "Quantum mechanics describes particle behavior.",
        "The algorithm has O(n) complexity.",
    ],
    "Narrative": [
        "The hero journeyed through the forest.",
        "She opened the ancient door slowly.",
        "The storm raged through the night.",
    ],
    "Technical": [
        "This function returns a boolean value.",
        "The database stores user information.",
        "The API endpoint accepts JSON.",
    ],
    "Conversational": [
        "How are you doing today?",
        "What do you think about this?",
        "I really appreciate your help.",
    ],
}


def load_model(ckpt_path, device='cuda'):
    """Load mHCWalsh model from checkpoint."""
    from src.model import Walsh, WalshConfig, mHCWalsh
    
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt.get('model', ckpt)
    
    # Strip _orig_mod prefix if present
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_state[k[10:]] = v
        else:
            clean_state[k] = v
    
    # Get config
    model_args = ckpt.get('model_args', {})
    config_dict = ckpt.get('config', {})
    
    valid_fields = ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 
                    'bias', 'dropout', 'head_mixing', 'algebra', 'hash_embeddings',
                    'use_moe', 'moe_threshold', 'moe_min_experts', 'moe_max_experts',
                    'use_channel_mod']
    arch_config = {k: v for k, v in model_args.items() if k in valid_fields}
    
    if 'vocab_size' not in arch_config:
        arch_config['vocab_size'] = 50304
    
    config = WalshConfig(**arch_config)
    
    use_mhc = config_dict.get('use_mhc', False)
    n_streams = config_dict.get('n_streams', 4)
    
    if use_mhc:
        print(f"Loading mHCWalsh with {n_streams} streams...")
        model = mHCWalsh(config, n_streams=n_streams)
    else:
        model = Walsh(config)
    
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    
    # Print architecture info
    print(f"Architecture: {config.n_layer}L × {config.n_head}H × {config.n_embd}D")
    print(f"MoE enabled: {getattr(config, 'use_moe', False)}")
    print(f"ChannelMod enabled: {getattr(config, 'use_channel_mod', False)}")
    
    return model, config


def analyze_moe_routing(model, enc, text, device='cuda'):
    """
    Get MoE routing decisions for each token in text.
    Returns: dict with expert_masks, expert_weights per layer
    """
    tokens = enc.encode(text)
    token_tensor = torch.tensor([tokens], device=device)
    
    layer_routing = []
    
    with torch.no_grad():
        # We need to hook into the MoE layers
        # Store routing info as we go
        routing_info = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(module, '_last_expert_mask'):
                    mask = module._last_expert_mask
                    if mask is not None:
                        routing_info.append({
                            'layer': layer_idx,
                            'mask': mask.detach().cpu().numpy(),
                        })
            return hook
        
        # Register hooks on MoE layers
        hooks = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, '_last_expert_mask'):
                h = layer.feed_forward.register_forward_hook(make_hook(i))
                hooks.append(h)
        
        # Forward pass
        _ = model(token_tensor)
        
        # Clean up hooks
        for h in hooks:
            h.remove()
    
    return routing_info, tokens


def analyze_channel_mod(model, enc, text, device='cuda'):
    """
    Get channel modulation scales for text.
    Returns: list of dicts with scale/shift info per layer
    """
    tokens = enc.encode(text)
    token_tensor = torch.tensor([tokens], device=device)
    
    # Forward pass to populate stats
    with torch.no_grad():
        _ = model(token_tensor)
    
    # Now collect stats from all channel_mod layers
    channel_info = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'channel_mod'):
            stats = layer.channel_mod.get_modulation_stats()
            channel_info.append({
                'layer': i,
                'scale_mean': stats['scale_mean'],
                'scale_std': stats['scale_std'],
                'shift_mean': stats['shift_mean'],
            })
    
    return channel_info


def plot_expert_heatmap(results, output_path, title="Expert Routing by Category"):
    """Plot heatmap of expert usage by word category."""
    categories = list(results.keys())
    n_experts = results[categories[0]].shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrix = np.array([results[cat] for cat in categories])
    
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f'E{i}' for i in range(n_experts)])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Word Category')
    ax.set_title(title)
    
    plt.colorbar(im, label='Activation Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_channel_scales(results, output_path):
    """Plot channel modulation scales by context type."""
    categories = list(results.keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Scale means
    for cat in categories:
        axes[0].plot(results[cat]['scale_means'], label=cat, marker='o')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Scale Mean')
    axes[0].set_title('Channel Modulation Scale by Layer and Context')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Scale stds
    for cat in categories:
        axes[1].plot(results[cat]['scale_stds'], label=cat, marker='o')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Scale Std')
    axes[1].set_title('Channel Modulation Variance by Layer and Context')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_word_categories(model, enc, device, output_dir):
    """Analyze MoE routing for different word categories."""
    print("\n=== Analyzing Word Categories ===")
    
    category_expert_usage = {}
    
    for category, words in WORD_CATEGORIES.items():
        expert_counts = None
        n_words = 0
        
        for word in words:
            # Try with space prefix
            text = " " + word
            routing_info, tokens = analyze_moe_routing(model, enc, text, device)
            
            if routing_info:
                # Get routing for the last token (the word itself)
                for info in routing_info:
                    mask = info['mask']  # [B, T, n_experts]
                    if mask.shape[1] > 0:
                        # Use last token
                        word_mask = mask[0, -1, :]  # [n_experts]
                        if expert_counts is None:
                            expert_counts = np.zeros_like(word_mask)
                        expert_counts += word_mask
                        n_words += 1
        
        if n_words > 0 and expert_counts is not None:
            category_expert_usage[category] = expert_counts / n_words
            print(f"  {category}: analyzed {n_words} words")
    
    if category_expert_usage:
        plot_expert_heatmap(
            category_expert_usage, 
            output_dir / 'moe_word_categories.png',
            "MoE Expert Activation by Word Category"
        )
    
    return category_expert_usage


def analyze_sentence_contexts(model, enc, device, output_dir):
    """Analyze channel modulation for different sentence types."""
    print("\n=== Analyzing Sentence Contexts ===")
    
    context_channel_info = {}
    
    for context_type, sentences in SENTENCE_TYPES.items():
        scale_means_per_layer = defaultdict(list)
        scale_stds_per_layer = defaultdict(list)
        
        for sentence in sentences:
            channel_info = analyze_channel_mod(model, enc, sentence, device)
            
            for info in channel_info:
                layer = info['layer']
                scale_means_per_layer[layer].append(info['scale_mean'])
                scale_stds_per_layer[layer].append(info['scale_std'])
        
        if scale_means_per_layer:
            n_layers = max(scale_means_per_layer.keys()) + 1
            context_channel_info[context_type] = {
                'scale_means': [np.mean(scale_means_per_layer[i]) for i in range(n_layers)],
                'scale_stds': [np.mean(scale_stds_per_layer[i]) for i in range(n_layers)],
            }
            print(f"  {context_type}: analyzed {len(sentences)} sentences")
    
    if context_channel_info:
        plot_channel_scales(context_channel_info, output_dir / 'channel_mod_contexts.png')
    
    return context_channel_info


def analyze_expert_specialization(model, enc, device, output_dir):
    """Analyze what each expert specializes in (function vs content)."""
    print("\n=== Analyzing Expert Specialization ===")
    
    # Broader word lists
    function_words = ["the", "a", "an", "in", "on", "at", "to", "for", "of", "with", 
                      "by", "as", "is", "was", "are", "were", "be", "been", "being",
                      "and", "or", "but", "if", "when", "where", "how", "why", "that"]
    
    content_words = ["scientist", "mountain", "discover", "beautiful", "algorithm",
                     "computer", "theory", "ocean", "create", "ancient", "power",
                     "journey", "analysis", "particle", "quantum", "energy", "story"]
    
    # Get expert usage for each category
    function_usage = np.zeros(16)  # Assume 16 experts for 512D
    content_usage = np.zeros(16)
    function_count = 0
    content_count = 0
    
    for word in function_words:
        routing_info, _ = analyze_moe_routing(model, enc, " " + word, device)
        if routing_info:
            for info in routing_info:
                mask = info['mask']
                if mask.shape[1] > 0:
                    function_usage += mask[0, -1, :].flatten()[:16]
                    function_count += 1
    
    for word in content_words:
        routing_info, _ = analyze_moe_routing(model, enc, " " + word, device)
        if routing_info:
            for info in routing_info:
                mask = info['mask']
                if mask.shape[1] > 0:
                    content_usage += mask[0, -1, :].flatten()[:16]
                    content_count += 1
    
    if function_count > 0:
        function_usage /= function_count
    if content_count > 0:
        content_usage /= content_count
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(function_usage))
    width = 0.35
    
    ax.bar(x - width/2, function_usage, width, label='Function Words', color='#377eb8')
    ax.bar(x + width/2, content_usage, width, label='Content Words', color='#e41a1c')
    
    ax.set_xlabel('Expert')
    ax.set_ylabel('Activation Frequency')
    ax.set_title('Expert Specialization: Function vs Content Words')
    ax.set_xticks(x)
    ax.set_xticklabels([f'E{i}' for i in range(len(function_usage))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'expert_specialization.png'}")
    
    # Print summary
    print(f"\n  Function word experts (top 3): {np.argsort(function_usage)[-3:][::-1]}")
    print(f"  Content word experts (top 3): {np.argsort(content_usage)[-3:][::-1]}")
    
    return {'function': function_usage, 'content': content_usage}


def main():
    parser = argparse.ArgumentParser(description='Analyze MoE and ChannelMod in Walsh models')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='analysis_moe', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model, config = load_model(args.ckpt, args.device)
    
    # Get tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # Check if MoE/ChannelMod are enabled
    use_moe = getattr(config, 'use_moe', False)
    use_channel_mod = getattr(config, 'use_channel_mod', False)
    
    if use_moe:
        # Analyze MoE routing
        analyze_word_categories(model, enc, args.device, output_dir)
        analyze_expert_specialization(model, enc, args.device, output_dir)
    else:
        print("MoE not enabled in this model")
    
    if use_channel_mod:
        # Analyze channel modulation
        analyze_sentence_contexts(model, enc, args.device, output_dir)
    else:
        print("ChannelMod not enabled in this model")
    
    print(f"\n✅ Analysis complete! Results in {output_dir}/")


if __name__ == '__main__':
    main()
