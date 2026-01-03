#!/usr/bin/env python3
"""
Comprehensive Semantic Analysis for Walsh/mHC Models
-----------------------------------------------------
Analyzes channel specialization, semantic structure, and mHC stream dynamics.

Analyses:
1. Analogy Channel Tracking (king - man + woman = queen)
2. Antonym/Synonym Channel Divergence
3. Compositional Semantics (adjective + noun)
4. Channel Trajectories Through Context
5. Part-of-Speech Channel Gating
6. Semantic Field Clustering
7. Channel Cross-Correlation Matrix
8. Temporal Channel Dynamics (per-layer)
9. mHC Stream Analysis (doubly-stochastic mixing)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import argparse
import tiktoken
import os
from collections import defaultdict

# -----------------------------------------------------------------------------
# SEMANTIC TEST SETS
# -----------------------------------------------------------------------------

ANALOGIES = [
    # (a, b, c, expected_d) - a is to b as c is to d
    ("king", "man", "queen", "woman"),
    ("man", "woman", "boy", "girl"),
    ("big", "bigger", "small", "smaller"),
    ("France", "Paris", "Germany", "Berlin"),
    ("walk", "walked", "run", "ran"),
]

ANTONYM_PAIRS = [
    ("big", "small"), ("hot", "cold"), ("fast", "slow"),
    ("good", "bad"), ("happy", "sad"), ("light", "dark"),
    ("old", "young"), ("high", "low"), ("rich", "poor"),
    ("love", "hate"), ("open", "close"), ("up", "down"),
]

SYNONYM_GROUPS = [
    ["big", "large", "huge", "enormous"],
    ["small", "tiny", "little", "minute"],
    ["fast", "quick", "rapid", "swift"],
    ["happy", "glad", "joyful", "pleased"],
    ["sad", "unhappy", "sorrowful", "melancholy"],
]

COMPOSITIONAL_TESTS = [
    # (adjective, noun, combined_meaning)
    ("red", "car", "color+object"),
    ("fast", "car", "speed+object"),
    ("big", "house", "size+object"),
    ("old", "man", "age+person"),
    ("new", "idea", "time+abstract"),
]

SEMANTIC_FIELDS = {
    "Emotions": ["happy", "sad", "angry", "afraid", "love", "hate", "joy", "fear"],
    "Body Parts": ["hand", "foot", "head", "heart", "eye", "arm", "leg", "brain"],
    "Animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep"],
    "Colors": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"],
    "Numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    "Family": ["mother", "father", "sister", "brother", "son", "daughter", "wife", "husband"],
    "Weather": ["rain", "sun", "cloud", "wind", "snow", "storm", "fog", "thunder"],
    "Food": ["bread", "meat", "fruit", "water", "milk", "rice", "fish", "egg"],
}

CONTEXT_SENTENCES = [
    "The cat sat on the mat",
    "She walked to the store",
    "The sun rises in the east",
    "He wrote a beautiful poem",
]


def load_model(ckpt_path, device='cuda'):
    """Load Walsh or mHCWalsh model from checkpoint."""
    from src.model import Walsh, WalshConfig, mHCWalsh
    
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt.get('model', ckpt)
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_state[k[10:]] = v
        else:
            clean_state[k] = v
    
    model_args = ckpt.get('model_args', {})
    config_dict = ckpt.get('config', {})
    
    # Updated fields to include MoE and ChannelMod
    valid_fields = [
        'n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 
        'bias', 'dropout', 'head_mixing', 'algebra', 'hash_embeddings',
        'use_moe', 'moe_threshold', 'moe_min_experts', 'moe_max_experts',
        'use_channel_mod'
    ]
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
    
    return model, config, use_mhc, n_streams


def get_embedding(model, enc, word, device='cuda'):
    """Get the final hidden state for a single word."""
    tokens = enc.encode(" " + word)
    if len(tokens) > 1:
        tokens = enc.encode(word)
    
    if len(tokens) != 1:
        return None
    
    token_tensor = torch.tensor([[tokens[0]]], device=device)
    
    with torch.no_grad():
        logits, loss, hidden = model(token_tensor, return_hidden=True)
        return hidden[0, 0].cpu().numpy()


def get_layer_embeddings(model, enc, word, device='cuda'):
    """Get hidden states at each layer for a word."""
    tokens = enc.encode(" " + word)
    if len(tokens) > 1:
        tokens = enc.encode(word)
    
    if len(tokens) != 1:
        return None
    
    token_tensor = torch.tensor([[tokens[0]]], device=device)
    layer_states = []
    
    with torch.no_grad():
        # Get embeddings
        h = model.tok_embeddings(token_tensor)
        
        # For mHC, need to expand to streams
        if hasattr(model, 'stream_proj'):
            h = model.stream_proj(h)
            h = h.view(1, 1, model.n_streams, model.config.n_embd)
        
        freqs_cis = model.freqs_cis[:1]
        
        for i, block in enumerate(model.layers):
            h = block(h, freqs_cis)
            
            # Merge streams if mHC
            if hasattr(model, 'stream_merge'):
                h_flat = h.reshape(1, 1, -1)
                h_merged = model.stream_merge(h_flat)
                layer_states.append(h_merged[0, 0].cpu().numpy())
            else:
                layer_states.append(h[0, 0].cpu().numpy())
    
    return layer_states


def get_sequence_embeddings(model, enc, sentence, device='cuda'):
    """Get hidden states for each token in a sentence."""
    tokens = enc.encode(sentence)
    token_states = []
    
    for i in range(len(tokens)):
        prefix = tokens[:i+1]
        token_tensor = torch.tensor([prefix], device=device)
        
        with torch.no_grad():
            logits, loss, hidden = model(token_tensor, return_hidden=True)
            token_states.append(hidden[0, -1].cpu().numpy())
    
    return token_states, [enc.decode([t]) for t in tokens]


def analyze_channels(state, hadamard_dim=32):
    """Reshape hidden state into Hadamard channel view."""
    n_blocks = len(state) // hadamard_dim
    return state.reshape(n_blocks, hadamard_dim).mean(axis=0)


# -----------------------------------------------------------------------------
# ANALYSIS 1: Analogy Channel Tracking
# -----------------------------------------------------------------------------

def analyze_analogies(model, enc, device, output_dir):
    """Track channel activations during analogy operations."""
    print("\n=== Analysis 1: Analogy Channel Tracking ===")
    
    fig, axes = plt.subplots(len(ANALOGIES), 4, figsize=(16, 3*len(ANALOGIES)))
    
    for i, (a, b, c, d) in enumerate(ANALOGIES):
        emb_a = get_embedding(model, enc, a, device)
        emb_b = get_embedding(model, enc, b, device)
        emb_c = get_embedding(model, enc, c, device)
        emb_d = get_embedding(model, enc, d, device)
        
        if any(e is None for e in [emb_a, emb_b, emb_c, emb_d]):
            continue
        
        # Compute analogy: a - b + c ≈ d
        analogy_result = emb_a - emb_b + emb_c
        
        # Channel patterns
        ch_a = analyze_channels(emb_a)
        ch_diff = analyze_channels(emb_a - emb_b)
        ch_result = analyze_channels(analogy_result)
        ch_d = analyze_channels(emb_d)
        
        axes[i, 0].bar(range(32), ch_a, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'"{a}"')
        axes[i, 0].set_ylabel(f'{a}:{b}::{c}:{d}')
        
        axes[i, 1].bar(range(32), ch_diff, color='red', alpha=0.7)
        axes[i, 1].set_title(f'"{a}" - "{b}"')
        
        axes[i, 2].bar(range(32), ch_result, color='green', alpha=0.7)
        axes[i, 2].set_title(f'({a} - {b}) + {c}')
        
        axes[i, 3].bar(range(32), ch_d, color='purple', alpha=0.7)
        axes[i, 3].set_title(f'"{d}" (target)')
        
        # Compute similarity
        sim = np.dot(ch_result, ch_d) / (np.linalg.norm(ch_result) * np.linalg.norm(ch_d))
        print(f"  {a}:{b}::{c}:{d} - Similarity: {sim:.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analogy_channels.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/analogy_channels.png")


# -----------------------------------------------------------------------------
# ANALYSIS 2: Antonym/Synonym Divergence
# -----------------------------------------------------------------------------

def analyze_antonyms_synonyms(model, enc, device, output_dir):
    """Compare channel patterns for antonyms vs synonyms."""
    print("\n=== Analysis 2: Antonym/Synonym Channel Divergence ===")
    
    # Antonym similarities
    antonym_sims = []
    for w1, w2 in ANTONYM_PAIRS:
        e1 = get_embedding(model, enc, w1, device)
        e2 = get_embedding(model, enc, w2, device)
        if e1 is not None and e2 is not None:
            ch1, ch2 = analyze_channels(e1), analyze_channels(e2)
            sim = np.dot(ch1, ch2) / (np.linalg.norm(ch1) * np.linalg.norm(ch2))
            antonym_sims.append(sim)
    
    # Synonym similarities
    synonym_sims = []
    for group in SYNONYM_GROUPS:
        embeddings = [get_embedding(model, enc, w, device) for w in group]
        embeddings = [e for e in embeddings if e is not None]
        if len(embeddings) >= 2:
            channels = [analyze_channels(e) for e in embeddings]
            for i in range(len(channels)):
                for j in range(i+1, len(channels)):
                    sim = np.dot(channels[i], channels[j]) / (np.linalg.norm(channels[i]) * np.linalg.norm(channels[j]))
                    synonym_sims.append(sim)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(antonym_sims, bins=20, alpha=0.7, label=f'Antonyms (mean={np.mean(antonym_sims):.3f})', color='red')
    ax.hist(synonym_sims, bins=20, alpha=0.7, label=f'Synonyms (mean={np.mean(synonym_sims):.3f})', color='green')
    ax.set_xlabel('Channel Pattern Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Antonym vs Synonym Channel Similarity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'antonym_synonym.png'), dpi=150)
    plt.close()
    
    print(f"  Antonym mean similarity: {np.mean(antonym_sims):.3f}")
    print(f"  Synonym mean similarity: {np.mean(synonym_sims):.3f}")
    print(f"  Saved: {output_dir}/antonym_synonym.png")


# -----------------------------------------------------------------------------
# ANALYSIS 3: Compositional Semantics
# -----------------------------------------------------------------------------

def analyze_composition(model, enc, device, output_dir):
    """Analyze how adjective + noun channels combine."""
    print("\n=== Analysis 3: Compositional Semantics ===")
    
    fig, axes = plt.subplots(len(COMPOSITIONAL_TESTS), 3, figsize=(14, 3*len(COMPOSITIONAL_TESTS)))
    
    for i, (adj, noun, desc) in enumerate(COMPOSITIONAL_TESTS):
        e_adj = get_embedding(model, enc, adj, device)
        e_noun = get_embedding(model, enc, noun, device)
        e_combined = get_embedding(model, enc, adj + noun, device)  # Try combined
        
        if e_adj is None or e_noun is None:
            continue
        
        ch_adj = analyze_channels(e_adj)
        ch_noun = analyze_channels(e_noun)
        ch_sum = analyze_channels(e_adj + e_noun)
        
        axes[i, 0].bar(range(32), ch_adj, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'"{adj}"')
        axes[i, 0].set_ylabel(f'{adj} + {noun}')
        
        axes[i, 1].bar(range(32), ch_noun, color='orange', alpha=0.7)
        axes[i, 1].set_title(f'"{noun}"')
        
        axes[i, 2].bar(range(32), ch_sum, color='green', alpha=0.7)
        axes[i, 2].set_title(f'Sum: "{adj}" + "{noun}"')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'composition.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/composition.png")


# -----------------------------------------------------------------------------
# ANALYSIS 4: Context Trajectories
# -----------------------------------------------------------------------------

def analyze_trajectories(model, enc, device, output_dir):
    """Plot channel trajectories through sentence context."""
    print("\n=== Analysis 4: Channel Trajectories Through Context ===")
    
    fig, axes = plt.subplots(len(CONTEXT_SENTENCES), 2, figsize=(16, 4*len(CONTEXT_SENTENCES)))
    
    for i, sentence in enumerate(CONTEXT_SENTENCES):
        states, tokens = get_sequence_embeddings(model, enc, sentence, device)
        
        # Channel activations over time
        channel_traj = np.array([analyze_channels(s) for s in states])
        
        # Heatmap
        im = axes[i, 0].imshow(channel_traj.T, aspect='auto', cmap='viridis')
        axes[i, 0].set_xlabel('Token position')
        axes[i, 0].set_ylabel('Channel')
        axes[i, 0].set_title(f'Channel activations: "{sentence[:30]}..."')
        axes[i, 0].set_xticks(range(len(tokens)))
        axes[i, 0].set_xticklabels([t.strip() for t in tokens], rotation=45, ha='right')
        plt.colorbar(im, ax=axes[i, 0])
        
        # PCA trajectory
        if len(states) >= 2:
            pca = PCA(n_components=2)
            states_2d = pca.fit_transform(states)
            
            axes[i, 1].plot(states_2d[:, 0], states_2d[:, 1], 'b-o', markersize=8)
            for j, token in enumerate(tokens):
                axes[i, 1].annotate(token.strip(), states_2d[j], fontsize=8)
            axes[i, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            axes[i, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            axes[i, 1].set_title('Trajectory in embedding space')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectories.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/trajectories.png")


# -----------------------------------------------------------------------------
# ANALYSIS 5: Semantic Field Clustering
# -----------------------------------------------------------------------------

def analyze_semantic_fields(model, enc, device, output_dir):
    """Cluster words by semantic field and analyze channel patterns."""
    print("\n=== Analysis 5: Semantic Field Clustering ===")
    
    # Collect embeddings by field
    field_embeddings = {}
    for field, words in SEMANTIC_FIELDS.items():
        embeddings = []
        for word in words:
            e = get_embedding(model, enc, word, device)
            if e is not None:
                embeddings.append(e)
        if embeddings:
            field_embeddings[field] = np.array(embeddings)
    
    # PCA all embeddings
    all_embeddings = []
    all_labels = []
    for field, embs in field_embeddings.items():
        all_embeddings.extend(embs)
        all_labels.extend([field] * len(embs))
    
    all_embeddings = np.array(all_embeddings)
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(all_embeddings)
    
    # Compute silhouette score
    unique_labels = list(set(all_labels))
    label_ints = [unique_labels.index(l) for l in all_labels]
    sil = silhouette_score(all_embeddings, label_ints) if len(unique_labels) > 1 else 0
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for j, field in enumerate(unique_labels):
        mask = [l == field for l in all_labels]
        axes[0].scatter(embs_2d[mask, 0], embs_2d[mask, 1], 
                       c=[colors[j]], label=field, alpha=0.7, s=80)
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title(f'Semantic Fields in Embedding Space (Silhouette: {sil:.3f})')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Channel patterns per field
    field_patterns = {}
    for field, embs in field_embeddings.items():
        channels = np.array([analyze_channels(e) for e in embs])
        field_patterns[field] = channels.mean(axis=0)
    
    pattern_matrix = np.array([field_patterns[f] for f in unique_labels])
    im = axes[1].imshow(pattern_matrix, aspect='auto', cmap='viridis')
    axes[1].set_yticks(range(len(unique_labels)))
    axes[1].set_yticklabels(unique_labels)
    axes[1].set_xlabel('Hadamard Channel')
    axes[1].set_title('Mean Channel Activation by Semantic Field')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_fields.png'), dpi=150)
    plt.close()
    
    print(f"  Silhouette score: {sil:.3f}")
    print(f"  Saved: {output_dir}/semantic_fields.png")


# -----------------------------------------------------------------------------
# ANALYSIS 6: Channel Cross-Correlation
# -----------------------------------------------------------------------------

def analyze_channel_correlation(model, enc, device, output_dir):
    """Compute channel cross-correlation matrix."""
    print("\n=== Analysis 6: Channel Cross-Correlation Matrix ===")
    
    # Gather all word embeddings
    all_channels = []
    all_words = []
    for field, words in SEMANTIC_FIELDS.items():
        for word in words:
            e = get_embedding(model, enc, word, device)
            if e is not None:
                all_channels.append(analyze_channels(e))
                all_words.append(word)
    
    all_channels = np.array(all_channels)  # [n_words, 32]
    
    # Compute correlation matrix between channels
    corr_matrix = np.corrcoef(all_channels.T)  # [32, 32]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    ax.set_title('Channel Cross-Correlation Matrix')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_correlation.png'), dpi=150)
    plt.close()
    
    # Find most correlated and anti-correlated pairs
    np.fill_diagonal(corr_matrix, 0)
    max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    min_idx = np.unravel_index(np.argmin(corr_matrix), corr_matrix.shape)
    
    print(f"  Most correlated: Ch{max_idx[0]}-Ch{max_idx[1]} (r={corr_matrix[max_idx]:.3f})")
    print(f"  Most anti-correlated: Ch{min_idx[0]}-Ch{min_idx[1]} (r={corr_matrix[min_idx]:.3f})")
    print(f"  Saved: {output_dir}/channel_correlation.png")


# -----------------------------------------------------------------------------
# ANALYSIS 7: Temporal Channel Dynamics (Per-Layer)
# -----------------------------------------------------------------------------

def analyze_layer_dynamics(model, enc, device, output_dir):
    """Analyze how channels evolve across layers."""
    print("\n=== Analysis 7: Temporal Channel Dynamics (Per-Layer) ===")
    
    test_words = ["king", "run", "happy", "computer", "democracy"]
    
    fig, axes = plt.subplots(len(test_words), 1, figsize=(14, 3*len(test_words)))
    
    for i, word in enumerate(test_words):
        layer_states = get_layer_embeddings(model, enc, word, device)
        if layer_states is None:
            continue
        
        # Channel patterns per layer
        layer_channels = np.array([analyze_channels(s) for s in layer_states])
        
        im = axes[i].imshow(layer_channels, aspect='auto', cmap='viridis')
        axes[i].set_xlabel('Hadamard Channel')
        axes[i].set_ylabel('Layer')
        axes[i].set_title(f'"{word}" - Channel evolution across layers')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_dynamics.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/layer_dynamics.png")


# -----------------------------------------------------------------------------
# ANALYSIS 8: mHC Stream Analysis
# -----------------------------------------------------------------------------

def analyze_mhc_streams(model, enc, device, output_dir, n_streams):
    """Analyze mHC doubly-stochastic stream mixing."""
    print("\n=== Analysis 8: mHC Stream Analysis ===")
    
    if not hasattr(model, 'stream_proj'):
        print("  Model is not mHC - skipping stream analysis")
        return
    
    # Get stream mixing matrices H_res from first layer
    test_words = ["the", "king", "run", "happy", "computer"]
    
    fig, axes = plt.subplots(len(test_words), 3, figsize=(12, 3*len(test_words)))
    
    for i, word in enumerate(test_words):
        tokens = enc.encode(" " + word)
        if len(tokens) > 1:
            tokens = enc.encode(word)
        if len(tokens) != 1:
            continue
        
        token_tensor = torch.tensor([[tokens[0]]], device=device)
        
        with torch.no_grad():
            # Get embeddings and expand to streams
            h = model.tok_embeddings(token_tensor)
            h = model.stream_proj(h)
            h = h.view(1, 1, model.n_streams, model.config.n_embd)
            x_flat = h.reshape(1, 1, -1)
            
            # Get H_res from first layer's attention stream
            layer = model.layers[0]
            H_res = layer.attn_stream.H_res(x_flat)  # [1, 1, n, n]
            H_pre = layer.attn_stream.H_pre(x_flat)  # [1, 1, 1, n]
            H_post = layer.attn_stream.H_post(x_flat)  # [1, 1, 1, n]
        
        H_res_np = H_res[0, 0].cpu().numpy()
        H_pre_np = H_pre[0, 0, 0].cpu().numpy()
        H_post_np = H_post[0, 0, 0].cpu().numpy()
        
        # Plot H_res
        im = axes[i, 0].imshow(H_res_np, cmap='Blues', vmin=0, vmax=1)
        axes[i, 0].set_xlabel('To Stream')
        axes[i, 0].set_ylabel('From Stream')
        axes[i, 0].set_title(f'"{word}" H_res (doubly stochastic)')
        plt.colorbar(im, ax=axes[i, 0])
        
        # Check doubly stochastic properties
        row_sums = H_res_np.sum(axis=1)
        col_sums = H_res_np.sum(axis=0)
        
        # Plot H_pre
        axes[i, 1].bar(range(n_streams), H_pre_np, color='green', alpha=0.7)
        axes[i, 1].set_xlabel('Stream')
        axes[i, 1].set_ylabel('Weight')
        axes[i, 1].set_title(f'H_pre (aggregation)')
        axes[i, 1].set_ylim(0, 1)
        
        # Plot H_post
        axes[i, 2].bar(range(n_streams), H_post_np, color='purple', alpha=0.7)
        axes[i, 2].set_xlabel('Stream')
        axes[i, 2].set_ylabel('Weight')
        axes[i, 2].set_title(f'H_post (distribution)')
        
        print(f"  '{word}': row_sums={row_sums}, col_sums={col_sums}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mhc_streams.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/mhc_streams.png")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

# Additional edge-case tests - items known to be single tokens in GPT-2
ADVERSARIAL_TESTS = {
    "Rare Words": ["aardvark", "zeppelin", "xylophone", "quasar", "fjord", "sphinx"],
    "Numbers": ["zero", "hundred", "million", "billion", "trillion", "infinity"],
    "Tech Terms": ["http", "www", "html", "json", "api", "cpu", "gpu", "ram"],
    "Emotions": ["happy", "sad", "angry", "afraid", "love", "hate", "joy", "fear"],
    "Function Words": ["the", "and", "but", "if", "or", "not", "yet", "so"],
    "Actions": ["run", "jump", "walk", "fly", "swim", "climb", "fall", "rise"],
}


def analyze_channel_ablation(model, enc, device, output_dir):
    """Zero out channels and measure impact on generation quality."""
    print("\n=== Analysis 9: Channel Ablation Study ===")
    
    prompt = "The history of science began when"
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], device=device)
    
    results = {}
    
    # Get baseline perplexity
    with torch.no_grad():
        logits_base, loss_base = model(x)
        baseline_loss = loss_base.item() if loss_base is not None else 0
    
    results['baseline'] = baseline_loss
    
    # Test ablating each channel (via embedding modification)
    # We'll measure embedding variance impact as proxy
    emb_weight = model.tok_embeddings.weight.data.clone()
    n_channels = 32
    n_blocks = emb_weight.shape[1] // n_channels
    
    channel_importance = []
    for ch in range(n_channels):
        # Zero out this channel in embeddings
        emb_reshaped = emb_weight.reshape(-1, n_blocks, n_channels)
        channel_energy = emb_reshaped[:, :, ch].abs().mean().item()
        channel_importance.append(channel_energy)
    
    # Plot importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of channel importance
    colors = ['red' if i == 10 else 'blue' for i in range(32)]
    axes[0].bar(range(32), channel_importance, color=colors, alpha=0.7)
    axes[0].set_xlabel('Hadamard Channel')
    axes[0].set_ylabel('Mean |Activation| in Embeddings')
    axes[0].set_title('Channel Importance (Red = Ch10)')
    axes[0].axhline(y=np.mean(channel_importance), color='gray', linestyle='--', label='Mean')
    axes[0].legend()
    
    # Sorted importance
    sorted_idx = np.argsort(channel_importance)[::-1]
    axes[1].bar(range(32), [channel_importance[i] for i in sorted_idx], alpha=0.7)
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Mean |Activation|')
    axes[1].set_title('Channel Importance (Sorted)')
    axes[1].set_xticks(range(32))
    axes[1].set_xticklabels([f'Ch{i}' for i in sorted_idx], rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_ablation.png'), dpi=150)
    plt.close()
    
    print(f"  Top 5 channels: {[f'Ch{sorted_idx[i]}' for i in range(5)]}")
    print(f"  Bottom 5 channels: {[f'Ch{sorted_idx[-(i+1)]}' for i in range(5)]}")
    print(f"  Saved: {output_dir}/channel_ablation.png")


def analyze_attention_heads(model, enc, device, output_dir):
    """Analyze what each attention head specializes in."""
    print("\n=== Analysis 10: Attention Head Specialization ===")
    
    # Get attention patterns for a sample sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    tokens = enc.encode(sentence)
    token_tensor = torch.tensor([tokens], device=device)
    
    # Hook to capture attention weights
    attention_patterns = []
    
    def hook_fn(module, input, output):
        # Try to capture attention weights if available
        if isinstance(output, tuple) and len(output) > 1:
            attention_patterns.append(output[1])
    
    # Register hooks on attention layers
    hooks = []
    for layer in model.layers:
        if hasattr(layer, 'attention'):
            hooks.append(layer.attention.register_forward_hook(hook_fn))
        elif hasattr(layer, 'attn'):
            hooks.append(layer.attn.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(token_tensor)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Analyze query/key/value patterns from weights instead
    n_heads = model.config.n_head
    n_layers = model.config.n_layer
    
    # Compute head diversity across layers
    head_norms = []
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'attention'):
            attn = layer.attention
        elif hasattr(layer, 'attn'):
            attn = layer.attn
        else:
            continue
        
        # Get Q, K, V weight norms per head
        if hasattr(attn, 'wq'):
            wq = attn.wq.weight.data
            wk = attn.wk.weight.data
            wv = attn.wv.weight.data
            
            head_dim = wq.shape[0] // n_heads
            for h in range(n_heads):
                q_norm = wq[h*head_dim:(h+1)*head_dim].norm().item()
                k_norm = wk[h*head_dim:(h+1)*head_dim].norm().item()
                v_norm = wv[h*head_dim:(h+1)*head_dim].norm().item()
                head_norms.append({
                    'layer': layer_idx,
                    'head': h,
                    'q_norm': q_norm,
                    'k_norm': k_norm,
                    'v_norm': v_norm
                })
    
    if not head_norms:
        print("  Could not extract attention head info")
        return
    
    # Plot head norms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    q_matrix = np.zeros((n_layers, n_heads))
    k_matrix = np.zeros((n_layers, n_heads))
    v_matrix = np.zeros((n_layers, n_heads))
    
    for entry in head_norms:
        q_matrix[entry['layer'], entry['head']] = entry['q_norm']
        k_matrix[entry['layer'], entry['head']] = entry['k_norm']
        v_matrix[entry['layer'], entry['head']] = entry['v_norm']
    
    im0 = axes[0].imshow(q_matrix, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Query Weight Norms')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(k_matrix, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Key Weight Norms')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(v_matrix, aspect='auto', cmap='viridis')
    axes[2].set_xlabel('Head')
    axes[2].set_ylabel('Layer')
    axes[2].set_title('Value Weight Norms')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_heads.png'), dpi=150)
    plt.close()
    print(f"  Analyzed {len(head_norms)} heads across {n_layers} layers")
    print(f"  Saved: {output_dir}/attention_heads.png")


def analyze_adversarial(model, enc, device, output_dir):
    """Test model on edge cases: neologisms, numbers, symbols, code."""
    print("\n=== Analysis 11: Adversarial/Edge Case Probes ===")
    
    results = {}
    
    for category, test_items in ADVERSARIAL_TESTS.items():
        category_results = []
        for item in test_items:
            tokens = enc.encode(item)
            n_tokens = len(tokens)
            
            # Get embedding if single token
            if n_tokens == 1:
                e = get_embedding(model, enc, item, device)
                if e is not None:
                    ch = analyze_channels(e)
                    category_results.append({
                        'item': item,
                        'n_tokens': n_tokens,
                        'channel_pattern': ch,
                        'max_channel': np.argmax(ch),
                        'entropy': -np.sum(ch/ch.sum() * np.log(ch/ch.sum() + 1e-8))
                    })
            else:
                category_results.append({
                    'item': item,
                    'n_tokens': n_tokens,
                    'channel_pattern': None,
                    'max_channel': None,
                    'entropy': None
                })
        
        results[category] = category_results
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (category, cat_results) in enumerate(results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        
        # Plot channel patterns for single-token items
        single_token = [r for r in cat_results if r['channel_pattern'] is not None]
        
        if single_token:
            for r in single_token:
                ax.plot(r['channel_pattern'], label=r['item'][:10], alpha=0.7)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Activation')
            ax.set_title(f'{category} ({len(single_token)} single-token)')
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, f'{category}\n(all multi-token)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(category)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adversarial.png'), dpi=150)
    plt.close()
    
    print(f"  Categories tested: {list(results.keys())}")
    for cat, res in results.items():
        single = sum(1 for r in res if r['channel_pattern'] is not None)
        print(f"    {cat}: {single}/{len(res)} single-token")
    print(f"  Saved: {output_dir}/adversarial.png")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive semantic analysis')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='semantic_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model, config, use_mhc, n_streams = load_model(args.ckpt, args.device)
    enc = tiktoken.get_encoding('gpt2')
    
    print(f"\nModel type: {'mHCWalsh' if use_mhc else 'Walsh'}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Layers: {config.n_layer}")
    if use_mhc:
        print(f"Streams: {n_streams}")
    
    # Run all analyses
    analyze_moe_specialization(model, enc, args.device, args.output)
    analyze_channel_modulation_dynamics(model, enc, args.device, args.output)
    
    analyze_analogies(model, enc, args.device, args.output)
    analyze_antonyms_synonyms(model, enc, args.device, args.output)
    analyze_composition(model, enc, args.device, args.output)
    analyze_trajectories(model, enc, args.device, args.output)
    analyze_semantic_fields(model, enc, args.device, args.output)
    analyze_channel_correlation(model, enc, args.device, args.output)
    analyze_layer_dynamics(model, enc, args.device, args.output)
    
    if use_mhc:
        analyze_mhc_streams(model, enc, args.device, args.output, n_streams)
    
    # New analyses
    analyze_channel_ablation(model, enc, args.device, args.output)
    analyze_attention_heads(model, enc, args.device, args.output)
    analyze_adversarial(model, enc, args.device, args.output)
    
    print(f"\n✅ All analyses complete! Results in {args.output}/")


def analyze_moe_specialization(model, enc, device, output_dir):
    """Analysis 9: MoE Expert Specialization."""
    print("\n--- Analysis 9: MoE Expert Specialization ---")
    
    if not hasattr(model.config, 'use_moe') or not model.config.use_moe:
        print("  MoE not enabled in this model, skipping.")
        return

    # Categorize words for MoE testing
    categories = {
        "Function": ["the", "a", "an", "in", "on", "at", "to", "for", "with", "and", "or", "but"],
        "Verbs": ["is", "are", "was", "be", "do", "have", "say", "go", "get", "make", "know"],
        "Nouns": ["time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand"],
        "Technical": ["quantum", "algorithm", "energy", "matter", "electron", "database", "binary"],
    }
    
    n_experts = model.config.n_embd // 32
    results = {}
    
    for cat, words in categories.items():
        expert_usage = np.zeros(n_experts)
        count = 0
        for word in words:
            token = enc.encode(" " + word)
            if len(token) != 1: continue
            
            token_tensor = torch.tensor([token], device=device)
            with torch.no_grad():
                _ = model(token_tensor)
            
            # Aggregate from all layers
            for layer in model.layers:
                if hasattr(layer.feed_forward, '_last_expert_mask'):
                    mask = layer.feed_forward._last_expert_mask # [1, 1, n_experts]
                    expert_usage += mask[0, 0].cpu().numpy()
            count += 1
        
        if count > 0:
            results[cat] = expert_usage / (count * model.config.n_layer)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(n_experts)
    width = 0.8 / len(results)
    for i, (cat, usage) in enumerate(results.items()):
        plt.bar(x + (i - len(results)/2)*width, usage, width, label=cat)
    
    plt.xlabel("Expert Index")
    plt.ylabel("Avg Activation Probability")
    plt.title("MoE Expert Specialization by Part-of-Speech")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/moe_specialization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/moe_specialization.png")

def analyze_channel_modulation_dynamics(model, enc, device, output_dir):
    """Analysis 10: Channel Modulation Dynamics."""
    print("\n--- Analysis 10: Channel Modulation Dynamics ---")
    
    if not hasattr(model.config, 'use_channel_mod') or not model.config.use_channel_mod:
        print("  Channel Modulation not enabled in this model, skipping.")
        return

    contexts = {
        "Scientific": "Quantum entanglement describes particles sharing state.",
        "Narrative": "The knight fought the dragon in the dark cave.",
        "Dialogue": "Hello there, how are you feeling today?",
        "Technical": "The database query returned an empty result set."
    }
    
    results = {}
    for name, text in contexts.items():
        tokens = enc.encode(text)
        token_tensor = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            _ = model(token_tensor)
            
        layer_scales = []
        for layer in model.layers:
            if hasattr(layer, 'channel_mod'):
                stats = layer.channel_mod.get_modulation_stats()
                layer_scales.append(stats['scale_mean'])
        results[name] = layer_scales

    # Plot
    plt.figure(figsize=(10, 6))
    for name, scales in results.items():
        plt.plot(scales, marker='o', label=name)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel("Layer")
    plt.ylabel("Mean Channel Scale")
    plt.title("Channel Modulation Dynamics by Context")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/channel_mod_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/channel_mod_dynamics.png")


if __name__ == '__main__':
    main()

