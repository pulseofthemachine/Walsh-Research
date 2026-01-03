# Walsh

**Status:** `EXPERIMENTAL` / `ACTIVE DEV`

Walsh is an exploratory transformer architecture combining **1.58-bit (Ternary) Quantization** (à la BitNet b1.58) with **Hyper-Complex Algebras** and **Interpretable Routing**. We replace standard linear layers with **Hadamard (32D)** or **Octonion (8D)** multiplications, achieving 32x/8x parameter compression while maintaining expressivity through structured geometric mixing.

The architecture incorporates:
- **mHC (Manifold Hyper-Connections)**: Multi-stream residuals with doubly-stochastic mixing
- **Hadamard MoE**: Sparse Mixture of Experts with dynamic routing
- **Channel Modulation**: Context-adaptive scaling of Hadamard channels

Currently running on **CUDA** (via custom Triton kernels) and **WebAssembly** (on the Internet Computer blockchain).

---

## 🧪 The Hypothesis

Standard LLMs treat dimensions as independent. We force dimensions to interact via structured algebras:

| Algebra | Dimension | Compression | Mixing | Complexity |
|---------|-----------|-------------|--------|------------|
| **Hadamard** | 32D | 1/32th params | Fast Hadamard Transform | O(n log n) |
| **Octonion** | 8D | 1/8th params | Cayley-Dickson | O(n²) |

This enables:
1. **Extreme Compression**: ~33% ternary sparsity (1/3 zeros, 1/3 +1, 1/3 -1)
2. **Memory Efficiency**: Ternary weights {-1, 0, +1} = 1.58 bits per weight
3. **Fast Mixing**: FHT provides structured mixing with O(n log n) complexity
4. **Interpretability**: Sparse MoE routing and channel modulation expose decision pathways

---

## 🏗️ Architecture Overview

### Core Configuration

```python
from src.model import WalshConfig, mHCWalsh

config = WalshConfig(
    n_layer=16,
    n_head=32,               # Must be divisible by algebra dimension
    n_embd=512,              # 512D = 16 × 32D Hadamard blocks
    algebra="hadamard",      # "hadamard" (32D) or "octonion" (8D)
    head_mixing=True,        # Enable algebra-based head mixing
    
    # MoE (Mixture of Experts)
    use_moe=True,            # Sparse expert routing in FFN
    moe_threshold=0.1,       # Dynamic routing threshold
    moe_max_experts=6,       # Maximum experts per token
    
    # Channel Modulation
    use_channel_mod=True,    # Context-adaptive channel scaling
)

# With mHC multi-stream residuals
model = mHCWalsh(config, n_streams=4)
```

### Architectural Layers

| Layer | Purpose | Key Feature |
|-------|---------|-------------|
| **Hadamard Linear** | 32x compressed projections | Ternary weights + FHT mixing |
| **mHC Streams** | Multi-stream residuals | Doubly-stochastic mixing matrices |
| **Hadamard MoE** | Sparse expert routing | Dynamic top-k expert selection |
| **Channel Modulation** | Context-adaptive scaling | Global state → per-channel scales |

---

## 🔄 mHC: Manifold Hyper-Connections

> Based on [arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880)

mHC replaces standard residual connections with **multi-stream residuals** using doubly-stochastic mixing matrices for guaranteed stability.

### Standard vs mHC Residual

```
Standard:  x = x + layer(x)              — single stream, can collapse

mHC:       x = H_res @ x + H_post^T @ layer(H_pre @ x)
                                          — n parallel streams with balanced mixing
```

### Key Properties

| Property | Benefit |
|----------|---------|
| **Doubly stochastic H_res** | No stream can dominate (prevents collapse) |
| **Spectral norm ≤ 1** | No gradient explosion |
| **Birkhoff polytope** | Mixing = convex combo of permutations |
| **Sinkhorn-Knopp projection** | Efficient manifold constraint (Triton kernel) |

### Emergent Stream Specialization

In trained models, the 4 streams naturally specialize:
- **Stream 0**: Primary syntax/structure pathway
- **Stream 1**: Main semantic content pathway
- **Stream 2**: Context/modulation pathway
- **Stream 3**: Specialist features (adjectives, action verbs)

---

## 🎯 Hadamard MoE (Mixture of Experts)

Sparse expert routing with ternary-quantized experts operating on 32D Hadamard blocks.

### How It Works

```
Input Token → Router → Select Top-K Experts → Weighted Sum
              ↓
         Threshold-based dynamic routing
         (min 1, max 6 experts per token)
```

### Key Features

| Feature | Implementation |
|---------|----------------|
| **Dynamic Top-K** | Threshold-based routing (not fixed top-k) |
| **Auto-scaling** | N_experts = n_embd / 32 (e.g., 512D = 16 experts) |
| **Load Balancing** | Auxiliary loss prevents expert collapse |
| **Ternary Experts** | Each expert uses ternary quantization |

### Observed Specialization

| Expert | Specialization |
|--------|----------------|
| E0 | Universal (fires for everything) |
| E3 | Function words ("the", "and", "with") |
| E9 | Adjectives and descriptors |
| E12, E15 | Content nouns and technical terms |

---

## 📡 Channel Modulation

Context-adaptive scaling of Hadamard channels based on global sequence state.

### Mechanism

```
Global State (mean pooling) → MLP → Scale & Shift per Channel
                                    ↓
                              Modulate hidden states
```

### Interpretation

The model learns to use channel modulation as a **"genre detector"**:

| Context | Channel Behavior |
|---------|------------------|
| Scientific | High scale in early layers (1.5x), drops to 0.8x |
| Narrative | Even higher early (1.7x), gradual decay |
| Dialogue | Lower overall (suppresses formality channels) |
| Technical | Peak at layer 6, high variance |

---

## 📊 Analysis Suite

Three analysis scripts for interpretability:

### `analyze_categories.py`
Semantic category clustering across 18 word types:
```bash
python analyze_categories.py --ckpt experiments/out-wikipedia-512/ckpt.pt \
    --output analysis/categories.png --hadamard-dim 32
```

### `analyze_semantic.py`
Comprehensive semantic analysis (analogies, synonyms, trajectories, mHC streams):
```bash
python analyze_semantic.py --ckpt experiments/out-wikipedia-512/ckpt.pt \
    --output analysis_semantic/
```

### `analyze_moe_channel.py`
MoE routing patterns and channel modulation dynamics:
```bash
python analyze_moe_channel.py --ckpt experiments/out-wikipedia-512/ckpt.pt \
    --output analysis_moe/
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy triton tiktoken datasets transformers tqdm matplotlib scikit-learn
```

### 2. Prepare Dataset
```bash
# Wikipedia (1B tokens)
python data/wikipedia/prepare.py

# TinyStories (for testing)
python data/tinystories/prepare.py
```

### 3. Train
```bash
# Full architecture: mHC + MoE + Channel Modulation (512D, 16 layers)
python train.py config/train_wikipedia_full.py

# Quick test (smaller model)
python train.py config/train_tinystories_hadamard.py
```

### 4. Generate
```bash
python generate.py --ckpt experiments/out-wikipedia-512/ckpt.pt \
    --prompt "The quantum physicist" --max_tokens 100
```

### 5. Check Sparsity
```bash
python check_sparsity.py
# Expected output: ~33% ternary sparsity (optimal for 1.58-bit)
```

---

## � Current Results

### Model Configuration (512D, 16L)
```
Total Parameters:  35.5M
Trainable:         35.5M
Ternary Sparsity:  ~33% (optimal)
Embedding:         512 dimensions
Experts:           16 (auto-scaled from 512/32)
Streams:           4 (mHC)
```

### Emergent Properties
- **Expert Specialization**: Function words, content words, and technical terms route to different experts
- **Stream Differentiation**: Syntax vs semantics vs specialist features
- **Context Adaptation**: Scientific text amplifies different channels than narrative text

### Training Metrics
At ~250 iterations on Wikipedia:
- Loss: 4.75
- Silhouette Score: 0.18 (semantic clustering)
- Distance Ratio: 1.18 (between-category vs within-category)

---

## 📂 Key Files

### Python Training & Inference
| File | Description |
|------|-------------|
| `src/model/chassis.py` | `Walsh` and `WalshConfig` base classes |
| `src/model/mhc.py` | `mHCWalsh` with multi-stream residuals |
| `src/model/moe.py` | `HadamardMoE` and `HadamardChannelModulation` |
| `src/model/fht_cuda.py` | Hadamard 32D kernels with FHT |
| `src/model/physics.py` | Ternary quantization (BitNet b1.58 style) |
| `train.py` | Training loop with MoE/ChannelMod logging |
| `generate.py` | Text generation with checkpoint loading |
| `check_sparsity.py` | Verify ternary weight distribution |

### Analysis Scripts
| File | Description |
|------|-------------|
| `analyze_categories.py` | 18-category semantic clustering |
| `analyze_semantic.py` | Comprehensive semantic analysis suite |
| `analyze_moe_channel.py` | MoE + Channel Modulation visualization |

### Rust/Wasm Inference (Octonion only)
| File | Description |
|------|-------------|
| `inference/src/model.rs` | Rust inference with KV cache |
| `inference/src/lib.rs` | IC Canister API |

---

## 🗺️ Roadmap

### Phase 1: Core Engine ✅
- [x] Hadamard 32D / Octonion 8D linear layers
- [x] Fused Triton kernels
- [x] KV Cache for fast inference
- [x] Variance-preserving initialization (α/β diagonals)

### Phase 2: mHC Integration ✅
- [x] Manifold Hyper-Connections (multi-stream residuals)
- [x] Sinkhorn-Knopp doubly-stochastic projection (Triton kernel)
- [x] Stream specialization emergence

### Phase 3: Sparse Experts ✅
- [x] Hadamard MoE with dynamic top-k routing
- [x] Ternary-quantized experts
- [x] Auto-scaling (n_experts = n_embd / 32)
- [x] Load balancing auxiliary loss

### Phase 4: Channel Modulation ✅
- [x] Context-adaptive per-channel scaling
- [x] Triton fused kernel with autograd backward
- [x] Genre/mode detection emergence

### Phase 5: Interpretability ✅
- [x] Semantic category analysis (18 types)
- [x] MoE routing visualization
- [x] Channel modulation dynamics plots
- [x] mHC stream analysis

### Phase 6: Scaling 🚧
- [ ] 1B+ token pretraining validation
- [ ] Instruction tuning
- [ ] Hadamard support in Rust/Wasm
- [ ] Client-side browser inference

---

## 📖 Technical Details

### Ternary Quantization (BitNet b1.58)

```python
# Absmean scaling: γ = mean(|W|)
# Quantize: W_ternary = clip(round(W / γ), -1, +1)
# Straight-through estimator for gradients
```

Optimal distribution: ~33% zeros, ~33% +1, ~33% -1

### Variance-Preserving Initialization

```python
# α: per-feature input scaling [32, in_o]
self.alpha = nn.Parameter(torch.ones(ALGEBRA_DIM, in_o))

# β: per-feature output scaling with variance-preserving init
beta_init = math.sqrt(3.0 / (2.0 * in_o))
self.beta = nn.Parameter(torch.ones(ALGEBRA_DIM, out_o) * beta_init)
```

### Channel-Balanced Initialization

```python
weight_init = torch.randn(32, out_o, in_o) * 0.02
channel_norms = weight_init.view(32, -1).norm(dim=1, keepdim=True)
weight_init = weight_init / channel_norms  # Equal magnitude per channel
```

---

## 📚 References

- [BitNet b1.58: 1-bit LLMs](https://arxiv.org/abs/2310.11453)
- [mHC: Manifold Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [Fast Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform)
- [Cayley-Dickson Construction](https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction)
- [Sinkhorn-Knopp Algorithm](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)