# Walsh

**Status:** `EXPERIMENTAL` / `ACTIVE DEV`

Walsh is an exploratory architecture combining **1.58-bit (Ternary) Quantization** (à la BitNet) with **Hyper-Complex Algebras**. We replace standard linear layers with **Octonion (8D)** or **Hadamard (32D)** multiplications, compressing the "brain" of the model by 8/32x while maintaining expressivity through structured geometric mixing.

Currently running on **CUDA** (via custom Triton kernels) and **WebAssembly** (on the Internet Computer blockchain).

---

## 🧪 The Hypothesis

Standard LLMs treat dimensions as independent. We force dimensions to interact via structured algebras:

| Algebra | Dimension | Compression | Mixing | Complexity |
|---------|-----------|-------------|--------|------------|
| **Octonion** | 8D | 1/8th params | Cayley-Dickson | O(n²) |
| **Hadamard** | 32D | 1/32th params | Fast Hadamard Transform | O(n log n) |

This allows:
1. **Extreme Compression**: 0.87M "brain" params for a 26M total model
2. **Memory Efficiency**: Ternary weights {-1, 0, +1} = 1.58 bits
3. **Fast Mixing**: FHT provides structured mixing with zero learned params

---

## 🏗️ Architecture Overview

### Algebra Selection

```python
WalshConfig(
    algebra="hadamard",      # or "octonion"
    head_mixing=True,        # Enable algebra-based head mixing
    hash_embeddings=False,   # Enable hash embeddings (experimental)
    n_head=32,               # Must be divisible by algebra dimension
    n_embd=512,
)
```

### Key Features

| Feature | Octonion (8D) | Hadamard (32D) |
|---------|---------------|----------------|
| Linear Compression | 8x | 32x |
| Head Groups | Groups of 8 | Groups of 32 |
| Mixing Method | Cayley-Dickson | FHT (O(n log n)) |
| Triton Kernels | ✅ Fused | ✅ FP32 accumulators |
| Inference Packing | ✅ 4x memory | ✅ 16x memory |

---

## 🧬 Hash Embeddings (Experimental)

> [!WARNING]
> Hash embeddings are highly experimental. Quality may be lower than standard embeddings due to hash collisions between tokens.

### The Idea

Standard embeddings dominate model size: `vocab_size × n_embd = 50K × 512 = 25.6M params`

Hash embeddings use **composite hashing** with two small tables:
```python
# Instead of: embedding[token_id]  → 25.6M params
# We use:    emb_1[token % 1021] + emb_2[token // 1021]  → 1.0M params
```

### Benefits

| Metric | Standard | Hash Embeddings |
|--------|----------|-----------------|
| Embedding params | 25.6M | 1.0M |
| Total model | 26.6M | **2.0M** |
| VRAM (training) | ~6GB | **2.8GB** |
| Compression | 1x | **25x** |

### The Trade-off

**What's preserved:**
- ✅ Grammar and syntax
- ✅ Common patterns and phrases
- ✅ Reasoning structure

**What's lost:**
- ❌ Rare word distinctions (hash collisions)
- ❌ Precise factual knowledge
- ❌ Unique token associations

### RAG-Native Architecture

Hash collisions **prevent memorizing facts**, making the model naturally suited for Retrieval-Augmented Generation:

```
┌─────────────────┐     ┌─────────────────┐
│  Walsh Brain  │────▶│  Vector DB      │
│  (2M params)    │     │  (Knowledge)    │
│  "How to think" │     │  "What to know" │
└─────────────────┘     └─────────────────┘
```

**This architecture cannot hallucinate facts it never stored!**

### Training with Hash Embeddings

```bash
# Ultra-compressed: ~2M params with 8L×512D
python train.py config/train_tinystories_hadamard_hash.py
```

### Chunked Cross Entropy

Hash embeddings include memory-efficient loss computation:
- Computes cross-entropy in vocab chunks (4K at a time)
- Avoids materializing full `[B×T, vocab_size]` logits tensor
- Reduces peak VRAM by ~60%

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy triton tiktoken datasets transformers tqdm
```

### 2. Prepare Dataset
```bash
# TinyStories (recommended for testing)
python data/tinystories/prepare.py
```

### 3. Train
```bash
# Hadamard 32D (faster convergence, 32x compression)
python train.py config/train_tinystories_hadamard.py

# Hadamard + Hash Embeddings (experimental, 13x total compression)
python train.py config/train_tinystories_hadamard_hash.py

# Octonion 8D (original)
python train.py config/train_tinystories_octonion.py
```

### 4. Generate
```bash
python generate.py --ckpt experiments/out-tinystories-hadamard/ckpt.pt \
    --prompt "Once upon a time" --max_tokens 100
```

---

## 📊 Current Status

### ✅ Verified Working
- **Hadamard 32D**: 0.87M brain params, loss 3.5 @ 200 iters, 25 tok/s
- **Hash Embeddings**: 2.0M total params, 2.8GB VRAM training
- **Octonion 8D**: Full training + inference pipeline
- **Head Mixing**: Both algebras support attention head mixing
- **KV Cache**: 4.6x speedup for autoregressive generation
- **ICP Deployment**: WebAssembly inference on Internet Computer

### ⚠️ Experimental Features
- **Hash Embeddings**: Quality vs compression trade-off under investigation
- **Chunked Cross Entropy**: Memory-efficient but slightly slower

### ⚠️ Rust/Wasm Status
The Rust inference engine (`inference/`) currently supports **Octonion (8D) only**.

```
inference/src/model.rs  - Octonion 8D ✅ | Hadamard 32D ❌ | Hash Embeddings ❌
```

---

## 📂 Key Files

### Python Training & Inference
| File | Description |
|------|-------------|
| `src/model/chassis.py` | Model architecture with algebra selection + hash embeddings |
| `src/model/fht_cuda.py` | Hadamard 32D kernels with FHT |
| `src/model/cayley_dickson_cuda.py` | Octonion 8D Triton kernels |
| `config/train_tinystories_hadamard.py` | Hadamard training config |
| `config/train_tinystories_hadamard_hash.py` | Hadamard + Hash embeddings config |
| `config/train_tinystories_octonion.py` | Octonion training config |

### Rust/Wasm Inference (Octonion only)
| File | Description |
|------|-------------|
| `inference/src/model.rs` | Rust inference with KV cache |
| `inference/src/tokenizer.rs` | GPT-2/char tokenizer |
| `inference/src/lib.rs` | IC Canister API |

---

## 🗺️ Roadmap

### Phase 1: Core Engine ✅
- [x] Octonion 8D linear layers + head mixer
- [x] Fused Triton kernels (6x speedup)
- [x] KV Cache for fast inference
- [x] Rust/Wasm inference engine

### Phase 2: Hadamard 32D ✅
- [x] Fast Hadamard Transform (FHT) kernel
- [x] 32D linear layers with O(n log n) mixing
- [x] Variance-preserving initialization (α/β diagonals)
- [x] FP32 accumulators for numerical stability

### Phase 3: Extreme Compression 🧪
- [x] Hash embeddings (25x embedding compression)
- [x] Chunked cross entropy (60% VRAM reduction)
- [ ] 3-table hash embeddings (better collision handling)
- [ ] RAG integration for fact retrieval

### Phase 4: Deployment 🚧
- [ ] Hadamard support in Rust/Wasm
- [ ] Hash embeddings in Rust/Wasm
- [ ] Client-side browser inference
- [ ] ICP mainnet deployment

---

## 📖 Technical Details

### Variance-Preserving Initialization
All layers use learned diagonal scaling (α input, β output) for ternary weights:
```python
# α: per-feature input scaling [32, in_o]
self.alpha = nn.Parameter(torch.ones(ALGEBRA_DIM, in_o))

# β: per-feature output scaling with variance-preserving init
beta_init = math.sqrt(3.0 / (2.0 * in_o))
self.beta = nn.Parameter(torch.ones(ALGEBRA_DIM, out_o) * beta_init)
```

### Parameter Breakdown

**Standard Hadamard 32D:**
```
Total:      26.60M
Embedding:  25.73M (97%)
Brain:       0.87M (3%)
```

**Hadamard + Hash Embeddings:**
```
Total:       2.00M
Embedding:   1.05M (52%)  ← 25x smaller!
Brain:       0.95M (48%)
```

---

## 📚 References

- [BitNet: 1-bit LLMs](https://arxiv.org/abs/2310.11453)
- [Fast Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform)
- [Cayley-Dickson Construction](https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction)
- [Hash Embeddings (Svenstrup et al.)](https://arxiv.org/abs/1709.03933)