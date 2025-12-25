# SpinNet (Research Preview)

**Status:** `EXPERIMENTAL` / `ACTIVE DEV`

SpinNet is an exploratory architecture combining **1.58-bit (Ternary) Quantization** (à la BitNet) with **Hyper-Complex (Cayley-Dickson) Algebras**. Specifically, we replace standard linear layers with **Octonion (8D)** multiplications, aiming to compress more "intelligence" into lower memory bandwidth by enforcing structured geometric relationships in the latent space.

It is currently running on **CUDA** (via custom Triton kernels) and **WebAssembly** (on the Internet Computer blockchain).

---

## 🧪 The Hypothesis
Standard LLMs treat dimensions as independent. We suspect that forcing dimensions to interact via Cayley-Dickson algebras (Complex -> Quaternion -> Octonion) might allow:
1.  **Higher Information Density**: 8 dimensions per "unit".
2.  **Memory Compression**: Weights are ternary \({-1, 0, 1}\), drastically reducing I/O.
3.  **Efficient Inference**: Structured math = fewer parameters for same effective dimensionality.

---

## 🏗️ Architecture Components

### 1. Python Training Stack (`src/`)
Built on PyTorch. Use this to train models.

- **Octonion Linear Layers**: Implements `y = x @ W^T` using the 8x8 Cayley-Dickson sign table.
- **Octonion Head Mixer**: Mixes 8 attention heads using Cayley-Dickson algebra for improved representation learning (~8% loss reduction).
- **Fused Triton Kernels**: ~6x speedup for linear layers, ~46x speedup for head mixer.
  - Custom autograd functions that fuse the Octonion algebra into a single kernel launch.

**Quick Start: Training**

1.  **Install Dependencies**:
    ```bash
    pip install torch numpy triton tiktoken datasets transformers tqdm requests pandas ta
    ```

2.  **Prepare a Dataset**: Run the preparation script for your chosen data source.
    ```bash
    # Option A: TinyShakespeare (Fastest, good for testing)
    python data/tinyshakespeare/prepare.py
    
    # Option B: TinyStories (Children's stories, 50k vocab)
    python data/tinystories/prepare.py
    
    # Option C: FineWeb (High quality, for larger models)
    python data/fineweb/prepare.py
    ```

3.  **Select a Config**: We provide configs for different scales.
    ```bash
    # Train small Shakespeare model
    python train.py config/train_tinyshakespeare.py
    
    # Train TinyStories with Octonion Head Mixer
    python train.py config/train_tinystories_octonion.py
    
    # Train 124M parameter model (GPT-2 Small scale)
    python train.py config/scholar_124m.py
    ```

4.  **Analyze Octonion Structure**: Visualize dimension specialization.
    ```bash
    python tools/analyze_octonion.py --ckpt experiments/out-tinystories-octonion/ckpt.pt
    ```

**Note**: The system *automatically* detects CUDA and switches to Fused Triton Kernels.

### 2. Rust / Wasm Inference Engine (`inference/`)
A bare-metal inference engine designed for the **DFINITY Internet Computer (IC)**. It runs entirely in WebAssembly within ICP's 40B instruction limits.

- **Weights**: Parses custom `.spinnet` sparse-ternary format with Octonion Head Mixer support.
- **Tokenizer**: Auto-detecting (char-level for vocab≤256, GPT-2 BPE otherwise).
- **Head Mixer**: Full Cayley-Dickson algebra implementation for attention mixing.
- **KV Cache**: Incremental attention (O(1) complexity per token) — also in Python!
- **Temperature Sampling**: Softmax with T=0.8 for diverse generation.
- **Adaptive Chunking**: Automatically pauses at 60% instruction budget.

---

## 🚀 ICP Canister Deployment

### Prerequisites
- Install [dfx](https://internetcomputer.org/docs/current/developer-docs/setup/install/)
- Have a trained model checkpoint

### Deploy to Local Replica

1.  **Compress the model**:
    ```bash
    python compress.py experiments/out-tinystories-octonion/ckpt.pt --output inference/ckpt_v2.spinnet
    ```

2.  **Start the replica and deploy**:
    ```bash
    cd inference
    dfx start --background
    dfx deploy
    ```

3.  **Test with verify script**:
    ```bash
    ./verify_single_user.sh "Once upon a time" 50
    ```
    
    Expected output (TinyStories):
    ```
    Once upon a time, there was a little girl named Lily. 
    She liked to play outside and play with her friends...
    ```

### API Endpoints
- `start_generation(prompt, max_tokens)` - Begin generation session
- `start_forward()` - Initialize forward pass
- `process_layers(n)` - Process n layers (adaptive chunking)
- `finish_forward()` - Sample next token
- `generate_n_tokens(n)` - Generate up to n tokens in one call

---

## 📊 Current Status

### Verified Working
- ✅ **Coherent Text Generation**: TinyStories produces children's story text
- ✅ **Octonion Head Mixer**: Reduces loss by ~8% vs baseline
- ✅ **KV Cache (Python)**: 4.6x speedup, 36+ tok/s generation
- ✅ **Model Compression**: 13x smaller files, 3.7x less GPU memory
- ✅ **ICP Deployment**: ~0.7 tok/s on local replica

### Recent Improvements
- **KV Cache in Python**: Incremental decoding for fast inference
- **Compression Pipeline**: Bitmask ternary format with 90% sparsity
- **GPT-2 Tokenizer**: Both Rust and Python support 50k vocab models
- **Head Mixer**: Full Cayley-Dickson implementation in Python + Rust

---

## 🗺️ Roadmap & Todos

### Phase 1: Engine Optimization (✅ Complete)
- [x] Implement Octonion logic in PyTorch
- [x] Write Fused Triton Kernels (linear + head mixer)
- [x] Port inference to Rust/Wasm with head mixer
- [x] Implement KV Cache & incremental attention
- [x] Implement GPT-2 tokenizer in Rust
- [x] Add temperature sampling

### Phase 2: Deployment & UX (🚧 In Progress)
- [ ] **Client-Side Wasm**: Port to browser via `wasm-bindgen`
- [ ] **Web Dashboard**: Visualizer for hyper-dimensional states
- [ ] **Data Quality**: Train 100M+ param model on FineWeb-Edu
- [.] **Mainnet**: Deploy canister to live IC network

---

## 📂 Key Files

### Python Training & Inference
- `src/model/chassis.py`: Model architecture with KV Cache and Octonion Head Mixer
- `src/model/cayley_dickson_cuda.py`: High-performance Triton kernels
- `train.py`: Training script with gradient checkpointing
- `generate.py`: Generate from `.pt` checkpoints
- `generate_compressed.py`: Generate from `.spinnet` files
- `compress.py`: Convert `.pt` → `.spinnet` format

### Rust/Wasm Inference
- `inference/src/model.rs`: Rust inference with KV cache and head mixer
- `inference/src/tokenizer.rs`: Auto-detecting GPT-2/char tokenizer
- `inference/src/lib.rs`: IC Canister API

### Docs
- `COMPRESSION_GUIDE.md`: Detailed compression workflow and API