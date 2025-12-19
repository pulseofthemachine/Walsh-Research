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
- **Fused Triton Kernels**: ~6x speedup over PyTorch.

**Quick Start: Training**

1.  **Prepare a Dataset**: Run the preparation script for your chosen data source.
    ```bash
    # Option A: TinyShakespeare (Fastest, good for testing)
    python data/tinyshakespeare/prepare.py
    
    # Option B: Crypto (Custom domain data)
    python data/crypto/prepare.py
    
    # Option C: FineWeb (High quality, for larger models)
    python data/fineweb/prepare.py
    ```

2.  **Select a Config**: We provide configs for different scales.
    ```bash
    # Train small Shakespeare model
    python train.py config/train_tinyshakespeare.py
    
    # Train 124M parameter model (GPT-2 Small scale)
    python train.py config/scholar_124m.py
    
    # Train Crypto model
    python train.py config/train_crypto.py
    ```

**Note**: The system *automatically* detects if you are running on CUDA and switches to the Fused Triton Kernel for maximum speed.

### 2. Rust / Wasm Inference Engine (`inference/`)
A bare-metal inference engine designed for the **DFINITY Internet Computer (IC)**. It runs entirely in WebAssembly within strict 20B instruction limits.
- **Weights**: Parses custom `.spinnet` sparse-ternary format.
- **Execution**: Splits the Forward pass into chunks to fit in a single block.
- **KV Cache**: Implemented incremental attention (O(1) complexity).
- **Batched Inference**: Using inverted loops and `BTreeMap` session management, we achieve **1.51 tok/sec** aggregate throughput (3x boost) for concurrent users.

**Usage (IC Replica)**:
```bash
cd inference
dfx deploy
./run_batch_inference.sh
```

---

## 📊 Current Status (The "Shat" Report)

We have verified **mechanical correctness** across the stack.
- The Python model trains and reduces loss.
- The Rust engine produces mathematically identical outputs to the Python reference.
- The Batched Engine handles concurrent sessions correctly.

**However**, the current checkpoint (`ckpt_v2.spinnet`) is trained on a tiny dataset (TinyShakespeare) for a very short time.
- **Expected Output**: "To be or not to be..."
- **Actual Output**: "To be or not to se the shat t..."

*Note: We are optimizing the **Engine** first. The **Model Intelligence** (fixing "the shat") is next on the roadmap.*

---

## 🗺️ Roadmap & Todos

### Phase 1: Engine Optimization (✅ Complete)
- [x] Implement Octonion logic in PyTorch
- [x] Write Fused Triton Kernels
- [x] Port inference to Rust/Wasm
- [x] Implement KV Cache & Flash Attention equivalent
- [x] Implement Batched Inference (Session Manager)

### Phase 2: Deployment & UX (🚧 In Progress)
- [ ] **Client-Side Wasm**: Port the Rust engine to run directly in the browser (via `wasm-bindgen`). This eliminates network latency.
- [ ] **Web Dashboard**: Visualizer for the hyper-dimensional states.
- [ ] **Data Quality**: Train a 100M+ param model on FineWeb-Edu to produce coherent English.
- [.] **Mainnet**: Deploy the canister to the live IC network.

---

## 📂 Key Files
- `src/model/cayley_dickson_cuda.py`: The high-performance CUDA kernel.
- `inference/src/model.rs`: The Rust Wasm inference logic.
- `inference/src/lib.rs`: The IC Canister API (Session Manager).
- `compress.py`: Tool to convert PyTorch `.pt` -> `.spinnet` format.