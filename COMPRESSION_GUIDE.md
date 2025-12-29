# Walsh Compression Guide

This guide covers how to compress Walsh models for efficient storage and deployment using a TinyStories-trained model as an example.

CURRENTLY SUPPORTS OCTONION (8D) ONLY

## Overview

The compression pipeline converts PyTorch checkpoints (`.pt`) to optimized `.walsh` format:

| Metric | ckpt.pt | model.walsh |
|--------|---------|---------------|
| **File size** | 334 MB | 26 MB (7.7%) |
| **GPU memory** | 445 MB | 120 MB (27%) |
| **Peak memory** | 512 MB | 178 MB (35%) |
| **Speed** | 36.6 tok/s | 38.3 tok/s |

## Quick Start

### 1. Compress a Trained Model

```bash
python compress.py experiments/out-tinystories-octonion/ckpt.pt -o model.walsh
```

### 2. Generate from Compressed Model

```bash
python generate_compressed.py model.walsh --prompt "Once upon a time" --max_tokens 100
```

## Format Details

The `.walsh` format uses multiple compression techniques:

### Weight Types

| Marker | Type | Compression |
|--------|------|-------------|
| `B` | Bitmask ternary | ~90% sparse → huge savings |
| `E` | Embeddings | INT8 quantized |
| `N` | Norm weights | FP16 |
| `S` | Scale (beta) | FP16 |
| `H` | Head mixer W | FP16 |
| `h` | Head mixer beta | FP16 |
| `R` | RoPE frequencies | FP32 complex |

### Bitmask Ternary Format

For ternary weights (`{-1, 0, +1}`), we use bitmask + sign encoding:
- **Bitmask**: 1 bit per position (1 = non-zero)
- **Sign bits**: 1 bit per non-zero (0 = -1, 1 = +1)

At ~90% sparsity (typical for trained Walsh), this is much smaller than 2-bit packed format.

## Usage Examples

### Basic Generation
```bash
python generate_compressed.py model.walsh --prompt "Hello world"
```

### Quiet Mode (suppress loading logs)
```bash
python generate_compressed.py model.walsh --prompt "Hello" --quiet
```

### Custom Parameters
```bash
python generate_compressed.py model.walsh \
    --prompt "The scientist discovered" \
    --max_tokens 200 \
    --temperature 0.9 \
    --top_k 40
```

### CPU Inference
```bash
python generate_compressed.py model.walsh --prompt "Hello" --device cpu
```

## Python API

```python
from generate_compressed import load_walsh
from src.model import Walsh, WalshConfig

# Load compressed model
state_dict, config = load_walsh("model.walsh")

# Create and initialize model
model = Walsh(WalshConfig(**config))
model.load_state_dict(state_dict, strict=False)
model.to("cuda").eval()

# Generate
tokens = model.generate(prompt_ids, max_new_tokens=100)
```

## ICP Deployment

For Internet Computer deployment, the `.walsh` format is loaded by the Rust inference engine:

```bash
# Compress
python compress.py experiments/out-tinystories-octonion/ckpt.pt -o inference/ckpt_v2.walsh

# Deploy
cd inference
dfx start --background
dfx deploy

# Test
./verify_single_user.sh "Once upon a time" 50
```
