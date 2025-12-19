# SpinNet CUDA Optimization Guide

This project includes custom CUDA kernels for the Cayley-Dickson octonion algebra, providing significant speedups for inference.

## Features

- **5x Faster Inference**: Fused Triton kernel reduces 64 separate kernel launches to 1.
- **Drop-in Compatible**: Works with existing checkpoints and models.
- **Ternary Weight Support**: Fully supports BitNet-style ternary weights.

## Usage

### 1. Training (Automatic)

The fused kernel is now **fully supported and recommended** for training. It provides a significant speedup per iteration (depending on model size) and cleaner gradients due to FP32 accumulation. The system automatically selects this implementation if CUDA is available.

```bash
python train.py config/train_crypto.py
```

### 2. Fast Inference

The fused kernel provides massive speedups for **both** latency and throughput workloads. In `btcsniper2.py`, we explicitly call `optimize_for_inference(model)` to enable zero-copy optimizations alongside the fused kernel.

## Benchmarks

### Inference (Forward Pass)

| Implementation | Latency (B=1) | Throughput (B=100) |
|----------------|---------------|--------------------|
| Original       | 2.10 ms       | ~260 ms            |
| Fused CUDA     | **0.35 ms**   | ~122 ms            |
| **Result**     | **6x Faster** | **2.1x Faster**    |

### Training (Forward + Backward) on TinyShakespeare

| Implementation | Iteration Time | Speedup |
|----------------|----------------|---------|
| Original       | 646 ms/step    | 1x      |
| Fused CUDA     | **106 ms/step**| **6.05x**|
