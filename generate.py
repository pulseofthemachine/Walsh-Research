"""
SpinNet Inference Engine
------------------------
Loads the "Wide Scholar" (1280-dim) and tests for coherence.
"""
import os
import torch
import tiktoken
from src.model import SpinNetConfig, SpinNet
from src.model.cayley_dickson_cuda import optimize_for_inference
import time

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ckpt_path = 'experiments/scholar_pilot/ckpt.pt'
start_prompt = "The capital of France is "
num_samples = 3
max_new_tokens = 100
temperature = 0.8  # < 1.0 makes it sharper/more coherent
top_k = 200        # Safety net against random garbage

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Sovereign from {ckpt_path}...")

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = SpinNetConfig(**checkpoint['model_args'])
model = SpinNet(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Apply CUDA inference optimizations (6x latency speedup)
optimize_for_inference(model)
print("CUDA inference optimization enabled.")

def benchmark_inference(model, prompt_tokens, num_tokens=100, warmup=3, runs=5):
    """
    Benchmark inference speed.
    
    Args:
        model: The SpinNet model
        prompt_tokens: Starting token tensor [1, seq_len]
        num_tokens: How many tokens to generate per run
        warmup: Number of warmup runs (not counted)
        runs: Number of timed runs
    
    Returns:
        Average tokens/sec
    """
    import time
    import torch
    
    x = prompt_tokens.clone()
    
    # Warmup (populates Triton cache)
    print("Warming up...")
    for _ in range(warmup):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate(x, num_tokens, temperature=0.8, top_k=200)
    
    # Timed runs
    torch.cuda.synchronize()
    times = []
    
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate(x, num_tokens, temperature=0.8, top_k=200)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        tps = num_tokens / (t1 - t0)
        times.append(tps)
        print(f"  Run {i+1}: {tps:.1f} tok/s")
    
    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f} tok/s")
    return avg

# -----------------------------------------------------------------------------
# GENERATE
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
print("=" * 60)
print(f"PROMPT: {start_prompt}")
print("=" * 60)
with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for k in range(num_samples):
            t0 = time.time()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            t1 = time.time()
            tps = max_new_tokens / (t1 - t0)
            print(f"\n--- SAMPLE {k+1} ({tps:.1f} tok/s) ---")
            print(decode(y[0].tolist()))
            print()