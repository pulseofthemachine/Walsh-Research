"""
SpinNet Text Generation
-----------------------
Generate text from any SpinNet checkpoint.

Usage:
    python generate.py --ckpt experiments/out-tinystories-octonion/ckpt.pt --prompt "Once upon a time"
    python generate.py --ckpt experiments/scholar_pilot/ckpt.pt --prompt "The capital of France"
"""
import os
import argparse
import time
import torch
import tiktoken
from src.model import SpinNetConfig, SpinNet

# Try to import CUDA optimizations (optional)
try:
    from src.model.cayley_dickson_cuda import optimize_for_inference
    HAS_CUDA_OPT = True
except ImportError:
    HAS_CUDA_OPT = False

def main():
    parser = argparse.ArgumentParser(description='Generate text from SpinNet checkpoint')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file (e.g., experiments/out-tinystories-octonion/ckpt.pt)')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='Starting prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more focused)')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling (0 = disabled)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Loading model from {args.ckpt}...")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = SpinNetConfig(**checkpoint['model_args'])
    
    print(f"Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Algebra: {getattr(config, 'algebra', 'octonion')} | Head mixing: {getattr(config, 'head_mixing', False)} | Hash embeddings: {getattr(config, 'hash_embeddings', False)}")
    
    model = SpinNet(config)
    
    # Load weights (handle compiled model prefix)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Apply CUDA inference optimizations if available
    if device == 'cuda' and HAS_CUDA_OPT:
        optimize_for_inference(model)
        print("CUDA inference optimization enabled.")

    # Set up tokenizer based on vocab size
    if config.vocab_size <= 256:
        # Char-level tokenizer
        print("Using char-level tokenizer")
        stoi = {chr(i): i for i in range(256)}
        itos = {i: chr(i) for i in range(256)}
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    else:
        # GPT-2 tokenizer
        print("Using GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special=set(enc.special_tokens_set))
        decode = lambda l: enc.decode(l)

    # Encode prompt
    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print("=" * 60)
    print(f"PROMPT: {args.prompt}")
    print("=" * 60)

    # Generate samples
    with torch.no_grad():
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad()
        with ctx:
            for k in range(args.num_samples):
                t0 = time.time()
                y = model.generate(x, args.max_tokens, temperature=args.temperature, top_k=args.top_k)
                t1 = time.time()
                tps = args.max_tokens / (t1 - t0)
                print(f"\n--- SAMPLE {k+1} ({tps:.1f} tok/s) ---")
                print(decode(y[0].tolist()))
                print()

if __name__ == "__main__":
    main()
