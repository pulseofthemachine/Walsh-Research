"""
Walsh Loader - Load .walsh files back into PyTorch
-------------------------------------------------------
Loads compressed .walsh models for inference/verification in PyTorch.

Supports all weight types:
- B: Bitmask ternary weights (octonion linear layers)
- E: Embeddings (INT8 quantized)
- S: Scale/beta parameters (FP16)
- N: Norm weights (FP16)
- R: RoPE frequencies (FP32 complex)
- H: Head mixer weights (FP16)
- h: Head mixer beta (FP16)

Usage:
    python generate_compressed.py inference/ckpt_v2.walsh --prompt "Once upon a time"
"""

import argparse
import json
import struct
import numpy as np
import torch
import tiktoken
from typing import Dict, Tuple, Any


# =============================================================================
# BINARY READING UTILITIES
# =============================================================================

def read_u8(f) -> int:
    return struct.unpack('B', f.read(1))[0]

def read_u16(f) -> int:
    return struct.unpack('<H', f.read(2))[0]

def read_u32(f) -> int:
    return struct.unpack('<I', f.read(4))[0]

def read_f32(f) -> float:
    return struct.unpack('<f', f.read(4))[0]

def read_name(f) -> str:
    """Read length-prefixed UTF-8 string."""
    name_len = read_u16(f)
    return f.read(name_len).decode('utf-8')

def read_array(f) -> np.ndarray:
    """Read numpy array with header: dtype_code(1) + ndim(1) + shape(4*ndim) + data"""
    dtype_map = {0: np.int8, 1: np.int32, 2: np.float16, 3: np.float32}
    dtype_code = read_u8(f)
    ndim = read_u8(f)
    shape = tuple(read_u32(f) for _ in range(ndim))
    dtype = dtype_map[dtype_code]
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    data = np.frombuffer(f.read(size), dtype=dtype).reshape(shape)
    return data.copy()


# =============================================================================
# WEIGHT UNPACKING
# =============================================================================

def unpack_bitmask_ternary(bitmask: bytes, sign_bits: bytes, 
                           shape: tuple, num_nonzero: int) -> torch.Tensor:
    """
    Reconstruct ternary weights from bitmask + sign representation.
    
    Format:
    - bitmask: 1 bit per position (1 = non-zero)
    - sign_bits: 1 bit per non-zero (0 = -1, 1 = +1)
    """
    total = int(np.prod(shape))
    positions = np.zeros(total, dtype=np.float32)
    
    # Extract non-zero indices from bitmask
    nonzero_indices = []
    for byte_idx, byte_val in enumerate(bitmask):
        for bit in range(8):
            pos = byte_idx * 8 + bit
            if pos >= total:
                break
            if byte_val & (1 << bit):
                nonzero_indices.append(pos)
    
    # Extract signs (0 = -1, 1 = +1)
    signs = []
    for byte_idx, byte_val in enumerate(sign_bits):
        for bit in range(8):
            idx = byte_idx * 8 + bit
            if idx >= num_nonzero:
                break
            signs.append(1.0 if (byte_val & (1 << bit)) else -1.0)
    
    # Reconstruct tensor
    for i, pos in enumerate(nonzero_indices[:num_nonzero]):
        positions[pos] = signs[i]
    
    return torch.tensor(positions.reshape(shape), dtype=torch.float32)


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_walsh(path: str, verbose: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a .walsh file and return state_dict + config.
    
    Returns:
        (state_dict, config): Model weights and configuration
    """
    if verbose:
        print(f"Loading Walsh model from {path}")
    
    state_dict = {}
    
    with open(path, 'rb') as f:
        # === HEADER ===
        magic = f.read(4)
        if magic != b'SPIN':
            raise ValueError(f"Invalid magic number: {magic}")
        
        version = read_u16(f)
        if verbose:
            print(f"  Format version: {version}")
        if version not in (3, 4):
            raise ValueError(f"Unsupported version: {version} (expected 3 or 4)")
        
        # Config JSON
        config_len = read_u32(f)
        config_json = f.read(config_len).decode('utf-8')
        config = json.loads(config_json)
        if verbose:
            print(f"  Config: {config}")
        
        # === WEIGHT CHUNKS ===
        while True:
            marker = f.read(1)
            if not marker:
                break
            
            # Check for END marker
            if marker == b'E':
                peek = f.read(3)
                if peek == b'ND!':
                    break
                else:
                    # It's an Embedding marker, rewind
                    f.seek(-3, 1)
            
            # Read weight name
            name = read_name(f)
            
            if marker == b'E':
                # === EMBEDDING (INT8 + scale) ===
                scale = read_f32(f)
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32) * scale)
                state_dict[name] = tensor
                if verbose:
                    print(f"  [EMBED] {name}: {tensor.shape}")
                
            elif marker == b'B':
                # === BITMASK TERNARY WEIGHT ===
                ndim = read_u8(f)
                shape = tuple(read_u32(f) for _ in range(ndim))
                num_nonzero = read_u32(f)
                
                bitmask_len = read_u32(f)
                bitmask = f.read(bitmask_len)
                
                sign_len = read_u32(f)
                sign_bits = f.read(sign_len)
                
                tensor = unpack_bitmask_ternary(bitmask, sign_bits, shape, num_nonzero)
                state_dict[name] = tensor
                if verbose:
                    print(f"  [WEIGHT] {name}: {tensor.shape}")
                
            elif marker == b'S':
                # === SCALE (beta) - FP16 ===
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                if verbose:
                    print(f"  [SCALE] {name}: {tensor.shape}")
                
            elif marker == b'N':
                # === NORM WEIGHT - FP16 ===
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                if verbose:
                    print(f"  [NORM] {name}: {tensor.shape}")
                
            elif marker == b'R':
                # === ROPE - FP32 complex ===
                data = read_array(f)
                # data is [seq_len, dim/2, 2] (real/imag pairs)
                tensor = torch.view_as_complex(torch.tensor(data))
                state_dict[name] = tensor
                if verbose:
                    print(f"  [ROPE] {name}: {tensor.shape}")
                    
            elif marker == b'H':
                # === HEAD MIXER WEIGHTS - FP16 [8, D, D] ===
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                if verbose:
                    print(f"  [HEAD_MIXER] {name}: {tensor.shape}")
                    
            elif marker == b'h':
                # === HEAD MIXER BETA - FP16 [D] ===
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                if verbose:
                    print(f"  [HEAD_MIXER_BETA] {name}: {tensor.shape}")
                
            else:
                if verbose:
                    print(f"  [UNKNOWN] marker={marker}, name={name}")
                # Try to skip unknown data gracefully
                break
    
    return state_dict, config


# =============================================================================
# GENERATION
# =============================================================================

def sample_from_walsh(walsh_path: str, prompt: str, max_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 50,
                        device: str = 'cuda', verbose: bool = True):
    """Load a .walsh and generate text."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import WalshConfig, Walsh
    
    # Load weights
    state_dict, config = load_walsh(walsh_path, verbose=verbose)
    
    # Create model
    model_config = WalshConfig(**config)
    model = Walsh(model_config)
    
    # Load state dict (strict=False to handle any missing buffers)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose and missing:
        print(f"  Missing keys: {missing}")
    if verbose and unexpected:
        print(f"  Unexpected keys: {unexpected}")
    
    model.to(device).eval()
    
    # Enable fused CUDA kernels for faster inference
    if device == 'cuda':
        try:
            from src.model.cayley_dickson_cuda import optimize_for_inference
            model = optimize_for_inference(model)
            if verbose:
                print("  Enabled fused CUDA kernels")
        except ImportError:
            pass
    
    # Tokenize
    if config.get('vocab_size', 50257) <= 256:
        # Char-level tokenizer
        if verbose:
            print("Using char-level tokenizer")
        encode = lambda s: [ord(c) for c in s]
        decode = lambda l: ''.join([chr(i) for i in l])
    else:
        # GPT-2 tokenizer
        if verbose:
            print("Using GPT-2 tokenizer")
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special=set(enc.special_tokens_set))
        decode = lambda l: enc.decode(l)
    
    tokens = encode(prompt)
    x = torch.tensor(tokens, device=device, dtype=torch.long)[None, ...]
    
    # Generate
    print(f"\nPrompt: '{prompt}'")
    print("-" * 60)
    
    import time
    t0 = time.time()
    with torch.no_grad():
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad()
        with ctx:
            y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
    t1 = time.time()
    
    output = decode(y[0].tolist())
    tps = max_tokens / (t1 - t0)
    
    print(f"\nOutput ({tps:.1f} tok/s):")
    print(output)
    
    return output


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and generate from .walsh model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_compressed.py inference/ckpt_v2.walsh --prompt "Once upon a time"
    python generate_compressed.py model.walsh --prompt "The quick brown" --max_tokens 50
        """
    )
    parser.add_argument("walsh", help="Path to .walsh file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--quiet", action="store_true", help="Suppress loading messages")
    
    args = parser.parse_args()
    
    sample_from_walsh(
        args.walsh, 
        args.prompt, 
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.device,
        verbose=not args.quiet
    )
