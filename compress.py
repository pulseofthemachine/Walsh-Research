"""
SpinNet Model Compression Script
---------------------------------
Converts a PyTorch checkpoint to optimized binary format for Rust inference.

Output format (.spinnet v2):
- Header: model config (JSON)
- Embeddings: INT8 quantized
- Ternary weights: PACKED format (4 values per byte = 16x smaller)
- LayerNorm weights: FP16
"""

import argparse
import json
import struct
import numpy as np
import torch
from pathlib import Path

# Import packing utility from our CUDA module
try:
    from src.model.cayley_dickson_cuda import pack_ternary_weights
except ImportError:
    # Fallback: inline implementation if module not available
    def pack_ternary_weights(w: torch.Tensor) -> torch.Tensor:
        """Pack ternary weights: 4 values per byte."""
        assert w.shape[-1] % 4 == 0
        encoded = (w.int() + 1).to(torch.uint8)  # -1->0, 0->1, +1->2
        shape = w.shape[:-1] + (w.shape[-1] // 4, 4)
        encoded = encoded.view(*shape)
        packed = (encoded[..., 0] << 6) | (encoded[..., 1] << 4) | \
                 (encoded[..., 2] << 2) | encoded[..., 3]
        return packed.to(torch.uint8)


def quantize_to_bitmask_ternary(weight: torch.Tensor) -> tuple:
    """
    Convert ghost weights to bitmask + sign representation.
    Returns (bitmask, sign_bits, original_shape, sparsity_pct)
    
    Format:
    - bitmask: 1 bit per position (1 = non-zero)
    - sign_bits: 1 bit per non-zero (0 = -1, 1 = +1)
    
    This is smaller than packed format AND enables sparse iteration.
    """
    w_ternary = torch.round(weight).clamp(-1, 1).flatten()
    original_shape = weight.shape
    total = w_ternary.numel()
    
    # Create bitmask (1 = non-zero)
    nonzero_mask = (w_ternary != 0)
    
    # Pack bitmask into bytes (8 positions per byte)
    # Pad to multiple of 8
    padded_len = (total + 7) // 8 * 8
    padded_mask = torch.zeros(padded_len, dtype=torch.bool)
    padded_mask[:total] = nonzero_mask
    
    bitmask_bytes = []
    for i in range(0, padded_len, 8):
        byte_val = 0
        for bit in range(8):
            if padded_mask[i + bit]:
                byte_val |= (1 << bit)
        bitmask_bytes.append(byte_val)
    bitmask = np.array(bitmask_bytes, dtype=np.uint8)
    
    # Get signs for non-zero elements only (0 = -1, 1 = +1)
    nonzero_values = w_ternary[nonzero_mask]
    signs = (nonzero_values == 1)  # True = +1, False = -1
    
    # Pack signs into bytes
    num_nonzero = len(signs)
    padded_signs_len = (num_nonzero + 7) // 8 * 8
    padded_signs = torch.zeros(padded_signs_len, dtype=torch.bool)
    padded_signs[:num_nonzero] = signs
    
    sign_bytes = []
    for i in range(0, padded_signs_len, 8):
        byte_val = 0
        for bit in range(8):
            if padded_signs[i + bit]:
                byte_val |= (1 << bit)
        sign_bytes.append(byte_val)
    sign_bits = np.array(sign_bytes, dtype=np.uint8)
    
    # Calculate sparsity
    sparsity_pct = 100.0 * (1 - num_nonzero / total)
    
    return bitmask, sign_bits, original_shape, sparsity_pct, num_nonzero


def quantize_embedding(embedding: torch.Tensor) -> tuple:
    """
    Quantize embedding to INT8 with scale factor.
    Returns (quantized_data, scale, zero_point).
    """
    # Compute scale and zero point for symmetric quantization
    max_val = embedding.abs().max().item()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    
    # Quantize
    quantized = torch.round(embedding / scale).clamp(-127, 127).to(torch.int8)
    
    return quantized.numpy(), np.float32(scale)


def write_array(f, arr: np.ndarray, dtype_code: str):
    """Write numpy array with header: dtype_code(1) + ndim(1) + shape(4*ndim) + data"""
    dtype_map = {'i8': 0, 'i32': 1, 'f16': 2, 'f32': 3}
    f.write(struct.pack('B', dtype_map[dtype_code]))
    f.write(struct.pack('B', arr.ndim))
    for dim in arr.shape:
        f.write(struct.pack('<I', dim))
    f.write(arr.tobytes())


def compress_model(checkpoint_path: str, output_path: str, vocab_path: str = None):
    """
    Compress a SpinNet checkpoint to .spinnet format.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model config
    model_args = checkpoint.get('model_args', {})
    state_dict = checkpoint['model']
    
    # Clean up compiled model prefixes
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            k = k[len('_orig_mod.'):]
        cleaned_state[k] = v
    state_dict = cleaned_state
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Compressing to: {output_path}")
    
    with open(output_path, 'wb') as f:
        # Magic number
        f.write(b'SPIN')
        
        # Version (4 = bitmask ternary format - smallest + sparse)
        f.write(struct.pack('<H', 4))
        
        # Config as JSON
        config_json = json.dumps(model_args).encode('utf-8')
        f.write(struct.pack('<I', len(config_json)))
        f.write(config_json)
        
        # Process layers
        layer_count = 0
        ternary_count = 0
        embed_count = 0
        norm_count = 0
        
        for name, param in state_dict.items():
            param_data = param.detach()
            
            # Determine parameter type
            if 'tok_embeddings' in name or 'output.weight' in name:
                # Embedding layer - INT8 quantize
                print(f"  [EMBED] {name}: {param_data.shape}")
                quantized, scale = quantize_embedding(param_data)
                
                # Write marker + name
                f.write(b'E')  # Embedding marker
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                
                # Write scale + data
                f.write(struct.pack('<f', scale))
                write_array(f, quantized, 'i8')
                embed_count += 1
                
            elif '.weight' in name and ('wq' in name or 'wk' in name or 'wv' in name or 
                                         'wo' in name or 'gate_proj' in name or 
                                         'up_proj' in name or 'down_proj' in name):
                # Ternary Octonion weight - BITMASK format (smallest size + sparse iteration)
                bitmask, sign_bits, shape, sparsity, num_nonzero = quantize_to_bitmask_ternary(param_data)
                total = 1
                for d in shape:
                    total *= d
                packed_size = total // 4  # What packed format would use
                bitmask_size = len(bitmask) + len(sign_bits)
                savings = 100 * (1 - bitmask_size / packed_size) if packed_size > 0 else 0
                print(f"  [BITMASK] {name}: {param_data.shape} ({sparsity:.1f}% sparse, {num_nonzero} non-zero, {savings:.0f}% smaller than packed)")
                
                # Write marker + name
                f.write(b'B')  # Bitmask Weight marker
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                
                # Write original shape
                f.write(struct.pack('<B', len(shape)))
                for dim in shape:
                    f.write(struct.pack('<I', dim))
                
                # Write num_nonzero
                f.write(struct.pack('<I', num_nonzero))
                
                # Write bitmask and sign arrays
                f.write(struct.pack('<I', len(bitmask)))
                f.write(bitmask.tobytes())
                f.write(struct.pack('<I', len(sign_bits)))
                f.write(sign_bits.tobytes())
                
                ternary_count += 1
                
            elif '.beta' in name:
                # Learnable scale (beta) - FP16
                print(f"  [SCALE] {name}: {param_data.shape}")
                f.write(b'S')
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                write_array(f, param_data.to(torch.float16).numpy(), 'f16')
                
            elif 'norm' in name or 'RMSNorm' in name.lower():
                # LayerNorm weights - FP16
                print(f"  [NORM] {name}: {param_data.shape}")
                f.write(b'N')
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                write_array(f, param_data.to(torch.float16).numpy(), 'f16')
                norm_count += 1
                
            elif 'freqs_cis' in name:
                # RoPE frequencies - FP32 complex
                print(f"  [ROPE] {name}: {param_data.shape}")
                f.write(b'R')
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                # Store as real/imag pairs
                rope_data = torch.view_as_real(param_data).to(torch.float32).numpy()
                write_array(f, rope_data, 'f32')
                
            else:
                print(f"  [SKIP] {name}: {param_data.shape}")
            
            layer_count += 1
        
        # End marker
        f.write(b'END!')
    
    # Report
    file_size = output_path.stat().st_size
    print(f"\nCompression complete!")
    print(f"  Embeddings: {embed_count}")
    print(f"  Ternary weights: {ternary_count}")
    print(f"  Norm layers: {norm_count}")
    print(f"  Output size: {file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress SpinNet model for Rust inference")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("-o", "--output", default=None, help="Output path (.spinnet)")
    parser.add_argument("--vocab", default=None, help="Path to tokenizer vocab (optional)")
    
    args = parser.parse_args()
    
    output = args.output
    if output is None:
        output = Path(args.checkpoint).with_suffix('.spinnet')
    
    compress_model(args.checkpoint, output, args.vocab)
