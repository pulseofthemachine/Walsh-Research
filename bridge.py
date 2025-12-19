"""
SpinNet Bridge (Python -> Rust)
-------------------------------
Converts the Pickle (.spin) into a raw Binary stream (.bin)
that Rust can read via memory mapping.
"""
import pickle
import struct
import numpy as np
import sys
import os

INPUT_PATH = 'experiments/out-crypto/sniper_v2.spin'
OUTPUT_PATH = 'inference/model.bin'

print(f"[BRIDGE] Loading Pickle: {INPUT_PATH}")
with open(INPUT_PATH, 'rb') as f:
    package = pickle.load(f)

args = package['meta']
layers = package['layers']

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, 'wb') as f:
    # 1. WRITE HEADER (Little Endian)
    # Magic (4 bytes) + Config (6 ints)
    print(f"[BRIDGE] Writing Header...")
    f.write(b'SPIN') 
    header = struct.pack('<IIIIII', 
        args['n_layer'], 
        args['n_embd'], 
        args['n_head'], 
        args['vocab_size'], 
        args['block_size'],
        1 if args.get('octonion', True) else 0
    )
    f.write(header)

    # 2. WRITE WEIGHTS SEQUENTIALLY
    # We follow a strict order: 
    # TokenEmb -> Layers (Attn -> MLP) -> Norm -> Head
    
    print(f"[BRIDGE] Serializing {len(layers)} tensors...")
    
    def write_dense(name, tensor_data):
        # ID: 0 = Dense
        f.write(struct.pack('<B', 0)) 
        # Shape Rank (how many dims)
        shape = tensor_data.shape
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<I', dim))
        # Data
        # Ensure float32 for safety in Rust v1, or float16 if you want to be adventurous
        # Let's stick to float32 for the 'dense' parts (Embeds/Norms) for easier Rust debug
        data = tensor_data.astype(np.float32).flatten()
        f.write(data.tobytes())

    def write_sparse(name, layer_data):
        # ID: 1 = Sparse CSR
        f.write(struct.pack('<B', 1))
        
        # Original 3D Shape
        shape = layer_data['orig_shape'] # (8, 48, 48) usually
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<I', dim))
            
        # CSR Components
        values = layer_data['values'].astype(np.int8)  # Weight values {-1, 0, 1}
        indices = layer_data['indices'].astype(np.int32) # Col indices
        indptr = layer_data['indptr'].astype(np.int32)   # Row pointers
        
        # Write sizes
        f.write(struct.pack('<III', len(values), len(indices), len(indptr)))
        
        # Write Payload
        f.write(values.tobytes())
        f.write(indices.tobytes())
        f.write(indptr.tobytes())

    # Sort keys to ensure deterministic order isn't required by Rust yet, 
    # but we will loop through known names in Rust or read dynamically.
    # For now, let's just dump everything and let Rust read key-names? 
    # Actually, simpler: Write the name length, name, then data.
    
    for name, data in layers.items():
        # Write Name Length + Name String
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('<I', len(name_bytes)))
        f.write(name_bytes)
        
        if 'type' in data and data['type'] == 'OCTONION_CSR':
            write_sparse(name, data)
        else:
            write_dense(name, data['data'])

print(f"[BRIDGE] Success. Binary payload written to {OUTPUT_PATH}")
print(f"File Size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")