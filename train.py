"""
SpinNet Training Script
-----------------------
Training loop for Octonion-Ternary Transformer models.
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch._dynamo import disable
from src.model import SpinNetConfig, SpinNet

# -----------------------------------------------------------------------------
# DEFAULT CONFIG (Overridden by config/train_fineweb.py)
# -----------------------------------------------------------------------------
out_dir = 'out-spinnet'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True 
init_from = 'scratch' 

# Data
dataset = 'fineweb'
gradient_accumulation_steps = 5 
batch_size = 12 
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False 
octonion_attention = True  # Enable octonion head mixing

# AdamW
learning_rate = 6e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

# Schedule
decay_lr = True
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 6e-5 

# System
backend = 'nccl' 
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 

# -----------------------------------------------------------------------------
# CONFIG LOADER
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides globals()
config = {k: globals()[k] for k in config_keys} 

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
if 'cudnn_benchmark' in config and config['cudnn_benchmark']:
    torch.backends.cudnn.benchmark = True
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# DATA LOADER (Memmap Optimized - FIXED)
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

# 1. INITIALIZE MEMMAPS ONCE (Globally)
# We keep these open so the OS caches the pages efficiently.
train_data = None
val_data = None

if os.path.exists(os.path.join(data_dir, 'train.bin')):
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)

if os.path.exists(os.path.join(data_dir, 'val.bin')):
    val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)

def get_batch(split):
    # Use the global open maps
    data = train_data if split == 'train' else val_data
    
    # Safety check for small datasets / missing splits
    if data is None: 
        # Fallback or error handling if val set is missing
        return torch.zeros((batch_size, block_size), dtype=torch.long, device=device), \
               torch.zeros((batch_size, block_size), dtype=torch.long, device=device)

    if len(data) <= block_size:
        ix = torch.randint(len(data)-1, (batch_size,))
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
    # OPTIMIZATION: Convert to Tensor directly from uint16, then cast to long on GPU if needed
    # This avoids creating a massive temporary int64 numpy array in CPU RAM
    x_stack = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y_stack = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # Pin memory speeds up host-to-device transfer
        if x_stack.device.type == 'cpu':
            x = x_stack.pin_memory().to(device, non_blocking=True)
            y = y_stack.pin_memory().to(device, non_blocking=True)
        else:
            x = x_stack.to(device)
            y = y_stack.to(device)
    else:
        x, y = x_stack.to(device), y_stack.to(device)
        
    return x, y

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size'] if 'vocab_size' in meta else len(meta['stoi'])
    print(f"Found vocab_size = {meta_vocab_size}")

iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 (50304)")
    # Get octonion_attention from config if set, default to False
    octonion_attention = config.get('octonion_attention', False)
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout, 
                      octonion_attention=octonion_attention)
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = SpinNetConfig(**model_args)
    model = SpinNet(gptconf)
    
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # Important: Update the global model_args so saving works later
    model_args = checkpoint_model_args

    # Force fix for bias if missing
    if 'bias' not in checkpoint_model_args:
        checkpoint_model_args['bias'] = False
        
    gptconf = SpinNetConfig(**checkpoint_model_args)
    model = SpinNet(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Extract iteration number
    iter_num = checkpoint['iter_num'] if 'iter_num' in checkpoint else 0
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']

model.to(device)

# -----------------------------------------------------------------------------
# COMMAND CENTER: METRICS REPORT
# -----------------------------------------------------------------------------
print("-" * 60)
print(">>> SPINNET: LAUNCH SEQUENCE <<<")
print("-" * 60)

# Parameter Count
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compute actual batch sizes and tokens
effective_batch = gradient_accumulation_steps * batch_size
tokens_per_iter = effective_batch * block_size

# Dataset size info
train_tokens = len(train_data) if train_data is not None else 0
val_tokens = len(val_data) if val_data is not None else 0
iters_per_epoch = train_tokens // tokens_per_iter if tokens_per_iter > 0 else 0

print(f"MODEL:")
print(f"  Params       : {n_params/1e6:.2f}M total ({n_trainable/1e6:.2f}M trainable)")
print(f"  Architecture : {n_layer} layers, {n_head} heads, {n_embd//8} dim × 8 octonion")
print(f"  Context      : {block_size} tokens")
print(f"  Octonion Attn: {'Enabled' if config.get('octonion_attention', False) else 'Disabled'}")

print(f"DATA:")
print(f"  Dataset      : {dataset}")
print(f"  Train Tokens : {train_tokens/1e6:.1f}M")
print(f"  Val Tokens   : {val_tokens/1e6:.1f}M")

print(f"TRAINING:")
print(f"  Batch Size   : {batch_size} x {gradient_accumulation_steps} = {effective_batch} effective")
print(f"  Tokens/Iter  : {tokens_per_iter:,}")
print(f"  Iters/Epoch  : ~{iters_per_epoch:,}")
print(f"  Max Iters    : {max_iters:,}")
print(f"  Est. Epochs  : ~{max_iters / iters_per_epoch:.1f}" if iters_per_epoch > 0 else "  Est. Epochs  : N/A")

print(f"COMPUTE:")
flops_per_token = 6 * n_params
flops_per_iter = flops_per_token * tokens_per_iter
print(f"  FLOPs/Iter   : {flops_per_iter:.2e}")
print(f"  Device       : {device} ({dtype})")
print(f"  Compile      : {'Enabled' if compile else 'Disabled'}")

print("-" * 60)


# -----------------------------------------------------------------------------
# OPTIMIZER
# -----------------------------------------------------------------------------
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile:
    print("Compiling SpinNet (Default Mode)...")
    model = torch.compile(model, mode="default")

# -----------------------------------------------------------------------------
# UTILS & LOOPS (Enhanced Telemetry)
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Evaluate train/val loss over eval_iters batches."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 

while True:
    # 1. Update LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Evaluation & Checkpointing
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        print(f"\n[EVAL] step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if always_save_checkpoint:
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'config': config,
                }
                print(f"[SAVE] saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                # RAM Cleanup
                del checkpoint
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    # 3. Forward / Backward Pass
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        X, Y = get_batch('train') 
        loss.backward()
    
    # 4. Gradient Clipping & Telemetry Extraction
    if grad_clip != 0.0:
        # clip_grad_norm_ returns the Total Norm of the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    else:
        # If clipping is off, calculate it manually for logging
        grad_norm = 0.0
        # (Skipping manual calc to save time if clip is off, usually clip is on)
        
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 5. Performance Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        
        # --- QUICK HEALTH CHECK (Middle Layer) ---
        # We peek at the Ghost Weights of the middle layer to ensure they aren't exploding
        # A healthy ternary model usually has w_std between 0.2 and 0.6
        mid_layer = model.layers[len(model.layers)//2]
        # Access the raw ghost weights (wq.weight)
        w_stats = mid_layer.attention.wq.weight.detach()
        w_std = w_stats.std().item()
        
        # Format: Iter | Loss | Time | GradNorm | WeightStd | LR
        print(f"iter {iter_num:>5} | loss {lossf:.4f} | time {dt*1000:.0f}ms | "
              f"grad_n {grad_norm:.4f} | w_std {w_std:.4f} | lr {lr:.2e}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        print(">>> MISSION COMPLETE <<<")
        break
