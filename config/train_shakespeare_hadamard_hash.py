# Walsh TinyShakespeare Config - Hadamard 32D with Hash Embeddings
# Small test config for validating architecture + Rust inference
#
# Character-level tokenization (vocab_size=65)
# Quick training for testing pipeline

out_dir = 'experiments/out-shakespeare-hadamard-hash'
eval_interval = 100
eval_iters = 50
log_interval = 10

# Data
dataset = 'tinyshakespeare'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# Model - Small Hadamard 32D with Hash Embeddings
n_layer = 4
n_head = 32           # Must be divisible by 32 for Hadamard
n_embd = 256          # Must be divisible by 32 for Hadamard
dropout = 0.0
bias = False
vocab_size = 65       # Character-level (a-z, A-Z, 0-9, punctuation)

# Walsh-specific
algebra = "hadamard"     # Use 32D Hadamard algebra (O(n log n) mixing)
head_mixing = True       # Enable Hadamard head mixing
hash_embeddings = True   # Use composite hash embeddings

# Optimizer
learning_rate = 3e-3     # Higher LR for smaller model
max_iters = 2000         # Quick test
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Schedule (Two-Stage BitNet-style)
decay_lr = True
warmup_iters = 0
lr_decay_iters = 2000
min_lr = 3e-4

# Two-stage schedule (BitNet b1.58)
two_stage_schedule = True       # Enable two-stage LR and WD
cooldown_start = 0.5            # Stage 2 starts at 50% of max_iters (iter 1000)
cooldown_lr = 3e-4              # Stage 2 peak LR (10x lower than stage 1)
stage1_wd_peak = 0.1            # Weight decay peaks at 0.1 during stage 1
stage2_wd = 0.0                 # Weight decay disabled during stage 2

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Disable compile for flexibility
