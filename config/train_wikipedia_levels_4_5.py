# Config for MoE + Channel Modulation at 512D
# Uses Hadamard algebra for Walsh-Hadamard structured layers
# 512D = 16 experts × 32D Hadamard blocks (auto-scaled)

# Experiment output
out_dir = 'experiments/out-wikipedia-512'

# Model architecture
n_layer = 16
n_head = 32       # 32 heads for Hadamard (32D blocks)
n_embd = 512      # 512D = 16 × 32D Hadamard blocks (auto-scaled!)
block_size = 512
bias = False
dropout = 0.0

# Hadamard algebra (32D vs 8D octonion)
head_mixing = True
algebra = 'hadamard'
hash_embeddings = False

# mHC (Manifold Hyper-Connections)
use_mhc = True
n_streams = 4

# MoE (Mixture of Experts)
# With 512D, we get 16 experts × 32D instead of 8 × 32D
use_moe = True
moe_threshold = 0.1
moe_min_experts = 1
moe_max_experts = 6  # Increased for more experts
moe_aux_loss_weight = 0.01

# Channel Modulation
use_channel_mod = True  # 16 blocks × 32D

# Gradient checkpointing for memory efficiency
gradient_checkpointing = True

# Training
learning_rate = 3e-3
max_iters = 1000
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 3e-5

# Two-stage weight decay schedule
two_stage_schedule = True
stage1_wd_peak = 0.1
stage2_wd = 0.0
cooldown_start = 0.5
cooldown_lr = 1e-4

# Batch size (reduced for larger model)
batch_size = 8
gradient_accumulation_steps = 16  # Same effective batch

# Eval
eval_interval = 250
eval_iters = 50
always_save_checkpoint = True

# Dataset
dataset = 'wikipedia'

# Compile (disable for debugging)
compile = False

# Logging
log_interval = 10
