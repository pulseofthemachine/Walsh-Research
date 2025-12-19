# SpinNet Phase 11 Config: "The Scholar" (Wide Variant)
# Optimized for ~5B token FineWeb-Edu training run
# -----------------------------------------------------------------------------

out_dir = 'experiments/scholar_pilot'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' 

# Data & Gradient Dynamics
# Effective batch = 8 * 8 * 1024 = 65,536 tokens/step
dataset = 'fineweb'
gradient_accumulation_steps = 8
batch_size = 10 
block_size = 1024

# Model Architecture (124M params)
n_layer = 24
n_head = 20        # Head Dim = 64
n_embd = 1280      # Wide Chassis
dropout = 0.0      # No dropout - ternary weights provide regularization
bias = False

# Optimizer
# Note: With stable fused kernel gradients, we can use higher LR
learning_rate = 3e-3
max_iters = 75000           # ~5B tokens / 65k tokens per step
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Schedule
decay_lr = True
warmup_iters = 1000         # Longer warmup for large model stability
lr_decay_iters = 75000      # Decay over full training
min_lr = 1e-5               # Final polish phase (was 6e-4, too high)

# Hardware Optimization
compile = False             # Fused Triton kernel is faster than torch.compile
cudnn_benchmark = True
device = 'cuda'
dtype = 'bfloat16'