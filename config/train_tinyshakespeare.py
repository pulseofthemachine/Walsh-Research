# SpinNet TinyShakespeare Config (Tiny Model for Testing)
# Designed for quick reproducibility testing of the CUDA kernel

out_dir = 'experiments/out-tinyshakespeare'
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True  # Save so we can test generation

# Data
dataset = 'tinyshakespeare'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512

# Tiny Model (Fits in any GPU, fast iteration)
n_layer = 8
n_head = 8
n_embd = 256  # Must be divisible by 8 for octonions

dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-2
max_iters = 5000
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR Schedule
decay_lr = True
warmup_iters = 50
lr_decay_iters = 5000
min_lr = 1e-5

# Hardware
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Fused kernel is faster than compile
