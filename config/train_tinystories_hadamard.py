# Walsh TinyStories Config with Hadamard 32D Algebra
# Extreme compression (1/32nd parameters) using FHT mixing

out_dir = 'experiments/out-tinystories-hadamard'
eval_interval = 200
log_interval = 20
eval_iters = 100
eval_only = False
always_save_checkpoint = True

# Data
dataset = 'tinystories'
gradient_accumulation_steps = 4  # Effective batch size: 4 * 32 = 128
batch_size = 32
block_size = 256  # Shorter context for faster iteration

# Model (Hadamard 32D)
n_layer = 8
n_head = 32  # Must be multiple of 32 for hadamard head mixing
n_embd = 512  # Must be multiple of 32 for hadamard linear
vocab_size = 50304  # GPT-2 vocab padded to multiple of 64
algebra = "hadamard"  # Use 32D Hadamard algebra
head_mixing = True    # Enable Hadamard head mixing

dropout = 0.0
bias = False

# Optimizer
learning_rate = 6e-3  # Same as octonion config
max_iters = 25000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR Schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 25000
min_lr = 1e-4

# Hardware
device = 'cuda'
dtype = 'bfloat16'
compile = False
