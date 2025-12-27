# SpinNet MNIST Image Generation Config
# --------------------------------------
# Autoregressive image generation using patch tokens.
# Each 28x28 MNIST image = 49 patches (7x7 of 4x4 patches)
# 
# Uses the same Hadamard + hash embedding architecture for images!

out_dir = 'experiments/out-mnist-hadamard'
eval_interval = 200
eval_iters = 50
log_interval = 20

# Data
dataset = 'mnist'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 50  # 49 patches + 1 separator per image

# Model - Small for fast iteration
n_layer = 6
n_head = 32           # Divisible by 32 for Hadamard
n_embd = 256          # Smaller for images
dropout = 0.0
bias = False
vocab_size = 257      # 0-255 grayscale + separator token

# SpinNet-specific
algebra = "hadamard"
head_mixing = True
hash_embeddings = True  # Test hash embeddings on images!

# Optimizer
learning_rate = 3e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-4

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
