# SpinNet TinyStories Config with SSM (State Space Model)
# Replaces attention with O(1) recurrent dynamics

out_dir = 'experiments/out-tinystories-ssm'
eval_interval = 200
log_interval = 20
eval_iters = 100
eval_only = False
always_save_checkpoint = True

# Data
dataset = 'tinystories'
gradient_accumulation_steps = 4  # Effective batch size: 4 * 32 = 128
batch_size = 32
block_size = 256  # Context window (SSM has no real limit but train on this)

# Model (SSM mode - no attention)
n_layer = 8
n_head = 8  # Not used by SSM but kept for config compatibility
n_embd = 512
vocab_size = 50304  # GPT-2 vocab padded to multiple of 64
octonion_attention = False  # Not needed for SSM
use_ssm = True  # Enable State Space Model

dropout = 0.0
bias = False

# Optimizer (slightly higher LR, SSMs often train faster)
learning_rate = 1e-3  # Higher than attention baseline
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR Schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 10000
min_lr = 1e-4

# Hardware
device = 'cuda'
dtype = 'bfloat16'
compile = False  # SSM uses torch.autograd.Function, compile may not help
