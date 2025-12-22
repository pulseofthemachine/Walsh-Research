# SpinNet TinyStories Config with Octonion Attention
# Token-level (GPT-2 tokenizer) training for better semantic evaluation

out_dir = 'experiments/out-tinystories-octonion'
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

# Model (with Octonion Attention)
n_layer = 8
n_head = 8  # Must be multiple of 8 for octonion attention
n_embd = 512
vocab_size = 50304  # GPT-2 vocab padded to multiple of 64
octonion_attention = True  # Octonion head mixing

dropout = 0.0
bias = False

# Optimizer
learning_rate = 6e-3  # Slightly lower for token-level
max_iters = 25000
weight_decay = 0.1  # Standard for token-level training
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
