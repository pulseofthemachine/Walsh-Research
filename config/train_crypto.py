    
# SpinNet Crypto Scout Configuration
out_dir = 'experiments/out-crypto'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

dataset = 'crypto'
gradient_accumulation_steps = 4
batch_size = 64
block_size = 1024

# Model Dimensions (Nano)
n_layer = 8
n_head = 8
n_embd = 768

bias = False

# The "Market Physics" Optimizer Settings
learning_rate = 1e-3 # Conservative start. Crypto is more volatile than English.
max_iters = 5000
lr_decay_iters = 5000
min_lr = 5e-5
weight_decay = 0.2
dropout = 0.2 # Crypto is noisy, we need slight regularization here to prevent memorizing the exact chart history

warmup_iters = 100
device = 'cuda'
compile = False

  
