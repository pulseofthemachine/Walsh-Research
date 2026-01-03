import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
# Use original for training (fused backward is slow)
# For inference, see cayley_dickson_cuda.py
from .physics import OctonionTernaryLinear

# Use fused head mixer on CUDA, pure PyTorch otherwise
_USE_FUSED_HEAD_MIXER = torch.cuda.is_available()
if _USE_FUSED_HEAD_MIXER:
    try:
        from .cayley_dickson_cuda import OctonionHeadMixerFused
    except ImportError:
        _USE_FUSED_HEAD_MIXER = False

# Hadamard 32D layers for extreme compression
_HADAMARD_AVAILABLE = False
_HADAMARD_HEAD_MIXER_AVAILABLE = False
try:
    from .fht_cuda import (HadamardLinear, HadamardTernaryLinear, 
                           HadamardTernaryLinearTuple, HadamardHeadMixer, 
                           HadamardHeadMixerFused)
    _HADAMARD_AVAILABLE = True
    _HADAMARD_HEAD_MIXER_AVAILABLE = True
except ImportError:
    pass

@dataclass
class WalshConfig:
    block_size: int = 256
    vocab_size: int = 65  
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    head_mixing: bool = False  # Enable algebra-based head mixing (auto-detects type)
    algebra: str = "octonion"  # "octonion" (8D) or "hadamard" (32D)
    hash_embeddings: bool = False  # Use composite hash embeddings (25x compression)
    # MoE settings
    use_moe: bool = False  # Use Hadamard Mixture of Experts in FFN
    moe_threshold: float = 0.1  # Dynamic routing threshold
    moe_min_experts: int = 1  # Minimum experts per token
    moe_max_experts: int = 4  # Maximum experts per token
    # Channel Modulation settings
    use_channel_mod: bool = False  # Use Hadamard Channel Modulation (bulk self-interaction)


def get_linear_layer(config: WalshConfig):
    """Get the appropriate linear layer class based on algebra config.
    
    Returns:
        (LayerClass, algebra_dim): Layer class and dimension for splitting
    """
    if config.algebra == "hadamard":
        if not _HADAMARD_AVAILABLE:
            raise ImportError("Hadamard layers not available. Check fht_cuda.py import.")
        return HadamardTernaryLinearTuple, 32
    else:
        return OctonionTernaryLinear, 8


def get_head_mixer(config: WalshConfig, head_dim: int):
    """Get the appropriate head mixer based on algebra config.
    
    Auto-detects whether to use Octonion (8-head) or Hadamard (32-head) mixing.
    
    Returns:
        HeadMixer module or None if head_mixing is disabled
    """
    if not config.head_mixing:
        return None
    
    if config.algebra == "hadamard":
        # Hadamard: requires n_head divisible by 32
        if config.n_head % 32 != 0:
            raise ValueError(f"Hadamard head_mixing requires n_head % 32 == 0, got {config.n_head}")
        if _HADAMARD_HEAD_MIXER_AVAILABLE:
            return HadamardHeadMixerFused(head_dim)
        else:
            raise ImportError("Hadamard head mixer not available")
    else:
        # Octonion: requires n_head divisible by 8
        if config.n_head % 8 != 0:
            raise ValueError(f"Octonion head_mixing requires n_head % 8 == 0, got {config.n_head}")
        if _USE_FUSED_HEAD_MIXER:
            return OctonionHeadMixerFused(head_dim)
        else:
            return OctonionHeadMixer(head_dim)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class HashEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, bucket_size: int = 1021, num_tables: int = 3):
        super().__init__()
        self.bucket_size = bucket_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_tables = num_tables
        
        # 1. The Tables (Shared Pool)
        self.emb_tables = nn.ModuleList([
            nn.Embedding(bucket_size, n_embd) for _ in range(num_tables)
        ])
        
        # 2. Importance Weights (The "Noise Gate")
        # Learned scalar for every token/table combo. 
        # Cost: 50k * 3 params = 0.15M params (Tiny, but vital for quality)
        self.importance = nn.Embedding(vocab_size, num_tables)
        
        # Init: Variance scaling for tables, 1.0 for importance
        for emb in self.emb_tables:
            nn.init.normal_(emb.weight, std=0.02 / math.sqrt(num_tables))
        nn.init.normal_(self.importance.weight, mean=1.0, std=0.01) # Start close to sum()
        
        # 3. Universal Hashing Constants (Fixed, Random)
        # Replaces the 'base decomposition' logic with true scrambling
        # We generate random coefficients a, b for each table
        rng = np.random.RandomState(42)
        self.register_buffer('hash_a', torch.from_numpy(rng.randint(1, 100000, size=(num_tables, 1))).long())
        self.register_buffer('hash_b', torch.from_numpy(rng.randint(0, 100000, size=(num_tables, 1))).long())
        self.register_buffer('prime', torch.tensor(1000000007)) # Large prime > vocab_size

        # Precompute all indices for speed (memory vs compute tradeoff)
        self._precompute_indices()

    def _precompute_indices(self):
        # Calculate ((ax + b) % p) % buckets for all vocab
        # Shape: [vocab_size, num_tables]
        all_ids = torch.arange(self.vocab_size, device=self.hash_a.device)
        
        # [num_tables, vocab_size]
        indices = ((all_ids * self.hash_a + self.hash_b) % self.prime) % self.bucket_size
        self.register_buffer('all_indices', indices.T) # [vocab_size, num_tables]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        
        # 1. Get Indices: [B, T, num_tables]
        indices = F.embedding(x, self.all_indices)
        
        # 2. Get Vectors: [B, T, num_tables, n_embd]
        vectors = []
        for i in range(self.num_tables):
            vectors.append(self.emb_tables[i](indices[:, :, i]))
        vectors = torch.stack(vectors, dim=2)
        
        # 3. Get Importance Weights: [B, T, num_tables]
        weights = self.importance(x)
        
        # 4. Weighted Sum: [B, T, n_embd]
        x_emb = torch.sum(vectors * weights.unsqueeze(-1), dim=2)

        # 5. Scale Signal
        # Multiplies by ~32 (for 1024 dim), bringing variance from 0.02 -> 0.64
        return x_emb * math.sqrt(self.n_embd)

    def _get_chunk_embeddings(self, start: int, end: int) -> torch.Tensor:
        """
        Reconstructs the full embedding matrix for a chunk of the vocab.
        Used for Output Projection / Chunked Loss.
        """
        indices = self.all_indices[start:end]
        weights = self.importance.weight[start:end]
        
        chunk_emb = torch.zeros(end - start, self.n_embd, device=indices.device, dtype=weights.dtype)
        
        for i in range(self.num_tables):
            v = self.emb_tables[i](indices[:, i])
            w = weights[:, i].unsqueeze(-1)
            chunk_emb += v * w
            
        return chunk_emb * math.sqrt(self.n_embd)
    
    def _get_all_embeddings(self) -> torch.Tensor:
        """Get embeddings for all vocab tokens."""
        return self._get_chunk_embeddings(0, self.vocab_size)

    def output_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from hidden states using the same hash embedding tables.
        
        This is the "reverse" of embedding lookup - we compute dot products
        between hidden states and all vocab embeddings efficiently.
        
        Args:
            hidden: [B, T, n_embd] hidden states
            
        Returns:
            [B, T, vocab_size] logits
        """
        all_emb = self._get_all_embeddings()
        return hidden @ all_emb.t()
    
    def chunked_cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor, 
                               chunk_size: int = 4096) -> torch.Tensor:
        """
        Compute cross entropy loss in chunks to avoid materializing full logits.
        
        Memory: O(B*T*chunk_size) instead of O(B*T*vocab_size)
        """
        B, T, D = hidden.shape
        hidden_flat = hidden.view(-1, D)
        targets_flat = targets.view(-1)
        
        # First pass: compute max logit for numerical stability
        max_logit = torch.full((B * T,), float('-inf'), device=hidden.device, dtype=hidden.dtype)
        
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            chunk_emb = self._get_chunk_embeddings(start, end)
            chunk_logits = hidden_flat @ chunk_emb.t()
            max_logit = torch.maximum(max_logit, chunk_logits.max(dim=-1).values)
        
        # Second pass: compute sum(exp(logits - max))
        sum_exp = torch.zeros((B * T,), device=hidden.device, dtype=torch.float32)
        
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            chunk_emb = self._get_chunk_embeddings(start, end)
            chunk_logits = hidden_flat @ chunk_emb.t()
            sum_exp += torch.exp(chunk_logits.float() - max_logit.float().unsqueeze(-1)).sum(dim=-1)
        
        log_sum_exp = max_logit.float() + torch.log(sum_exp)
        
        # Third pass: get target logits
        target_logits = torch.zeros((B * T,), device=hidden.device, dtype=torch.float32)
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            mask = (targets_flat >= start) & (targets_flat < end)
            if mask.any():
                chunk_emb = self._get_chunk_embeddings(start, end)
                chunk_logits = hidden_flat[mask] @ chunk_emb.t()
                local_targets = targets_flat[mask] - start
                target_logits[mask] = chunk_logits.float().gather(1, local_targets.unsqueeze(-1)).squeeze(-1)
        
        loss = -target_logits + log_sum_exp
        return loss.mean()
    
    def extra_repr(self) -> str:
        std_params = self.vocab_size * self.n_embd
        hash_params = self.bucket_size * self.n_embd * self.num_tables
        return f"vocab={self.vocab_size}, dim={self.n_embd}, bucket={self.bucket_size}, compression={std_params/hash_params:.1f}x"


class KVCache:
    """
    KV Cache for efficient autoregressive generation.
    
    Stores K and V tensors per layer to avoid recomputation during generation.
    Only used for inference (generate method), not training.
    """
    
    def __init__(self, n_layers: int, max_seq_len: int, n_head: int, head_dim: int, 
                 device: torch.device, dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length (block_size)
            n_head: Number of attention heads
            head_dim: Dimension per head (n_embd // n_head)
            device: Device to allocate tensors on
            dtype: Data type for cache tensors
        """
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        
        # Pre-allocate cache tensors: [max_seq_len, n_head, head_dim] per layer
        # Using list instead of stacking for easier per-layer updates
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(max_seq_len, n_head, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(max_seq_len, n_head, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.cached_len = 0
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V and return full K, V for attention.
        
        Args:
            layer_idx: Which layer's cache to update
            k: New key tensor [batch, new_len, n_head, head_dim]
            v: New value tensor [batch, new_len, n_head, head_dim]
            
        Returns:
            (k_full, v_full): Full K/V including cache [batch, total_len, n_head, head_dim]
        """
        batch, new_len, n_head, head_dim = k.shape
        
        # Store new K, V in cache (squeeze batch dim since batch=1 for generation)
        start_pos = self.cached_len
        end_pos = start_pos + new_len
        self.k_cache[layer_idx][start_pos:end_pos] = k[0]  # [new_len, n_head, head_dim]
        self.v_cache[layer_idx][start_pos:end_pos] = v[0]
        
        # Return full K, V up to current position (add batch dim back)
        k_full = self.k_cache[layer_idx][:end_pos].unsqueeze(0)  # [1, total_len, n_head, head_dim]
        v_full = self.v_cache[layer_idx][:end_pos].unsqueeze(0)
        
        return k_full, v_full
    
    def increment(self, n: int = 1):
        """Increment cached length after processing new tokens."""
        self.cached_len += n
    
    def reset(self):
        """Reset cache for new generation session."""
        self.cached_len = 0

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class OctonionHeadMixer(nn.Module):
    """
    Mix attention heads using octonion (Cayley-Dickson) structure.
    
    After standard attention computes outputs for 8 heads, this module
    mixes them using the same algebraic structure as OctonionTernaryLinear,
    introducing non-commutativity at the head interaction level.
    
    Input: [B, 8, T, head_dim] (8 heads)
    Output: [B, 8, T, head_dim] (mixed heads)
    """
    
    def __init__(self, head_dim):
        super().__init__()
        from .constants import SIGN_TABLE, WEIGHT_IDX
        
        self.head_dim = head_dim
        # Learnable mixing weights: 8 weight matrices [head_dim, head_dim]
        self.W = nn.Parameter(torch.randn(8, head_dim, head_dim) * 0.02)
        
        # Variance-preserving beta for head mixing
        beta_init = math.sqrt(3.0 / (2.0 * head_dim))
        self.beta = nn.Parameter(torch.ones(head_dim) * beta_init)
        
        # Pre-register sign and widx tables as buffers
        self.register_buffer('signs', torch.tensor(SIGN_TABLE, dtype=torch.float32))
        self.register_buffer('widx', torch.tensor(WEIGHT_IDX, dtype=torch.long))
    
    def forward(self, x):
        """
        x: [B, 8, T, head_dim] - 8 attention head outputs
        Returns: [B, 8, T, head_dim] - mixed via octonion algebra
        """
        B, H, T, D = x.shape
        assert H == 8, f"OctonionHeadMixer requires exactly 8 heads, got {H}"
        
        # Split into 8 head tensors
        heads = [x[:, i] for i in range(8)]  # List of [B, T, D]
        
        # Compute mixed outputs using Cayley-Dickson structure
        # y_i = sum_j (sign[i,j] * heads[j] @ W[widx[i,j]].T)
        outputs = []
        for i in range(8):
            acc = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
            for j in range(8):
                sign = self.signs[i, j]
                w_idx = self.widx[i, j]
                # heads[j]: [B, T, D], W[w_idx]: [D, D]
                acc = acc + sign * (heads[j] @ self.W[w_idx].to(x.dtype))
            outputs.append(acc * self.beta.to(x.dtype))
        
        # Stack back to [B, 8, T, D]
        return torch.stack(outputs, dim=1)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.head_mixing = config.head_mixing
        
        # Get algebra-specific layer and dimension
        LinearLayer, self.algebra_dim = get_linear_layer(config)
        
        # Mixer dimension for head grouping (8 for octonion, 32 for hadamard)
        self.mixer_dim = 32 if config.algebra == "hadamard" else 8
        
        self.wq = LinearLayer(config.n_embd, config.n_embd)
        self.wk = LinearLayer(config.n_embd, config.n_embd)
        self.wv = LinearLayer(config.n_embd, config.n_embd)
        self.wo = LinearLayer(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        
        # Head mixer (auto-detects octonion vs hadamard based on algebra config)
        self.head_mixer = get_head_mixer(config, self.head_dim)

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None, layer_idx: int = 0):
        B, T, C = x.size()
        x_parts = x.split(C // self.algebra_dim, dim=-1)
        q_parts = self.wq(x_parts)
        k_parts = self.wk(x_parts)
        v_parts = self.wv(x_parts)
        
        q = torch.cat(q_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        k = torch.cat(k_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        v = torch.cat(v_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # KV Cache: store new K,V and retrieve full sequence
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            # k, v now have shape [B, cached_len + T, n_head, head_dim]
        
        # Transpose for attention: [B, n_head, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # When using cache, we still need causal mask for the query positions
        # but with respect to all cached + new positions
        # is_causal=True only works when q_len == k_len, so use manual mask when cached
        if kv_cache is not None and kv_cache.cached_len > 0:
            # New tokens can attend to all cached tokens plus themselves (causally)
            # q: [B, n_head, T, head_dim], k: [B, n_head, cached_len + T, head_dim]
            total_len = k.size(2)
            # Build causal mask: each query position can see all previous + itself
            # Query position i (relative to cached_len) can see positions 0..cached_len+i
            causal_mask = torch.ones(T, total_len, dtype=torch.bool, device=q.device).tril(diagonal=total_len - T)
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=causal_mask,
                dropout_p=self.dropout if self.training else 0
            )
        else:
            # No cache or first forward: use standard causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        
        # y: [B, n_head, T, head_dim]
        
        # Algebra-based head mixing (auto-detected: octonion=8 heads, hadamard=32 heads)
        if self.head_mixer is not None:
            # Group heads into sets for mixing
            n_groups = self.n_head // self.mixer_dim
            y_groups = y.view(B, n_groups, self.mixer_dim, T, self.head_dim)
            y_mixed = []
            for g in range(n_groups):
                y_mixed.append(self.head_mixer(y_groups[:, g]))
            y = torch.cat(y_mixed, dim=1)  # [B, n_head, T, head_dim]
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y_parts = y.split(C // self.algebra_dim, dim=-1)
        y_out_parts = self.wo(y_parts)
        return torch.cat(y_out_parts, dim=-1)

class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get algebra-specific layer and dimension
        LinearLayer, self.algebra_dim = get_linear_layer(config)
        
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3) 
        hidden_dim = math.ceil(hidden_dim / self.algebra_dim) * self.algebra_dim
        self.gate_proj = LinearLayer(config.n_embd, hidden_dim)
        self.up_proj = LinearLayer(config.n_embd, hidden_dim)
        self.down_proj = LinearLayer(hidden_dim, config.n_embd)

    def forward(self, x):
        in_dim = x.shape[-1] // self.algebra_dim
        x_parts = x.split(in_dim, dim=-1)
        gate_out = torch.cat(self.gate_proj(x_parts), dim=-1)
        up_out = torch.cat(self.up_proj(x_parts), dim=-1)
        h = F.silu(gate_out) * up_out
        h_dim = h.shape[-1] // self.algebra_dim
        h_parts = h.split(h_dim, dim=-1)
        out_parts = self.down_proj(h_parts)
        return torch.cat(out_parts, dim=-1)

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        
        # Choose FFN type: MoE or dense
        if getattr(config, 'use_moe', False):
            from .moe import HadamardMoE
            self.feed_forward = HadamardMoE(
                config,
                threshold=getattr(config, 'moe_threshold', 0.1),
                min_experts=getattr(config, 'moe_min_experts', 1),
                max_experts=getattr(config, 'moe_max_experts', 4),
            )
        else:
            self.feed_forward = SwiGLUMLP(config)

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None, layer_idx: int = 0):
        h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache=kv_cache, layer_idx=layer_idx)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Walsh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding: hash or standard
        if config.hash_embeddings:
            self.tok_embeddings = HashEmbedding(config.vocab_size, config.n_embd)
            self.output = None  # Use tok_embeddings.output_projection()
        else:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
            self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying for standard embeddings
            self.tok_embeddings.weight = self.output.weight
        
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size * 2)
        self.register_buffer("freqs_cis", freqs_cis)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('.beta'):
                torch.nn.init.constant_(p, 0.1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache: Optional[KVCache] = None, start_pos: int = 0, return_hidden: bool = False):
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        
        # Get RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]
        
        for layer_idx, layer in enumerate(self.layers):
            if self.training:
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, kv_cache=kv_cache, layer_idx=layer_idx)

        h = self.norm(h)
        
        # Training with targets
        if targets is not None:
            if self.output is not None:
                # Standard embeddings: compute full logits
                logits = self.output(h)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                # Hash embeddings: use chunked cross entropy (memory efficient)
                loss = self.tok_embeddings.chunked_cross_entropy(h, targets)
                logits = None  # Don't compute logits during training (saves memory)
        else:
            # Inference: compute logits for last position only
            if self.output is not None:
                logits = self.output(h[:, [-1], :])
            else:
                logits = self.tok_embeddings.output_projection(h[:, [-1], :])
            loss = None
        
        if return_hidden:
            return logits, loss, h
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Input token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_k: Top-k sampling (None = disabled)
            use_cache: Whether to use KV cache for efficient generation
        """
        device = idx.device
        B, T = idx.shape
        
        # Initialize KV cache if using cached generation
        if use_cache:
            kv_cache = KVCache(
                n_layers=self.config.n_layer,
                max_seq_len=self.config.block_size,
                n_head=self.config.n_head,
                head_dim=self.config.n_embd // self.config.n_head,
                device=device,
                dtype=next(self.parameters()).dtype
            )
            
            # Process prompt through all layers (fills the cache)
            logits, _ = self(idx, kv_cache=kv_cache, start_pos=0)
            kv_cache.increment(T)
            
            # Sample first new token
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Generate remaining tokens one at a time using cache
            for i in range(1, max_new_tokens):
                if kv_cache.cached_len >= self.config.block_size:
                    break
                
                # Only forward the new token
                logits, _ = self(idx_next, kv_cache=kv_cache, start_pos=kv_cache.cached_len)
                kv_cache.increment(1)
                
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        else:
            # Original non-cached generation (for comparison/fallback)
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx