import math
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

@dataclass
class SpinNetConfig:
    block_size: int = 256
    vocab_size: int = 65  
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    octonion_attention: bool = False  # Enable octonion head mixing

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


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
        self.beta = nn.Parameter(torch.ones(head_dim) * 0.1)
        
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
        self.octonion_attention = config.octonion_attention
        
        self.wq = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wk = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wv = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wo = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        
        # Octonion head mixer (only if enabled and n_head is multiple of 8)
        if self.octonion_attention:
            assert config.n_head % 8 == 0, f"octonion_attention requires n_head % 8 == 0, got {config.n_head}"
            # Use fused CUDA kernel when available
            if _USE_FUSED_HEAD_MIXER:
                self.head_mixer = OctonionHeadMixerFused(self.head_dim)
            else:
                self.head_mixer = OctonionHeadMixer(self.head_dim)

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None, layer_idx: int = 0):
        B, T, C = x.size()
        x_parts = x.split(C // 8, dim=-1)
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
        
        # Octonion mixing across heads
        if self.octonion_attention:
            # Group heads into sets of 8 for octonion mixing
            n_groups = self.n_head // 8
            y_groups = y.view(B, n_groups, 8, T, self.head_dim)
            y_mixed = []
            for g in range(n_groups):
                y_mixed.append(self.head_mixer(y_groups[:, g]))  # [B, 8, T, head_dim]
            y = torch.cat(y_mixed, dim=1)  # [B, n_head, T, head_dim]
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y_parts = y.split(C // 8, dim=-1)
        y_out_parts = self.wo(y_parts)
        return torch.cat(y_out_parts, dim=-1)

class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3) 
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.gate_proj = OctonionTernaryLinear(config.n_embd, hidden_dim)
        self.up_proj = OctonionTernaryLinear(config.n_embd, hidden_dim)
        self.down_proj = OctonionTernaryLinear(hidden_dim, config.n_embd)

    def forward(self, x):
        in_dim = x.shape[-1] // 8
        x_parts = x.split(in_dim, dim=-1)
        gate_out = torch.cat(self.gate_proj(x_parts), dim=-1)
        up_out = torch.cat(self.up_proj(x_parts), dim=-1)
        h = F.silu(gate_out) * up_out
        h_dim = h.shape[-1] // 8
        h_parts = h.split(h_dim, dim=-1)
        out_parts = self.down_proj(h_parts)
        return torch.cat(out_parts, dim=-1)

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = SwiGLUMLP(config)

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None, layer_idx: int = 0):
        h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache=kv_cache, layer_idx=layer_idx)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SpinNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
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

    def forward(self, idx, targets=None, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
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
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.output(h[:, [-1], :])
            loss = None
        
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