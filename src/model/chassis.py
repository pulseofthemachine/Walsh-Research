import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
# Use original for training (fused backward is slow)
# For inference, see cayley_dickson_cuda.py
from .physics import OctonionTernaryLinear
from .ssm import OctonionSSM

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
    use_ssm: bool = False  # Use SSM instead of attention (infinite context)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

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
    # Cayley-Dickson sign table (same as physics.py)
    SIGNS = [
        [+1, -1, -1, -1, -1, -1, -1, -1],  # y0
        [+1, +1, +1, -1, +1, -1, -1, +1],  # y1
        [+1, -1, +1, +1, +1, +1, -1, -1],  # y2
        [+1, +1, -1, +1, +1, -1, +1, -1],  # y3
        [+1, -1, -1, -1, +1, +1, +1, +1],  # y4
        [+1, +1, -1, +1, -1, +1, -1, +1],  # y5
        [+1, +1, +1, -1, -1, +1, +1, -1],  # y6
        [+1, -1, +1, +1, -1, -1, +1, +1],  # y7
    ]
    
    # Weight index table
    WIDX = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # y0
        [1, 0, 3, 2, 5, 4, 7, 6],  # y1
        [2, 3, 0, 1, 6, 7, 4, 5],  # y2
        [3, 2, 1, 0, 7, 6, 5, 4],  # y3
        [4, 5, 6, 7, 0, 1, 2, 3],  # y4
        [5, 4, 7, 6, 1, 0, 3, 2],  # y5
        [6, 7, 4, 5, 2, 3, 0, 1],  # y6
        [7, 6, 5, 4, 3, 2, 1, 0],  # y7
    ]
    
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        # Learnable mixing weights: 8 weight matrices [head_dim, head_dim]
        self.W = nn.Parameter(torch.randn(8, head_dim, head_dim) * 0.02)
        self.beta = nn.Parameter(torch.ones(head_dim) * 0.1)
        
        # Pre-register sign and widx tables as buffers
        self.register_buffer('signs', torch.tensor(self.SIGNS, dtype=torch.float32))
        self.register_buffer('widx', torch.tensor(self.WIDX, dtype=torch.long))
    
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

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        x_parts = x.split(C // 8, dim=-1)
        q_parts = self.wq(x_parts)
        k_parts = self.wk(x_parts)
        v_parts = self.wv(x_parts)
        
        q = torch.cat(q_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        k = torch.cat(k_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        v = torch.cat(v_parts, dim=-1).view(B, T, self.n_head, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
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
        self.use_ssm = config.use_ssm
        self.attention_norm = RMSNorm(config.n_embd)
        
        if self.use_ssm:
            # SSM mode: no attention, just state space dynamics
            self.ssm = OctonionSSM(config.n_embd)
        else:
            # Attention mode: standard causal self-attention
            self.attention = CausalSelfAttention(config)
            
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = SwiGLUMLP(config)

    def forward(self, x, freqs_cis=None, h_prev=None):
        """
        Forward pass.
        
        For attention mode: uses freqs_cis for RoPE, ignores h_prev
        For SSM mode: uses h_prev for state, ignores freqs_cis
        
        Returns:
            out: output tensor
            h_new: new hidden state (SSM mode only, None for attention)
        """
        h_new = None
        
        if self.use_ssm:
            # SSM path
            ssm_out, h_new = self.ssm(self.attention_norm(x), h_prev)
            h = x + ssm_out
        else:
            # Attention path  
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, h_new

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

    def forward(self, idx, targets=None, h_states=None):
        """
        Forward pass.
        
        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices for loss computation
            h_states: list of [B, D] hidden states per layer (SSM mode only)
            
        Returns:
            logits: [B, T, vocab_size] or [B, 1, vocab_size]
            loss: scalar loss if targets provided
            h_states_new: list of new hidden states (SSM mode only)
        """
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T] if not self.config.use_ssm else None
        
        # Initialize hidden states for SSM mode
        if self.config.use_ssm and h_states is None:
            h_states = [None] * len(self.layers)
        
        h_states_new = []
        
        # --- Layer forward (with gradient checkpointing for attention) ---
        for i, layer in enumerate(self.layers):
            h_prev = h_states[i] if self.config.use_ssm else None
            
            if self.training and not self.config.use_ssm:
                # Gradient checkpointing for attention mode only
                # SSM has different memory patterns, checkpointing less useful
                h, h_new = checkpoint(layer, h, freqs_cis, None, use_reentrant=False)
            else:
                h, h_new = layer(h, freqs_cis, h_prev)
            
            if self.config.use_ssm:
                h_states_new.append(h_new)
        # ----------------------------------------

        h = self.norm(h)
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.output(h[:, [-1], :])
            loss = None
        
        # Return hidden states for SSM mode (for stateful generation)
        if self.config.use_ssm:
            return logits, loss, h_states_new
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.
        
        For attention mode: Uses sliding window, recomputes attention each step.
        For SSM mode: Maintains hidden state, O(1) per token.
        """
        h_states = None  # SSM hidden states
        
        for _ in range(max_new_tokens):
            if self.config.use_ssm:
                # SSM mode: process one token at a time with state
                idx_input = idx[:, -1:]  # Just the last token
                result = self(idx_input, h_states=h_states)
                logits, _, h_states = result
            else:
                # Attention mode: sliding window
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