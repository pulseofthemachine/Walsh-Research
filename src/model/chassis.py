import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
# Use original for training (fused backward is slow)
# For inference, see cayley_dickson_cuda.py
from .physics import OctonionTernaryLinear

@dataclass
class SpinNetConfig:
    block_size: int = 256
    vocab_size: int = 65  
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False 

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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.wq = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wk = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wv = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.wo = OctonionTernaryLinear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

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

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T]

        # --- MODIFIED: Gradient Checkpointing ---
        for layer in self.layers:
            if self.training:
                # We save VRAM by not storing activations, re-computing them during backward pass.
                # use_reentrant=False is the modern, stable standard.
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)
        # ----------------------------------------

        h = self.norm(h)
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.output(h[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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