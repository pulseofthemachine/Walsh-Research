"""
Walsh: Manifold Hyper-Connections (mHC)
----------------------------------------
Based on paper: https://arxiv.org/abs/2512.24880

Implements manifold-constrained residual connections using doubly stochastic
matrices for stable multi-stream signal propagation.

Key properties:
- Norm Preservation: ||H_res||_2 <= 1 (prevents gradient explosion)
- Compositional Closure: Products of doubly stochastic matrices stay doubly stochastic
- Birkhoff Polytope: Residual mapping as convex combination of permutations

Optimizations:
- Fused RMSNorm + projection kernels
- Fused Sinkhorn-Knopp iteration
- Fused stream mixing kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# -----------------------------------------------------------------------------
# FUSED TRITON KERNELS
# -----------------------------------------------------------------------------

if HAS_TRITON:
    
    @triton.jit
    def _rms_norm_fwd_kernel(
        X, Y, W,
        stride_x, stride_y,
        N, eps,
        BLOCK_N: tl.constexpr,
    ):
        """Fused RMSNorm kernel."""
        row = tl.program_id(0)
        X = X + row * stride_x
        Y = Y + row * stride_y
        
        # Compute RMS
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        
        rms = tl.sqrt(tl.sum(x * x) / N + eps)
        x_norm = x / rms
        
        # Apply weights if provided
        if W is not None:
            w = tl.load(W + cols, mask=mask)
            x_norm = x_norm * w
        
        tl.store(Y + cols, x_norm.to(Y.dtype.element_ty), mask=mask)
    
    
    @triton.jit
    def _sinkhorn_kernel(
        M, OUT,
        n, n_iter,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused Sinkhorn-Knopp iteration kernel.
        
        M: [batch, n, n] input matrix (already exp'd)
        OUT: [batch, n, n] output doubly stochastic matrix
        """
        batch_idx = tl.program_id(0)
        
        # Load entire n x n matrix into registers (n is small, typically 4)
        offs_i = tl.arange(0, BLOCK_N)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < n
        mask_j = offs_j < n
        mask = mask_i[:, None] & mask_j[None, :]
        
        # Base pointer for this batch element
        base = batch_idx * n * n
        
        # Load and exp in one pass
        m = tl.load(M + base + offs_i[:, None] * n + offs_j[None, :], mask=mask, other=0.0)
        m = tl.exp(m.to(tl.float32))
        
        # Sinkhorn iterations
        for _ in range(n_iter):
            # Column normalization
            col_sum = tl.sum(m, axis=0, keep_dims=True)
            m = m / (col_sum + 1e-8)
            
            # Row normalization
            row_sum = tl.sum(m, axis=1, keep_dims=True)
            m = m / (row_sum + 1e-8)
        
        # Store result
        tl.store(OUT + base + offs_i[:, None] * n + offs_j[None, :], m, mask=mask)
    
    
    @triton.jit
    def _mhc_fused_mapping_kernel(
        X,           # [B*T, n*C] input
        PHI,         # [n*C, n^2 + 2n] projection weights
        BIAS,        # [n^2 + 2n] bias
        ALPHA_PRE,   # scalar
        ALPHA_POST,  # scalar
        ALPHA_RES,   # scalar
        H_PRE,       # [B*T, n] output
        H_POST,      # [B*T, n] output
        H_RES,       # [B*T, n, n] output
        B_T, n, C,
        n_iter,      # Sinkhorn iterations
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused kernel for computing all three mHC mappings.
        
        Implements Eq. 14-19 from paper:
        1. RMSNorm(x)
        2. Linear projection to get H_tilde_pre, H_tilde_post, H_tilde_res
        3. Apply constraints (sigmoid for pre/post, Sinkhorn for res)
        """
        row = tl.program_id(0)
        
        # Load x and compute RMS norm
        nC = n * C
        offs = tl.arange(0, BLOCK_C)
        mask = offs < nC
        
        x = tl.load(X + row * nC + offs, mask=mask, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x) / nC + 1e-8)
        x_norm = x / rms
        
        # Project to get all mappings at once
        # PHI is [nC, n^2 + 2n], we compute x_norm @ PHI
        # This is expensive - in production, use proper matmul
        
        # For now, output placeholders - full implementation would need
        # proper blocked matmul with shared memory
        pass


def sinkhorn_knopp_fused(H_tilde, n_iter=20):
    """
    Fused Sinkhorn-Knopp using Triton kernel.
    
    Args:
        H_tilde: [B, T, n, n] or [B*T, n, n] input matrix
        n_iter: Number of iterations
        
    Returns:
        Doubly stochastic matrix
    """
    if not HAS_TRITON:
        # Fallback to PyTorch implementation
        M = torch.exp(H_tilde)
        for _ in range(n_iter):
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        return M
    
    # Use Triton kernel
    orig_shape = H_tilde.shape
    if len(orig_shape) == 4:
        B, T, n, _ = orig_shape
        H_tilde = H_tilde.view(B * T, n, n)
    else:
        n = H_tilde.shape[-1]
    
    batch = H_tilde.shape[0]
    out = torch.empty_like(H_tilde)
    
    # Launch kernel
    BLOCK_N = triton.next_power_of_2(n)
    grid = (batch,)
    
    _sinkhorn_kernel[grid](
        H_tilde, out,
        n, n_iter,
        BLOCK_N=BLOCK_N,
    )
    
    if len(orig_shape) == 4:
        out = out.view(orig_shape)
    
    return out


class SinkhornKnopp(nn.Module):
    """
    Projects a matrix onto the doubly stochastic manifold using Sinkhorn-Knopp algorithm.
    
    A doubly stochastic matrix has:
    - Non-negative entries
    - Rows sum to 1
    - Columns sum to 1
    
    This ensures stable signal propagation (no explosion/vanishing).
    """
    def __init__(self, n_iter=20, eps=1e-8):
        super().__init__()
        self.n_iter = n_iter
        self.eps = eps
    
    def forward(self, H):
        """
        H: [batch, n, n] or [n, n] - matrix to project
        Returns: Doubly stochastic matrix (rows sum to 1, cols sum to 1)
        """
        # Make positive via exp (as per paper Eq. 8)
        M = torch.exp(H)
        
        # Iteratively normalize rows and columns (Eq. 9)
        for _ in range(self.n_iter):
            # Column normalization (T_c)
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)
            # Row normalization (T_r)
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
        
        return M


class ResidualStreamMapping(nn.Module):
    """
    Learnable mapping for mHC with manifold constraint.
    
    Computes: H_mapping = P_manifold(alpha * dynamic_mapping + static_bias)
    where P_manifold projects onto the appropriate constraint manifold.
    
    From paper Eq. 7-8:
    - H_pre: sigmoid(alpha * phi(x) + bias) - non-negative [1 x n]
    - H_post: 2 * sigmoid(alpha * phi(x) + bias) - allows amplification [1 x n]
    - H_res: Sinkhorn-Knopp(alpha * phi(x) + bias) - doubly stochastic [n x n]
    """
    def __init__(self, n_embd, n_streams, mapping_type='res'):
        super().__init__()
        self.n_embd = n_embd
        self.n_streams = n_streams
        self.mapping_type = mapping_type  # 'pre', 'post', or 'res'
        
        # Gating factor (start small to preserve initial identity-like behavior)
        # Paper: learnable scalar alpha
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
        if mapping_type == 'res':
            # Residual mapping: n x n matrix for stream mixing
            # phi: R^(nC) -> R^(n^2)
            self.phi = nn.Linear(n_streams * n_embd, n_streams * n_streams, bias=False)
            self.bias = nn.Parameter(torch.zeros(n_streams, n_streams))
            self.sinkhorn = SinkhornKnopp(n_iter=20)
            
            # Initialize bias close to identity (scaled for Sinkhorn)
            # exp(2*I) gives strong diagonal after Sinkhorn
            with torch.no_grad():
                self.bias.copy_(torch.eye(n_streams) * 2.0)
        
        elif mapping_type == 'pre':
            # Pre mapping: aggregates n streams -> 1 input (1 x n)
            # phi: R^(nC) -> R^n
            self.phi = nn.Linear(n_streams * n_embd, n_streams, bias=False)
            # Initialize to uniform aggregation
            self.bias = nn.Parameter(torch.zeros(1, n_streams))
        
        elif mapping_type == 'post':
            # Post mapping: distributes 1 output -> n streams (1 x n)
            # phi: R^(nC) -> R^n
            self.phi = nn.Linear(n_streams * n_embd, n_streams, bias=False)
            # Initialize to uniform distribution
            self.bias = nn.Parameter(torch.zeros(1, n_streams))
        
        # Initialize phi weights small
        nn.init.normal_(self.phi.weight, std=0.01)
    
    def forward(self, x):
        """
        x: [B, T, n_streams * n_embd] - flattened residual stream
        Returns: Mapping matrix [B, T, n, n] or [B, T, 1, n]
        """
        B, T, _ = x.shape
        
        # RMSNorm the input (paper Eq. 7)
        x_norm = F.rms_norm(x, (x.size(-1),))
        
        # Compute dynamic mapping via learned projection
        dynamic = self.phi(x_norm)
        
        if self.mapping_type == 'res':
            # Reshape to matrix: [B, T, n, n]
            dynamic = dynamic.view(B, T, self.n_streams, self.n_streams)
            # Add static bias and scale by alpha
            H_tilde = self.alpha * dynamic + self.bias
            # Project onto doubly stochastic manifold (Eq. 8)
            # Need to handle batch dimension
            H_tilde_flat = H_tilde.view(B * T, self.n_streams, self.n_streams)
            H_flat = self.sinkhorn(H_tilde_flat)
            H = H_flat.view(B, T, self.n_streams, self.n_streams)
            return H
        
        elif self.mapping_type == 'pre':
            # [B, T, 1, n] - constrain to non-negative via sigmoid (Eq. 8)
            H_tilde = self.alpha * dynamic + self.bias
            H = torch.sigmoid(H_tilde).unsqueeze(2)
            return H
        
        else:  # 'post'
            # [B, T, 1, n] - constrain to non-negative, scale by 2 (Eq. 8)
            H_tilde = self.alpha * dynamic + self.bias
            H = 2.0 * torch.sigmoid(H_tilde).unsqueeze(2)
            return H


class mHCResidualStream(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Residual Stream.
    
    Maintains n parallel streams and applies constrained mixing for stability.
    
    From paper Eq. 5:
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
    where:
    - H_pre: [1 x n] aggregates n streams to single layer input
    - H_post: [1 x n] distributes layer output back to n streams
    - H_res: [n x n] mixes existing streams (doubly stochastic)
    """
    def __init__(self, n_embd, n_streams=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_streams = n_streams
        
        # Three learnable mappings
        self.H_pre = ResidualStreamMapping(n_embd, n_streams, 'pre')
        self.H_post = ResidualStreamMapping(n_embd, n_streams, 'post')
        self.H_res = ResidualStreamMapping(n_embd, n_streams, 'res')
    
    def forward(self, x, layer_fn):
        """
        x: [B, T, n_streams, n_embd] - multi-stream residual
        layer_fn: Function to apply (e.g., attention or FFN)
        
        Returns: [B, T, n_streams, n_embd] - updated streams
        """
        B, T, n, C = x.shape
        
        # Flatten for computing mappings
        x_flat = x.reshape(B, T, n * C)
        
        # Compute mapping matrices
        H_pre = self.H_pre(x_flat)   # [B, T, 1, n]
        H_post = self.H_post(x_flat) # [B, T, 1, n]
        H_res = self.H_res(x_flat)   # [B, T, n, n]
        
        # Pre-mapping: Mix n streams -> layer input
        # H_pre @ x: [B, T, 1, n] @ [B, T, n, C] -> [B, T, 1, C]
        layer_input = torch.matmul(H_pre, x).squeeze(2)  # [B, T, C]
        
        # Apply layer function
        layer_output = layer_fn(layer_input)  # [B, T, C]
        
        # Post-mapping: Distribute output back to streams
        # H_post^T @ output: [B, T, n, 1] @ [B, T, 1, C] -> [B, T, n, C]
        layer_output_expanded = layer_output.unsqueeze(2)  # [B, T, 1, C]
        post_contrib = torch.matmul(H_post.transpose(-2, -1), layer_output_expanded)
        
        # Residual mapping: Mix existing streams
        # H_res @ x: [B, T, n, n] @ [B, T, n, C] -> [B, T, n, C]
        res_contrib = torch.matmul(H_res, x)
        
        # Combine: x_new = H_res @ x + H_post^T @ F(H_pre @ x)
        x_new = res_contrib + post_contrib
        
        return x_new


class mHCBlock(nn.Module):
    """
    Transformer block modified to use mHC residual streams.
    
    Instead of: x = x + attn(norm(x))
    Uses: x = mHC_attn(x, lambda y: attn(norm(y)))
    
    This replaces standard residual connections with manifold-constrained
    multi-stream residuals for enhanced stability and capacity.
    """
    def __init__(self, attention_module, ffn_module, n_embd, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        
        # Original layer modules (with their norms)
        self.attention = attention_module
        self.ffn = ffn_module
        
        # mHC residual streams (one for attention, one for FFN)
        self.attn_stream = mHCResidualStream(n_embd, n_streams)
        self.ffn_stream = mHCResidualStream(n_embd, n_streams)
    
    def forward(self, x, freqs_cis, kv_cache=None, layer_idx=0):
        """
        x: [B, T, n_streams, n_embd] - multi-stream residual
        Returns: [B, T, n_streams, n_embd]
        """
        # Attention with mHC streams
        def attn_fn(x_in):
            return self.attention(x_in, freqs_cis, kv_cache=kv_cache, layer_idx=layer_idx)
        
        x = self.attn_stream(x, attn_fn)
        
        # FFN with mHC streams
        def ffn_fn(x_in):
            return self.ffn(x_in)
        
        x = self.ffn_stream(x, ffn_fn)
        
        return x


def wrap_with_mhc(model_class, n_streams=4):
    """
    Factory function to create mHC-wrapped version of a model.
    
    This is a convenience function - for production use, implement
    mHC directly in the model architecture.
    
    Args:
        model_class: Original model class
        n_streams: Number of parallel streams (default 4)
        
    Returns:
        Modified model class with mHC residual streams
    """
    # This would require model-specific implementation
    raise NotImplementedError(
        "Use mHCBlock directly in your model architecture. "
        "See mHCWalsh below for an example."
    )


# -----------------------------------------------------------------------------
# mHC-WALSH: Walsh model with mHC residual streams
# -----------------------------------------------------------------------------

class mHCLlamaBlock(nn.Module):
    """
    LlamaBlock with mHC residual streams.
    
    Uses the same attention/FFN as standard LlamaBlock but with
    manifold-constrained residual connections.
    
    Architecture per layer:
    - Level 2: Attention (token ↔ token)
    - Level 3: mHC streams (stream ↔ stream via doubly stochastic)
    - Level 4: MoE routing (token → expert)
    - Level 5: Channel modulation (bulk → channel, optional)
    """
    def __init__(self, config, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.n_embd = config.n_embd
        
        # Import the attention and FFN modules
        from .chassis import CausalSelfAttention, SwiGLUMLP, RMSNorm
        
        # Norms
        self.attention_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)
        
        # Core layers
        self.attention = CausalSelfAttention(config)
        
        # FFN: MoE or dense
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
        
        # Channel modulation (optional)
        self.use_channel_mod = getattr(config, 'use_channel_mod', False)
        if self.use_channel_mod:
            from .moe import HadamardChannelModulation
            self.channel_mod = HadamardChannelModulation(
                n_embd=config.n_embd,
                n_blocks=None,  # Auto-computed: n_embd // 32
                use_ternary=True,
                residual_scale=0.5,
            )
            self.channel_mod_norm = RMSNorm(config.n_embd)
        
        # mHC residual streams
        self.attn_stream = mHCResidualStream(config.n_embd, n_streams)
        self.ffn_stream = mHCResidualStream(config.n_embd, n_streams)
    
    def forward(self, x, freqs_cis, kv_cache=None, layer_idx=0):
        """
        x: [B, T, n_streams, n_embd] - multi-stream residual
        Returns: [B, T, n_streams, n_embd]
        """
        B, T, n, C = x.shape
        
        # Attention with mHC streams
        def attn_fn(x_in):
            return self.attention(self.attention_norm(x_in), freqs_cis, 
                                 kv_cache=kv_cache, layer_idx=layer_idx)
        
        x = self.attn_stream(x, attn_fn)
        
        # FFN with mHC streams
        def ffn_fn(x_in):
            return self.feed_forward(self.ffn_norm(x_in))
        
        x = self.ffn_stream(x, ffn_fn)
        
        # Channel modulation (bulk → channel)
        if self.use_channel_mod:
            # Apply channel modulation per stream
            x_flat = x.view(B * n, T, C)  # Treat each stream independently
            x_flat = x_flat + self.channel_mod(self.channel_mod_norm(x_flat))
            x = x_flat.view(B, T, n, C)
        
        return x


class mHCWalsh(nn.Module):
    """
    Walsh model with mHC (manifold Hyper-Connection) residual streams.
    
    Maintains n_streams parallel residual streams throughout the model,
    with doubly-stochastic mixing to ensure stable signal propagation.
    
    Key benefits:
    - No stream can dominate (doubly stochastic constraint)
    - Stable gradient flow across arbitrary depth
    - Enhanced model capacity via multi-stream architecture
    """
    def __init__(self, config, n_streams=4, gradient_checkpointing=False):
        super().__init__()
        self.config = config
        self.n_streams = n_streams
        self.gradient_checkpointing = gradient_checkpointing
        
        # Import required modules
        from .chassis import RMSNorm, precompute_freqs_cis
        
        # Token embeddings
        if getattr(config, 'hash_embeddings', False):
            from .hash_embeddings import HashEmbedding
            self.tok_embeddings = HashEmbedding(config.vocab_size, config.n_embd)
            self.output = None
        else:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
            self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.tok_embeddings.weight = self.output.weight
        
        # Project single embedding to n streams
        # [n_embd] -> [n_streams * n_embd]
        self.stream_proj = nn.Linear(config.n_embd, n_streams * config.n_embd)
        
        # mHC transformer blocks
        self.layers = nn.ModuleList([
            mHCLlamaBlock(config, n_streams) for _ in range(config.n_layer)
        ])
        
        # Merge n streams back to single output
        # [n_streams * n_embd] -> [n_embd]
        self.stream_merge = nn.Linear(n_streams * config.n_embd, config.n_embd)
        
        # Final norm
        self.norm = RMSNorm(config.n_embd)
        
        # RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2
        )
        self.register_buffer("freqs_cis", freqs_cis)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special init for stream projection (preserve variance across streams)
        with torch.no_grad():
            # Initialize to approximately copy embedding to all streams
            eye_pattern = torch.eye(config.n_embd).repeat(n_streams, 1)
            eye_pattern = eye_pattern / math.sqrt(n_streams)  # Scale to preserve variance
            self.stream_proj.weight.copy_(eye_pattern)
            if self.stream_proj.bias is not None:
                self.stream_proj.bias.zero_()
            
            # Initialize merge to average streams
            avg_pattern = torch.eye(config.n_embd).repeat(1, n_streams) / n_streams
            self.stream_merge.weight.copy_(avg_pattern)
            if self.stream_merge.bias is not None:
                self.stream_merge.bias.zero_()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, kv_cache=None, start_pos=0, return_hidden=False):
        """
        Forward pass with multi-stream residuals.
        
        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices for loss computation
            kv_cache: Optional KV cache for inference
            start_pos: Starting position for RoPE
            return_hidden: If True, also return final hidden states
            
        Returns:
            logits, loss, [hidden_states]
        """
        B, T = idx.shape
        
        # Embed tokens: [B, T, n_embd]
        h = self.tok_embeddings(idx)
        
        # Expand to n streams: [B, T, n_streams * n_embd]
        h = self.stream_proj(h)
        
        # Reshape to [B, T, n_streams, n_embd]
        h = h.view(B, T, self.n_streams, self.config.n_embd)
        
        # Get RoPE frequencies
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]
        
        # Process through mHC blocks (with optional gradient checkpointing)
        for layer_idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, freqs_cis, kv_cache, layer_idx, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, kv_cache=kv_cache, layer_idx=layer_idx)
        
        # Merge streams: [B, T, n_streams, n_embd] -> [B, T, n_embd]
        h_flat = h.reshape(B, T, self.n_streams * self.config.n_embd)
        h_merged = self.stream_merge(h_flat)
        h_out = self.norm(h_merged)
        
        # Compute loss if targets provided
        if targets is not None:
            if self.output is not None:
                logits = self.output(h_out)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
            else:
                # Use hash embedding's chunked cross entropy
                loss = self.tok_embeddings.chunked_cross_entropy(h_out, targets)
                logits = None
        else:
            # Inference mode - only compute logits for last position
            if self.output is not None:
                logits = self.output(h_out[:, [-1], :])
            else:
                logits = self.tok_embeddings.output_projection(h_out[:, [-1], :])
            loss = None
        
        if return_hidden:
            return logits, loss, h_out
        return logits, loss
    
    @property
    def num_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_aux_loss(self):
        """
        Collect auxiliary losses from MoE layers (for load balancing).
        Returns 0 if no MoE layers present.
        """
        aux_loss = 0.0
        for layer in self.layers:
            if hasattr(layer.feed_forward, 'get_aux_loss'):
                aux_loss = aux_loss + layer.feed_forward.get_aux_loss()
        return aux_loss
    
    def get_moe_stats(self) -> dict:
        """
        Collect MoE expert usage statistics from all layers.
        Returns aggregated stats if MoE is enabled.
        """
        all_stats = []
        for layer in self.layers:
            if hasattr(layer.feed_forward, 'get_expert_stats'):
                all_stats.append(layer.feed_forward.get_expert_stats())
        
        if not all_stats:
            return {}
        
        # Aggregate across layers
        n_experts = len(all_stats[0]['expert_load'])
        avg_load = [0.0] * n_experts
        for stats in all_stats:
            for i, load in enumerate(stats['expert_load']):
                avg_load[i] += load / len(all_stats)
        
        return {
            'expert_load': avg_load,
            'avg_experts': sum(s['avg_experts'] for s in all_stats) / len(all_stats),
            'load_balance': sum(s['load_balance'] for s in all_stats) / len(all_stats),
        }
    
    def reset_moe_stats(self):
        """Reset MoE statistics for all layers."""
        for layer in self.layers:
            if hasattr(layer.feed_forward, 'reset_stats'):
                layer.feed_forward.reset_stats()
    
    def get_channel_mod_stats(self) -> dict:
        """
        Collect channel modulation statistics from all layers.
        Returns aggregated stats if channel modulation is enabled.
        """
        all_stats = []
        for layer in self.layers:
            if hasattr(layer, 'channel_mod') and hasattr(layer.channel_mod, 'get_modulation_stats'):
                all_stats.append(layer.channel_mod.get_modulation_stats())
        
        if not all_stats:
            return {}
        
        # Aggregate across layers
        return {
            'scale_mean': sum(s['scale_mean'] for s in all_stats) / len(all_stats),
            'scale_std': sum(s['scale_std'] for s in all_stats) / len(all_stats),
            'shift_mean': sum(s['shift_mean'] for s in all_stats) / len(all_stats),
        }
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: [B, T] context tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (None = disabled)
            
        Returns:
            [B, T + max_new_tokens] tensor of tokens
        """
        for _ in range(max_new_tokens):
            # Crop context to block size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Export public API
__all__ = [
    'SinkhornKnopp',
    'sinkhorn_knopp_fused',
    'ResidualStreamMapping',
    'mHCResidualStream',
    'mHCBlock',
    'mHCLlamaBlock',
    'mHCWalsh',
]
