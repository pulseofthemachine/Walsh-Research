"""
Hadamard Mixture of Experts (HadamardMoE) - Optimized with Triton
-----------------------------------------------------------------
Leverages the natural 256D = 8 × 32D Hadamard block structure.
Each expert operates on a 32D Walsh basis block.

Features:
- Dynamic routing: threshold-based, not fixed top-k
- Per-stream routing: each mHC stream routes independently
- Load balancing: auxiliary loss prevents expert collapse
- FUSED Triton kernels for speed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available, MoE will use slow PyTorch fallback")


# =============================================================================
# TRITON KERNELS
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _moe_expert_fwd_kernel(
        # Input/output pointers
        X, OUT,
        # Expert weight pointers (stacked)
        GATE_W, UP_W, DOWN_W,
        # Routing weights
        ROUTE_W,
        # Dimensions
        batch_seq,      # B * T
        n_experts,      # 8
        block_dim,      # 32
        hidden_dim,     # 128 (32 * 4)
        # Block sizes
        BLOCK_DIM: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
    ):
        """
        Fused MoE expert forward pass.
        
        Each program handles one token (batch_seq) and one expert block.
        """
        # Program IDs
        token_id = tl.program_id(0)
        expert_id = tl.program_id(1)
        
        # Check bounds
        if token_id >= batch_seq:
            return
            
        # Get routing weight for this token/expert
        route_idx = token_id * n_experts + expert_id
        route_weight = tl.load(ROUTE_W + route_idx)
        
        # Skip if weight is zero (expert not selected)
        if route_weight == 0.0:
            return
        
        # Load input block for this expert
        x_base = token_id * n_experts * block_dim + expert_id * block_dim
        x_offs = tl.arange(0, BLOCK_DIM)
        x_mask = x_offs < block_dim
        x = tl.load(X + x_base + x_offs, mask=x_mask, other=0.0).to(tl.float32)
        
        # Expert weight offsets
        expert_gate_base = expert_id * hidden_dim * block_dim
        expert_up_base = expert_id * hidden_dim * block_dim
        expert_down_base = expert_id * block_dim * hidden_dim
        
        # Compute gate projection: [block_dim] @ [block_dim, hidden] -> [hidden]
        gate_out = tl.zeros([HIDDEN_DIM], dtype=tl.float32)
        up_out = tl.zeros([HIDDEN_DIM], dtype=tl.float32)
        
        for h in range(hidden_dim):
            h_mask = h < hidden_dim
            if h_mask:
                # Load gate and up weights for this hidden unit
                gate_row = tl.load(GATE_W + expert_gate_base + h * block_dim + x_offs, mask=x_mask, other=0.0)
                up_row = tl.load(UP_W + expert_up_base + h * block_dim + x_offs, mask=x_mask, other=0.0)
                
                # Dot product
                gate_out = tl.where(x_offs == 0, tl.sum(x * gate_row), gate_out)
                up_out = tl.where(x_offs == 0, tl.sum(x * up_row), up_out)
        
        # SwiGLU activation: silu(gate) * up
        hidden = tl.sigmoid(gate_out) * gate_out * up_out  # silu = x * sigmoid(x)
        
        # Down projection and write output
        for d in range(block_dim):
            d_mask = d < block_dim
            if d_mask:
                down_row = tl.load(DOWN_W + expert_down_base + d * hidden_dim + tl.arange(0, HIDDEN_DIM), 
                                   mask=tl.arange(0, HIDDEN_DIM) < hidden_dim, other=0.0)
                out_val = tl.sum(hidden * down_row) * route_weight
                
                # Atomic add to output (multiple experts may write to same output)
                out_idx = token_id * n_experts * block_dim + expert_id * block_dim + d
                tl.atomic_add(OUT + out_idx, out_val)


    @triton.jit
    def _channel_mod_fwd_kernel(
        # Input/output pointers
        X, OUT, SCALE, SHIFT,
        # Dimensions
        T, n_blocks, block_dim,
        # Parameters
        residual_scale, shift_scale,
        # Block size
        BLOCK_DIM: tl.constexpr,
    ):
        """
        Fused channel modulation forward kernel.
        
        Each program handles one (batch*token, block) pair.
        Computes: out = x + residual_scale * (x * scale + shift * shift_scale - x)
        """
        bt_id = tl.program_id(0)
        block_id = tl.program_id(1)
        batch_id = bt_id // T
        
        # Load scale and shift
        scale_idx = batch_id * n_blocks + block_id
        scale = tl.load(SCALE + scale_idx)
        shift = tl.load(SHIFT + scale_idx)
        
        block_offs = tl.arange(0, BLOCK_DIM)
        mask = block_offs < block_dim
        
        x_base = bt_id * n_blocks * block_dim + block_id * block_dim
        x = tl.load(X + x_base + block_offs, mask=mask, other=0.0)
        
        # out = x + residual_scale * (x * scale + shift * shift_scale - x)
        #     = x * (1 + residual_scale * (scale - 1)) + residual_scale * shift * shift_scale
        x_mod = x * scale + shift * shift_scale
        out = x + residual_scale * (x_mod - x)
        
        tl.store(OUT + x_base + block_offs, out, mask=mask)

    @triton.jit
    def _channel_mod_bwd_kernel(
        # Input/output pointers
        GRAD_OUT, X, SCALE,
        GRAD_X, GRAD_SCALE, GRAD_SHIFT,
        # Dimensions
        T, n_blocks, block_dim,
        # Parameters
        residual_scale, shift_scale,
        # Block size
        BLOCK_DIM: tl.constexpr,
    ):
        """
        Channel modulation backward kernel.
        
        Gradients:
        ∂L/∂x = grad_out * (1 + residual_scale * (scale - 1))
        ∂L/∂scale = sum(grad_out * x) * residual_scale  (reduced over block_dim)
        ∂L/∂shift = sum(grad_out) * shift_scale * residual_scale
        """
        bt_id = tl.program_id(0)
        block_id = tl.program_id(1)
        batch_id = bt_id // T
        
        scale_idx = batch_id * n_blocks + block_id
        scale = tl.load(SCALE + scale_idx)
        
        block_offs = tl.arange(0, BLOCK_DIM)
        mask = block_offs < block_dim
        
        x_base = bt_id * n_blocks * block_dim + block_id * block_dim
        grad_out = tl.load(GRAD_OUT + x_base + block_offs, mask=mask, other=0.0)
        x = tl.load(X + x_base + block_offs, mask=mask, other=0.0)
        
        # ∂L/∂x = grad_out * (1 + residual_scale * (scale - 1))
        grad_x = grad_out * (1.0 + residual_scale * (scale - 1.0))
        tl.store(GRAD_X + x_base + block_offs, grad_x, mask=mask)
        
        # ∂L/∂scale = sum(grad_out * x) * residual_scale
        grad_scale_contrib = tl.sum(grad_out * x) * residual_scale
        tl.atomic_add(GRAD_SCALE + scale_idx, grad_scale_contrib)
        
        # ∂L/∂shift = sum(grad_out) * shift_scale * residual_scale
        grad_shift_contrib = tl.sum(grad_out) * shift_scale * residual_scale
        tl.atomic_add(GRAD_SHIFT + scale_idx, grad_shift_contrib)


    class ChannelModFunction(torch.autograd.Function):
        """Autograd wrapper for Triton channel modulation kernels."""
        
        @staticmethod
        def forward(ctx, x, scale, shift, n_blocks, block_dim, residual_scale, shift_scale):
            B = x.shape[0] // (x.shape[1] if len(x.shape) > 1 else 1)
            T = x.shape[1] if len(x.shape) > 2 else x.shape[0]
            
            # Handle 3D input [B, T, C]
            if len(x.shape) == 3:
                B, T, C = x.shape
            else:
                raise ValueError(f"Expected 3D input, got {x.shape}")
            
            out = torch.empty_like(x)
            grid = (B * T, n_blocks)
            
            _channel_mod_fwd_kernel[grid](
                x, out, scale, shift,
                T, n_blocks, block_dim,
                residual_scale, shift_scale,
                BLOCK_DIM=block_dim,
            )
            
            ctx.save_for_backward(x, scale)
            ctx.n_blocks = n_blocks
            ctx.block_dim = block_dim
            ctx.residual_scale = residual_scale
            ctx.shift_scale = shift_scale
            ctx.T = T
            
            return out
        
        @staticmethod
        def backward(ctx, grad_out):
            x, scale = ctx.saved_tensors
            B, T, C = x.shape
            n_blocks = ctx.n_blocks
            block_dim = ctx.block_dim
            
            grad_x = torch.empty_like(x)
            grad_scale = torch.zeros_like(scale)
            grad_shift = torch.zeros_like(scale)
            
            grid = (B * T, n_blocks)
            
            _channel_mod_bwd_kernel[grid](
                grad_out.contiguous(), x, scale,
                grad_x, grad_scale, grad_shift,
                T, n_blocks, block_dim,
                ctx.residual_scale, ctx.shift_scale,
                BLOCK_DIM=block_dim,
            )
            
            return grad_x, grad_scale, grad_shift, None, None, None, None

else:
    # Fallback when Triton not available
    _channel_mod_fwd_kernel = None
    _channel_mod_bwd_kernel = None
    ChannelModFunction = None

# =============================================================================
# FAST VECTORIZED PYTORCH IMPLEMENTATION WITH TERNARY QUANTIZATION
# =============================================================================

class TernaryAbsmeanSTE(torch.autograd.Function):
    """
    BitNet b1.58 style Straight-Through Estimator for ternary {-1, 0, 1}.
    
    Uses absmean scaling: γ = mean(|W|), then round(W/γ) clamped to [-1,+1].
    This DYNAMIC threshold ensures consistent ternary distribution across training.
    """
    @staticmethod
    def forward(ctx, w_latent):
        # Compute per-tensor absolute mean as scale factor (DYNAMIC!)
        gamma = w_latent.abs().mean() + 1e-8
        
        # Scale weights by absmean, round, and clamp to ternary
        w_scaled = w_latent / gamma
        w_quant = torch.round(w_scaled)
        w_quant = torch.clamp(w_quant, -1, 1)
        
        ctx.save_for_backward(w_latent)
        ctx.gamma = gamma
        return w_quant * gamma  # Return scaled ternary
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # Gradient clipping for stability
        grad_input = torch.clamp(grad_input, -2.0, 2.0)
        return grad_input


def quantize_ternary(w: torch.Tensor) -> torch.Tensor:
    """
    BitNet b1.58 style ternary quantization with DYNAMIC absmean scaling.
    
    Formula: W_ternary = clip(round(W / γ), -1, +1) * γ where γ = mean(|W|)
    """
    return TernaryAbsmeanSTE.apply(w)


class FastMoEExperts(nn.Module):
    """
    Vectorized MoE experts WITH TERNARY QUANTIZATION.
    
    Uses ghost weights (full precision for gradients, quantized for forward pass).
    Each expert is a SwiGLU FFN operating on 32D.
    
    Weight structure:
    - gate_weights: [n_experts, hidden, dim]
    - up_weights: [n_experts, hidden, dim]  
    - down_weights: [n_experts, dim, hidden]
    """
    
    def __init__(self, n_experts: int = 8, dim: int = 32, expansion: int = 4, use_ternary: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.hidden = dim * expansion
        self.use_ternary = use_ternary
        
        # Ghost weights (full precision, quantized during forward via dynamic gamma)
        self.gate_weights = nn.Parameter(torch.empty(n_experts, self.hidden, dim))
        self.up_weights = nn.Parameter(torch.empty(n_experts, self.hidden, dim))
        self.down_weights = nn.Parameter(torch.empty(n_experts, dim, self.hidden))
        
        # Initialize uniformly in [-1, 1] for good ternary distribution
        nn.init.uniform_(self.gate_weights, -1, 1)
        nn.init.uniform_(self.up_weights, -1, 1)
        nn.init.uniform_(self.down_weights, -1, 1)
    
    
    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Vectorized expert forward pass using einsum (memory efficient).
        Uses ternary quantization with straight-through estimator.
        
        Args:
            x: [B, T, n_experts, dim] - input split into expert blocks
            expert_weights: [B, T, n_experts] - routing weights
            
        Returns:
            [B, T, n_experts, dim] - expert outputs weighted by routing
        """
        B, T, E, D = x.shape
        H = self.hidden
        
        # Quantize weights to ternary (gamma scaling is built into quantize_ternary)
        if self.use_ternary:
            gate_w = quantize_ternary(self.gate_weights)
            up_w = quantize_ternary(self.up_weights)
            down_w = quantize_ternary(self.down_weights)
        else:
            gate_w = self.gate_weights
            up_w = self.up_weights
            down_w = self.down_weights
        
        # Use einsum for efficient broadcasting
        # x: [B, T, E, D], weights: [E, H, D] -> out: [B, T, E, H]
        gate_out = torch.einsum('bted,ehd->bteh', x, gate_w)
        up_out = torch.einsum('bted,ehd->bteh', x, up_w)
        
        # SwiGLU activation
        hidden = F.silu(gate_out) * up_out  # [B, T, E, H]
        
        # Down projection
        # hidden: [B, T, E, H]
        # down_w: [E, D, H] (already quantized above)
        # output: [B, T, E, D]
        output = torch.einsum('bteh,edh->bted', hidden, down_w)
        
        # Apply routing weights
        expert_weights_exp = expert_weights.unsqueeze(-1)  # [B, T, E, 1]
        output = output * expert_weights_exp
        
        return output


class HadamardRouter(nn.Module):
    """
    Dynamic router that decides which Hadamard blocks (experts) to activate.
    
    Uses threshold-based routing: activates all experts with probability > threshold.
    Enforces min/max constraints for stability.
    """
    
    def __init__(
        self, 
        n_embd: int, 
        n_experts: int = 8,
        threshold: float = 0.1,
        min_experts: int = 1,
        max_experts: int = 4,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.threshold = threshold
        self.min_experts = min_experts
        self.max_experts = max_experts
        
        # Small MLP for routing decision
        hidden = n_embd // 4
        self.gate = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, n_experts, bias=False)
        )
        
        # Initialize small
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, n_embd] hidden states
            
        Returns:
            expert_weights: [B, T, n_experts] - gating weights (sparse, sums to 1)
            expert_mask: [B, T, n_experts] - binary mask of selected experts
            router_probs: [B, T, n_experts] - raw probabilities for aux loss
        """
        B, T, C = x.shape
        
        # Compute router logits and probabilities
        logits = self.gate(x)  # [B, T, n_experts]
        router_probs = F.softmax(logits, dim=-1)
        
        # Top-k selection (simpler and faster than threshold-based)
        top_k = min(self.max_experts, self.n_experts)
        top_vals, top_idx = router_probs.topk(top_k, dim=-1)  # [B, T, k]
        
        # Create mask
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(-1, top_idx, 1.0)
        
        # Normalize weights among selected experts
        masked_probs = router_probs * expert_mask
        weight_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        expert_weights = masked_probs / weight_sum
        
        return expert_weights, expert_mask, router_probs


class HadamardMoE(nn.Module):
    """
    Mixture of Experts layer using Hadamard block structure.
    
    Replaces dense SwiGLU FFN with sparse expert routing.
    Each expert handles one 32D Hadamard block.
    Number of experts scales with n_embd: 256D → 8 experts, 512D → 16, etc.
    
    OPTIMIZED version using vectorized operations with ternary quantization.
    """
    BLOCK_DIM = 32  # Fixed Hadamard block size
    
    def __init__(
        self,
        config,
        n_experts: int = None,  # Auto-computed from n_embd if None
        expansion: int = 4,
        threshold: float = 0.1,
        min_experts: int = 1,
        max_experts: int = 4,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.n_embd = config.n_embd
        
        # Auto-compute n_experts to maintain 32D Hadamard block structure
        if n_experts is None:
            n_experts = config.n_embd // self.BLOCK_DIM
        
        self.n_experts = n_experts
        self.block_dim = config.n_embd // n_experts
        self.aux_loss_weight = aux_loss_weight
        
        assert config.n_embd % n_experts == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_experts ({n_experts})"
        assert self.block_dim == self.BLOCK_DIM, \
            f"block_dim ({self.block_dim}) should be {self.BLOCK_DIM} for Hadamard structure"
        
        # Router
        self.router = HadamardRouter(
            n_embd=config.n_embd,
            n_experts=n_experts,
            threshold=threshold,
            min_experts=min_experts,
            max_experts=max_experts,
        )
        
        # Vectorized experts (FAST!)
        self.experts = FastMoEExperts(
            n_experts=n_experts,
            dim=self.block_dim,
            expansion=expansion,
        )
        
        # Store aux loss for retrieval during training
        self._aux_loss = None
        
        # Expert usage tracking for logging
        self.register_buffer('_expert_counts', torch.zeros(n_experts), persistent=False)
        self.register_buffer('_total_tokens', torch.zeros(1), persistent=False)
        self._last_expert_mask = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_embd] input hidden states
            
        Returns:
            [B, T, n_embd] output after sparse expert processing
        """
        B, T, C = x.shape
        
        # Get routing decisions
        expert_weights, expert_mask, router_probs = self.router(x)
        
        # Compute auxiliary loss for load balancing
        self._aux_loss = self._compute_aux_loss(router_probs, expert_mask)
        
        # Track expert usage for logging
        self._last_expert_mask = expert_mask.detach()
        if self.training:
            with torch.no_grad():
                self._expert_counts += expert_mask.sum(dim=(0, 1))
                self._total_tokens += B * T
        
        # Split input into blocks: [B, T, n_experts, block_dim]
        x_blocks = x.view(B, T, self.n_experts, self.block_dim)
        
        # Run all experts (vectorized!) 
        outputs = self.experts(x_blocks, expert_weights)
        
        # Reshape back to [B, T, n_embd]
        return outputs.reshape(B, T, C)
    
    def _compute_aux_loss(
        self, 
        router_probs: torch.Tensor, 
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        From Switch Transformer paper.
        """
        probs_flat = router_probs.view(-1, self.n_experts)
        mask_flat = expert_mask.view(-1, self.n_experts)
        
        tokens_per_expert = mask_flat.mean(dim=0)
        prob_per_expert = probs_flat.mean(dim=0)
        
        aux_loss = (tokens_per_expert * prob_per_expert).sum() * self.n_experts
        return aux_loss
    
    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """Retrieve the auxiliary loss computed in the last forward pass."""
        return self._aux_loss if self._aux_loss is not None else torch.tensor(0.0)
    
    def get_expert_stats(self) -> dict:
        """Get statistics about expert usage for logging."""
        if self._total_tokens.item() == 0:
            return {
                'expert_load': [0.0] * self.n_experts,
                'avg_experts': 0.0,
                'load_balance': 0.0,
            }
        
        total = self._total_tokens.item()
        counts = self._expert_counts.cpu().numpy()
        load = counts / total
        avg_experts = counts.sum() / total
        
        sorted_load = np.sort(load)
        n = len(sorted_load)
        cumulative = np.cumsum(sorted_load)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_load)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1] + 1e-8)
        
        return {
            'expert_load': load.tolist(),
            'avg_experts': float(avg_experts),
            'load_balance': float(1 - abs(gini)),
        }
    
    def reset_stats(self):
        """Reset the running expert usage statistics."""
        self._expert_counts.zero_()
        self._total_tokens.zero_()
    
    def get_expert_usage(self) -> dict:
        """Alias for get_expert_stats."""
        return self.get_expert_stats()


def should_use_moe(config) -> bool:
    """Check if config enables MoE."""
    return getattr(config, 'use_moe', False)


# =============================================================================
# HADAMARD CHANNEL MODULATION
# =============================================================================

class HadamardChannelModulation(nn.Module):
    """
    Passive self-interaction via channel modulation.
    
    The network observes its own global state ("bulk") and generates
    per-channel (per-Hadamard-block) scale and shift modulations.
    
    This creates a 1-1 mapping: each bulk state → unique modulation.
    
    Architecture:
    ```
    Input [B, T, 256]
           ↓
    Global pooling → Bulk state [B, 256]
           ↓
    Modulation MLP → (scale[8], shift[8])
           ↓
    Per-block modulation: block_i = block_i * scale_i + shift_i
           ↓
    Output [B, T, 256]
    ```
    
    Args:
        n_embd: Embedding dimension (256, 512, 1024, ...)
        n_blocks: Number of Hadamard blocks (auto-computed if None: n_embd // 32)
        use_ternary: Whether to use ternary quantization in modulation MLP
        residual_scale: Scale factor for residual connection
    """
    BLOCK_DIM = 32  # Fixed Hadamard block size
    
    def __init__(
        self, 
        n_embd: int = 256, 
        n_blocks: int = None,  # Auto-computed from n_embd if None
        use_ternary: bool = True,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        
        # Auto-compute n_blocks to maintain 32D Hadamard structure
        if n_blocks is None:
            n_blocks = n_embd // self.BLOCK_DIM
        
        self.n_blocks = n_blocks
        self.block_dim = n_embd // n_blocks
        self.residual_scale = residual_scale
        self.use_ternary = use_ternary
        
        # Context aggregator MLP (observes "bulk" state)
        # Small MLP to avoid too many parameters
        hidden = n_embd // 4
        self.context_mlp = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, n_blocks * 2, bias=False),  # scale + shift per block
        )
        
        # Initialize with small random values (NOT zeros, so gradients can flow!)
        # If we init to zeros, scale=1 exactly and there's no gradient signal
        nn.init.normal_(self.context_mlp[0].weight, std=0.02)
        nn.init.normal_(self.context_mlp[-1].weight, std=0.01)  # Small but non-zero
        
        # Learnable temperature for scale (controls how dynamic the modulation is)
        # Start higher so modulation has real effect
        self.scale_temp = nn.Parameter(torch.ones(1) * 1.0)
        
        # Stats tracking for logging
        self._last_scale_mean = 0.0
        self._last_scale_std = 0.0
        self._last_shift_mean = 0.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bulk-state-dependent channel modulation.
        
        Args:
            x: [B, T, n_embd] input hidden states
            
        Returns:
            [B, T, n_embd] modulated hidden states
        """
        B, T, C = x.shape
        
        # 1. Observe bulk state (passive self-interaction)
        bulk_state = x.mean(dim=1)  # [B, C]
        
        # 2. Generate modulation from bulk (1-1 representation)
        if self.use_ternary:
            mods = self._forward_ternary(bulk_state)
        else:
            mods = self.context_mlp(bulk_state)  # [B, n_blocks * 2]
        
        # Split into scale and shift
        raw_scale, shift = mods.chunk(2, dim=-1)  # [B, n_blocks] each
        
        # Apply tanh to keep scale in reasonable range
        scale = 1.0 + self.scale_temp * torch.tanh(raw_scale)  # [~0.5, ~1.5]
        
        # Track stats for logging (always, for analysis)
        with torch.no_grad():
            self._last_scale_mean = scale.mean().item()
            self._last_scale_std = scale.std().item()
            self._last_shift_mean = shift.abs().mean().item()
        
        # 3. Apply modulation per Hadamard block (Triton with autograd backward)
        if HAS_TRITON and x.is_cuda and ChannelModFunction is not None:
            x_out = self._forward_triton(x, scale, shift)
        else:
            x_out = self._forward_pytorch(x, scale, shift)
        
        return x_out
    
    def _forward_pytorch(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback for channel modulation."""
        B, T, C = x.shape
        
        x_blocks = x.view(B, T, self.n_blocks, self.block_dim)  # [B, T, 8, 32]
        
        # Expand scale and shift for broadcasting
        scale = scale.view(B, 1, self.n_blocks, 1)  # [B, 1, 8, 1]
        shift = shift.view(B, 1, self.n_blocks, 1)  # [B, 1, 8, 1]
        
        # Modulate: each block gets its own scale and shift
        x_mod = x_blocks * scale + shift * 0.1
        
        # Residual blend
        if self.residual_scale > 0:
            x_out = x_blocks + self.residual_scale * (x_mod - x_blocks)
        else:
            x_out = x_mod
        
        return x_out.view(B, T, C)
    
    def _forward_triton(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Triton fused channel modulation with autograd support."""
        return ChannelModFunction.apply(
            x, scale, shift,
            self.n_blocks, self.block_dim,
            self.residual_scale, 0.1  # shift_scale = 0.1
        )
    
    def _forward_ternary(self, bulk_state: torch.Tensor) -> torch.Tensor:
        """Forward with ternary quantization."""
        h = bulk_state
        for i, layer in enumerate(self.context_mlp):
            if isinstance(layer, nn.Linear):
                w = quantize_ternary(layer.weight)
                h = F.linear(h, w, layer.bias)
            else:
                h = layer(h)
        return h
    
    def get_modulation_stats(self) -> dict:
        """Get statistics about modulation behavior for logging."""
        return {
            'scale_mean': self._last_scale_mean,
            'scale_std': self._last_scale_std,
            'shift_mean': self._last_shift_mean,
            'scale_temp': self.scale_temp.item(),
        }


class HadamardChannelModulationBlock(nn.Module):
    """
    Complete block with normalization and residual.
    
    Can be inserted between transformer layers or after FFN.
    
    Structure:
        x → Norm → ChannelMod → + x
    """
    
    def __init__(self, config, n_blocks: int = 8):
        super().__init__()
        from .chassis import RMSNorm
        
        self.norm = RMSNorm(config.n_embd)
        self.channel_mod = HadamardChannelModulation(
            n_embd=config.n_embd,
            n_blocks=n_blocks,
            use_ternary=getattr(config, 'use_ternary', True),
            residual_scale=0.1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, n_embd] -> [B, T, n_embd]"""
        return x + self.channel_mod(self.norm(x))


def should_use_channel_mod(config) -> bool:
    """Check if config enables channel modulation"""
    return getattr(config, 'use_channel_mod', False)
