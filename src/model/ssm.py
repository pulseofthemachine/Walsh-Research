"""
Octonion State Space Model (OctonionSSM)
=========================================

Replaces attention with recurrent state space dynamics.
Achieves O(1) memory and compute per token.

Based on Mamba (Gu & Dao, 2023) with octonion structure for the state transitions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics import OctonionTernaryLinear


class OctonionSSM(nn.Module):
    """
    Octonion State Space Model layer.
    
    Replaces attention with recurrent state update:
        h[t] = (1 - delta) * h[t-1] + delta * (A(h[t-1]) + B(x[t]))
        y[t] = C(h[t])
    
    Where:
        - A: State transition (how past state evolves) - OctonionTernary
        - B: Input injection (how new input enters state) - OctonionTernary
        - C: Output readout (what to output from state) - OctonionTernary
        - delta: Selective gating (controls update rate) - Dense FP32
    
    The octonion structure brings:
        - Non-commutativity: order of state updates matters
        - Dimension specialization: e_i can specialize for different memory types
    """
    
    def __init__(self, n_embd, expand_factor=1):
        super().__init__()
        self.n_embd = n_embd
        self.d_state = n_embd * expand_factor  # State dimension (can be larger than embedding)
        
        # State transition: how past state evolves
        self.A = OctonionTernaryLinear(self.d_state, self.d_state)
        
        # Input injection: how new input enters state
        self.B = OctonionTernaryLinear(n_embd, self.d_state)
        
        # Output readout: what to output from state
        self.C = OctonionTernaryLinear(self.d_state, n_embd)
        
        # Selective gating: controls update rate per dimension
        # This is the key Mamba innovation - content-aware gating
        self.delta_proj = nn.Linear(n_embd, self.d_state, bias=False)
        
        # Initialize delta bias to encourage slow updates initially
        # (helps with training stability)
        self.delta_bias = nn.Parameter(torch.ones(self.d_state) * -2.0)
        
        # Optional: learnable decay rate per octonion dimension
        # Different e_i can have different memory persistence
        self.decay_scale = nn.Parameter(torch.ones(8))
        
    def _apply_octonion_linear(self, layer, x_2d):
        """
        Apply OctonionTernaryLinear to 2D tensor by adding/removing time dim.
        OctonionTernaryLinear expects tuple of [B, T, D/8] tensors.
        
        Args:
            layer: OctonionTernaryLinear layer
            x_2d: [B, D] input tensor
        Returns:
            [B, D_out] output tensor
        """
        B, D = x_2d.shape
        # Add time dimension: [B, D] -> [B, 1, D]
        x_3d = x_2d.unsqueeze(1)
        # Split into parts
        x_parts = tuple(p for p in x_3d.split(D // 8, dim=-1))
        # Apply layer
        y_parts = layer(x_parts)
        # Concat and remove time dim: [B, 1, D_out] -> [B, D_out]
        return torch.cat(y_parts, dim=-1).squeeze(1)
        
    def forward(self, x, h_prev=None):
        """
        Forward pass.
        
        Args:
            x: [B, T, D] input sequence
            h_prev: [B, D_state] previous hidden state (None = zeros)
            
        Returns:
            y: [B, T, D] output sequence
            h_new: [B, D_state] final hidden state
        """
        B, T, D = x.shape
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        
        # Compute gating for entire sequence
        # delta controls how much to update state at each position
        delta_raw = self.delta_proj(x) + self.delta_bias  # [B, T, D_state]
        delta = torch.sigmoid(delta_raw)  # [B, T, D_state], range [0, 1]
        
        # Process sequence step by step (will parallelize with scan later)
        outputs = []
        h = h_prev
        
        for t in range(T):
            x_t = x[:, t, :]  # [B, D]
            delta_t = delta[:, t, :]  # [B, D_state]
            
            # Apply octonion operations (with dimension handling)
            Ah = self._apply_octonion_linear(self.A, h)  # State evolution
            Bx = self._apply_octonion_linear(self.B, x_t)  # Input injection
            
            # Selective state update
            # h_new = (1 - delta) * h + delta * (Ah + Bx)
            h_candidate = Ah + Bx
            h = (1 - delta_t) * h + delta_t * h_candidate
            
            # Output readout
            y_t = self._apply_octonion_linear(self.C, h)  # [B, D]
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, T, D]
        h_new = h  # Final hidden state
        
        return y, h_new
    
    def forward_single(self, x, h_prev):
        """
        Single-step forward for inference.
        
        Args:
            x: [B, D] single token embedding
            h_prev: [B, D_state] previous hidden state
            
        Returns:
            y: [B, D] output
            h_new: [B, D_state] new hidden state
        """
        B, D = x.shape
        
        # Compute gating
        delta = torch.sigmoid(self.delta_proj(x) + self.delta_bias)  # [B, D_state]
        
        # Split into octonion parts
        x_parts = x.split(D // 8, dim=-1)
        h_parts = h_prev.split(self.d_state // 8, dim=-1)
        
        # Apply octonion operations
        Ah = torch.cat(self.A(h_parts), dim=-1)
        Bx = torch.cat(self.B(x_parts), dim=-1)
        
        # Selective state update
        h_candidate = Ah + Bx
        h_new = (1 - delta) * h_prev + delta * h_candidate
        
        # Output readout
        h_parts_new = h_new.split(self.d_state // 8, dim=-1)
        y = torch.cat(self.C(h_parts_new), dim=-1)
        
        return y, h_new


class ParallelScan(torch.autograd.Function):
    """
    Parallel associative scan for efficient SSM training.
    
    Computes h[t] = a[t] * h[t-1] + b[t] for all t in O(log T) parallel steps.
    This is the key to making SSM training as fast as attention.
    
    For now, we use the sequential version. This can be optimized later
    with Triton or CUDA kernels.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Args:
            a: [B, T, D] - decay factors
            b: [B, T, D] - input contributions
            
        Returns:
            h: [B, T, D] - hidden states for all timesteps
        """
        B, T, D = a.shape
        h = torch.zeros(B, T, D, device=a.device, dtype=a.dtype)
        
        # Sequential scan (will parallelize later)
        h_prev = torch.zeros(B, D, device=a.device, dtype=a.dtype)
        for t in range(T):
            h[:, t, :] = a[:, t, :] * h_prev + b[:, t, :]
            h_prev = h[:, t, :]
        
        ctx.save_for_backward(a, h)
        return h
    
    @staticmethod
    def backward(ctx, grad_h):
        """Backward pass using reverse scan."""
        a, h = ctx.saved_tensors
        B, T, D = a.shape
        
        grad_a = torch.zeros_like(a)
        grad_b = torch.zeros_like(grad_h)
        
        # Reverse scan for gradients
        grad_h_acc = torch.zeros(B, D, device=a.device, dtype=a.dtype)
        for t in range(T - 1, -1, -1):
            grad_h_acc = grad_h_acc + grad_h[:, t, :]
            grad_b[:, t, :] = grad_h_acc
            if t > 0:
                grad_a[:, t, :] = grad_h_acc * h[:, t - 1, :]
            grad_h_acc = grad_h_acc * a[:, t, :]
        
        return grad_a, grad_b


def parallel_scan(a, b):
    """Parallel scan wrapper."""
    return ParallelScan.apply(a, b)
