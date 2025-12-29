"""
Walsh Physics Layer
---------------------
Ternary quantization and reference Octonion linear layer.

The OctonionTernaryLinear exported from this module uses CUDA-optimized
kernels when available, falling back to the reference implementation.
"""
import torch
import torch.nn as nn
import math

class TernaryAbsmeanSTE(torch.autograd.Function):
    """
    BitNet b1.58 style Straight-Through Estimator for ternary {-1, 0, 1}.
    
    Uses absmean scaling: γ = mean(|W|), then round(W/γ) clamped to [-1,+1].
    This adaptive threshold ensures consistent ternary distribution across training.
    """
    @staticmethod
    def forward(ctx, w_latent, training=True):
        # Compute per-tensor absolute mean as scale factor
        # Add small epsilon to avoid division by zero
        gamma = w_latent.abs().mean() + 1e-8
        
        # Scale weights by absmean, round, and clamp to ternary
        w_scaled = w_latent / gamma
        w_quant = torch.round(w_scaled)
        w_quant = torch.clamp(w_quant, -1, 1)
        
        ctx.save_for_backward(w_latent, torch.tensor([gamma]))
        return w_quant

    @staticmethod
    def backward(ctx, grad_output):
        w_latent, gamma_tensor = ctx.saved_tensors
        grad_input = grad_output.clone()
        # The Scream Clamp - prevent gradient explosion
        grad_input = torch.clamp(grad_input, -2.0, 2.0)
        return grad_input, None

def quantize_ternary(w, training=True):
    """
    BitNet b1.58 style ternary quantization with absmean scaling.
    
    Formula: W_ternary = clip(round(W / γ), -1, +1) where γ = mean(|W|)
    """
    w_quant = TernaryAbsmeanSTE.apply(w, training)
    # Soft clamp on latent weights to bound gradient magnitudes
    w_clamped = w.clamp(-1.5, 1.5)
    return w_clamped + (w_quant - w_clamped).detach()

class OctonionTernaryLinearRef(nn.Module):
    """Vectorized Octonion Layer (8x Kernel Reduction) - Reference Implementation."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        assert in_features % 8 == 0 and out_features % 8 == 0
        self.in_o = in_features // 8
        self.out_o = out_features // 8
        self.weight = nn.Parameter(torch.rand(8, self.out_o, self.in_o) * 2 - 1)
        
        # Variance-preserving beta: ternary E[w²] ≈ 2/3, so scale = sqrt(3/(2*in_o))
        beta_init = math.sqrt(3.0 / (2.0 * self.in_o))
        self.beta = nn.Parameter(torch.ones(self.out_o) * beta_init)

    def forward(self, x_parts):
        w_q = quantize_ternary(self.weight, self.training)
        W = [w_q[i].t() for i in range(8)]
        x0, x1, x2, x3, x4, x5, x6, x7 = x_parts
        
        # Cayley-Dickson Algebra
        y0 = (x0@W[0] - x1@W[1] - x2@W[2] - x3@W[3] - x4@W[4] - x5@W[5] - x6@W[6] - x7@W[7])
        y1 = (x0@W[1] + x1@W[0] + x2@W[3] - x3@W[2] + x4@W[5] - x5@W[4] - x6@W[7] + x7@W[6])
        y2 = (x0@W[2] - x1@W[3] + x2@W[0] + x3@W[1] + x4@W[6] + x5@W[7] - x6@W[4] - x7@W[5])
        y3 = (x0@W[3] + x1@W[2] - x2@W[1] + x3@W[0] + x4@W[7] - x5@W[6] + x6@W[5] - x7@W[4])
        y4 = (x0@W[4] - x1@W[5] - x2@W[6] - x3@W[7] + x4@W[0] + x5@W[1] + x6@W[2] + x7@W[3])
        y5 = (x0@W[5] + x1@W[4] - x2@W[7] + x3@W[6] - x4@W[1] + x5@W[0] - x6@W[3] + x7@W[2])
        y6 = (x0@W[6] + x1@W[7] + x2@W[4] - x3@W[5] - x4@W[2] + x5@W[3] + x6@W[0] - x7@W[1])
        y7 = (x0@W[7] - x1@W[6] + x2@W[5] + x3@W[4] - x4@W[3] - x5@W[2] + x6@W[1] + x7@W[0])

        return (y0*self.beta, y1*self.beta, y2*self.beta, y3*self.beta,
                y4*self.beta, y5*self.beta, y6*self.beta, y7*self.beta)

# Select Implementation
try:
    from .cayley_dickson_cuda import OctonionFusedLinear
    # print("Walsh: Using CUDA Fused Kernels") # Silence optimized print to avoid log spam
    OctonionTernaryLinear = OctonionFusedLinear
except ImportError:
    # print("Walsh: Using Python Reference Implementation")
    OctonionTernaryLinear = OctonionTernaryLinearRef