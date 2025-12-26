"""
SpinNet: Fast Hadamard Transform (FHT) CUDA Kernel
---------------------------------------------------
O(n log n) mixing layer for 32D Hadamard algebra.

The FHT replaces full Clifford multiplication (O(n²)) with recursive
butterfly operations, enabling 32D compression with minimal speed penalty.

Key properties:
- Self-inverse: FHT(FHT(x)) = n * x (self-adjoint)
- Orthogonal: H @ H.T = n * I
- Pure +/- operations: No complex arithmetic needed

Optimizations:
- FP32 accumulators for numerical stability
- Ternary weight packing (4 values per byte, 4x memory reduction)
- INT8 activation quantization for inference
- Zero-copy as_strided views for interleaved tensors
"""

import torch
import torch.nn as nn
import torch._dynamo
import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# HADAMARD MATRIX GENERATION
# -----------------------------------------------------------------------------

def hadamard_matrix(n: int, normalize: bool = True) -> torch.Tensor:
    """
    Generate n×n Hadamard matrix (n must be power of 2).
    
    Args:
        n: Dimension (must be power of 2)
        normalize: If True, scale by 1/sqrt(n) for orthonormality
        
    Returns:
        Hadamard matrix as float32 tensor
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    
    # Sylvester construction: H_2k = [[H_k, H_k], [H_k, -H_k]]
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    if normalize:
        H = H / math.sqrt(n)
    
    return H


# -----------------------------------------------------------------------------
# TERNARY WEIGHT PACKING (4x memory reduction)
# -----------------------------------------------------------------------------
# Pack 4 ternary values {-1, 0, +1} into a single byte.
# Encoding: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10 (2 bits per value)

def pack_ternary_weights(w: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights from [32, N, K] float -> [32, N, K//4] uint8.
    
    Each byte contains 4 ternary values:
        byte = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
    
    Args:
        w: Ternary weight tensor [32, N, K] with values in {-1, 0, +1}
        
    Returns:
        Packed tensor [32, N, K//4] as uint8
    """
    assert w.shape[-1] % 4 == 0, f"K dimension must be divisible by 4, got {w.shape[-1]}"
    
    # Map: -1 -> 0, 0 -> 1, +1 -> 2
    encoded = (w.int() + 1).to(torch.uint8)
    
    # Reshape to group 4 consecutive values
    shape = w.shape[:-1] + (w.shape[-1] // 4, 4)
    encoded = encoded.view(*shape)
    
    # Pack 4 values into 1 byte
    packed = (encoded[..., 0] << 6) | \
             (encoded[..., 1] << 4) | \
             (encoded[..., 2] << 2) | \
             (encoded[..., 3])
    
    return packed.to(torch.uint8)


def unpack_ternary_weights(packed: torch.Tensor, original_k: int) -> torch.Tensor:
    """
    Unpack uint8 tensor back to ternary float tensor.
    
    Args:
        packed: Packed tensor [32, N, K//4] as uint8
        original_k: Original K dimension size
        
    Returns:
        Unpacked tensor [32, N, K] as float32 with values in {-1, 0, +1}
    """
    # Extract 4 values from each byte
    v0 = ((packed >> 6) & 0b11).to(torch.int8) - 1
    v1 = ((packed >> 4) & 0b11).to(torch.int8) - 1
    v2 = ((packed >> 2) & 0b11).to(torch.int8) - 1
    v3 = (packed & 0b11).to(torch.int8) - 1
    
    # Interleave back to original layout
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1)
    unpacked = unpacked.view(*packed.shape[:-1], original_k)
    
    return unpacked.float()


# -----------------------------------------------------------------------------
# INT8 ACTIVATION QUANTIZATION
# -----------------------------------------------------------------------------

def quantize_activations_int8(x: torch.Tensor) -> tuple:
    """
    Quantize activations to INT8 using absmax scaling.
    
    Args:
        x: Activation tensor [*, K] in bfloat16/float32
        
    Returns:
        (x_int8, scale): Quantized tensor and per-row scale factor
    """
    # Per-row absmax (token-wise scaling)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_scaled = x / scale * 127.0
    x_int8 = x_scaled.round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale / 127.0  # Return scale for dequantization


def dequantize_output(y_int32: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize INT32 accumulator output back to floating point.
    
    For ternary weights, the effective weight scale is 1.0 (since values are -1/0/+1).
    So: y_float = y_int32 * x_scale
    """
    return y_int32.float() * x_scale



# -----------------------------------------------------------------------------
# TRITON KERNEL: FAST HADAMARD TRANSFORM (In-Place Butterfly)
# -----------------------------------------------------------------------------

@triton.jit
def fht_butterfly_kernel(
    x_ptr, y_ptr,
    stride_batch, stride_dim,
    N: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Single stage of FHT butterfly operation.
    
    For stage s, each element k pairs with k ^ (1 << s):
        x[k], x[partner] = x[k] + x[partner], x[k] - x[partner]
    """
    batch_idx = tl.program_id(0)
    elem_idx = tl.program_id(1)
    
    # Compute butterfly partner
    partner = elem_idx ^ (1 << STAGE)
    
    # Only lower index of pair does the computation (avoid double processing)
    if elem_idx < partner:
        # Load pair
        x_k = tl.load(x_ptr + batch_idx * stride_batch + elem_idx * stride_dim)
        x_p = tl.load(x_ptr + batch_idx * stride_batch + partner * stride_dim)
        
        # Butterfly: sum and difference
        y_k = x_k + x_p
        y_p = x_k - x_p
        
        # Store
        tl.store(y_ptr + batch_idx * stride_batch + elem_idx * stride_dim, y_k)
        tl.store(y_ptr + batch_idx * stride_batch + partner * stride_dim, y_p)
    elif partner < elem_idx:
        # Partner already computed this pair, just copy if needed
        pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fht_fused_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_m, stride_n,
    LOG_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused FHT kernel - all log2(N) stages in one kernel.
    
    Grid: (cdiv(M, BLOCK_M),)
    Each block processes BLOCK_M rows of dimension N.
    """
    pid = tl.program_id(0)
    
    # Row indices this block handles
    row_start = pid * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < M
    
    # Load entire row into registers (N must be small enough)
    # For N=32, this is 32 floats = 128 bytes per row
    col_offs = tl.arange(0, BLOCK_N)
    
    # Load input rows
    x_ptrs = x_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n
    x_mask = row_mask[:, None] & (col_offs[None, :] < N)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    
    # Apply all butterfly stages in-place
    for stage in range(LOG_N):
        stride = 1 << stage
        
        # Vectorized butterfly over columns
        for k in range(0, N, 2 * stride):
            for i in range(stride):
                idx_a = k + i
                idx_b = k + i + stride
                
                if idx_a < N and idx_b < N:
                    # Extract elements
                    a = tl.load(x_ptr + row_offs[:, None] * stride_m + idx_a * stride_n, mask=row_mask[:, None])[:, 0]
                    b = tl.load(x_ptr + row_offs[:, None] * stride_m + idx_b * stride_n, mask=row_mask[:, None])[:, 0]
                    
                    # Butterfly
                    sum_ab = a + b
                    diff_ab = a - b
                    
                    # Store back
                    tl.store(y_ptr + row_offs * stride_m + idx_a * stride_n, sum_ab, mask=row_mask)
                    tl.store(y_ptr + row_offs * stride_m + idx_b * stride_n, diff_ab, mask=row_mask)
    
    # Final normalization
    norm = 1.0 / tl.sqrt(float(N))
    y_ptrs = y_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n
    y = tl.load(y_ptrs, mask=x_mask)
    tl.store(y_ptrs, y * norm, mask=x_mask)


# -----------------------------------------------------------------------------
# PYTORCH REFERENCE IMPLEMENTATION (Used for correctness verification)
# -----------------------------------------------------------------------------

def fht_reference(x: torch.Tensor) -> torch.Tensor:
    """
    Reference FHT using matrix multiplication (for testing).
    
    Args:
        x: Input tensor [..., N] where N is power of 2
        
    Returns:
        Transformed tensor [..., N]
    """
    N = x.shape[-1]
    H = hadamard_matrix(N, normalize=True).to(x.device, x.dtype)
    return x @ H.T


def fht_butterfly(x: torch.Tensor) -> torch.Tensor:
    """
    FHT using in-place butterfly operations (O(n log n)).
    
    Args:
        x: Input tensor [..., N] where N is power of 2
        
    Returns:
        Transformed tensor [..., N]
    """
    N = x.shape[-1]
    assert N > 0 and (N & (N - 1)) == 0, f"N must be power of 2, got {N}"
    
    y = x.clone()
    log_n = int(math.log2(N))
    
    # Butterfly stages
    for stage in range(log_n):
        stride = 1 << stage
        for k in range(0, N, 2 * stride):
            for i in range(stride):
                idx_a, idx_b = k + i, k + i + stride
                a, b = y[..., idx_a].clone(), y[..., idx_b].clone()
                y[..., idx_a] = a + b
                y[..., idx_b] = a - b
    
    # Normalize
    y = y / math.sqrt(N)
    return y


# -----------------------------------------------------------------------------
# AUTOGRAD FUNCTION
# -----------------------------------------------------------------------------

class FHTFunction(torch.autograd.Function):
    """
    Autograd wrapper for Fast Hadamard Transform.
    
    FHT is self-adjoint (symmetric), so backward = forward (up to scaling).
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(x.shape[-1]))
        return fht_butterfly(x)
    
    @staticmethod
    def backward(ctx, grad_y: torch.Tensor) -> torch.Tensor:
        # FHT is self-adjoint: H = H.T
        # So d/dx (x @ H) = grad_y @ H.T = grad_y @ H = FHT(grad_y)
        return fht_butterfly(grad_y)


def fht(x: torch.Tensor) -> torch.Tensor:
    """Apply Fast Hadamard Transform with autograd support."""
    return FHTFunction.apply(x)


# -----------------------------------------------------------------------------
# HADAMARD TERNARY LINEAR LAYER
# -----------------------------------------------------------------------------

class HadamardTernaryLinear(nn.Module):
    """
    32D Linear layer with FHT mixing.
    
    Replaces full Clifford multiplication O(n²) with FHT O(n log n).
    Uses 32 sub-weight matrices mixed via Hadamard transform.
    
    Comparison:
    - Octonion (8D): 64 ops per interaction, 1/8th parameter compression
    - Hadamard (32D): 160 ops per interaction, 1/32nd parameter compression
    
    Args:
        in_features: Input dimension (must be divisible by 32)
        out_features: Output dimension (must be divisible by 32)
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert in_features % self.ALGEBRA_DIM == 0, f"in_features must be divisible by {self.ALGEBRA_DIM}"
        assert out_features % self.ALGEBRA_DIM == 0, f"out_features must be divisible by {self.ALGEBRA_DIM}"
        
        self.in_o = in_features // self.ALGEBRA_DIM   # 32 input groups
        self.out_o = out_features // self.ALGEBRA_DIM  # 32 output groups
        
        # 32 ternary weight matrices [32, out_o, in_o]
        self.weight = nn.Parameter(torch.rand(self.ALGEBRA_DIM, self.out_o, self.in_o) * 2 - 1)
        
        # Alpha (input diagonal): per-feature scaling BEFORE FHT [32, in_o]
        self.alpha = nn.Parameter(torch.ones(self.ALGEBRA_DIM, self.in_o))
        
        # Beta (output diagonal): per-feature scaling AFTER FHT [32, out_o]
        beta_init = math.sqrt(3.0 / (2.0 * self.in_o))
        self.beta = nn.Parameter(torch.ones(self.ALGEBRA_DIM, self.out_o) * beta_init)
        
        # Quantization function (lazy import)
        self._quantize_fn = None
    
    def _get_quantize_fn(self):
        if self._quantize_fn is None:
            from .physics import quantize_ternary
            self._quantize_fn = quantize_ternary
        return self._quantize_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FHT mixing.
        
        Architecture: y = β · FHT(W_ternary · FHT(α · x))
        """
        B, T, D = x.shape
        assert D == self.in_o * self.ALGEBRA_DIM
        
        # Quantize weights to ternary
        quantize_fn = self._get_quantize_fn()
        w_q = quantize_fn(self.weight, self.training)
        
        # Split into 32 sub-groups: [B, T, 32, in_o]
        x_parts = x.view(B, T, self.ALGEBRA_DIM, self.in_o)
        
        # Apply input diagonal scaling (α · x) - [32, in_o] broadcasts with [B, T, 32, in_o]
        x_scaled = x_parts * self.alpha
        
        # Apply FHT to mix across the 32 dimensions: [B, T, 32, in_o]
        x_mixed = fht(x_scaled.transpose(-2, -1)).transpose(-2, -1)  # FHT on dim=-2
        
        # Matrix multiply each group: y_i = x_mixed_i @ W[i].T
        # [B, T, 32, in_o] @ [32, in_o, out_o] -> [B, T, 32, out_o]
        y_parts = torch.einsum('btgi,gio->btgo', x_mixed, w_q.transpose(-1, -2))
        
        # Apply FHT again to mix outputs
        y_mixed = fht(y_parts.transpose(-2, -1)).transpose(-2, -1)
        
        # Apply output scaling (β) - [32, out_o] broadcasts with [B, T, 32, out_o]
        y_scaled = y_mixed * self.beta
        
        return y_scaled.reshape(B, T, -1)  # [B, T, out_features]
    
    def extra_repr(self) -> str:
        return f"in={self.in_o * self.ALGEBRA_DIM}, out={self.out_o * self.ALGEBRA_DIM}, algebra=hadamard32"


# NOTE: Fused FHT+MatMul Triton kernel is not implemented.
# FHT requires cross-column butterfly communication while MatMul accumulates over K.
# These operations cannot be efficiently fused in a single kernel.
# The torch.einsum approach below leverages cuBLAS/Tensor Cores effectively for N=32.


class HadamardLinear(nn.Module):
    """
    32D Hadamard Linear layer with FHT mixing and learned diagonals.
    
    Architecture: y = β · H · W_ternary · H · (α · x)
    
    The learned diagonals (α, β) provide fine-grained feature selection,
    compensating for the coarse ternary weights and fixed Hadamard mixing.
    
    NOT truly fused - issues 3 separate kernels: einsum(FHT) -> einsum(MatMul) -> einsum(FHT).
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert in_features % self.ALGEBRA_DIM == 0
        assert out_features % self.ALGEBRA_DIM == 0
        
        self.in_o = in_features // self.ALGEBRA_DIM
        self.out_o = out_features // self.ALGEBRA_DIM
        self.in_features = in_features
        self.out_features = out_features
        
        # Ternary weight matrix
        self.weight = nn.Parameter(torch.rand(self.ALGEBRA_DIM, self.out_o, self.in_o) * 2 - 1)
        
        # Alpha (input diagonal): per-feature scaling BEFORE FHT
        # Shape [32, in_o] for full resolution within each Hadamard group
        self.alpha = nn.Parameter(torch.ones(self.ALGEBRA_DIM, self.in_o))
        
        # Beta (output diagonal): per-feature scaling AFTER FHT
        # Shape [32, out_o] for full resolution, with variance-preserving init
        beta_init = math.sqrt(3.0 / (2.0 * self.in_o))
        self.beta = nn.Parameter(torch.ones(self.ALGEBRA_DIM, self.out_o) * beta_init)
        
        # Cache Hadamard matrix for FHT
        self.register_buffer('H', hadamard_matrix(self.ALGEBRA_DIM, normalize=True))
        
        self._quantize_fn = None
    
    def _get_quantize_fn(self):
        if self._quantize_fn is None:
            from .physics import quantize_ternary
            self._quantize_fn = quantize_ternary
        return self._quantize_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        orig_dtype = x.dtype
        
        # Quantize weights
        quantize_fn = self._get_quantize_fn()
        w_q = quantize_fn(self.weight, self.training)
        
        # Reshape: [B, T, 32, in_o]
        x_parts = x.view(B, T, self.ALGEBRA_DIM, self.in_o)
        
        # Apply input diagonal scaling (α · x) - per-feature before FHT
        # alpha is [32, in_o], broadcasts with [B, T, 32, in_o]
        x_scaled = x_parts * self.alpha
        
        # FHT via matrix multiply with FP32 accumulator for stability
        # [B, T, 32, in_o] with H [32, 32] on dim 2
        x_mixed = torch.einsum('btgi,gh->bthi', x_scaled.float(), self.H)
        
        # Batched matmul with FP32 accumulator: [B, T, 32, in_o] @ [32, out_o, in_o].T -> [B, T, 32, out_o]
        y_parts = torch.einsum('btgi,goi->btgo', x_mixed, w_q.float())
        
        # FHT on output
        y_mixed = torch.einsum('btgo,gh->btho', y_parts, self.H)
        
        # Apply output scaling (β) - per-feature after FHT
        # beta is [32, out_o], broadcasts with [B, T, 32, out_o]
        y_scaled = y_mixed * self.beta
        
        # Reshape and restore original dtype
        return y_scaled.reshape(B, T, -1).to(orig_dtype)


# -----------------------------------------------------------------------------
# TUPLE-COMPATIBLE WRAPPER (Drop-in for OctonionTernaryLinear)
# -----------------------------------------------------------------------------

class HadamardTernaryLinearTuple(nn.Module):
    """
    Wrapper for HadamardLinear with tuple interface.
    
    Accepts tuple of 32 tensors like OctonionTernaryLinear accepts tuple of 8.
    This enables drop-in replacement in existing model code.
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.inner = HadamardLinear(in_features, out_features, bias)
        self.in_o = self.inner.in_o
        self.out_o = self.inner.out_o
    
    @property
    def weight(self):
        return self.inner.weight
    
    @property
    def beta(self):
        return self.inner.beta
    
    def forward(self, x_parts: tuple) -> tuple:
        """
        Forward pass with tuple interface.
        
        Args:
            x_parts: Tuple of 32 tensors, each [B, T, in_o]
            
        Returns:
            Tuple of 32 tensors, each [B, T, out_o]
        """
        assert len(x_parts) == self.ALGEBRA_DIM, f"Expected {self.ALGEBRA_DIM} parts, got {len(x_parts)}"
        
        # Stack parts: [B, T, 32 * in_o]
        x = torch.cat(x_parts, dim=-1)
        
        # Apply Hadamard linear
        y = self.inner(x)
        
        # Split output back to tuple
        return tuple(y.split(self.out_o, dim=-1))


# -----------------------------------------------------------------------------
# HADAMARD HEAD MIXER (Analogous to OctonionHeadMixer)
# -----------------------------------------------------------------------------

class HadamardHeadMixer(nn.Module):
    """
    Mix attention heads using Hadamard (FHT) structure.
    
    After standard attention computes outputs for 32 heads, this module
    mixes them using the Hadamard transform for O(n log n) mixing.
    
    Input: [B, 32, T, head_dim] (32 heads)
    Output: [B, 32, T, head_dim] (mixed heads)
    
    This is the reference implementation - use HadamardHeadMixerFused for speed.
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        
        # Learnable mixing weights: 32 weight matrices [head_dim, head_dim]
        self.W = nn.Parameter(torch.randn(self.ALGEBRA_DIM, head_dim, head_dim) * 0.02)
        
        # Variance-preserving beta for head mixing
        beta_init = math.sqrt(3.0 / (2.0 * head_dim))
        self.beta = nn.Parameter(torch.ones(head_dim) * beta_init)
        
        # Cache normalized Hadamard matrix
        self.register_buffer('H', hadamard_matrix(self.ALGEBRA_DIM, normalize=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 32, T, head_dim] - 32 attention head outputs
        Returns: [B, 32, T, head_dim] - mixed via Hadamard transform
        """
        B, H, T, D = x.shape
        assert H == self.ALGEBRA_DIM, f"HadamardHeadMixer requires {self.ALGEBRA_DIM} heads, got {H}"
        
        # Apply FHT to mix across heads: [B, 32, T, D]
        # FHT on the head dimension (dim 1)
        x_mixed = torch.einsum('bhti,hg->bgti', x, self.H)
        
        # Apply learnable per-head weights: [B, 32, T, D] @ [32, D, D] -> [B, 32, T, D]
        y = torch.einsum('bhti,hio->bhto', x_mixed, self.W)
        
        # Apply FHT again for output mixing
        y_mixed = torch.einsum('bhti,hg->bgti', y, self.H)
        
        # Apply beta scaling
        return y_mixed * self.beta


class HadamardHeadMixerFused(nn.Module):
    """
    High-performance Hadamard Head Mixer using fused operations.
    
    Same semantics as HadamardHeadMixer but with optimized implementation.
    Uses matrix multiply for FHT (cached Hadamard buffer) instead of butterfly
    for better GPU utilization at small sizes.
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        
        # Learnable mixing weights
        self.W = nn.Parameter(torch.randn(self.ALGEBRA_DIM, head_dim, head_dim) * 0.02)
        
        # Variance-preserving beta for head mixing
        beta_init = math.sqrt(3.0 / (2.0 * head_dim))
        self.beta = nn.Parameter(torch.ones(head_dim) * beta_init)
        
        # Cache normalized Hadamard matrix for FHT
        self.register_buffer('H', hadamard_matrix(self.ALGEBRA_DIM, normalize=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 32, T, head_dim] - 32 attention head outputs
        Returns: [B, 32, T, head_dim] - mixed via Hadamard transform
        """
        B, H_heads, T, D = x.shape
        assert H_heads == self.ALGEBRA_DIM
        
        # Fused FHT -> Linear -> FHT using einsum for efficiency
        # Step 1: FHT on head dimension
        x_fht = torch.einsum('bhti,hg->bgti', x, self.H.to(x.dtype))
        
        # Step 2: Per-head linear transform
        y = torch.einsum('bhti,hio->bhto', x_fht, self.W.to(x.dtype))
        
        # Step 3: FHT on output
        y_fht = torch.einsum('bhti,hg->bgti', y, self.H.to(x.dtype))
        
        return y_fht * self.beta.to(x.dtype)


# -----------------------------------------------------------------------------
# ZERO-COPY FUSED FUNCTION (with as_strided optimization)
# -----------------------------------------------------------------------------

@torch._dynamo.disable  # Prevent graph breaks with torch.compile
def hadamard_fused(x_parts: tuple, weight: torch.Tensor, H: torch.Tensor) -> tuple:
    """
    Fused Hadamard linear with FHT mixing.
    
    Args:
        x_parts: tuple of 32 tensors, each [B, T, K]
        weight: [32, N, K] weight tensor (quantized ternary)
        H: [32, 32] normalized Hadamard matrix
        
    Returns:
        tuple of 32 tensors, each [B, T, N]
    """
    B, T, K = x_parts[0].shape
    N = weight.shape[1]
    
    # ---------------------------------------------------------
    # Zero-Copy Optimization for Interleaved Inputs
    # ---------------------------------------------------------
    # SAFETY: Only use as_strided if we can verify the memory layout.
    # This check ensures x_parts are contiguous slices of an interleaved tensor.
    x0 = x_parts[0].view(-1, K)
    use_fast_path = False
    
    if len(x_parts) == 32 and x0.is_contiguous():
        try:
            # Check specific stride pattern: (32*K, 1) implies interleaved rows
            expected_stride = (32 * K, 1)
            if x0.stride() == expected_stride:
                # Verify all 32 chunks are memory-adjacent
                base_ptr = x_parts[0].data_ptr()
                elem_size = x_parts[0].element_size()
                all_adjacent = True
                for i in range(1, 32):
                    expected_ptr = base_ptr + i * K * elem_size
                    if x_parts[i].data_ptr() != expected_ptr:
                        all_adjacent = False
                        break
                use_fast_path = all_adjacent
        except Exception:
            pass  # Fall back to safe path on any error

    M = B * T
    if use_fast_path:
        # Create virtual stacked tensor using stride tricks (Zero Copy)
        # Stride: dim0(32)=K, dim1(M)=32*K, dim2(K)=1
        x_stacked = torch.as_strided(x0, size=(32, M, K), stride=(K, 32*K, 1))
    else:
        # Fallback: Physical stack (Copy) - always safe
        x_stacked = torch.stack([x.view(-1, K) for x in x_parts], dim=0)
        x_stacked = x_stacked.contiguous().to(torch.bfloat16)
    
    # Reshape for FHT: [M, 32, K]
    x_grouped = x_stacked.permute(1, 0, 2)  # [M, 32, K]
    
    # FHT on dim 1 (32D mixing) - use FP32 accumulator
    x_mixed = torch.einsum('mgi,gh->mhi', x_grouped.float(), H).to(x_grouped.dtype)
    
    # Per-group matmul: [M, 32, K] @ [32, K, N] -> [M, 32, N]
    # Use FP32 accumulator for numerical stability
    y = torch.einsum('mgi,gki->mgk', x_mixed.float(), weight.transpose(-1, -2).float())
    
    # FHT on output
    y_mixed = torch.einsum('mgi,gh->mhi', y, H).to(torch.bfloat16)
    
    # Unstack outputs: tuple of [B, T, N]
    y_stacked = y_mixed.permute(1, 0, 2)  # [32, M, N]
    return tuple(y_stacked[i].view(B, T, -1) for i in range(32))


# -----------------------------------------------------------------------------
# PACKED INFERENCE LAYER (4x memory reduction)
# -----------------------------------------------------------------------------

class HadamardPackedLinear(nn.Module):
    """
    Inference-optimized Hadamard Linear with packed ternary weights.
    
    - Weights stored as packed uint8 (4x memory reduction)
    - Uses FP32 accumulators for numerical stability
    - as_strided for zero-copy input views
    
    Use this for deployment after training with HadamardLinear.
    """
    
    ALGEBRA_DIM = 32
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features % self.ALGEBRA_DIM == 0
        assert out_features % self.ALGEBRA_DIM == 0
        
        self.in_o = in_features // self.ALGEBRA_DIM
        self.out_o = out_features // self.ALGEBRA_DIM
        self.in_features = in_features
        self.out_features = out_features
        
        # Packed weights: [32, out_o, in_o//4] as uint8
        self.register_buffer('weight_packed', torch.zeros(
            self.ALGEBRA_DIM, self.out_o, self.in_o // 4, dtype=torch.uint8
        ))
        
        # Scale factor (from training)
        self.register_buffer('beta', torch.ones(self.out_o))
        
        # Cached Hadamard matrix
        self.register_buffer('H', hadamard_matrix(self.ALGEBRA_DIM, normalize=True))
    
    @classmethod
    def from_trained(cls, trained_layer) -> 'HadamardPackedLinear':
        """
        Create packed layer from trained HadamardLinear.
        
        Args:
            trained_layer: Trained HadamardLinear or HadamardTernaryLinear
            
        Returns:
            HadamardPackedLinear with packed weights
        """
        # Get dimensions
        in_features = trained_layer.in_o * trained_layer.ALGEBRA_DIM
        out_features = trained_layer.out_o * trained_layer.ALGEBRA_DIM
        
        packed = cls(in_features, out_features)
        
        # Quantize weights to ternary and pack
        from .physics import quantize_ternary
        w_ternary = quantize_ternary(trained_layer.weight, training=False)
        packed.weight_packed.copy_(pack_ternary_weights(w_ternary))
        packed.beta.copy_(trained_layer.beta)
        
        return packed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with packed weights.
        
        Args:
            x: Input [B, T, in_features]
            
        Returns:
            Output [B, T, out_features]
        """
        B, T, D = x.shape
        
        # Unpack weights on-the-fly (could cache if memory permits)
        w = unpack_ternary_weights(self.weight_packed, self.in_o)
        
        # Reshape input: [B, T, 32, in_o]
        x_parts = x.view(B, T, self.ALGEBRA_DIM, self.in_o)
        
        # FHT mixing with FP32 accumulator
        x_mixed = torch.einsum('btgi,gh->bthi', x_parts.float(), self.H)
        
        # Matmul with FP32 accumulator
        y_parts = torch.einsum('btgi,gio->btgo', x_mixed, w.transpose(-1, -2))
        
        # FHT on output
        y_mixed = torch.einsum('btgo,gh->btho', y_parts, self.H)
        
        # Reshape and scale
        y = y_mixed.contiguous().view(B, T, -1).to(x.dtype)
        beta_expanded = self.beta.repeat(self.ALGEBRA_DIM)
        return y * beta_expanded


def optimize_for_inference(model: nn.Module) -> None:
    """
    Convert HadamardLinear layers to HadamardPackedLinear for inference.
    
    Modifies model in-place.
    
    Args:
        model: Model containing HadamardLinear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, (HadamardLinear, HadamardTernaryLinear)):
            # Get parent module
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name
            
            # Replace with packed version
            packed = HadamardPackedLinear.from_trained(module)
            setattr(parent, attr_name, packed)
            print(f"  Converted {name} to HadamardPackedLinear")


# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------

__all__ = [
    'hadamard_matrix',
    'fht',
    'fht_reference',
    'fht_butterfly',
    'pack_ternary_weights',
    'unpack_ternary_weights',
    'quantize_activations_int8',
    'dequantize_output',
    'hadamard_fused',
    'HadamardTernaryLinear',
    'HadamardLinear',
    'HadamardTernaryLinearTuple',
    'HadamardHeadMixer',
    'HadamardHeadMixerFused',
    'HadamardPackedLinear',
    'optimize_for_inference',
]



