//! Fast Hadamard Transform (FHT) for 32D Algebra
//!
//! O(n log n) mixing using butterfly operations.
//! Self-inverse: FHT(FHT(x)) = x (up to normalization)

/// In-place Fast Hadamard Transform for 32 elements.
/// Uses butterfly pattern with 5 stages (log2(32) = 5).
pub fn fht_32(x: &mut [f32; 32]) {
    // 5 butterfly stages
    for stage in 0..5 {
        let stride = 1 << stage;
        let mut k = 0;
        while k < 32 {
            for i in 0..stride {
                let a = x[k + i];
                let b = x[k + i + stride];
                x[k + i] = a + b;
                x[k + i + stride] = a - b;
            }
            k += 2 * stride;
        }
    }
    
    // Normalize by 1/sqrt(32)
    let norm = 1.0 / 5.656854249492381; // sqrt(32)
    for v in x.iter_mut() {
        *v *= norm;
    }
}


/// Hadamard linear multiplication for 32D algebra.
/// 
/// Input: x [32, in_per_part] (32 groups of input features)
/// Weight: w [32, out_per_part, in_per_part] (32 weight matrices, stored sparse)
/// Alpha: [32, in_per_part] input scaling
/// Beta: [32, out_per_part] output scaling
/// 
/// Output: [32, out_per_part]
/// 
/// Architecture: y = β · FHT(W · FHT(α · x))
pub fn hadamard_linear(
    x: &[f32],           // [32 * in_per_part]
    bitmask: &[u8],      // Sparse ternary weight bitmask
    sign_bits: &[u8],    // Sign bits for non-zeros
    alpha: &[f32],       // [32 * in_per_part] input scaling
    beta: &[f32],        // [32 * out_per_part] output scaling (or [out_per_part] broadcast)
    in_per_part: usize,
    out_per_part: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; 32 * out_per_part];
    
    // Step 1: Apply input scaling (α · x) and FHT across 32 groups
    // Process each feature position across all 32 groups
    let mut x_scaled = vec![0.0f32; 32 * in_per_part];
    for g in 0..32 {
        for i in 0..in_per_part {
            let idx = g * in_per_part + i;
            x_scaled[idx] = x[idx] * alpha.get(idx).copied().unwrap_or(1.0);
        }
    }
    
    // FHT on each feature position across the 32 groups
    // x_mixed[i * 32 + g] = FHT applied to x_scaled[g * in_per_part + i] across g
    let mut x_mixed = vec![0.0f32; 32 * in_per_part];
    for i in 0..in_per_part {
        let mut group: [f32; 32] = [0.0; 32];
        for g in 0..32 {
            group[g] = x_scaled[g * in_per_part + i];
        }
        fht_32(&mut group);
        for g in 0..32 {
            x_mixed[g * in_per_part + i] = group[g];
        }
    }
    
    // Step 2: Matrix multiply per group with sparse ternary weights
    // For each group g: y[g] = x_mixed[g] @ W[g].T
    // Weight is stored in bitmask format: iterate over non-zeros
    let total_weights = 32 * out_per_part * in_per_part;
    let mut bit_idx = 0usize;
    let mut sign_idx = 0usize;
    
    for flat_idx in 0..total_weights {
        // Check if this position is non-zero
        let byte_idx = flat_idx / 8;
        let bit_pos = flat_idx % 8;
        
        if byte_idx < bitmask.len() && (bitmask[byte_idx] & (1 << bit_pos)) != 0 {
            // Non-zero weight - get sign
            let sign_byte_idx = sign_idx / 8;
            let sign_bit_pos = sign_idx % 8;
            let weight_val = if sign_byte_idx < sign_bits.len() && 
                               (sign_bits[sign_byte_idx] & (1 << sign_bit_pos)) != 0 {
                1.0f32  // +1
            } else {
                -1.0f32 // -1
            };
            sign_idx += 1;
            
            // Decode position: flat_idx = g * out_per_part * in_per_part + o * in_per_part + i
            let g = flat_idx / (out_per_part * in_per_part);
            let rem = flat_idx % (out_per_part * in_per_part);
            let o = rem / in_per_part;
            let i = rem % in_per_part;
            
            // y[g, o] += x_mixed[g, i] * weight
            let y_idx = g * out_per_part + o;
            let x_idx = g * in_per_part + i;
            y[y_idx] += x_mixed[x_idx] * weight_val;
        }
        bit_idx += 1;
    }
    
    // Step 3: FHT on output across 32 groups
    let mut y_mixed = vec![0.0f32; 32 * out_per_part];
    for o in 0..out_per_part {
        let mut group: [f32; 32] = [0.0; 32];
        for g in 0..32 {
            group[g] = y[g * out_per_part + o];
        }
        fht_32(&mut group);
        for g in 0..32 {
            y_mixed[g * out_per_part + o] = group[g];
        }
    }
    
    // Step 4: Apply output scaling (β)
    for g in 0..32 {
        for o in 0..out_per_part {
            let idx = g * out_per_part + o;
            // Beta can be [32 * out_per_part] or [out_per_part] broadcast
            let b = if beta.len() == out_per_part {
                beta[o]
            } else {
                beta.get(idx).copied().unwrap_or(1.0)
            };
            y_mixed[idx] *= b;
        }
    }
    
    y_mixed
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fht_self_inverse() {
        let mut x: [f32; 32] = [0.0; 32];
        for i in 0..32 {
            x[i] = i as f32;
        }
        let original = x.clone();
        
        // Apply FHT twice
        fht_32(&mut x);
        fht_32(&mut x);
        
        // Should get back original (self-inverse property)
        for i in 0..32 {
            assert!((x[i] - original[i]).abs() < 1e-5, 
                    "FHT self-inverse failed at {}: {} vs {}", i, x[i], original[i]);
        }
    }
    
    #[test]
    fn test_fht_orthogonal() {
        // FHT preserves norm (orthogonal transform)
        let mut x: [f32; 32] = [0.0; 32];
        for i in 0..32 {
            x[i] = (i as f32 + 1.0) / 32.0;
        }
        
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        fht_32(&mut x);
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        assert!((norm_before - norm_after).abs() < 1e-5,
                "FHT should preserve norm: {} vs {}", norm_before, norm_after);
    }
}
