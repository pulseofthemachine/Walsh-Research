/// In-place Fast Hadamard Transform for 32 elements (unnormalized).
/// Skips the 1/sqrt(32) normalization for use in hadamard_linear.
/// The normalization is deferred to be applied once via beta scaling.
#[inline(always)]
pub fn fht_32_unnorm(x: &mut [f32; 32]) {
    // Stage 1 (Stride 1)
    for i in (0..32).step_by(2) {
        let a = x[i];
        let b = x[i+1];
        x[i] = a + b;
        x[i+1] = a - b;
    }
    
    // Stage 2 (Stride 2)
    for i in (0..32).step_by(4) {
        let a0 = x[i];   let b0 = x[i+2]; x[i] = a0+b0;   x[i+2] = a0-b0;
        let a1 = x[i+1]; let b1 = x[i+3]; x[i+1] = a1+b1; x[i+3] = a1-b1;
    }
    
    // Stage 3 (Stride 4)
    for i in (0..32).step_by(8) {
        for j in 0..4 {
            let a = x[i+j];
            let b = x[i+j+4];
            x[i+j] = a + b;
            x[i+j+4] = a - b;
        }
    }
    
    // Stage 4 (Stride 8)
    for i in (0..32).step_by(16) {
        for j in 0..8 {
            let a = x[i+j];
            let b = x[i+j+8];
            x[i+j] = a + b;
            x[i+j+8] = a - b;
        }
    }
    
    // Stage 5 (Stride 16)
    for j in 0..16 {
        let a = x[j];
        let b = x[j+16];
        x[j] = a + b;
        x[j+16] = a - b;
    }
    // No normalization - caller handles it
}

/// In-place Fast Hadamard Transform for 32 elements (normalized).
/// Includes 1/sqrt(32) normalization.
#[inline(always)]
pub fn fht_32(x: &mut [f32; 32]) {
    fht_32_unnorm(x);
    // Normalize by 1/sqrt(32)
    let norm = 0.1767766952966369; // 1/sqrt(32)
    for v in x.iter_mut() {
        *v *= norm;
    }
}



use std::cell::RefCell;

thread_local! {
    // Reusable buffers to avoid allocations per call
    static SCRATCH_X_SCALED: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static SCRATCH_X_MIXED: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static SCRATCH_Y: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static SCRATCH_Y_MIXED: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

pub fn hadamard_linear(
    x: &[f32],           // [32 * in_per_part]
    bitmask: &[u8],      // Sparse ternary weight bitmask
    sign_bits: &[u8],    // Sign bits for non-zeros
    alpha: &[f32],       // [32 * in_per_part] input scaling
    beta: &[f32],        // [32 * out_per_part] output scaling (or [out_per_part] broadcast)
    in_per_part: usize,
    out_per_part: usize,
) -> Vec<f32> {
    // 1. Prepare buffers using scratchpads
    let in_size = 32 * in_per_part;
    let out_size = 32 * out_per_part;
    
    // We return a new Vec, but we use scratch for intermediates
    let mut y_result = vec![0.0f32; out_size];
    
    SCRATCH_X_SCALED.with(|s_xs| {
    SCRATCH_X_MIXED.with(|s_xm| {
    // Note: We use y_result directly as accumulator 'y' to avoid copying
    SCRATCH_Y_MIXED.with(|s_ym| {
        let mut x_scaled = s_xs.borrow_mut();
        if x_scaled.len() < in_size { x_scaled.resize(in_size, 0.0); }
        
        let mut x_mixed = s_xm.borrow_mut();
        if x_mixed.len() < in_size { x_mixed.resize(in_size, 0.0); }
        
        let mut y_mixed = s_ym.borrow_mut();
        if y_mixed.len() < out_size { y_mixed.resize(out_size, 0.0); }
        
        // Zero out reused buffers if needed (x_scaled/mixed are overwritten, so no need)
        // y_result is already zeroed
        
        // Step 1: Apply input scaling (α · x)
        for g in 0..32 {
            let base = g * in_per_part;
            for i in 0..in_per_part {
                let idx = base + i;
                x_scaled[idx] = x[idx] * alpha.get(idx).copied().unwrap_or(1.0);
            }
        }
        
        // FHT Step 1
        for i in 0..in_per_part {
            let mut group: [f32; 32] = [0.0; 32];
            for g in 0..32 {
                group[g] = x_scaled[g * in_per_part + i];
            }
            fht_32_unnorm(&mut group);
            for g in 0..32 {
                x_mixed[g * in_per_part + i] = group[g];
            }
        }
        
        // Step 2: Matrix multiply
        let mut bitmask_idx = 0;
        let mut bitmask_byte_val = if !bitmask.is_empty() { bitmask[0] } else { 0 };
        let mut bitmask_bit = 0;
        
        let mut sign_idx = 0;
        let mut sign_byte_val = if !sign_bits.is_empty() { sign_bits[0] } else { 0 };
        let mut sign_bit = 0;

        for g in 0..32 {
            let y_base = g * out_per_part;
            let x_base = g * in_per_part;
            
            // Iterate in weight storage order: [32, out_o, in_o]
            // But Python transposes to [32, in_o, out_o] before einsum
            // So we iterate o->i to match the [g, o, i] flattening order,
            // then for each (g,o,i), we add x[g,i] * w[g,o,i] to y[g,o].
            // This is WRONG! The einsum does x[g,i] * w_transposed[g,i,o] = x[g,i] * w[g,o,i]
            // Wait, let me re-check:
            // - Python stores w as [32, out_o, in_o]
            // - Python einsum uses w.transpose(-1,-2) = [32, in_o, out_o]
            // - einsum('btgi,gio->btgo', x, w_t) means: y[o] += x[i] * w_t[i,o] = x[i] * w[o,i]
            // So y[g,o] += x[g,i] * w[g,o,i] -- the original storage order!
            // That means our iteration order IS correct, but maybe the bitmask
            // is stored differently... Let me iterate to match exactly how compress.py flattens.
            
            for o in 0..out_per_part {
                let y_idx = y_base + o;
                let mut acc = 0.0;
                
                for i in 0..in_per_part {
                    // Check bitmask
                    let is_nonzero = (bitmask_byte_val & (1 << bitmask_bit)) != 0;
                    
                    bitmask_bit += 1;
                    if bitmask_bit == 8 {
                        bitmask_bit = 0;
                        bitmask_idx += 1;
                        // Safe reload
                        bitmask_byte_val = if bitmask_idx < bitmask.len() { bitmask[bitmask_idx] } else { 0 };
                    }
                    
                    if is_nonzero {
                        let is_positive = (sign_byte_val & (1 << sign_bit)) != 0;
                        let val = x_mixed[x_base + i];
                        
                        if is_positive {
                            acc += val;
                        } else {
                            acc -= val;
                        }
                        
                        sign_bit += 1;
                        if sign_bit == 8 {
                            sign_bit = 0;
                            sign_idx += 1;
                            // Safe reload
                            sign_byte_val = if sign_idx < sign_bits.len() { sign_bits[sign_idx] } else { 0 };
                        }
                    }
                }
                y_result[y_idx] += acc;
            }
        }

        
        // Step 3: FHT on output
        for o in 0..out_per_part {
            let mut group: [f32; 32] = [0.0; 32];
            for g in 0..32 {
                group[g] = y_result[g * out_per_part + o];
            }
            fht_32_unnorm(&mut group);
            for g in 0..32 {
                y_mixed[g * out_per_part + o] = group[g];
            }
        }
        
        // Step 4: Beta scaling + deferred FHT normalization (1/32 = 1/sqrt(32)^2)
        const FHT_NORM_SQ: f32 = 0.03125; // 1/32 = (1/sqrt(32))^2
        let broadcast_beta = beta.len() == out_per_part;
        for g in 0..32 {
            let base = g * out_per_part;
            for o in 0..out_per_part {
                let idx = base + o;
                let b = if broadcast_beta { beta[o] } else { beta.get(idx).copied().unwrap_or(1.0) };
                y_mixed[idx] *= b * FHT_NORM_SQ;
            }
        }
        
        // Copy back to result? No, y_mixed IS the result, but we need to return a Vec.
        // We can reuse y_result to store the final result if we want, but y_mixed is already there.
        // Wait, y_mixed is scratch. We must return a new Vec (the return type forces it).
        // Let's just return a copy of y_mixed.
        y_mixed[..out_size].to_vec()
    
    }) // end s_ym
    }) // end s_xm
    }) // end s_xs
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
