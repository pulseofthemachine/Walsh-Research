//! Cayley-Dickson Octonion Multiplication
//!
//! Implements the 8-way octonion multiplication for ternary weights.
//! y_i = sum_j (sign[i][j] * x_j @ W[idx[i][j]])

/// Sign table for Cayley-Dickson multiplication
pub const SIGN_TABLE: [[i8; 8]; 8] = [
    [1, -1, -1, -1, -1, -1, -1, -1],
    [1,  1,  1, -1,  1, -1, -1,  1],
    [1, -1,  1,  1,  1,  1, -1, -1],
    [1,  1, -1,  1,  1, -1,  1, -1],
    [1, -1, -1, -1,  1,  1,  1,  1],
    [1,  1, -1,  1, -1,  1, -1,  1],
    [1,  1,  1, -1, -1,  1,  1, -1],
    [1, -1,  1,  1, -1, -1,  1,  1],
];

/// Weight index table for Cayley-Dickson multiplication
pub const WEIGHT_IDX: [[usize; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 3, 2, 5, 4, 7, 6],
    [2, 3, 0, 1, 6, 7, 4, 5],
    [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [5, 4, 7, 6, 1, 0, 3, 2],
    [6, 7, 4, 5, 2, 3, 0, 1],
    [7, 6, 5, 4, 3, 2, 1, 0],
];

/// Ternary sparse matrix-vector multiplication
/// 
/// Computes y = x @ W^T where W is in CSR format with ternary values.
pub fn ternary_matvec(
    x: &[f32],      // Input vector [K]
    indices: &[u32],  // Non-zero indices (flattened)
    signs: &[i8],     // Non-zero signs (+1 or -1)
    shape: &[usize],  // [N, K]
    beta: &[f32],     // Scale factors [N]
) -> Vec<f32> {
    let n = shape[0];
    let k = shape[1];
    let mut y = vec![0.0f32; n];
    
    // Iterate through sparse entries
    // This is a simplified CSR-like iteration
    let mut idx = 0;
    for row in 0..n {
        let mut acc = 0.0f32;
        // Find entries for this row (simplified - assumes sorted by row)
        while idx < indices.len() {
            let flat_idx = indices[idx] as usize;
            let row_idx = flat_idx / k;
            let col_idx = flat_idx % k;
            
            if row_idx > row {
                break;
            }
            if row_idx == row {
                let sign = signs[idx] as f32;
                acc += sign * x[col_idx];
            }
            idx += 1;
        }
        y[row] = acc * beta.get(row).copied().unwrap_or(1.0);
    }
    
    y
}

/// Octonion linear layer forward pass
///
/// Takes 8 input parts, applies Cayley-Dickson multiplication, returns 8 output parts.
pub fn octonion_linear(
    x_parts: &[Vec<f32>; 8],  // 8 input vectors, each [K]
    weights: &[TernaryWeight; 8],  // 8 weight matrices
    betas: &[Vec<f32>; 8],    // 8 scale vectors
) -> [Vec<f32>; 8] {
    let n = weights[0].shape[1];  // Output dim per octonion part
    let mut y_parts: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0.0; n]);
    
    // For each output component
    for i in 0..8 {
        let mut acc = vec![0.0f32; n];
        
        // Sum over 8 input components with sign and weight selection
        for j in 0..8 {
            let sign = SIGN_TABLE[i][j] as f32;
            let w_idx = WEIGHT_IDX[i][j];
            
            // Multiply x_j @ W[w_idx]^T
            let contribution = ternary_matvec(
                &x_parts[j],
                &weights[w_idx].indices,
                &weights[w_idx].signs,
                &weights[w_idx].shape[1..],  // [N, K]
                &betas[w_idx],
            );
            
            // Accumulate with sign
            for k in 0..n {
                acc[k] += sign * contribution[k];
            }
        }
        
        y_parts[i] = acc;
    }
    
    y_parts
}

/// Ternary weight container (re-exported from model)
pub struct TernaryWeight {
    pub indices: Vec<u32>,
    pub signs: Vec<i8>,
    pub shape: Vec<usize>,  // [8, N, K] for octonion
}
