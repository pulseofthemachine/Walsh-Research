//! Causal Self-Attention
//!
//! Implements scaled dot-product attention with causal masking.

/// Scaled dot-product attention with causal mask
pub fn attention(
    q: &[f32],  // [seq_len, n_head, head_dim]
    k: &[f32],
    v: &[f32],
    n_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let seq_len = q.len() / (n_head * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];
    
    for h in 0..n_head {
        for i in 0..seq_len {
            // Compute attention scores for position i
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            
            for j in 0..=i {  // Causal: only attend to past + current
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = i * n_head * head_dim + h * head_dim + d;
                    let k_idx = j * n_head * head_dim + h * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                scores[j] = dot * scale;
            }
            
            // Softmax
            let max_score = scores.iter().take(i + 1).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_score).exp();
                sum += scores[j];
            }
            for j in 0..=i {
                scores[j] /= sum;
            }
            
            // Weighted sum of values
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    let v_idx = j * n_head * head_dim + h * head_dim + d;
                    acc += scores[j] * v[v_idx];
                }
                let out_idx = i * n_head * head_dim + h * head_dim + d;
                output[out_idx] = acc;
            }
        }
    }
    
    output
}

/// Cached attention: Q is just new token(s), K,V include full history
/// q: [new_len, n_head * head_dim] - Q for new positions
/// k_full, v_full: [full_len, n_head * head_dim] - full K,V including new
/// Returns: [new_len, n_head * head_dim]
pub fn attention_cached(
    q: &[f32],
    k_full: &[f32],
    v_full: &[f32],
    new_len: usize,
    full_len: usize,
    n_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_embd = n_head * head_dim;
    
    let mut output = vec![0.0f32; new_len * n_embd];
    
    // For each new position
    for new_pos in 0..new_len {
        let abs_pos = full_len - new_len + new_pos; // Absolute position in sequence
        
        for h in 0..n_head {
            // Compute attention scores: this Q attends to K[0..abs_pos+1]
            let mut scores = vec![0.0f32; abs_pos + 1];
            
            for j in 0..=abs_pos {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = new_pos * n_embd + h * head_dim + d;
                    let k_idx = j * n_embd + h * head_dim + d;
                    dot += q[q_idx] * k_full[k_idx];
                }
                scores[j] = dot * scale;
            }
            
            // Softmax
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in scores.iter_mut() {
                *score /= sum;
            }
            
            // Weighted sum of values
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=abs_pos {
                    let v_idx = j * n_embd + h * head_dim + d;
                    acc += scores[j] * v_full[v_idx];
                }
                let out_idx = new_pos * n_embd + h * head_dim + d;
                output[out_idx] = acc;
            }
        }
    }
    
    output
}

/// RMSNorm
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    
    // Compute RMS
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    
    // Normalize and scale
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi / rms * wi)
        .collect()
}

/// SiLU activation (Swish)
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
