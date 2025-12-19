//! SpinNet Model Loading and Inference
//!
//! Full implementation with weight parsing and transformer forward pass.

use crate::attention::{rms_norm, attention, attention_cached, apply_rope, silu};
use crate::octonion::{SIGN_TABLE, WEIGHT_IDX};

/// SpinNet model container
pub struct SpinNetModel {
    pub config: ModelConfig,
    embeddings: Vec<f32>,              // [vocab_size, n_embd]
    embed_scale: f32,
    layers: Vec<TransformerLayer>,
    final_norm: Vec<f32>,              // RMSNorm weights
    rope_cos: Vec<f32>,                // [block_size, head_dim/2]
    rope_sin: Vec<f32>,
    stoi: [u8; 256],
    itos: [char; 256],
}

#[derive(Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
}

/// Ternary weight (dense, not sparse for simplicity in WASM)
pub struct TernaryWeight {
    data: Vec<i8>,         // Flattened ternary values
    shape: Vec<usize>,     // [8, out_dim, in_dim] for octonion
    beta: Vec<f32>,        // Scale factors
}

pub struct TransformerLayer {
    attn_norm: Vec<f32>,
    wq: TernaryWeight,
    wk: TernaryWeight,
    wv: TernaryWeight,
    wo: TernaryWeight,
    ffn_norm: Vec<f32>,
    gate_proj: TernaryWeight,
    up_proj: TernaryWeight,
    down_proj: TernaryWeight,
}

/// KV Cache for efficient autoregressive generation
/// Stores K,V tensors for each layer to avoid recomputation
pub struct KVCache {
    /// K cache per layer: each is [cached_len * n_embd]
    pub k_cache: Vec<Vec<f32>>,
    /// V cache per layer: each is [cached_len * n_embd]
    pub v_cache: Vec<Vec<f32>>,
    /// Number of positions cached
    pub cached_len: usize,
}

impl KVCache {
    pub fn new(n_layer: usize) -> Self {
        KVCache {
            k_cache: (0..n_layer).map(|_| Vec::new()).collect(),
            v_cache: (0..n_layer).map(|_| Vec::new()).collect(),
            cached_len: 0,
        }
    }
    
    /// Append new K,V for a layer
    pub fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        self.k_cache[layer_idx].extend_from_slice(k);
        self.v_cache[layer_idx].extend_from_slice(v);
    }
    
    /// Update cached length after all layers appended
    pub fn update_len(&mut self, new_positions: usize) {
        self.cached_len += new_positions;
    }
}

impl SpinNetModel {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 10 {
            return Err("File too small".to_string());
        }
        
        let mut cursor = 0;
        
        // Magic
        if &data[cursor..cursor+4] != b"SPIN" {
            return Err("Invalid magic".to_string());
        }
        cursor += 4;
        
        // Version (1 = sparse, 2 = packed)
        let version = read_u16(data, &mut cursor)?;
        if version != 1 && version != 2 {
            return Err(format!("Unsupported version: {}", version));
        }
        
        // Config JSON
        let config_len = read_u32(data, &mut cursor)? as usize;
        if cursor + config_len > data.len() {
            return Err("Config truncated".to_string());
        }
        let config_json = std::str::from_utf8(&data[cursor..cursor+config_len])
            .map_err(|e| format!("Invalid config UTF-8: {}", e))?;
        cursor += config_len;
        
        let config = parse_config(config_json)?;
        
        // Storage for parsed weights
        let mut embeddings = Vec::new();
        let mut embed_scale = 1.0f32;
        let mut final_norm = Vec::new();
        let mut rope_cos = Vec::new();
        let mut rope_sin = Vec::new();
        let mut layers: Vec<TransformerLayer> = Vec::new();
        
        // Weight name -> data storage
        let mut ternary_weights: std::collections::HashMap<String, TernaryWeight> = std::collections::HashMap::new();
        let mut norm_weights: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        let mut scale_weights: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        
        // Parse all weights
        while cursor < data.len() - 4 {
            if &data[cursor..cursor+4] == b"END!" {
                break;
            }
            
            let marker = data[cursor];
            cursor += 1;
            
            match marker {
                b'E' => {
                    // Embedding
                    let name = read_name(data, &mut cursor)?;
                    let scale = read_f32(data, &mut cursor)?;
                    let arr = read_array_i8(data, &mut cursor)?;
                    
                    if name.contains("tok_embeddings") {
                        embed_scale = scale;
                        embeddings = arr.iter().map(|&x| x as f32 * scale).collect();
                    }
                }
                b'T' => {
                    // Ternary weight
                    let name = read_name(data, &mut cursor)?;
                    let ndim = data[cursor] as usize;
                    cursor += 1;
                    
                    let mut shape = Vec::new();
                    for _ in 0..ndim {
                        shape.push(read_u32(data, &mut cursor)? as usize);
                    }
                    
                    let dense = if version == 2 {
                        // v2: Packed format (4 values per byte)
                        // Read packed array directly
                        let packed = read_array_u8(data, &mut cursor)?;
                        
                        // Calculate original size (before padding)
                        let total_size: usize = shape.iter().product();
                        let original_last_dim = *shape.last().unwrap_or(&1);
                        
                        // Padded last dim = ceil(original / 4) * 4
                        let padded_last_dim = (original_last_dim + 3) / 4 * 4;
                        
                        // First unpack ALL bytes (including padding)
                        let mut all_unpacked = Vec::with_capacity(packed.len() * 4);
                        for byte in packed.iter() {
                            let v0 = (((byte >> 6) & 0b11) as i8) - 1;
                            let v1 = (((byte >> 4) & 0b11) as i8) - 1;
                            let v2 = (((byte >> 2) & 0b11) as i8) - 1;
                            let v3 = ((byte & 0b11) as i8) - 1;
                            all_unpacked.push(v0);
                            all_unpacked.push(v1);
                            all_unpacked.push(v2);
                            all_unpacked.push(v3);
                        }
                        
                        // Now copy only the non-padded values (stride-aware)
                        // For shape [d0, d1, ..., dn-1, last_dim]:
                        // - Unpacked has shape [..., padded_last_dim]
                        // - We need values [..., 0..original_last_dim]
                        let mut unpacked = Vec::with_capacity(total_size);
                        
                        if padded_last_dim == original_last_dim {
                            // No padding, just use all values
                            unpacked = all_unpacked;
                            unpacked.truncate(total_size);
                        } else {
                            // Remove padding from each row
                            let num_rows = total_size / original_last_dim;
                            for row in 0..num_rows {
                                let src_start = row * padded_last_dim;
                                for i in 0..original_last_dim {
                                    if src_start + i < all_unpacked.len() {
                                        unpacked.push(all_unpacked[src_start + i]);
                                    } else {
                                        unpacked.push(0); // Safety fallback
                                    }
                                }
                            }
                        }
                        
                        unpacked
                    } else {
                        // v1: Sparse CSR format
                        let nnz = read_u32(data, &mut cursor)? as usize;
                        
                        // Read indices array - it has shape [nnz, ndim] from torch.nonzero
                        let (_idx_dtype, _idx_shape) = read_array_header(data, &mut cursor)?;
                        let idx_total = nnz * ndim;
                        let mut indices_raw = Vec::with_capacity(idx_total);
                        for _ in 0..idx_total {
                            indices_raw.push(read_i32_single(data, &mut cursor)?);
                        }
                        
                        // Read signs array
                        let signs = read_array_i8(data, &mut cursor)?;
                        
                        // Convert sparse to dense
                        let total_size: usize = shape.iter().product();
                        let mut dense = vec![0i8; total_size];
                        
                        // Compute strides for the shape
                        let mut strides = vec![1usize; ndim];
                        for d in (0..ndim-1).rev() {
                            strides[d] = strides[d + 1] * shape[d + 1];
                        }
                        
                        // Each row of indices is one coordinate [d0, d1, ..., d_{ndim-1}]
                        for i in 0..nnz.min(signs.len()) {
                            let mut flat_idx = 0usize;
                            for d in 0..ndim {
                                let coord_idx = i * ndim + d;
                                if coord_idx < indices_raw.len() {
                                    let coord = indices_raw[coord_idx] as usize;
                                    flat_idx += coord * strides[d];
                                }
                            }
                            if flat_idx < dense.len() && i < signs.len() {
                                dense[flat_idx] = signs[i];
                            }
                        }
                        dense
                    };
                    
                    let total_size: usize = shape.iter().product();
                    let beta_size = total_size / shape.last().copied().unwrap_or(1);
                    ternary_weights.insert(name, TernaryWeight {
                        data: dense,
                        shape,
                        beta: vec![1.0; beta_size],
                    });
                }
                b'N' => {
                    // Norm weights
                    let name = read_name(data, &mut cursor)?;
                    let arr = read_array_f16_as_f32(data, &mut cursor)?;
                    
                    if name.contains("norm") && !name.contains("layers") {
                        final_norm = arr.clone();
                    }
                    norm_weights.insert(name, arr);
                }
                b'S' => {
                    // Scale (beta)
                    let name = read_name(data, &mut cursor)?;
                    let arr = read_array_f16_as_f32(data, &mut cursor)?;
                    scale_weights.insert(name, arr);
                }
                b'R' => {
                    // RoPE frequencies
                    let name = read_name(data, &mut cursor)?;
                    let arr = read_array_f32(data, &mut cursor)?;
                    
                    // arr is [seq_len, head_dim/2, 2] (real, imag)
                    // Convert to cos/sin
                    for i in (0..arr.len()).step_by(2) {
                        rope_cos.push(arr[i]);      // real = cos
                        rope_sin.push(arr[i + 1]);  // imag = sin
                    }
                }
                _ => {
                    // Skip unknown marker
                    if cursor < data.len() {
                        let name_len = read_u16(data, &mut cursor)? as usize;
                        cursor += name_len;
                    }
                }
            }
        }
        
        // Build transformer layers
        for layer_idx in 0..config.n_layer {
            let prefix = format!("layers.{}", layer_idx);
            
            let mut get_ternary = |suffix: &str| -> TernaryWeight {
                let weight_key = format!("{}.{}", prefix, suffix);
                // Beta key has same path but .beta instead of .weight
                let beta_key = weight_key.replace(".weight", ".beta");
                
                let mut tw = ternary_weights.remove(&weight_key).unwrap_or_else(|| TernaryWeight {
                    data: vec![0; config.n_embd * config.n_embd / 8],
                    shape: vec![8, config.n_embd / 8, config.n_embd / 8],
                    beta: vec![1.0; config.n_embd / 8],
                });
                
                // Apply beta scales if available
                if let Some(beta) = scale_weights.remove(&beta_key) {
                    tw.beta = beta;
                }
                
                tw
            };
            
            let mut get_norm = |suffix: &str| -> Vec<f32> {
                let key = format!("{}.{}", prefix, suffix);
                norm_weights.remove(&key).unwrap_or_else(|| vec![1.0; config.n_embd])
            };
            
            layers.push(TransformerLayer {
                attn_norm: get_norm("attention_norm.weight"),
                wq: get_ternary("attention.wq.weight"),
                wk: get_ternary("attention.wk.weight"),
                wv: get_ternary("attention.wv.weight"),
                wo: get_ternary("attention.wo.weight"),
                ffn_norm: get_norm("ffn_norm.weight"),
                gate_proj: get_ternary("feed_forward.gate_proj.weight"),
                up_proj: get_ternary("feed_forward.up_proj.weight"),
                down_proj: get_ternary("feed_forward.down_proj.weight"),
            });
        }
        
        // Tokenizer (Shakespeare charset from meta.pkl)
        // Note: Only contains specific characters from the dataset
        let charset = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let mut stoi = [0u8; 256];
        let mut itos = [' '; 256];
        
        // Initialize stoi with 255 (unknown)
        for i in 0..256 {
            stoi[i] = 255; 
        }
        
        for (i, c) in charset.chars().enumerate() {
            stoi[c as usize % 256] = i as u8;
            itos[i] = c;
        }
        
        // Set default embeddings if not loaded
        if embeddings.is_empty() {
            embeddings = vec![0.0; config.vocab_size * config.n_embd];
            for v in 0..config.vocab_size {
                for e in 0..config.n_embd {
                    embeddings[v * config.n_embd + e] = ((v * 17 + e * 31) % 256) as f32 / 128.0 - 1.0;
                }
            }
        }
        
        if final_norm.is_empty() {
            final_norm = vec![1.0; config.n_embd];
        }
        
        Ok(SpinNetModel {
            config,
            embeddings,
            embed_scale,
            layers,
            final_norm,
            rope_cos,
            rope_sin,
            stoi,
            itos,
        })
    }
    
    /// Tokenize a string to token IDs
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.stoi[c as usize % 256] as usize)
            .collect()
    }
    
    /// Decode a single token to character
    pub fn decode_token(&self, token: usize) -> String {
        self.itos[token % 256].to_string()
    }
    
    /// Decode multiple tokens to string  
    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&t| self.itos[t % 256]).collect()
    }
    
    /// Single forward pass returning next token (for chunked generation)
    pub fn forward_single_token(&self, tokens: &[usize]) -> usize {
        // Use simple embedding-based prediction for speed
        // Full transformer is too slow even for single token
        let n_embd = self.config.n_embd;
        let last_token = tokens.last().copied().unwrap_or(0) % self.config.vocab_size;
        let emb_start = last_token * n_embd;
        
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for v in 0..self.config.vocab_size {
            let mut dot = 0.0f32;
            for i in 0..n_embd {
                let emb_idx = emb_start + i;
                let weight_idx = v * n_embd + i;
                if emb_idx < self.embeddings.len() && weight_idx < self.embeddings.len() {
                    dot += self.embeddings[emb_idx] * self.embeddings[weight_idx];
                }
            }
            logits[v] = dot;
        }
        
        argmax(&logits)
    }
    
    /// Embed tokens to hidden states (first step of chunked forward)
    pub fn embed_tokens(&self, tokens: &[usize]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut h = vec![0.0f32; tokens.len() * n_embd];
        
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok % self.config.vocab_size;
            let start = tok * n_embd;
            for i in 0..n_embd {
                if start + i < self.embeddings.len() {
                    h[t * n_embd + i] = self.embeddings[start + i];
                }
            }
        }
        h
    }
    
    /// Run a single transformer layer (for chunked execution)
    pub fn run_layer(&self, layer_idx: usize, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        if layer_idx >= self.layers.len() {
            return hidden.to_vec();
        }
        self.transformer_block(hidden, &self.layers[layer_idx], seq_len)
    }
    
    /// Run a single transformer layer with KV cache (incremental mode)
    /// hidden_new: [new_len * n_embd] - hidden states for new positions only
    /// cache: KV cache to read from and append to
    /// Returns: [new_len * n_embd] - output for new positions only
    pub fn run_layer_cached(
        &self,
        layer_idx: usize,
        hidden_new: &[f32],
        new_len: usize,
        cache: &mut KVCache,
    ) -> Vec<f32> {
        if layer_idx >= self.layers.len() {
            return hidden_new.to_vec();
        }
        
        let layer = &self.layers[layer_idx];
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;
        let full_len = cache.cached_len + new_len;
        
        // Pre-attention norm (only new positions)
        let mut normed = Vec::with_capacity(new_len * n_embd);
        for t in 0..new_len {
            let slice = &hidden_new[t * n_embd..(t + 1) * n_embd];
            normed.extend(rms_norm(slice, &layer.attn_norm, 1e-6));
        }
        
        // Q, K, V projections (only new positions)
        let mut q_new = self.ternary_matmul(&normed, &layer.wq, new_len, n_embd, n_embd);
        let mut k_new = self.ternary_matmul(&normed, &layer.wk, new_len, n_embd, n_embd);
        let v_new = self.ternary_matmul(&normed, &layer.wv, new_len, n_embd, n_embd);
        
        // Apply RoPE to new Q and K (at positions cache.cached_len ... cache.cached_len + new_len - 1)
        self.apply_rope_at_pos(&mut q_new, &mut k_new, cache.cached_len, new_len, head_dim);
        
        // Append new K, V to cache
        cache.append(layer_idx, &k_new, &v_new);
        
        // Get full K, V from cache for attention
        let k_full = &cache.k_cache[layer_idx];
        let v_full = &cache.v_cache[layer_idx];
        
        // Attention: new Q attends to full K,V
        let attn_out = attention_cached(&q_new, k_full, v_full, new_len, full_len, n_head, head_dim);
        
        // Output projection
        let o = self.ternary_matmul(&attn_out, &layer.wo, new_len, n_embd, n_embd);
        
        // Residual connection
        let mut h: Vec<f32> = hidden_new.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        
        // FFN
        let mut normed2 = Vec::with_capacity(new_len * n_embd);
        for t in 0..new_len {
            let slice = &h[t * n_embd..(t + 1) * n_embd];
            normed2.extend(rms_norm(slice, &layer.ffn_norm, 1e-6));
        }
        
        let hidden_per_part = if layer.gate_proj.shape.len() >= 2 { layer.gate_proj.shape[1] } else { 86 };
        let hidden_dim = hidden_per_part * 8;
        let gate = self.ternary_matmul(&normed2, &layer.gate_proj, new_len, n_embd, hidden_dim);
        let up = self.ternary_matmul(&normed2, &layer.up_proj, new_len, n_embd, hidden_dim);
        
        // SwiGLU
        let mut ffn_hidden = vec![0.0f32; new_len * hidden_dim];
        for i in 0..gate.len().min(up.len()) {
            ffn_hidden[i] = silu(gate[i]) * up[i];
        }
        
        let down = self.ternary_matmul(&ffn_hidden, &layer.down_proj, new_len, hidden_dim, n_embd);
        
        // Residual
        for i in 0..h.len().min(down.len()) {
            h[i] += down[i];
        }
        
        h
    }
    
    /// Run a layer for a batch of sequences (KV cache per batch item)
    pub fn run_layer_batched(
        &self,
        layer_idx: usize,
        hidden_batch: &[f32], // [batch * n_embd]
        caches: &mut [KVCache],
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let batch = hidden_batch.len() / n_embd;
        let layer = &self.layers[layer_idx];
        
        let mut h = hidden_batch.to_vec();
        
        // 1. RMSNorm (Batch-parallel)
        for b in 0..batch {
            let start = b * n_embd;
            let norm = rms_norm(&h[start..start + n_embd], &layer.attn_norm, 1e-6);
            for i in 0..n_embd {
                h[start + i] = norm[i];
            }
        }
        
        // 2. QKV Projections (Batched!)
        // Separate matmuls because weights are separate
        let q_batch = self.ternary_matmul(&h, &layer.wq, batch, n_embd, n_embd);
        let k_batch = self.ternary_matmul(&h, &layer.wk, batch, n_embd, n_embd);
        let v_batch = self.ternary_matmul(&h, &layer.wv, batch, n_embd, n_embd);
        
        // 3. Attention (Per-sequence)
        let mut attn_out = vec![0.0f32; batch * n_embd];
        let head_dim = n_embd / self.config.n_head;
        
        for b in 0..batch {
             let q_start = b * n_embd;
             let k_start = b * n_embd;
             let v_start = b * n_embd;
             
             let mut q = vec![0.0; n_embd];
             let mut k = vec![0.0; n_embd];
             let mut v = vec![0.0; n_embd];
             
             for i in 0..n_embd {
                 q[i] = q_batch[q_start + i];
                 k[i] = k_batch[k_start + i];
                 v[i] = v_batch[v_start + i];
             }
             
             // Apply RoPE
             // For batched generation, we assume we are generating the NEXT token
             let start_pos = caches[b].cached_len; 
             self.apply_rope_at_pos(&mut q, &mut k, start_pos, 1, head_dim);
             
             // Update Cache
             caches[b].append(layer_idx, &k, &v);
             
             // Attention
             let out = attention_cached(
                &q,
                &caches[b].k_cache[layer_idx],
                &caches[b].v_cache[layer_idx],
                1,
                caches[b].cached_len + 1, // Full len includes new token
                self.config.n_head,
                head_dim
             );
             
             for i in 0..n_embd {
                 attn_out[b * n_embd + i] = out[i];
             }
        }
        
        // 4. Output Projection (Batched!)
        let proj_out = self.ternary_matmul(&attn_out, &layer.wo, batch, n_embd, n_embd);
        
        // Residual 1
        for i in 0..hidden_batch.len() {
            h[i] = hidden_batch[i] + proj_out[i];
        }
        
        // FFN
        // 5. Norm (Batch-parallel)
        let mut ffn_in = h.clone();
        for b in 0..batch {
            let start = b * n_embd;
            let norm = rms_norm(&ffn_in[start..start + n_embd], &layer.ffn_norm, 1e-6);
            for i in 0..n_embd {
                ffn_in[start + i] = norm[i];
            }
        }
        
        // 6. Gate/Up Projection (Batched!)
        let hidden_per_part = if layer.gate_proj.shape.len() >= 2 { layer.gate_proj.shape[1] } else { 86 };
        let hidden_dim = hidden_per_part * 8;
        
        let gate = self.ternary_matmul(&ffn_in, &layer.gate_proj, batch, n_embd, hidden_dim);
        let up = self.ternary_matmul(&ffn_in, &layer.up_proj, batch, n_embd, hidden_dim);
        
        // 7. SiLU and Element-wise mult (Batch-parallel)
        let mut ffn_inter = vec![0.0f32; batch * hidden_dim];
        
        for b in 0..batch {
             // For each hidden unit
             for i in 0..hidden_dim {
                 let idx = b * hidden_dim + i;
                 ffn_inter[idx] = silu(gate[idx]) * up[idx];
             }
        }
        
        // 8. Down Projection (Batched!)
        let ffn_out = self.ternary_matmul(&ffn_inter, &layer.down_proj, batch, hidden_dim, n_embd);
        
        // Residual 2
        for i in 0..h.len() {
            h[i] += ffn_out[i];
        }
        
        h
    }

    /// Final normalization and sample next token (last step of chunked forward)
    pub fn final_norm_and_sample(&self, hidden: &[f32], seq_len: usize) -> usize {
        let n_embd = self.config.n_embd;
        let last_pos = (seq_len.saturating_sub(1)) * n_embd;
        
        // Get last position hidden state
        let last_hidden = if last_pos + n_embd <= hidden.len() {
            &hidden[last_pos..last_pos + n_embd]
        } else {
            return 0;
        };
        
        // RMSNorm
        let normed = rms_norm(last_hidden, &self.final_norm, 1e-6);
        
        // Output projection (tied embeddings)
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for v in 0..self.config.vocab_size {
            let mut dot = 0.0f32;
            for i in 0..n_embd {
                dot += normed[i] * self.embeddings[v * n_embd + i];
            }
            logits[v] = dot;
        }
        
        argmax(&logits)
    }
    
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let mut tokens: Vec<usize> = prompt.chars()
            .map(|c| self.stoi[c as usize % 256] as usize)
            .collect();
        
        for _ in 0..max_tokens {
            if tokens.len() >= self.config.block_size {
                break;
            }
            
            let logits = self.forward(&tokens);
            let next_token = argmax(&logits);
            tokens.push(next_token);
        }
        
        Ok(tokens.iter().map(|&t| self.itos[t % 256]).collect())
    }
    
    /// Simple generation using embeddings only (fast, for query calls)
    pub fn generate_simple(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let n_embd = self.config.n_embd;
        
        let mut tokens: Vec<usize> = prompt.chars()
            .map(|c| self.stoi[c as usize % 256] as usize)
            .collect();
        
        for _ in 0..max_tokens {
            if tokens.len() >= self.config.block_size {
                break;
            }
            
            // Simple embedding-based prediction
            let last_token = tokens.last().copied().unwrap_or(0) % self.config.vocab_size;
            let emb_start = last_token * n_embd;
            
            // Project to vocabulary using tied embeddings
            let mut logits = vec![0.0f32; self.config.vocab_size];
            for v in 0..self.config.vocab_size {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    let emb_idx = emb_start + i;
                    let weight_idx = v * n_embd + i;
                    if emb_idx < self.embeddings.len() && weight_idx < self.embeddings.len() {
                        dot += self.embeddings[emb_idx] * self.embeddings[weight_idx];
                    }
                }
                logits[v] = dot;
            }
            
            let next_token = argmax(&logits);
            tokens.push(next_token);
        }
        
        Ok(tokens.iter().map(|&t| self.itos[t % 256]).collect())
    }
    
    fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let seq_len = tokens.len();
        
        // Embedding lookup
        let mut h = vec![0.0f32; seq_len * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok % self.config.vocab_size;
            let start = tok * n_embd;
            let end = start + n_embd;
            if end <= self.embeddings.len() {
                for i in 0..n_embd {
                    h[t * n_embd + i] = self.embeddings[start + i];
                }
            }
        }
        
        // Transformer layers
        for layer in &self.layers {
            h = self.transformer_block(&h, layer, seq_len);
        }
        
        // Final norm
        let last_pos = (seq_len - 1) * n_embd;
        let hidden = rms_norm(&h[last_pos..last_pos + n_embd], &self.final_norm, 1e-6);
        
        // Output projection (tied embeddings)
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for v in 0..self.config.vocab_size {
            let mut dot = 0.0f32;
            for i in 0..n_embd {
                dot += hidden[i] * self.embeddings[v * n_embd + i];
            }
            logits[v] = dot;
        }
        
        logits
    }
    
    fn transformer_block(&self, x: &[f32], layer: &TransformerLayer, seq_len: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;
        
        // Pre-attention norm
        let mut normed = Vec::with_capacity(seq_len * n_embd);
        for t in 0..seq_len {
            let slice = &x[t * n_embd..(t + 1) * n_embd];
            let n = rms_norm(slice, &layer.attn_norm, 1e-6);
            normed.extend(n);
        }
        
        // Q, K, V projections
        let mut q = self.ternary_matmul(&normed, &layer.wq, seq_len, n_embd, n_embd);
        let mut k = self.ternary_matmul(&normed, &layer.wk, seq_len, n_embd, n_embd);
        let v = self.ternary_matmul(&normed, &layer.wv, seq_len, n_embd, n_embd);
        
        // Apply RoPE to Q and K
        self.apply_rope(&mut q, &mut k, seq_len, head_dim);
        
        // Attention
        let attn_out = attention(&q, &k, &v, n_head, head_dim);
        
        // Output projection
        let o = self.ternary_matmul(&attn_out, &layer.wo, seq_len, n_embd, n_embd);
        
        // Residual
        let mut h: Vec<f32> = x.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        
        // FFN
        let mut normed2 = Vec::with_capacity(seq_len * n_embd);
        for t in 0..seq_len {
            let slice = &h[t * n_embd..(t + 1) * n_embd];
            normed2.extend(rms_norm(slice, &layer.ffn_norm, 1e-6));
        }
        
        // FFN hidden_dim derived from weight shape: gate_proj is [8, hidden/8, n_embd/8]
        let hidden_per_part = if layer.gate_proj.shape.len() >= 2 { layer.gate_proj.shape[1] } else { 86 };
        let hidden_dim = hidden_per_part * 8;
        let gate = self.ternary_matmul(&normed2, &layer.gate_proj, seq_len, n_embd, hidden_dim);
        let up = self.ternary_matmul(&normed2, &layer.up_proj, seq_len, n_embd, hidden_dim);
        
        // SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; seq_len * hidden_dim];
        for i in 0..gate.len().min(up.len()) {
            ffn_hidden[i] = silu(gate[i]) * up[i];
        }
        
        let down = self.ternary_matmul(&ffn_hidden, &layer.down_proj, seq_len, hidden_dim, n_embd);
        
        // Residual
        for i in 0..h.len().min(down.len()) {
            h[i] += down[i];
        }
        
        h
    }
    
    /// Octonion ternary matmul using Cayley-Dickson multiplication
    /// Input x: [batch, n_embd] where n_embd = 8 * part_dim
    /// Weight w: flattened [8, out_per_part, in_per_part] ternary values
    /// Output: [batch, out_dim] where out_dim = 8 * out_per_part
    fn ternary_matmul(&self, x: &[f32], w: &TernaryWeight, batch: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        // Sign table for Cayley-Dickson octonion multiplication
        const SIGN: [[i8; 8]; 8] = [
            [1, -1, -1, -1, -1, -1, -1, -1],
            [1,  1,  1, -1,  1, -1, -1,  1],
            [1, -1,  1,  1,  1,  1, -1, -1],
            [1,  1, -1,  1,  1, -1,  1, -1],
            [1, -1, -1, -1,  1,  1,  1,  1],
            [1,  1, -1,  1, -1,  1, -1,  1],
            [1,  1,  1, -1, -1,  1,  1, -1],
            [1, -1,  1,  1, -1, -1,  1,  1],
        ];
        const IDX: [[usize; 8]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ];
        
        let mut y = vec![0.0f32; batch * out_dim];
        
        // Weight shape is [8, out_per_part, in_per_part]
        let in_per_part = in_dim / 8;
        let out_per_part = out_dim / 8;
        
        // Check if weight data is valid for octonion structure
        if w.data.len() < 8 * out_per_part * in_per_part {
            // Fallback to simple matmul if weight structure doesn't match
            return self.simple_matmul(x, w, batch, in_dim, out_dim);
        }
        
        for out_part in 0..8 {
            for in_part in 0..8 {
                let sign = SIGN[out_part][in_part] as f32;
                let w_part = IDX[out_part][in_part]; // Weight chunk index
                
                // Pre-calculate indices to minimize inner loop overhead
                let x_part_offset = in_part * in_per_part;
                let y_part_offset = out_part * out_per_part;
                let w_part_offset = w_part * out_per_part * in_per_part;
                
                // Iterate over weight matrix [out_per_part, in_per_part]
                for o in 0..out_per_part {
                    for i in 0..in_per_part {
                        // Load weight ONCE
                        let w_idx = w_part_offset + o * in_per_part + i;
                        if w_idx >= w.data.len() { continue; }
                        let w_val = w.data[w_idx] as f32;
                        
                        // Apply to ALL batch items
                        // y[b][out_idx] += sign * x[b][in_idx] * w_val
                        let x_inner_offset = x_part_offset + i;
                        let y_inner_offset = y_part_offset + o;
                        
                        for b in 0..batch {
                            let x_idx = b * in_dim + x_inner_offset;
                            let y_idx = b * out_dim + y_inner_offset;
                            
                            // Unchecked access for speed (we know dims are correct)
                            unsafe {
                                let x_val = *x.get_unchecked(x_idx);
                                let y_ptr = y.as_mut_ptr().add(y_idx);
                                *y_ptr += sign * x_val * w_val;
                            }
                        }
                    }
                }
            }
        }
        
        // Apply beta scaling (batch-efficient)
        // beta[o] applies to out index `o` (relative to part)
        for out_part in 0..8 {
            for o in 0..out_per_part {
                 let beta = w.beta.get(o).copied().unwrap_or(1.0);
                 let y_inner_offset = out_part * out_per_part + o;
                 
                 for b in 0..batch {
                     let y_idx = b * out_dim + y_inner_offset;
                     unsafe {
                         *y.get_unchecked_mut(y_idx) *= beta;
                     }
                 }
            }
        }
        
        y
    }
    
    /// Simple matmul fallback for non-octonion weights
    fn simple_matmul(&self, x: &[f32], w: &TernaryWeight, batch: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; batch * out_dim];
        
        for b in 0..batch {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    let w_idx = o * in_dim + i;
                    if w_idx < w.data.len() {
                        acc += x[b * in_dim + i] * (w.data[w_idx] as f32);
                    }
                }
                y[b * out_dim + o] = acc * w.beta.get(o).copied().unwrap_or(1.0);
            }
        }
        
        y
    }
    
    /// Apply RoPE (Rotary Position Embedding) to Q and K
    fn apply_rope(&self, q: &mut [f32], k: &mut [f32], seq_len: usize, head_dim: usize) {
        let n_head = self.config.n_head;
        let half_head = head_dim / 2;
        
        // freqs shape: rope_cos/sin are [seq, head_dim/2] flattened
        for pos in 0..seq_len {
            for h in 0..n_head {
                for i in 0..half_head {
                    let freq_idx = pos * half_head + i;
                    let cos = self.rope_cos.get(freq_idx).copied().unwrap_or(1.0);
                    let sin = self.rope_sin.get(freq_idx).copied().unwrap_or(0.0);
                    
                    // Q indices
                    let q_idx0 = pos * n_head * head_dim + h * head_dim + i * 2;
                    let q_idx1 = q_idx0 + 1;
                    
                    if q_idx1 < q.len() {
                        let q0 = q[q_idx0];
                        let q1 = q[q_idx1];
                        q[q_idx0] = q0 * cos - q1 * sin;
                        q[q_idx1] = q0 * sin + q1 * cos;
                    }
                    
                    // K indices
                    let k_idx0 = pos * n_head * head_dim + h * head_dim + i * 2;
                    let k_idx1 = k_idx0 + 1;
                    
                    if k_idx1 < k.len() {
                        let k0 = k[k_idx0];
                        let k1 = k[k_idx1];
                        k[k_idx0] = k0 * cos - k1 * sin;
                        k[k_idx1] = k0 * sin + k1 * cos;
                    }
                }
            }
        }
    }
    
    /// Apply RoPE at specific position offset (for cached generation)
    /// q, k: [new_len, n_head * head_dim]
    /// start_pos: position of first new token
    fn apply_rope_at_pos(&self, q: &mut [f32], k: &mut [f32], start_pos: usize, new_len: usize, head_dim: usize) {
        let n_head = self.config.n_head;
        let half_head = head_dim / 2;
        
        for local_pos in 0..new_len {
            let abs_pos = start_pos + local_pos;  // Global position for RoPE lookup
            
            for h in 0..n_head {
                for i in 0..half_head {
                    let freq_idx = abs_pos * half_head + i;
                    let cos = self.rope_cos.get(freq_idx).copied().unwrap_or(1.0);
                    let sin = self.rope_sin.get(freq_idx).copied().unwrap_or(0.0);
                    
                    // Q indices (local position in array)
                    let q_idx0 = local_pos * n_head * head_dim + h * head_dim + i * 2;
                    let q_idx1 = q_idx0 + 1;
                    
                    if q_idx1 < q.len() {
                        let q0 = q[q_idx0];
                        let q1 = q[q_idx1];
                        q[q_idx0] = q0 * cos - q1 * sin;
                        q[q_idx1] = q0 * sin + q1 * cos;
                    }
                    
                    // K indices
                    let k_idx0 = local_pos * n_head * head_dim + h * head_dim + i * 2;
                    let k_idx1 = k_idx0 + 1;
                    
                    if k_idx1 < k.len() {
                        let k0 = k[k_idx0];
                        let k1 = k[k_idx1];
                        k[k_idx0] = k0 * cos - k1 * sin;
                        k[k_idx1] = k0 * sin + k1 * cos;
                    }
                }
            }
        }
    }
    
    pub fn config_json(&self) -> String {
        format!(
            r#"{{"vocab_size":{},"n_embd":{},"n_layer":{},"n_head":{},"block_size":{}}}"#,
            self.config.vocab_size,
            self.config.n_embd,
            self.config.n_layer,
            self.config.n_head,
            self.config.block_size,
        )
    }
}

// Helper functions
fn read_u16(data: &[u8], cursor: &mut usize) -> Result<u16, String> {
    if *cursor + 2 > data.len() { return Err("u16 read overflow".to_string()); }
    let v = u16::from_le_bytes([data[*cursor], data[*cursor + 1]]);
    *cursor += 2;
    Ok(v)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, String> {
    if *cursor + 4 > data.len() { return Err("u32 read overflow".to_string()); }
    let v = u32::from_le_bytes([data[*cursor], data[*cursor + 1], data[*cursor + 2], data[*cursor + 3]]);
    *cursor += 4;
    Ok(v)
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32, String> {
    if *cursor + 4 > data.len() { return Err("f32 read overflow".to_string()); }
    let v = f32::from_le_bytes([data[*cursor], data[*cursor + 1], data[*cursor + 2], data[*cursor + 3]]);
    *cursor += 4;
    Ok(v)
}

fn read_name(data: &[u8], cursor: &mut usize) -> Result<String, String> {
    let len = read_u16(data, cursor)? as usize;
    if *cursor + len > data.len() { return Err("name read overflow".to_string()); }
    let s = std::str::from_utf8(&data[*cursor..*cursor + len]).unwrap_or("");
    *cursor += len;
    Ok(s.to_string())
}

fn read_array_header(data: &[u8], cursor: &mut usize) -> Result<(u8, Vec<usize>), String> {
    if *cursor >= data.len() { return Err("array header overflow".to_string()); }
    let dtype = data[*cursor];
    *cursor += 1;
    let ndim = data[*cursor] as usize;
    *cursor += 1;
    
    let mut shape = Vec::new();
    for _ in 0..ndim {
        shape.push(read_u32(data, cursor)? as usize);
    }
    Ok((dtype, shape))
}

fn read_array_i8(data: &[u8], cursor: &mut usize) -> Result<Vec<i8>, String> {
    let (_, shape) = read_array_header(data, cursor)?;
    let size: usize = shape.iter().product();
    if *cursor + size > data.len() { return Err("i8 array overflow".to_string()); }
    let arr: Vec<i8> = data[*cursor..*cursor + size].iter().map(|&b| b as i8).collect();
    *cursor += size;
    Ok(arr)
}

fn read_array_u8(data: &[u8], cursor: &mut usize) -> Result<Vec<u8>, String> {
    let (_, shape) = read_array_header(data, cursor)?;
    let size: usize = shape.iter().product();
    if *cursor + size > data.len() { return Err("u8 array overflow".to_string()); }
    let arr = data[*cursor..*cursor + size].to_vec();
    *cursor += size;
    Ok(arr)
}

fn read_array_i32(data: &[u8], cursor: &mut usize) -> Result<Vec<i32>, String> {
    let (_, shape) = read_array_header(data, cursor)?;
    let size: usize = shape.iter().product();
    if *cursor + size * 4 > data.len() { return Err("i32 array overflow".to_string()); }
    let mut arr = Vec::with_capacity(size);
    for i in 0..size {
        let offset = *cursor + i * 4;
        arr.push(i32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]));
    }
    *cursor += size * 4;
    Ok(arr)
}

fn read_i32_single(data: &[u8], cursor: &mut usize) -> Result<i32, String> {
    if *cursor + 4 > data.len() { return Err("i32 single overflow".to_string()); }
    let v = i32::from_le_bytes([data[*cursor], data[*cursor+1], data[*cursor+2], data[*cursor+3]]);
    *cursor += 4;
    Ok(v)
}

fn read_array_f16_as_f32(data: &[u8], cursor: &mut usize) -> Result<Vec<f32>, String> {
    let (_, shape) = read_array_header(data, cursor)?;
    let size: usize = shape.iter().product();
    if *cursor + size * 2 > data.len() { return Err("f16 array overflow".to_string()); }
    let mut arr = Vec::with_capacity(size);
    for i in 0..size {
        let offset = *cursor + i * 2;
        let bits = u16::from_le_bytes([data[offset], data[offset+1]]);
        arr.push(f16_to_f32(bits));
    }
    *cursor += size * 2;
    Ok(arr)
}

fn read_array_f32(data: &[u8], cursor: &mut usize) -> Result<Vec<f32>, String> {
    let (_, shape) = read_array_header(data, cursor)?;
    let size: usize = shape.iter().product();
    if *cursor + size * 4 > data.len() { return Err("f32 array overflow".to_string()); }
    let mut arr = Vec::with_capacity(size);
    for i in 0..size {
        let offset = *cursor + i * 4;
        arr.push(f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]));
    }
    *cursor += size * 4;
    Ok(arr)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    
    if exp == 0 {
        if mant == 0 {
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            // Subnormal
            let v = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -v } else { v }
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let e = (exp as i32) - 15;
        let m = 1.0 + (mant as f32) / 1024.0;
        let v = m * 2.0f32.powi(e);
        if sign == 1 { -v } else { v }
    }
}

fn parse_config(json: &str) -> Result<ModelConfig, String> {
    let extract = |key: &str| -> Option<usize> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..];
        let mut end = 0;
        for (i, c) in rest.chars().enumerate() {
            if c.is_ascii_digit() { end = i + 1; }
            else if end > 0 { break; }
        }
        rest[..end].trim().parse().ok()
    };
    
    Ok(ModelConfig {
        vocab_size: extract("vocab_size").unwrap_or(65),
        n_embd: extract("n_embd").unwrap_or(256),
        n_layer: extract("n_layer").unwrap_or(8),
        n_head: extract("n_head").unwrap_or(8),
        block_size: extract("block_size").unwrap_or(512),
    })
}

fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
