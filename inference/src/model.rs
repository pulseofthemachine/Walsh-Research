//! Walsh Model Loading and Inference
//!
//! Full implementation with weight parsing and transformer forward pass.
//! Supports both 8D Octonion and 32D Hadamard algebra modes.

use crate::attention::{rms_norm, attention, attention_cached, silu};
use crate::tokenizer::Tokenizer;

/// Hash embedding for ultra-compressed vocab representation
/// Uses 3 embedding tables with learned importance weights
pub struct HashEmbedding {
    tables: Vec<Vec<f32>>,      // 3 tables: [bucket_size, n_embd] each
    table_scales: Vec<f32>,     // INT8 dequantization scales per table
    importance: Vec<f32>,       // [vocab_size, 3] learned weights per token/table
    indices: Vec<i32>,          // [vocab_size, 3] precomputed hash indices
    bucket_size: usize,
    num_tables: usize,
    n_embd: usize,
}

/// Walsh model container
pub struct WalshModel {
    pub config: ModelConfig,
    // Standard embeddings (for octonion/non-hash models)
    embeddings: Vec<f32>,              // [vocab_size, n_embd]
    embed_scale: f32,
    // Hash embeddings (for hadamard + hash_embeddings=true)
    hash_embedding: Option<HashEmbedding>,
    layers: Vec<TransformerLayer>,
    final_norm: Vec<f32>,              // RMSNorm weights
    rope_cos: Vec<f32>,                // [block_size, head_dim/2]
    rope_sin: Vec<f32>,
    tokenizer: Tokenizer,              // Auto-detecting tokenizer
}

#[derive(Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub algebra: String,           // "octonion" (8D) or "hadamard" (32D)
    pub hash_embeddings: bool,     // Use hash embedding tables
}

/// Bitmask ternary weight - smallest format with sparse iteration
/// bitmask: 1 bit per position (1 = non-zero)
/// sign_bits: 1 bit per non-zero (0 = -1, 1 = +1)
pub struct BitmaskWeight {
    bitmask: Vec<u8>,      // 1 bit per position marking non-zero
    sign_bits: Vec<u8>,    // 1 bit per non-zero, packed (0=-1, 1=+1)
    num_nonzero: usize,    // Total count of non-zeros
    shape: (usize, usize), // (out_dim, in_dim) logical dimensions
    beta: Vec<f32>,        // Per-output-part scaling
}

pub struct TransformerLayer {
    attn_norm: Vec<f32>,
    wq: BitmaskWeight,
    wk: BitmaskWeight,
    wv: BitmaskWeight,
    wo: BitmaskWeight,
    ffn_norm: Vec<f32>,
    gate_proj: BitmaskWeight,
    up_proj: BitmaskWeight,
    down_proj: BitmaskWeight,
    // Octonion head mixer (optional, for octonion_attention models)
    head_mixer_w: Option<Vec<Vec<Vec<f32>>>>,  // [8, head_dim, head_dim]
    head_mixer_beta: Option<Vec<f32>>,         // [head_dim]
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

impl WalshModel {
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
        
        // Version (1 = old sparse, 2 = packed, 3 = new sparse)
        let version = read_u16(data, &mut cursor)?;
        if version != 1 && version != 2 && version != 3 && version != 4 {
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
        
        // Hash embedding storage (for hadamard + hash_embeddings mode)
        let mut hash_tables: Vec<(String, Vec<f32>, f32)> = Vec::new();  // (name, data, scale)
        let mut hash_importance: Vec<f32> = Vec::new();
        let mut hash_indices: Vec<i32> = Vec::new();
        
        // Weight name -> data storage
        let mut bitmask_weights: std::collections::HashMap<String, BitmaskWeight> = std::collections::HashMap::new();
        let mut norm_weights: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        let mut scale_weights: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        // Head mixer weights: name -> flattened [8, D, D] or [D]
        let mut head_mixer_w_raw: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        let mut head_mixer_beta_raw: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        
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
                b'B' => {
                    // Bitmask ternary weight (smallest + sparse iteration)
                    let name = read_name(data, &mut cursor)?;
                    
                    // Read shape
                    let ndim = data[cursor] as usize;
                    cursor += 1;
                    
                    let mut shape_vec = Vec::new();
                    for _ in 0..ndim {
                        shape_vec.push(read_u32(data, &mut cursor)? as usize);
                    }
                    
                    // Shape is [8, out_per_part, in_per_part] for 3D octonion weights
                    let (out_dim, in_dim) = if ndim == 3 {
                        let out_per_part = shape_vec[1];
                        let in_per_part = shape_vec[2];
                        (8 * out_per_part, 8 * in_per_part)
                    } else if ndim == 2 {
                        (shape_vec[0], shape_vec[1])
                    } else {
                        (1, 1)
                    };
                    
                    // Read num_nonzero
                    let num_nonzero = read_u32(data, &mut cursor)? as usize;
                    
                    // Read bitmask bytes
                    let bitmask_len = read_u32(data, &mut cursor)? as usize;
                    let bitmask = data[cursor..cursor + bitmask_len].to_vec();
                    cursor += bitmask_len;
                    
                    // Read sign bytes
                    let sign_len = read_u32(data, &mut cursor)? as usize;
                    let sign_bits = data[cursor..cursor + sign_len].to_vec();
                    cursor += sign_len;
                    
                    bitmask_weights.insert(name, BitmaskWeight {
                        bitmask,
                        sign_bits,
                        num_nonzero,
                        shape: (out_dim, in_dim),
                        beta: vec![1.0; out_dim / 8],  // beta is per out_per_part
                    });
                }
                b'W' => {
                    // Legacy sparse format - skip with warning
                    let name = read_name(data, &mut cursor)?;
                    let ndim = data[cursor] as usize;
                    cursor += 1;
                    for _ in 0..ndim {
                        cursor += 4;
                    }
                    let num_pos = read_u32(data, &mut cursor)? as usize;
                    let num_neg = read_u32(data, &mut cursor)? as usize;
                    cursor += (num_pos + num_neg) * 4;
                    ic_cdk::println!("Warning: Skipping legacy sparse weight: {}", name);
                }
                b'T' => {
                    // Legacy packed ternary - skip with warning
                    let name = read_name(data, &mut cursor)?;
                    let ndim = data[cursor] as usize;
                    cursor += 1;
                    for _ in 0..ndim {
                        cursor += 4; // Skip u32 dims
                    }
                    // Skip the array data
                    let (_dtype, arr_shape) = read_array_header(data, &mut cursor)?;
                    let arr_size: usize = arr_shape.iter().product();
                    cursor += arr_size;
                    ic_cdk::println!("Warning: Skipping legacy packed weight: {}", name);
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
                b'H' => {
                    // Head mixer weights [8, D, D] - FP16
                    let name = read_name(data, &mut cursor)?;
                    let arr = read_array_f16_as_f32(data, &mut cursor)?;
                    head_mixer_w_raw.insert(name, arr);
                }
                b'h' => {
                    // Head mixer beta [D] - FP16
                    let name = read_name(data, &mut cursor)?;
                    let arr = read_array_f16_as_f32(data, &mut cursor)?;
                    head_mixer_beta_raw.insert(name, arr);
                }
                b'A' => {
                    // Hash embedding table - INT8 with scale
                    let name = read_name(data, &mut cursor)?;
                    let scale = read_f32(data, &mut cursor)?;
                    let arr = read_array_i8(data, &mut cursor)?;
                    let arr_f32: Vec<f32> = arr.iter().map(|&x| x as f32 * scale).collect();
                    hash_tables.push((name, arr_f32, scale));
                }
                b'I' => {
                    // Hash importance weights - FP16
                    let _name = read_name(data, &mut cursor)?;
                    let arr = read_array_f16_as_f32(data, &mut cursor)?;
                    hash_importance = arr;
                }
                b'J' => {
                    // Hash indices - INT32
                    let _name = read_name(data, &mut cursor)?;
                    let arr = read_array_i32(data, &mut cursor)?;
                    hash_indices = arr;
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
            
            let mut get_bitmask = |suffix: &str| -> BitmaskWeight {
                let weight_key = format!("{}.{}", prefix, suffix);
                let beta_key = weight_key.replace(".weight", ".beta");
                
                let out_per_part = config.n_embd / 8;
                let mut bw = bitmask_weights.remove(&weight_key).unwrap_or_else(|| BitmaskWeight {
                    bitmask: Vec::new(),
                    sign_bits: Vec::new(),
                    num_nonzero: 0,
                    shape: (config.n_embd, config.n_embd),
                    beta: vec![1.0; out_per_part],
                });
                
                // Apply beta scales if available
                if let Some(beta) = scale_weights.remove(&beta_key) {
                    bw.beta = beta;
                }
                
                bw
            };
            
            let mut get_norm = |suffix: &str| -> Vec<f32> {
                let key = format!("{}.{}", prefix, suffix);
                norm_weights.remove(&key).unwrap_or_else(|| vec![1.0; config.n_embd])
            };
            
            // Extract head mixer weights if present
            let head_dim = config.n_embd / config.n_head;
            let hm_w_key = format!("{}.attention.head_mixer.W", prefix);
            let hm_beta_key = format!("{}.attention.head_mixer.beta", prefix);
            
            let head_mixer_w = head_mixer_w_raw.remove(&hm_w_key).map(|flat| {
                // Reshape from flat [8 * D * D] to [8][D][D]
                let mut w = vec![vec![vec![0.0f32; head_dim]; head_dim]; 8];
                for i in 0..8 {
                    for j in 0..head_dim {
                        for k in 0..head_dim {
                            let idx = i * head_dim * head_dim + j * head_dim + k;
                            if idx < flat.len() {
                                w[i][j][k] = flat[idx];
                            }
                        }
                    }
                }
                w
            });
            
            let head_mixer_beta = head_mixer_beta_raw.remove(&hm_beta_key);
            
            layers.push(TransformerLayer {
                attn_norm: get_norm("attention_norm.weight"),
                wq: get_bitmask("attention.wq.weight"),
                wk: get_bitmask("attention.wk.weight"),
                wv: get_bitmask("attention.wv.weight"),
                wo: get_bitmask("attention.wo.weight"),
                ffn_norm: get_norm("ffn_norm.weight"),
                gate_proj: get_bitmask("feed_forward.gate_proj.weight"),
                up_proj: get_bitmask("feed_forward.up_proj.weight"),
                down_proj: get_bitmask("feed_forward.down_proj.weight"),
                head_mixer_w,
                head_mixer_beta,
            });
        }
        
        // Create auto-detecting tokenizer based on vocab_size
        let tokenizer = Tokenizer::new(config.vocab_size);
        
        // Build hash embedding if we loaded hash tables
        let hash_embedding = if !hash_tables.is_empty() && hash_tables.len() == 3 {
            // Sort tables by name to ensure consistent ordering
            let mut sorted_tables: Vec<_> = hash_tables.into_iter().collect();
            sorted_tables.sort_by(|a, b| a.0.cmp(&b.0));
            
            // Determine bucket size from first table
            let bucket_size = sorted_tables[0].1.len() / config.n_embd;
            let num_tables = 3;
            
            let tables: Vec<Vec<f32>> = sorted_tables.iter().map(|(_, data, _)| data.clone()).collect();
            let table_scales: Vec<f32> = sorted_tables.iter().map(|(_, _, scale)| *scale).collect();
            
            Some(HashEmbedding {
                tables,
                table_scales,
                importance: hash_importance,
                indices: hash_indices,
                bucket_size,
                num_tables,
                n_embd: config.n_embd,
            })
        } else {
            None
        };
        
        // Set default embeddings if not loaded and not using hash
        if embeddings.is_empty() && hash_embedding.is_none() {
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
        
        Ok(WalshModel {
            config,
            embeddings,
            embed_scale,
            hash_embedding,
            layers,
            final_norm,
            rope_cos,
            rope_sin,
            tokenizer,
        })
    }
    
    /// Tokenize a string to token IDs
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        self.tokenizer.tokenize(text)
    }
    
    /// Decode a single token to string
    pub fn decode_token(&self, token: usize) -> String {
        self.tokenizer.decode_token(token)
    }
    
    /// Decode multiple tokens to string  
    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        self.tokenizer.decode_tokens(tokens)
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
        
        sample_with_temperature(&logits, 0.8)
    }
    
    /// Embed tokens to hidden states (first step of chunked forward)
    pub fn embed_tokens(&self, tokens: &[usize]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut h = vec![0.0f32; tokens.len() * n_embd];
        
        // Use hash embeddings if available
        if let Some(ref hash_emb) = self.hash_embedding {
            for (t, &tok) in tokens.iter().enumerate() {
                let tok = tok % self.config.vocab_size;
                let emb = self.hash_embed_single(hash_emb, tok);
                for i in 0..n_embd {
                    h[t * n_embd + i] = emb[i];
                }
            }
        } else {
            // Standard embedding lookup
            for (t, &tok) in tokens.iter().enumerate() {
                let tok = tok % self.config.vocab_size;
                let start = tok * n_embd;
                for i in 0..n_embd {
                    if start + i < self.embeddings.len() {
                        h[t * n_embd + i] = self.embeddings[start + i];
                    }
                }
            }
        }
        h
    }
    
    /// Hash embedding lookup for a single token
    fn hash_embed_single(&self, hash_emb: &HashEmbedding, tok: usize) -> Vec<f32> {
        let n_embd = hash_emb.n_embd;
        let num_tables = hash_emb.num_tables;
        let mut result = vec![0.0f32; n_embd];
        
        // Get importance weights for this token: [num_tables]
        let imp_base = tok * num_tables;
        
        // Sum weighted vectors from each table
        for table_idx in 0..num_tables {
            // Get precomputed hash index for this token/table: indices[tok * num_tables + table_idx]
            let indices_idx = tok * num_tables + table_idx;
            if indices_idx >= hash_emb.indices.len() { continue; }
            let bucket_idx = hash_emb.indices[indices_idx] as usize;
            if bucket_idx >= hash_emb.bucket_size { continue; }
            
            // Get importance weight
            let imp = if imp_base + table_idx < hash_emb.importance.len() {
                hash_emb.importance[imp_base + table_idx]
            } else {
                1.0
            };
            
            // Add weighted embedding from table
            let emb_start = bucket_idx * n_embd;
            for i in 0..n_embd {
                if emb_start + i < hash_emb.tables[table_idx].len() {
                    result[i] += imp * hash_emb.tables[table_idx][emb_start + i];
                }
            }
        }
        
        // Scale by sqrt(n_embd) as in Python
        let scale = (n_embd as f32).sqrt();
        for v in result.iter_mut() {
            *v *= scale;
        }
        
        result
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
        let mut q_new = self.bitmask_matmul(&normed, &layer.wq, new_len, n_embd, n_embd);
        let mut k_new = self.bitmask_matmul(&normed, &layer.wk, new_len, n_embd, n_embd);
        let v_new = self.bitmask_matmul(&normed, &layer.wv, new_len, n_embd, n_embd);
        
        // Apply RoPE to new Q and K (at positions cache.cached_len ... cache.cached_len + new_len - 1)
        self.apply_rope_at_pos(&mut q_new, &mut k_new, cache.cached_len, new_len, head_dim);
        
        // Append new K, V to cache
        cache.append(layer_idx, &k_new, &v_new);
        
        // Get full K, V from cache for attention
        let k_full = &cache.k_cache[layer_idx];
        let v_full = &cache.v_cache[layer_idx];
        
        // Attention: new Q attends to full K,V
        let mut attn_out = attention_cached(&q_new, k_full, v_full, new_len, full_len, n_head, head_dim);
        
        // Apply head mixer if present (octonion attention mixing)
        if let (Some(ref w), Some(ref beta)) = (&layer.head_mixer_w, &layer.head_mixer_beta) {
            attn_out = self.apply_head_mixer(&attn_out, w, beta, new_len, n_head, head_dim);
        }
        
        // Output projection
        let o = self.bitmask_matmul(&attn_out, &layer.wo, new_len, n_embd, n_embd);
        
        // Residual connection
        let mut h: Vec<f32> = hidden_new.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        
        // FFN
        let mut normed2 = Vec::with_capacity(new_len * n_embd);
        for t in 0..new_len {
            let slice = &h[t * n_embd..(t + 1) * n_embd];
            normed2.extend(rms_norm(slice, &layer.ffn_norm, 1e-6));
        }
        
        let hidden_dim = layer.gate_proj.shape.0;
        let gate = self.bitmask_matmul(&normed2, &layer.gate_proj, new_len, n_embd, hidden_dim);
        let up = self.bitmask_matmul(&normed2, &layer.up_proj, new_len, n_embd, hidden_dim);
        
        // SwiGLU
        let mut ffn_hidden = vec![0.0f32; new_len * hidden_dim];
        for i in 0..gate.len().min(up.len()) {
            ffn_hidden[i] = silu(gate[i]) * up[i];
        }
        
        let down = self.bitmask_matmul(&ffn_hidden, &layer.down_proj, new_len, hidden_dim, n_embd);
        
        // Residual
        for i in 0..h.len().min(down.len()) {
            h[i] += down[i];
        }
        
        h
    }
    
    /// Apply octonion head mixer using Cayley-Dickson algebra
    /// Mixes 8 attention heads using learned weight matrices
    fn apply_head_mixer(
        &self,
        attn_out: &[f32],  // [new_len * n_embd]
        w: &[Vec<Vec<f32>>],  // [8][head_dim][head_dim]
        beta: &[f32],  // [head_dim]
        new_len: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        use crate::octonion::{SIGN_TABLE, WEIGHT_IDX};
        
        let n_embd = n_head * head_dim;
        let mut y = vec![0.0f32; new_len * n_embd];
        
        // Process each token position
        for t in 0..new_len {
            // Extract 8 head inputs for this position: x[h] = attn_out[t * n_embd + h * head_dim .. +head_dim]
            let base = t * n_embd;
            
            // For each output head
            for i in 0..8 {
                let y_base = base + i * head_dim;
                
                // Sum over 8 input heads with Cayley-Dickson structure
                for j in 0..8 {
                    let sign = SIGN_TABLE[i][j] as f32;
                    let w_idx = WEIGHT_IDX[i][j];
                    
                    let x_base = base + j * head_dim;
                    
                    // y[i] += sign * x[j] @ W[w_idx]
                    // PyTorch: y[k] = sum_j x[j] * W[j][k]
                    for d_out in 0..head_dim {
                        let mut dot = 0.0f32;
                        for d_in in 0..head_dim {
                            let x_val = attn_out.get(x_base + d_in).copied().unwrap_or(0.0);
                            // W is [8][head_dim][head_dim], indexed as W[part][row][col]
                            // For x @ W: y[d_out] = sum_d_in x[d_in] * W[d_in][d_out]
                            let w_val = w.get(w_idx)
                                .and_then(|m| m.get(d_in))
                                .and_then(|row| row.get(d_out))
                                .copied()
                                .unwrap_or(0.0);
                            dot += x_val * w_val;
                        }
                        y[y_base + d_out] += sign * dot;
                    }
                }
                
                // Apply beta scaling
                for d in 0..head_dim {
                    let b = beta.get(d).copied().unwrap_or(1.0);
                    y[y_base + d] *= b;
                }
            }
        }
        
        y
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
        
        // Output projection
        let mut logits = vec![0.0f32; self.config.vocab_size];
        
        if let Some(ref hash_emb) = self.hash_embedding {
            // Hash embedding output projection: dot product with reconstructed embeddings
            for v in 0..self.config.vocab_size {
                let emb = self.hash_embed_single(hash_emb, v);
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += normed[i] * emb[i];
                }
                logits[v] = dot;
            }
        } else {
            // Standard tied embeddings
            for v in 0..self.config.vocab_size {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += normed[i] * self.embeddings[v * n_embd + i];
                }
                logits[v] = dot;
            }
        }
        
        sample_with_temperature(&logits, 0.8)
    }
    
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let mut tokens: Vec<usize> = self.tokenize(prompt);
        
        for _ in 0..max_tokens {
            if tokens.len() >= self.config.block_size {
                break;
            }
            
            let logits = self.forward(&tokens);
            let next_token = sample_with_temperature(&logits, 0.8);
            tokens.push(next_token);
        }
        
        Ok(self.decode_tokens(&tokens))
    }
    
    /// Simple generation using embeddings only (fast, for query calls)
    pub fn generate_simple(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let n_embd = self.config.n_embd;
        
        let mut tokens: Vec<usize> = self.tokenize(prompt);
        
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
            
            let next_token = sample_with_temperature(&logits, 0.8);
            tokens.push(next_token);
        }
        
        Ok(self.decode_tokens(&tokens))
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
        let mut q = self.bitmask_matmul(&normed, &layer.wq, seq_len, n_embd, n_embd);
        let mut k = self.bitmask_matmul(&normed, &layer.wk, seq_len, n_embd, n_embd);
        let v = self.bitmask_matmul(&normed, &layer.wv, seq_len, n_embd, n_embd);
        
        // Apply RoPE to Q and K
        self.apply_rope(&mut q, &mut k, seq_len, head_dim);
        
        // Attention
        let mut attn_out = attention(&q, &k, &v, n_head, head_dim);
        
        // Apply head mixer if present (octonion attention mixing)
        if let (Some(ref w), Some(ref beta)) = (&layer.head_mixer_w, &layer.head_mixer_beta) {
            attn_out = self.apply_head_mixer(&attn_out, w, beta, seq_len, n_head, head_dim);
        }
        
        // Output projection
        let o = self.bitmask_matmul(&attn_out, &layer.wo, seq_len, n_embd, n_embd);
        
        // Residual
        let mut h: Vec<f32> = x.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        
        // FFN
        let mut normed2 = Vec::with_capacity(seq_len * n_embd);
        for t in 0..seq_len {
            let slice = &h[t * n_embd..(t + 1) * n_embd];
            normed2.extend(rms_norm(slice, &layer.ffn_norm, 1e-6));
        }
        
        // FFN hidden_dim derived from weight shape: gate_proj is [8, hidden/8, n_embd/8]
        let hidden_dim = layer.gate_proj.shape.0;
        let gate = self.bitmask_matmul(&normed2, &layer.gate_proj, seq_len, n_embd, hidden_dim);
        let up = self.bitmask_matmul(&normed2, &layer.up_proj, seq_len, n_embd, hidden_dim);
        
        // SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; seq_len * hidden_dim];
        for i in 0..gate.len().min(up.len()) {
            ffn_hidden[i] = silu(gate[i]) * up[i];
        }
        
        let down = self.bitmask_matmul(&ffn_hidden, &layer.down_proj, seq_len, hidden_dim, n_embd);
        
        // Residual
        for i in 0..h.len().min(down.len()) {
            h[i] += down[i];
        }
        
        h
    }
    
    /// Bitmask ternary matmul with octonion structure (OPTIMIZED)
    /// Uses counter-based iteration to avoid expensive division operations.
    /// Input x: [batch, in_dim] where in_dim = 8 * in_per_part
    /// Weight w: BitmaskWeight with bitmask and sign_bits
    /// Output: [batch, out_dim] where out_dim = 8 * out_per_part
    fn bitmask_matmul(&self, x: &[f32], w: &BitmaskWeight, batch: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
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
        // REV_IDX[out_part][w_part] = in_part such that IDX[out_part][in_part] == w_part
        const REV_IDX: [[usize; 8]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ];
        
        // Pre-compute constants
        let in_per_part = in_dim / 8;
        let out_per_part = out_dim / 8;
        let mut y = vec![0.0f32; batch * out_dim];
        
        // State counters (avoid division/modulo inside loop)
        let mut w_part = 0usize;
        let mut o = 0usize;
        let mut i = 0usize;
        let mut sign_idx = 0usize;
        
        // Helper to advance counters by n positions
        let advance_counters = |w_part: &mut usize, o: &mut usize, i: &mut usize, n: usize, in_per_part: usize, out_per_part: usize| {
            *i += n;
            while *i >= in_per_part {
                *i -= in_per_part;
                *o += 1;
                if *o >= out_per_part {
                    *o = 0;
                    *w_part += 1;
                }
            }
        };
        
        // Flattened loop with counter-based indexing
        for &mask_byte in w.bitmask.iter() {
            if mask_byte == 0 {
                // Optimization: Skip 8 positions at once
                advance_counters(&mut w_part, &mut o, &mut i, 8, in_per_part, out_per_part);
                continue;
            }
            
            for bit in 0..8u8 {
                if (mask_byte & (1 << bit)) != 0 {
                    // Bounds check
                    if w_part >= 8 { break; }
                    
                    // --- HOT LOOP START ---
                    
                    // 1. Get Sign
                    let sign_byte = w.sign_bits.get(sign_idx / 8).copied().unwrap_or(0xFF);
                    let is_positive = (sign_byte & (1 << (sign_idx % 8))) != 0;
                    let base_sign = if is_positive { 1.0f32 } else { -1.0f32 };
                    sign_idx += 1;
                    
                    // 2. Octonion Permutation (Unrolled for all 8 output parts)
                    // y[k][o] += sign_table[k][in_part] * x[in_part][i] * base_sign
                    
                    if batch == 1 {
                        // Fast path: batch=1 (common for inference)
                        for k in 0..8 {
                            let in_part = REV_IDX[k][w_part];
                            let sign_val = SIGN[k][in_part] as f32;
                            let combined_sign = sign_val * base_sign;
                            
                            let x_idx = in_part * in_per_part + i;
                            let y_idx = k * out_per_part + o;
                            
                            // Safety: bounds are checked by w_part < 8, o < out_per_part, i < in_per_part
                            if x_idx < x.len() && y_idx < y.len() {
                                unsafe {
                                    *y.get_unchecked_mut(y_idx) += combined_sign * *x.get_unchecked(x_idx);
                                }
                            }
                        }
                    } else {
                        // General case: batch > 1
                        for k in 0..8 {
                            let in_part = REV_IDX[k][w_part];
                            let sign_val = SIGN[k][in_part] as f32;
                            let combined_sign = sign_val * base_sign;
                            
                            let x_offset = in_part * in_per_part + i;
                            let y_offset = k * out_per_part + o;
                            
                            for b in 0..batch {
                                let x_idx = b * in_dim + x_offset;
                                let y_idx = b * out_dim + y_offset;
                                if x_idx < x.len() && y_idx < y.len() {
                                    y[y_idx] += combined_sign * x[x_idx];
                                }
                            }
                        }
                    }
                    // --- HOT LOOP END ---
                }
                
                // Increment counters for each bit position
                i += 1;
                if i == in_per_part {
                    i = 0;
                    o += 1;
                    if o == out_per_part {
                        o = 0;
                        w_part += 1;
                    }
                }
            }
        }
        
        // Apply beta scaling - beta has length out_per_part, applies to all 8 output parts
        for out_part in 0..8 {
            for o_idx in 0..out_per_part {
                let beta = w.beta.get(o_idx).copied().unwrap_or(1.0);
                if beta != 1.0 {
                    let y_inner_offset = out_part * out_per_part + o_idx;
                    for b in 0..batch {
                        let y_idx = b * out_dim + y_inner_offset;
                        if y_idx < y.len() {
                            y[y_idx] *= beta;
                        }
                    }
                }
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
    
    // Extract string values for algebra
    let extract_str = |key: &str| -> Option<String> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..].trim_start();
        if rest.starts_with('"') {
            let rest = &rest[1..];
            let end = rest.find('"')?;
            Some(rest[..end].to_string())
        } else {
            None
        }
    };
    
    // Extract boolean values
    let extract_bool = |key: &str| -> Option<bool> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..].trim_start();
        if rest.starts_with("true") {
            Some(true)
        } else if rest.starts_with("false") {
            Some(false)
        } else {
            None
        }
    };
    
    Ok(ModelConfig {
        vocab_size: extract("vocab_size").unwrap_or(65),
        n_embd: extract("n_embd").unwrap_or(256),
        n_layer: extract("n_layer").unwrap_or(8),
        n_head: extract("n_head").unwrap_or(8),
        block_size: extract("block_size").unwrap_or(512),
        algebra: extract_str("algebra").unwrap_or_else(|| "octonion".to_string()),
        hash_embeddings: extract_bool("hash_embeddings").unwrap_or(false),
    })
}

fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// Simple LCG random number generator (deterministic seeded RNG for ICP)
// Uses the MINSTD parameters
thread_local! {
    static RNG_STATE: std::cell::RefCell<u64> = std::cell::RefCell::new(12345);
}

fn rand_u64() -> u64 {
    RNG_STATE.with(|state| {
        let mut s = state.borrow_mut();
        // LCG: x' = (a * x + c) mod m
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    })
}

fn rand_f32() -> f32 {
    (rand_u64() >> 40) as f32 / (1u64 << 24) as f32
}

/// Seed the RNG (call with time or counter for non-determinism)
pub fn seed_rng(seed: u64) {
    RNG_STATE.with(|state| {
        *state.borrow_mut() = seed;
    });
}

/// Sample from logits with temperature
/// temperature = 0.0 -> greedy (argmax)
/// temperature = 1.0 -> standard softmax sampling
/// temperature > 1.0 -> more random
pub fn sample_with_temperature(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 0.0 || logits.is_empty() {
        return argmax(logits);
    }
    
    // Apply temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    
    // Compute softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
    
    // Sample from distribution
    let r = rand_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    
    // Fallback to last token
    probs.len() - 1
}

