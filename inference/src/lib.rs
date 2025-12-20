//! SpinNet Inference Engine for Internet Computer (Single-User Mode)
//!
//! Implements layer-by-layer chunked execution with singleton state.

mod model;
mod octonion;
mod attention;

use candid::{CandidType, Deserialize};
use ic_cdk_macros::{init, post_upgrade, query, update};
use std::cell::RefCell;

use model::{SpinNetModel, KVCache};

const MODEL_BYTES: &[u8] = include_bytes!("../ckpt_v2.spinnet");

/// Generation state for chunked execution
struct GenerationState {
    tokens: Vec<usize>,
    max_tokens: usize,
    generated_count: usize,
}

/// Forward pass state for layer-by-layer execution
struct ForwardState {
    hidden: Vec<f32>,      // Current hidden states (for new positions only when caching)
    layer_idx: usize,      // Current layer (0..n_layer)
    seq_len: usize,        // Total sequence length (including cached)
    new_len: usize,        // Number of new positions (1 after prompt)
    phase: ForwardPhase,
    kv_cache: Option<KVCache>,  // KV cache for efficient generation
    // Scratchpad buffers (avoid allocations in hot loops)
    scratch_norm: Vec<f32>,     // For normalized hidden states
    scratch_ffn: Vec<f32>,      // For FFN intermediate activations
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ForwardPhase {
    Idle,
    Embedding,
    Layers,
    FinalNorm,
    Done,
}

thread_local! {
    static MODEL: RefCell<Option<SpinNetModel>> = RefCell::new(None);
    // Single-user state (Singleton)
    static GEN_STATE: RefCell<Option<GenerationState>> = RefCell::new(None);
    static FWD_STATE: RefCell<Option<ForwardState>> = RefCell::new(None);
}

#[init]
fn init() {
    load_embedded_model();
}

#[post_upgrade]
fn post_upgrade() {
    load_embedded_model();
}

fn load_embedded_model() {
    ic_cdk::println!("SpinNet: Loading model...");
    match SpinNetModel::from_bytes(MODEL_BYTES) {
        Ok(model) => {
            MODEL.with(|m| *m.borrow_mut() = Some(model));
            ic_cdk::println!("SpinNet: Model loaded ({} bytes)", MODEL_BYTES.len());
        }
        Err(e) => ic_cdk::println!("SpinNet: Load failed: {}", e),
    }
}

/// Generate full text in one update call (for small models)
#[update]
fn generate_full(prompt: String, max_tokens: u32) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => match model.generate(&prompt, max_tokens as usize) {
                Ok(text) => text,
                Err(e) => format!("Error: {}", e),
            },
            None => "Error: No model".to_string(),
        }
    })
}

/// Start generating with layer-by-layer execution (resets state)
#[update]
fn start_generation(prompt: String, max_tokens: u32) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                let tokens = model.tokenize(&prompt);
                
                // Clear any existing forward state
                FWD_STATE.with(|fwd| {
                    *fwd.borrow_mut() = None;
                });
                
                GEN_STATE.with(|s| {
                    *s.borrow_mut() = Some(GenerationState {
                        tokens,
                        max_tokens: max_tokens as usize,
                        generated_count: 0,
                    });
                });
                format!("Started: {} prompt tokens, {} to generate", prompt.len(), max_tokens)
            }
            None => "Error: No model".to_string(),
        }
    })
}


/// Start forward pass for current tokens (call before process_layer)
/// First call after start_generation: processes entire prompt
/// Subsequent calls: processes only the new token using KV cache
#[update]
fn start_forward() -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                GEN_STATE.with(|gen| {
                    let gen_ref = gen.borrow();
                    match gen_ref.as_ref() {
                        Some(gen_state) => {
                            FWD_STATE.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                
                                // Check if we have an existing cache in the *previous* state
                                // We extract it to reuse it
                                let existing_cache = fwd_ref.as_mut().and_then(|s| s.kv_cache.take());
                                
                                if let Some(mut cache) = existing_cache {
                                    // Continuation: only embed the NEW token (last one)
                                    let new_token = gen_state.tokens.last().copied().unwrap_or(0);
                                    let hidden = model.embed_tokens(&[new_token]);
                                    let seq_len = gen_state.tokens.len();
                                    
                                    *fwd_ref = Some(ForwardState {
                                        hidden,
                                        layer_idx: 0,
                                        seq_len,
                                        new_len: 1,  // Just the new token
                                        phase: ForwardPhase::Layers,
                                        kv_cache: Some(cache),
                                        scratch_norm: Vec::new(),
                                        scratch_ffn: Vec::new(),
                                    });
                                    
                                    format!("Forward: cached, new_len=1")
                                } else {
                                    // First call: embed ALL tokens, initialize cache
                                    let hidden = model.embed_tokens(&gen_state.tokens);
                                    let seq_len = gen_state.tokens.len();
                                    let cache = KVCache::new(model.config.n_layer);
                                    
                                    *fwd_ref = Some(ForwardState {
                                        hidden,
                                        layer_idx: 0,
                                        seq_len,
                                        new_len: seq_len,  // All tokens are new
                                        phase: ForwardPhase::Layers,
                                        kv_cache: Some(cache),
                                        scratch_norm: Vec::with_capacity(seq_len * model.config.n_embd),
                                        scratch_ffn: Vec::with_capacity(seq_len * model.config.n_embd * 4),
                                    });
                                    
                                    format!("Forward: init, seq_len={}", seq_len)
                                }
                            })
                        }
                        None => "Error: No generation started".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    });
    result
}

/// Process multiple transformer layers (uses KV cache when available)
/// Includes adaptive instruction monitoring to prevent exceeding ICP limits
#[update]
fn process_layers(count: u32) -> String {
    // Adaptive chunking: monitor instruction usage
    const MAX_INSTRUCTIONS: u64 = 2_000_000_000;  // Safe threshold (ICP limit is higher)
    let start_counter = ic_cdk::api::performance_counter(0);
    
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                FWD_STATE.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.as_mut() {
                        Some(state) => {
                            if state.phase != ForwardPhase::Layers {
                                return format!("Not in layer phase (phase: {:?})", state.phase);
                            }
                            
                            let mut processed = 0u32;
                            
                            // Use cached execution if we have a cache
                            if let Some(ref mut cache) = state.kv_cache {
                                for _ in 0..count {
                                    if state.layer_idx >= model.config.n_layer {
                                        state.phase = ForwardPhase::FinalNorm;
                                        break;
                                    }
                                    
                                    // Adaptive: Check instruction usage before processing next layer
                                    let used = ic_cdk::api::performance_counter(0) - start_counter;
                                    if used > MAX_INSTRUCTIONS * 80 / 100 {
                                        // Approaching limit, pause and let caller retry
                                        return format!("Paused at layer {} ({}% budget)", 
                                            state.layer_idx, 
                                            used * 100 / MAX_INSTRUCTIONS);
                                    }
                                    
                                    // Process layer with cache
                                    let new_hidden = model.run_layer_cached(
                                        state.layer_idx,
                                        &state.hidden,
                                        state.new_len,
                                        cache,
                                    );
                                    state.hidden = new_hidden;
                                    state.layer_idx += 1;
                                    processed += 1;
                                }
                                
                                // After all layers, update cached length
                                if state.layer_idx >= model.config.n_layer {
                                    cache.update_len(state.new_len);
                                    state.phase = ForwardPhase::FinalNorm;
                                }
                            } else {
                                // Fallback: non-cached (shouldn't happen normally)
                                for _ in 0..count {
                                    if state.layer_idx >= model.config.n_layer {
                                        state.phase = ForwardPhase::FinalNorm;
                                        break;
                                    }
                                    
                                    // Adaptive: Check instruction usage
                                    let used = ic_cdk::api::performance_counter(0) - start_counter;
                                    if used > MAX_INSTRUCTIONS * 80 / 100 {
                                        return format!("Paused at layer {} ({}% budget)", 
                                            state.layer_idx, 
                                            used * 100 / MAX_INSTRUCTIONS);
                                    }
                                    
                                    let new_hidden = model.run_layer(state.layer_idx, &state.hidden, state.seq_len);
                                    state.hidden = new_hidden;
                                    state.layer_idx += 1;
                                    processed += 1;
                                }
                                
                                if state.layer_idx >= model.config.n_layer {
                                    state.phase = ForwardPhase::FinalNorm;
                                }
                            }
                            
                            if state.phase == ForwardPhase::FinalNorm {
                                "Done".to_string()
                            } else {
                                format!("Layer {}/{}", state.layer_idx, model.config.n_layer)
                            }
                        }
                        None => "Error: No forward state".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    });
    result
}

/// Finish forward pass and sample next token
#[update]
fn finish_forward() -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                FWD_STATE.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.as_mut() {
                        Some(state) => {
                            // Final norm and sample from the last position in hidden
                            // In cached mode, hidden only has new_len positions
                            let sample_len = state.new_len;  // Sample from position new_len-1
                            let next_token = model.final_norm_and_sample(&state.hidden, sample_len);
                            
                            
                            // Add token to generation state
                            GEN_STATE.with(|gen| {
                                let mut gen_ref = gen.borrow_mut();
                                if let Some(gen_state) = gen_ref.as_mut() {
                                    gen_state.tokens.push(next_token);
                                    gen_state.generated_count += 1;
                                }
                            });
                            
                            
                            // Clear forward state
                            state.phase = ForwardPhase::Done;
                            
                            model.decode_token(next_token)
                        }
                        None => "Error: No forward state".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    });
    result
}


/// Generate a single token in one update call (optimized for KV cache)
/// Must be called AFTER the prompt has been processed
#[update]
fn generate_next_token() -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                GEN_STATE.with(|gen| {
                    let mut gen_ref = gen.borrow_mut();
                    match gen_ref.as_mut() {
                        Some(gen_state) => {
                            // 1. Get KV Cache from ForwardState
                            let mut cache_opt = FWD_STATE.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                fwd_ref.as_mut().and_then(|s| s.kv_cache.take())
                            });
                            
                            if let Some(mut cache) = cache_opt {
                                // 2. Run Forward Pass (Cached)
                                let new_token_id = gen_state.tokens.last().copied().unwrap_or(0);
                                let hidden = model.embed_tokens(&[new_token_id]);
                                let new_len = 1;
                                
                                let mut curr_hidden = hidden;
                                for layer_idx in 0..model.config.n_layer {
                                    curr_hidden = model.run_layer_cached(
                                        layer_idx,
                                        &curr_hidden,
                                        new_len,
                                        &mut cache
                                    );
                                }
                                
                                // Update cache length
                                cache.update_len(new_len);
                                
                                // 3. Sample
                                let next_token = model.final_norm_and_sample(&curr_hidden, new_len);
                                gen_state.tokens.push(next_token);
                                gen_state.generated_count += 1;
                                
                                // 4. Restore Cache to ForwardState
                                FWD_STATE.with(|fwd| {
                                    *fwd.borrow_mut() = Some(ForwardState {
                                        hidden: curr_hidden, // Not typically needed next time but good for consistency
                                        layer_idx: model.config.n_layer,
                                        seq_len: cache.cached_len, // Full length including new
                                        new_len: 0,
                                        phase: ForwardPhase::Done,
                                        kv_cache: Some(cache),
                                        scratch_norm: Vec::new(),
                                        scratch_ffn: Vec::new(),
                                    });
                                });
                                
                                model.decode_token(next_token)
                            } else {
                                "Error: No KV cache found. Must process prompt first.".to_string()
                            }
                        }
                        None => "Error: No generation started".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    })
}

/// Generate N tokens in a single call for maximum single-session throughput
/// This reduces round-trip overhead by generating multiple tokens per update call
#[update]
fn generate_n_tokens(n: u32) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                GEN_STATE.with(|gen| {
                    let mut gen_ref = gen.borrow_mut();
                    match gen_ref.as_mut() {
                        Some(gen_state) => {
                            let mut cache_opt = FWD_STATE.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                fwd_ref.as_mut().and_then(|s| s.kv_cache.take())
                            });
                            
                            if let Some(mut cache) = cache_opt {
                                let mut result = String::new();
                                let tokens_remaining = gen_state.max_tokens.saturating_sub(gen_state.generated_count);
                                let tokens_to_generate = std::cmp::min(n, tokens_remaining as u32);
                                
                                for _ in 0..tokens_to_generate {
                                    // Get last token to embed
                                    let last_token = gen_state.tokens.last().copied().unwrap_or(0);
                                    let hidden = model.embed_tokens(&[last_token]);
                                    
                                    // Run through all layers with cache
                                    let mut curr_hidden = hidden;
                                    for layer_idx in 0..model.config.n_layer {
                                        curr_hidden = model.run_layer_cached(
                                            layer_idx,
                                            &curr_hidden,
                                            1,
                                            &mut cache
                                        );
                                    }
                                    
                                    cache.update_len(1);
                                    
                                    // Sample next token
                                    let next_token = model.final_norm_and_sample(&curr_hidden, 1);
                                    gen_state.tokens.push(next_token);
                                    gen_state.generated_count += 1;
                                    
                                    // Decode and append to result
                                    result.push_str(&model.decode_token(next_token));
                                }
                                
                                // Restore cache
                                FWD_STATE.with(|fwd| {
                                    *fwd.borrow_mut() = Some(ForwardState {
                                        hidden: vec![],
                                        layer_idx: model.config.n_layer,
                                        seq_len: cache.cached_len,
                                        new_len: 0,
                                        phase: ForwardPhase::Done,
                                        kv_cache: Some(cache),
                                        scratch_norm: Vec::new(),
                                        scratch_ffn: Vec::new(),
                                    });
                                });
                                
                                result
                            } else {
                                "Error: No KV cache".to_string()
                            }
                        }
                        None => "Error: No generation started".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    })
}

/// Get current result (Session 0)
#[query]
fn get_result() -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                GEN_STATE.with(|gen| {
                    let gen_ref = gen.borrow();
                    match gen_ref.as_ref() {
                        Some(gen_state) => model.decode_tokens(&gen_state.tokens),
                        None => "No generation".to_string(),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    })
}

/// Check if more tokens needed
#[query]
fn generation_status() -> String {
    GEN_STATE.with(|gen| {
        let gen_ref = gen.borrow();
        match gen_ref.as_ref() {
            Some(s) => format!(
                "{{\"tokens\":{},\"generated\":{},\"max\":{},\"done\":{}}}",
                s.tokens.len(), s.generated_count, s.max_tokens,
                s.generated_count >= s.max_tokens
            ),
            None => "{\"active\":false}".to_string(),
        }
    })
}

/// Quick embedding-only generation
#[query]
fn generate_quick(prompt: String) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => model.generate_simple(&prompt, 20).unwrap_or_else(|e| e),
            None => "Error: No model".to_string(),
        }
    })
}

#[query]
fn get_config() -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => model.config_json(),
            None => "{}".to_string(),
        }
    })
}
