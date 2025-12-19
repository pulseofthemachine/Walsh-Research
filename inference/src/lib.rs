//! SpinNet Inference Engine for Internet Computer
//!
//! Implements layer-by-layer chunked execution to work within IC instruction limits.

mod model;
mod octonion;
mod attention;

use candid::{CandidType, Deserialize};
use ic_cdk_macros::{init, post_upgrade, query, update};
use std::cell::RefCell;
use std::collections::BTreeMap;

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
    // Map from SessionID (u32) to State
    static GEN_STATES: RefCell<BTreeMap<u32, GenerationState>> = RefCell::new(BTreeMap::new());
    static FWD_STATES: RefCell<BTreeMap<u32, ForwardState>> = RefCell::new(BTreeMap::new());
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

/// Start generating with layer-by-layer execution
#[update]
fn start_generation(prompt: String, max_tokens: u32) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                let tokens = model.tokenize(&prompt);
                
                // Clear any existing forward state for session 0
                FWD_STATES.with(|fwd| {
                    fwd.borrow_mut().remove(&0);
                });
                
                GEN_STATES.with(|s| {
                    s.borrow_mut().insert(0, GenerationState {
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
                GEN_STATES.with(|gen| {
                    let gen_ref = gen.borrow();
                    match gen_ref.get(&0) {
                        Some(gen_state) => {
                            FWD_STATES.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                
                                // Check if we have an existing cache
                                let existing_cache = fwd_ref.get_mut(&0).and_then(|s| s.kv_cache.take());
                                
                                if let Some(mut cache) = existing_cache {
                                    // Continuation: only embed the NEW token (last one)
                                    let new_token = gen_state.tokens.last().copied().unwrap_or(0);
                                    let hidden = model.embed_tokens(&[new_token]);
                                    let seq_len = gen_state.tokens.len();
                                    
                                    fwd_ref.insert(0, ForwardState {
                                        hidden,
                                        layer_idx: 0,
                                        seq_len,
                                        new_len: 1,  // Just the new token
                                        phase: ForwardPhase::Layers,
                                        kv_cache: Some(cache),
                                    });
                                    
                                    format!("Forward: cached, new_len=1")
                                } else {
                                    // First call: embed ALL tokens, initialize cache
                                    let hidden = model.embed_tokens(&gen_state.tokens);
                                    let seq_len = gen_state.tokens.len();
                                    let cache = KVCache::new(model.config.n_layer);
                                    
                                    fwd_ref.insert(0, ForwardState {
                                        hidden,
                                        layer_idx: 0,
                                        seq_len,
                                        new_len: seq_len,  // All tokens are new
                                        phase: ForwardPhase::Layers,
                                        kv_cache: Some(cache),
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
#[update]
fn process_layers(count: u32) -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                FWD_STATES.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.get_mut(&0) {
                        Some(state) => {
                            if state.phase != ForwardPhase::Layers {
                                return format!("Not in layer phase (phase: {:?})", state.phase);
                            }
                            
                            let mut processed = 0;
                            
                            // Use cached execution if we have a cache
                            if let Some(ref mut cache) = state.kv_cache {
                                for _ in 0..count {
                                    if state.layer_idx >= model.config.n_layer {
                                        state.phase = ForwardPhase::FinalNorm;
                                        break;
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
                                // Fallback: non-cached execution (shouldn't happen normally)
                                for _ in 0..count {
                                    if state.layer_idx >= model.config.n_layer {
                                        state.phase = ForwardPhase::FinalNorm;
                                        break;
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

/// Start forward pass for a specific session
#[update]
fn start_forward_session(session_id: u32) -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                GEN_STATES.with(|gen| {
                    let gen_ref = gen.borrow();
                    match gen_ref.get(&session_id) {
                        Some(gen_state) => {
                            FWD_STATES.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                
                                // Always start fresh for prompt processing in batched mode setup
                                // Assume previous cache is invalid/cleared by start_session
                                
                                let hidden = model.embed_tokens(&gen_state.tokens);
                                let seq_len = gen_state.tokens.len();
                                let cache = KVCache::new(model.config.n_layer);
                                
                                fwd_ref.insert(session_id, ForwardState {
                                    hidden,
                                    layer_idx: 0,
                                    seq_len,
                                    new_len: seq_len,  // All tokens act as new for prompt
                                    phase: ForwardPhase::Layers,
                                    kv_cache: Some(cache),
                                });
                                
                                format!("Session {}: Forward initialized", session_id)
                            })
                        }
                        None => format!("Error: No generation state for session {}", session_id),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    });
    result
}

/// Process multiple transformer layers for a specific session
#[update]
fn process_layers_session(session_id: u32, count: u32) -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                FWD_STATES.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.get_mut(&session_id) {
                        Some(state) => {
                            if state.phase != ForwardPhase::Layers {
                                return format!("Not in layer phase (phase: {:?})", state.phase);
                            }
                            
                            let mut processed = 0;
                            
                            // Use cached execution if we have a cache
                            if let Some(ref mut cache) = state.kv_cache {
                                for _ in 0..count {
                                    if state.layer_idx >= model.config.n_layer {
                                        state.phase = ForwardPhase::FinalNorm;
                                        break;
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
                            }
                            
                            if state.phase == ForwardPhase::FinalNorm {
                                "Done".to_string()
                            } else {
                                format!("Session {}: Layer {}/{}", session_id, state.layer_idx, model.config.n_layer)
                            }
                        }
                        None => format!("Error: No forward state for session {}", session_id),
                    }
                })
            }
            None => "Error: No model".to_string(),
        }
    });
    result
}



/// Finish forward pass for a specific session
#[update]
fn finish_forward_session(session_id: u32) -> String {
    let result = MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                FWD_STATES.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.get_mut(&session_id) {
                        Some(state) => {
                            // Final norm and sample from the last position in hidden
                            let sample_len = state.new_len;
                            let next_token = model.final_norm_and_sample(&state.hidden, sample_len);
                            
                            // Add token to generation state
                            GEN_STATES.with(|gen| {
                                let mut gen_ref = gen.borrow_mut();
                                if let Some(gen_state) = gen_ref.get_mut(&session_id) {
                                    gen_state.tokens.push(next_token);
                                    gen_state.generated_count += 1;
                                }
                            });
                            
                            // Mark as done but keep cache
                            state.phase = ForwardPhase::Done;
                            
                            model.decode_token(next_token)
                        }
                        None => format!("Error: No forward state for session {}", session_id),
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
                FWD_STATES.with(|fwd| {
                    let mut fwd_ref = fwd.borrow_mut();
                    match fwd_ref.get_mut(&0) {
                        Some(state) => {
                            // Final norm and sample from the last position in hidden
                            // In cached mode, hidden only has new_len positions
                            let sample_len = state.new_len;  // Sample from position new_len-1
                            let next_token = model.final_norm_and_sample(&state.hidden, sample_len);
                            
                            
                            // Add token to generation state
                            GEN_STATES.with(|gen| {
                                let mut gen_ref = gen.borrow_mut();
                                if let Some(gen_state) = gen_ref.get_mut(&0) {
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
                GEN_STATES.with(|gen| {
                    let mut gen_ref = gen.borrow_mut();
                    match gen_ref.get_mut(&0) {
                        Some(gen_state) => {
                            // 1. Get KV Cache from ForwardState
                            let mut cache_opt = FWD_STATES.with(|fwd| {
                                let mut fwd_ref = fwd.borrow_mut();
                                fwd_ref.get_mut(&0).and_then(|s| s.kv_cache.take())
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
                                FWD_STATES.with(|fwd| {
                                    fwd.borrow_mut().insert(0, ForwardState {
                                        hidden: curr_hidden, // Not typically needed next time but good for consistency
                                        layer_idx: model.config.n_layer,
                                        seq_len: cache.cached_len, // Full length including new
                                        new_len: 0,
                                        phase: ForwardPhase::Done,
                                        kv_cache: Some(cache),
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

/// Start a new session with a specific ID
#[update]
fn start_session(session_id: u32, prompt: String, max_tokens: u32) -> String {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                let tokens = model.tokenize(&prompt);
                
                // Clear state for this session
                FWD_STATES.with(|fwd| {
                    fwd.borrow_mut().remove(&session_id);
                });
                
                GEN_STATES.with(|s| {
                    s.borrow_mut().insert(session_id, GenerationState {
                        tokens,
                        max_tokens: max_tokens as usize,
                        generated_count: 0,
                    });
                });
                format!("Session {}: Started", session_id)
            }
            None => "Error: No model".to_string(),
        }
    })
}

/// Generate next tokens for a batch of sessions
#[update]
fn generate_batch(session_ids: Vec<u32>) -> Vec<String> {
    MODEL.with(|m| {
        let model_ref = m.borrow();
        match model_ref.as_ref() {
            Some(model) => {
                // 1. Collect states and prepare batch
                let mut batch_hidden = Vec::new();
                let mut batch_caches = Vec::new(); // Will hold KVCache
                let mut active_ids = Vec::new();
                
                // We must take the states out to mutate them together
                FWD_STATES.with(|fwd| {
                    let mut fwd_map = fwd.borrow_mut();
                    
                    for &id in &session_ids {
                        // We need the cache. If it's in FWD_STATES, take it.
                        // If not, check if we have a GEN_STATE (and need to init cache? No, prompt must be done).
                        // Note: generate_next_token leaves cache in FWD_STATES.
                        
                        if let Some(mut state) = fwd_map.remove(&id) {
                            if let Some(cache) = state.kv_cache.take() {
                                // Get last token from GEN_STATES to embed
                                let last_token = GEN_STATES.with(|gen| {
                                    gen.borrow().get(&id).and_then(|g| g.tokens.last().copied()).unwrap_or(0)
                                });
                                
                                let hidden = model.embed_tokens(&[last_token]);
                                batch_hidden.extend_from_slice(&hidden);
                                batch_caches.push(cache);
                                active_ids.push(id);
                                
                                // Put "shell" state back or keep it out? Keeping it out is easier.
                                // We will reconstruct it later.
                            }
                        }
                    }
                });
                
                if active_ids.is_empty() {
                    return session_ids.iter().map(|_| "Error: No active state".to_string()).collect();
                }

                // 2. Run Batched Layers
                let mut curr_hidden = batch_hidden;
                let batch_size = active_ids.len();
                let new_len = 1;
                
                for layer_idx in 0..model.config.n_layer {
                    curr_hidden = model.run_layer_batched(
                        layer_idx,
                        &curr_hidden,
                        &mut batch_caches
                    );
                }
                
                // Update cache lengths (all proceeded by 1)
                for cache in &mut batch_caches {
                    cache.update_len(new_len);
                }
                
                // 3. Sample and Update
                let mut results = BTreeMap::new(); // Map ID -> Result string
                let n_embd = model.config.n_embd;
                
                for (i, &id) in active_ids.iter().enumerate() {
                    let start = i * n_embd;
                    let hidden_slice = &curr_hidden[start..start + n_embd];
                    
                    // Sample
                    let next_token = model.final_norm_and_sample(hidden_slice, 1);
                    
                    // Decode
                    let token_str = model.decode_token(next_token);
                    results.insert(id, token_str);
                    
                    // Update Gen State
                    GEN_STATES.with(|gen| {
                       if let Some(gs) = gen.borrow_mut().get_mut(&id) {
                           gs.tokens.push(next_token);
                           gs.generated_count += 1;
                       } 
                    });
                }
                
                // Actually, let's consume batch_caches
                for (i, cache) in batch_caches.into_iter().enumerate() {
                    let id = active_ids[i];
                    // Reconstruct state
                     FWD_STATES.with(|fwd| {
                        fwd.borrow_mut().insert(id, ForwardState {
                            hidden: Vec::new(), // optimizing output size
                            layer_idx: model.config.n_layer,
                            seq_len: cache.cached_len,
                            new_len: 0,
                            phase: ForwardPhase::Done,
                            kv_cache: Some(cache),
                        });
                    });
                }
                
                // 4. Return results strictly matching input order
                session_ids.iter().map(|id| {
                    results.get(id).cloned().unwrap_or_else(|| "Error".to_string())
                }).collect()
            }
            None => vec!["Error: No model".to_string(); session_ids.len()],
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
                GEN_STATES.with(|gen| {
                    let gen_ref = gen.borrow();
                    match gen_ref.get(&0) {
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
    GEN_STATES.with(|gen| {
        let gen_ref = gen.borrow();
        match gen_ref.get(&0) {
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
