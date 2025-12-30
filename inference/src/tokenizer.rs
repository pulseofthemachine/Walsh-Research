//! GPT-2 Compatible Tokenizer
//!
//! Provides flexible tokenization that auto-detects between:
//! - Char-level (vocab_size <= 256): Simple character mapping
//! - GPT-2 (vocab_size > 256): Full BPE tokenization

use std::collections::HashMap;

/// GPT-2 vocab as base64-encoded token bytes (one per line)
const GPT2_VOCAB_B64: &str = include_str!("../gpt2_tokens_b64.txt");

/// Tokenizer that auto-detects mode based on vocab size
pub enum Tokenizer {
    CharLevel {
        stoi: [u8; 256],
        itos: [char; 256],
    },
    Gpt2 {
        /// id -> decoded string
        decoder: Vec<String>,
        /// string -> id (for encoding)
        encoder: HashMap<String, usize>,
    },
}

impl Tokenizer {
    /// Create a new tokenizer based on vocab size
    pub fn new(vocab_size: usize) -> Self {
        if vocab_size <= 256 {
            // Char-level tokenizer
            let charset = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            let charset_bytes: Vec<char> = charset.chars().collect();
            
            let mut stoi = [0u8; 256];
            let mut itos = [' '; 256];
            
            for (i, &c) in charset_bytes.iter().enumerate().take(vocab_size.min(256)) {
                stoi[c as usize % 256] = i as u8;
                itos[i] = c;
            }
            
            Tokenizer::CharLevel { stoi, itos }
        } else {
            // GPT-2 tokenizer
            let mut decoder = Vec::with_capacity(vocab_size);
            let mut encoder = HashMap::new();
            
            // Parse base64-encoded vocab
            for (i, line) in GPT2_VOCAB_B64.lines().enumerate() {
                if i >= vocab_size {
                    break;
                }
                let token_str = if line.is_empty() {
                    String::new()
                } else {
                    // Decode base64 to bytes, then to UTF-8
                    match base64_decode(line) {
                        Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                        Err(_) => String::new(),
                    }
                };
                encoder.insert(token_str.clone(), i);
                decoder.push(token_str);
            }
            
            // Fill remaining slots if vocab_size > 50257
            while decoder.len() < vocab_size {
                decoder.push(String::new());
            }
            
            Tokenizer::Gpt2 { decoder, encoder }
        }
    }
    
    /// Tokenize a string to token IDs
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        match self {
            Tokenizer::CharLevel { stoi, .. } => {
                text.chars()
                    .map(|c| stoi[c as usize % 256] as usize)
                    .collect()
            }
            Tokenizer::Gpt2 { encoder, decoder } => {
                // Simple greedy BPE: try longest matching tokens
                let mut tokens = Vec::new();
                let mut remaining = text;
                
                while !remaining.is_empty() {
                    // Try to find longest matching token
                    let mut found = false;
                    let max_len = remaining.len().min(50); // Max token length
                    
                    for len in (1..=max_len).rev() {
                        let candidate = &remaining[..len.min(remaining.len())];
                        if let Some(&id) = encoder.get(candidate) {
                            tokens.push(id);
                            remaining = &remaining[len..];
                            found = true;
                            break;
                        }
                    }
                    
                    if !found {
                        // Fallback: encode as single byte token (byte fallback)
                        let b = remaining.bytes().next().unwrap();
                        // GPT-2 uses tokens 0-255 for byte fallback
                        if b < 128 {
                            tokens.push(b as usize);
                        } else {
                            // Non-ASCII - skip or use special token
                            tokens.push(0);
                        }
                        remaining = &remaining[1..];
                    }
                }
                tokens
            }
        }
    }
    
    /// Decode a single token to string
    pub fn decode_token(&self, token: usize) -> String {
        match self {
            Tokenizer::CharLevel { itos, .. } => {
                itos[token % 256].to_string()
            }
            Tokenizer::Gpt2 { decoder, .. } => {
                decoder.get(token).cloned().unwrap_or_default()
            }
        }
    }
    
    /// Decode multiple tokens to string
    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        match self {
            Tokenizer::CharLevel { itos, .. } => {
                tokens.iter().map(|&t| itos[t % 256]).collect()
            }
            Tokenizer::Gpt2 { decoder, .. } => {
                tokens.iter()
                    .map(|&t| decoder.get(t).cloned().unwrap_or_default())
                    .collect()
            }
        }
    }

    /// Return debug info about tokenizer state
    pub fn debug_info(&self) -> String {
        match self {
            Tokenizer::CharLevel { .. } => "CharLevel".to_string(),
            Tokenizer::Gpt2 { decoder, encoder } => {
                format!("Gpt2(decoder_len={}, encoder_len={})", decoder.len(), encoder.len())
            }
        }
    }
}

/// Simple base64 decoder (avoid external dependency)
fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    const DECODE_TABLE: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
        -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
        -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    ];
    
    let bytes: Vec<u8> = input.bytes().collect();
    let mut output = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    
    for b in bytes {
        if b == b'=' {
            break;
        }
        if b >= 128 {
            return Err(());
        }
        let val = DECODE_TABLE[b as usize];
        if val < 0 {
            continue; // Skip whitespace
        }
        buffer = (buffer << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xFF) as u8);
        }
    }
    
    Ok(output)
}
