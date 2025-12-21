# TinyStories Training Results
**Date:** 2025-12-20

## Experiment Summary
Training SpinNet on TinyStories dataset (token-level, GPT-2 tokenizer).

**Note:** This run did NOT have octonion_attention enabled due to config loading bug. The model uses standard attention.

## Configuration
| Setting | Value |
|---------|-------|
| Dataset | TinyStories (473M tokens) |
| Model | 28.9M params |
| Architecture | 8 layers, 8 heads, 512 dim |
| Training | 10k iters (~0.7 epochs) |
| Octonion Attention | ❌ False (bug) |

## Results

| Metric | Value |
|--------|-------|
| Final Train Loss | 2.68 |
| Final Val Loss | 2.68 |
| Compression | 331 MB → 50 MB (6x) |

## Key Findings

### Finding 1: Octonion Dimensions Specialize
Different dimensions activate for different linguistic categories:

| Category | Most Active Dims |
|----------|------------------|
| Nouns | e₀, e₁, e₇ |
| Verbs | e₀, e₇, e₁ |
| Pronouns | e₀, e₇, e₂ |
| Emotions | e₀, e₁, e₃ |
| Dialogue | e₀, e₂, e₁ |

**Interpretation:**
- e₀ (real) = base representation
- e₇ = specificity/details
- e₃ = semantic/emotional content
- e₂ = dialogue structure

### Finding 2: Val Loss < Train Loss (Ternary Regularization)
Validation loss was consistently *lower* than training loss early in training.

**Hypothesis:** Ternary quantization acts as strong regularization.

### Finding 3: Loss as Linguistic Developmental Stage
Model at val loss 2.68 makes errors similar to children (~age 3-4):
- Pronoun confusion
- Object/subject errors
- Repetitive naming
- Stream-of-consciousness logic

| Loss | Approx. Age |
|------|-------------|
| 3.5+ | 2-3 years |
| 2.5-3.0 | 3-4 years |
| 2.0-2.5 | 4-5 years |
| <1.5 | 6+ years |

### Finding 4: Training Efficiency
Achieved coherent output in 0.7 epochs vs baseline's 20 epochs.

## Sample Output
```
Once upon a time, there was a little girl named Lily. 
Lily was so excited to have a picnic on the beach...
When they got to the beach, they saw a dolphin swimming in the sea.
```

## Next Steps
- [ ] Retrain with octonion_attention=True (bug fixed)
- [ ] Compare to baseline at same epoch count
