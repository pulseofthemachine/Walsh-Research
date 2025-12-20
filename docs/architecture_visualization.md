# SpinNet Architecture Visualization

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOKEN ID (integer)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DENSE EMBEDDING (64.4M params)                       │
│                     nn.Embedding(50304, 1280)                            │
│                     [One learned vector per token]                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   1280-dim Continuous Vector   │
                    │                               │
                    │  ╔══╦══╦══╦══╦══╦══╦══╦══╗   │
                    │  ║🔴║🟠║🟡║🟢║🔵║🟣║⚪║⚫║   │
                    │  ║x₀║x₁║x₂║x₃║x₄║x₅║x₆║x₇║   │
                    │  ╚══╩══╩══╩══╩══╩══╩══╩══╝   │
                    │     8 × 160 = 1280 dims       │
                    └───────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
    ╔═══════════╗            ╔═══════════╗            ╔═══════════╗
    ║  LAYER 1  ║            ║  LAYER 2  ║    ...     ║ LAYER 24  ║
    ║ (2.46M)   ║     →      ║ (2.46M)   ║     →      ║ (2.46M)   ║
    ╚═══════════╝            ╚═══════════╝            ╚═══════════╝
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DENSE OUTPUT (tied with embedding)                 │
│                       nn.Linear(1280, 50304)                             │
│                       [Logit for each possible next token]               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    50,304 LOGITS → SOFTMAX → TOKEN                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Inside an Octonion Transformer Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          LLAMABLOCK (2.46M params)                         │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          INPUT: x (1280-dim)                          │  │
│  │                  ╔══╦══╦══╦══╦══╦══╦══╦══╗                           │  │
│  │                  ║x₀║x₁║x₂║x₃║x₄║x₅║x₆║x₇║                           │  │
│  │                  ╚══╩══╩══╩══╩══╩══╩══╩══╝                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     RMSNorm (1280 params)                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│          ┌─────────────────────────┼─────────────────────────┐             │
│          ▼                         ▼                         ▼             │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│  │ OCTO-LINEAR  │          │ OCTO-LINEAR  │          │ OCTO-LINEAR  │      │
│  │     Wq       │          │     Wk       │          │     Wv       │      │
│  │  (205K)      │          │  (205K)      │          │  (205K)      │      │
│  └──────────────┘          └──────────────┘          └──────────────┘      │
│          │                         │                         │             │
│          ▼                         ▼                         ▼             │
│       Q (1280)                  K (1280)                  V (1280)         │
│                                    │                                       │
│                   ┌────────────────┼────────────────┐                      │
│                   ▼                ▼                ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              MULTI-HEAD ATTENTION (20 heads × 64 dim)                 │  │
│  │                                                                       │  │
│  │   Score = softmax(Q · Kᵀ / √64) · V                                  │  │
│  │   [Standard attention - this is where order matters!]                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     OCTO-LINEAR Wo (205K params)                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    + ◄───── RESIDUAL CONNECTION            │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     RMSNorm (1,280 params)                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│          ┌─────────────────────────┴─────────────────────────┐             │
│          ▼                                                   ▼             │
│  ┌──────────────────┐                              ┌──────────────────┐    │
│  │   OCTO-LINEAR    │                              │   OCTO-LINEAR    │    │
│  │   gate_proj      │                              │   up_proj        │    │
│  │   (547K)         │                              │   (547K)         │    │
│  │ 1280 → 3416      │                              │ 1280 → 3416      │    │
│  └──────────────────┘                              └──────────────────┘    │
│          │                                                   │             │
│          ▼                                                   ▼             │
│      SiLU(gate)                            ×                 up            │
│          │                                                   │             │
│          └─────────────────► ELEMENT-WISE ◄──────────────────┘             │
│                              MULTIPLY                                      │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                  OCTO-LINEAR down_proj (547K params)                  │  │
│  │                         3416 → 1280                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    + ◄───── RESIDUAL CONNECTION            │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       OUTPUT: x' (1280-dim)                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## The Octonion-Linear Magic

### Ternary Weights: Only 3 Possible Values
```
    WEIGHT VALUES
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          │           │           │
         -1           0          +1
          ●           ●           ●
       INHIBIT     IGNORE     EXCITE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Only 3 states per weight → 1.58 bits per parameter
    (vs 32 bits for float32)
```

### The Cayley-Dickson Multiplication Structure
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    OCTONION MULTIPLICATION TABLE                          ║
║                    (How 8 inputs combine to 8 outputs)                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   y₀ = x₀⊗W₀ - x₁⊗W₁ - x₂⊗W₂ - x₃⊗W₃ - x₄⊗W₄ - x₅⊗W₅ - x₆⊗W₆ - x₇⊗W₇   ║
║   y₁ = x₀⊗W₁ + x₁⊗W₀ + x₂⊗W₃ - x₃⊗W₂ + x₄⊗W₅ - x₅⊗W₄ - x₆⊗W₇ + x₇⊗W₆   ║
║   y₂ = x₀⊗W₂ - x₁⊗W₃ + x₂⊗W₀ + x₃⊗W₁ + x₄⊗W₆ + x₅⊗W₇ - x₆⊗W₄ - x₇⊗W₅   ║
║   y₃ = x₀⊗W₃ + x₁⊗W₂ - x₂⊗W₁ + x₃⊗W₀ + x₄⊗W₇ - x₅⊗W₆ + x₆⊗W₅ - x₇⊗W₄   ║
║   y₄ = x₀⊗W₄ - x₁⊗W₅ - x₂⊗W₆ - x₃⊗W₇ + x₄⊗W₀ + x₅⊗W₁ + x₆⊗W₂ + x₇⊗W₃   ║
║   y₅ = x₀⊗W₅ + x₁⊗W₄ - x₂⊗W₇ + x₃⊗W₆ - x₄⊗W₁ + x₅⊗W₀ - x₆⊗W₃ + x₇⊗W₂   ║
║   y₆ = x₀⊗W₆ + x₁⊗W₇ + x₂⊗W₄ - x₃⊗W₅ - x₄⊗W₂ + x₅⊗W₃ + x₆⊗W₀ - x₇⊗W₁   ║
║   y₇ = x₀⊗W₇ - x₁⊗W₆ + x₂⊗W₅ + x₃⊗W₄ - x₄⊗W₃ - x₅⊗W₂ + x₆⊗W₁ + x₇⊗W₀   ║
║                                                                           ║
║   WHERE: ⊗ = matrix multiply, +/- = accumulation sign                     ║
║                                                                           ║
║   KEY INSIGHT:                                                            ║
║   • Each Wₖ is a [160 × 160] ternary matrix                              ║
║   • Each Wₖ is reused 8 times with different signs                        ║
║   • This gives 8× parameter efficiency                                    ║
║   • Non-commutative: y(a,b) ≠ y(b,a)                                      ║
║   • Non-associative: y((a,b),c) ≠ y(a,(b,c))                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Visualizing the Sign Pattern
```
                    INPUT COMPONENT (x₀ to x₇)
                    
                    x₀   x₁   x₂   x₃   x₄   x₅   x₆   x₇
                    ─────────────────────────────────────
              y₀ │   +    -    -    -    -    -    -    -  │
              y₁ │   +    +    +    -    +    -    -    +  │
   OUTPUT     y₂ │   +    -    +    +    +    +    -    -  │
              y₃ │   +    +    -    +    +    -    +    -  │
              y₄ │   +    -    -    -    +    +    +    +  │
              y₅ │   +    +    -    +    -    +    -    +  │
              y₆ │   +    +    +    -    -    +    +    -  │
              y₇ │   +    -    +    +    -    -    +    +  │
                    ─────────────────────────────────────
                    
    + = Add contribution      - = Subtract contribution
    
    This fixed pattern encodes the algebraic structure of octonions,
    the largest normed division algebra with non-commutativity and
    non-associativity—properties that mirror natural language.
```

---

## Parameter Summary

```
┌───────────────────────────────────────────────────────────────────┐
│                    SPINNET "SCHOLAR" (123.5M)                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░  52%       │
│   EMBEDDING: 64.4M (Dense, tied with output)                      │
│                                                                   │
│   ░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  48%       │
│   LAYERS: 59.1M (24× Octonion Blocks, 8× compressed)              │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   EQUIVALENT DENSE MODEL: 537M parameters                         │
│   COMPRESSION RATIO: 4.3×                                         │
│   WEIGHT BITS: 1.58 (ternary) vs 32 (float)                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## The Cayley-Dickson Ladder: Why Octonions?

```
            REAL NUMBERS (ℝ)
                  │
                  │ Double dimensions, lose nothing
                  ▼
            COMPLEX NUMBERS (ℂ)
                  │
                  │ 
                  │ 
                  ▼
            QUATERNIONS (ℍ) ──────► Lose: COMMUTATIVITY (ab ≠ ba)
                  │
                  │ Double dimensions
                  ▼
    ┌─────► OCTONIONS (𝕆) ──────────► Lose: ASSOCIATIVITY ((ab)c ≠ a(bc))
    │             │
    │             │ Double dimensions
    │             ▼
    │       SEDENIONS (𝕊) ──────────► Lose: DIVISION (zero divisors!)
    │                                        ✗ Numerically unstable
    │
    └─── SWEET SPOT: Maximum algebraic richness
                     while retaining division (numerical stability)
                     
    Language is non-commutative: "dog bites man" ≠ "man bites dog"
    Language is non-associative: "(old man) river" ≠ "old (man river)"
    
    Octonions naturally encode these properties!
```
