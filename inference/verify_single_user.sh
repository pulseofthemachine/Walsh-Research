#!/bin/bash
# Verify single-user functionality with generate_n_tokens
# Updated for TinyStories (GPT-2 tokenizer) model
# Usage: ./verify_single_user.sh ["Prompt"] [N]

PROMPT="${1:-Once upon a time}"
N="${2:-50}"

echo "Testing Single-User Mode (TinyStories GPT-2)"
echo "Prompt: $PROMPT"
echo "Target: $N tokens"
echo ""

# 1. Start Generation
echo "Starting generation..."
START_RESULT=$(dfx canister call walsh_backend start_generation "(\"$PROMPT\", $((N+10)))")
echo "$START_RESULT"

# 2. Start Forward (Prompt)
echo "Start forward..."
dfx canister call walsh_backend start_forward

# 3. Process Layers (Prompt) - loop until done
echo "Processing prompt layers..."
STATUS="Layer"
while [[ "$STATUS" != *"Done"* ]]; do
  STATUS=$(dfx canister call walsh_backend process_layers '(8)')
  echo "  $STATUS"
  
  if [[ "$STATUS" == *"Error"* ]]; then
      echo "Error in processing layers!"
      exit 1
  fi
done

# 4. Finish Forward (Sample first token)
echo "Finish forward (First token)..."
FIRST_TOKEN=$(dfx canister call walsh_backend finish_forward --output json | jq -r '.')
echo "Token 1: $FIRST_TOKEN"

# 5. Generate N Tokens with Adaptive Looping
echo ""
echo "Generating $N tokens (adaptive chunking)..."
TOTAL_OUTPUT="$FIRST_TOKEN"
GENERATED=1
BURST_START=$(date +%s.%N)

while [ $GENERATED -lt $N ]; do
    REMAINING=$((N - GENERATED))
    
    # Request remaining tokens (will get partial if budget exceeded)
    RAW_JSON=$(dfx canister call walsh_backend generate_n_tokens "($REMAINING)" --output json)
    
    # Extract just the text using jq
    CHUNK=$(echo "$RAW_JSON" | jq -r '.')
    CHUNK_LEN=${#CHUNK}
    
    if [ $CHUNK_LEN -eq 0 ]; then
        echo "  [No tokens returned, breaking]"
        break
    fi
    
    TOTAL_OUTPUT+="$CHUNK"
    # For character-level model: 1 char = 1 token
    TOKEN_ESTIMATE=$CHUNK_LEN
    GENERATED=$((GENERATED + TOKEN_ESTIMATE))
    echo "  Generated $TOKEN_ESTIMATE tokens (total: $GENERATED/$N)"
done

BURST_END=$(date +%s.%N)

# Calculate stats
DURATION=$(echo "$BURST_END - $BURST_START" | bc)
TPS=$(echo "scale=2; ($GENERATED - 1) / $DURATION" | bc 2>/dev/null || echo "N/A")

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "OUTPUT (~$GENERATED tokens):"
echo "═══════════════════════════════════════════════════════════════"
echo "$TOTAL_OUTPUT"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STATS:"
echo "  Estimated tokens: $GENERATED"
echo "  Duration: ${DURATION}s"
echo "  Speed: $TPS tok/s"
echo "═══════════════════════════════════════════════════════════════"
