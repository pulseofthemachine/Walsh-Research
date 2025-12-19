#!/bin/bash
# Verify single-user functionality with generate_n_tokens
# Usage: ./verify_single_user.sh "Prompt" [N]

PROMPT="${1:-To be or not to be}"
N="${2:-50}"

echo "Testing Single-User Mode (Singleton Refactor)"
echo "Prompt: $PROMPT"

# 1. Start Generation
echo "Starting generation..."
dfx canister call spinnet_backend start_generation "(\"$PROMPT\", $((N+10)))"

# 2. Start Forward (Prompt)
echo "Start forward..."
dfx canister call spinnet_backend start_forward

# 3. Process Layers (Prompt)
echo "Processing prompt layers..."
STATUS="Layer"
while [[ "$STATUS" != *"Done"* ]]; do
  STATUS=$(dfx canister call spinnet_backend process_layers '(8)')
  echo "Status: $STATUS"
  
  if [[ "$STATUS" == *"Error"* ]]; then
      echo "Error in processing layers!"
      exit 1
  fi
done

# 4. Finish Forward (Sample first token)
echo "Finish forward (First token)..."
TOKEN=$(dfx canister call spinnet_backend finish_forward)
echo "Token 1: $TOKEN"

# 5. Generate N Tokens (Burst)
echo "Generating $N tokens in burst..."
START=$(date +%s.%N)
BURST=$(dfx canister call spinnet_backend generate_n_tokens "($N)")
END=$(date +%s.%N)

# Clean output
CLEAN_BURST=$(echo "$BURST" | sed 's/("\|")//g' | sed 's/\\n/\n/g')
echo "Burst Output: $CLEAN_BURST"

DURATION=$(echo "$END - $START" | bc)
TPS=$(echo "scale=2; $N / $DURATION" | bc)

echo ""
echo "Burst Speed: $TPS tok/s (N=$N)"
