#!/bin/bash
# Optimized inference script for SpinNet on IC (using generate_next_token)
# Usage: ./run_fast_inference.sh "Prompt text" [num_tokens]

PROMPT="$1"
TOKENS="${2:-20}"

if [ -z "$PROMPT" ]; then
    echo "Usage: $0 \"Prompt text\" [num_tokens]"
    exit 1
fi

echo "SpinNet: Generating $TOKENS tokens (Fast Mode)..."
echo "Prompt: $PROMPT"

START_TIME=$(date +%s.%N)

# Initialize generation
dfx canister call spinnet_backend start_generation "(\"$PROMPT\", $TOKENS)" >/dev/null

# --- Token 1: Prompt Processing (Must be chunked) ---
# 1. Start forward pass (Embeddings)
dfx canister call spinnet_backend start_forward >/dev/null

# 2. Process layers (8 layers)
dfx canister call spinnet_backend process_layers '(8)' > /dev/null

# 3. Finish forward pass (Norm + Sample)
TOKEN=$(dfx canister call spinnet_backend finish_forward)
CLEAN_TOKEN=$(echo "$TOKEN" | sed 's/("\|")//g' | sed 's/\\n/\n/g')
echo -n "$CLEAN_TOKEN"

# --- Tokens 2..N: Cached Generation (Single Call) ---
for ((i=2; i<=TOKENS; i++)); do
    TOKEN=$(dfx canister call spinnet_backend generate_next_token)
    CLEAN_TOKEN=$(echo "$TOKEN" | sed 's/("\|")//g' | sed 's/\\n/\n/g')
    echo -n "$CLEAN_TOKEN"
done

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)
TPS=$(echo "scale=2; $TOKENS / $DURATION" | bc)

echo "" # Newline
echo "Done!"
echo "Generated $TOKENS tokens in $(printf "%.2f" $DURATION)s"
echo "Speed: $TPS tok/s"
