#!/bin/bash
# Optimized inference script for SpinNet on IC
# Usage: ./run_inference.sh "Prompt text" [num_tokens]

PROMPT="$1"
TOKENS="${2:-20}"

if [ -z "$PROMPT" ]; then
    echo "Usage: $0 \"Prompt text\" [num_tokens]"
    exit 1
fi

echo "SpinNet: Generating $TOKENS tokens..."
echo "Prompt: $PROMPT"

START_TIME=$(date +%s.%N)

# Initialize generation
dfx canister call spinnet_backend start_generation "(\"$PROMPT\", $TOKENS)"

# Generation loop
for ((i=1; i<=TOKENS; i++)); do
    # 1. Start forward pass (Embeddings)
    dfx canister call spinnet_backend start_forward >/dev/null
    
    # 2. Process layers in batches (Optimization: 8 layers per call)
    # This processes all 8 layers in a single call for Tinyshakespeare model
    dfx canister call spinnet_backend process_layers '(8)' > /dev/null
    
    # 3. Finish forward pass (Norm + Sample)
    # This returns the generated token
    TOKEN=$(dfx canister call spinnet_backend finish_forward)
    
    # Extract just the token text
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
