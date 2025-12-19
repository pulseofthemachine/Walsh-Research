#!/bin/bash
# Ultra-fast inference using single-call generation
# Usage: ./run_fast.sh "Prompt text" [num_tokens]

PROMPT="$1"
TOKENS="${2:-20}"

if [ -z "$PROMPT" ]; then
    echo "Usage: $0 \"Prompt text\" [num_tokens]"
    exit 1
fi

echo "SpinNet Fast: Generating $TOKENS tokens..."
echo "Prompt: $PROMPT"

START_TIME=$(date +%s.%N)

# Single update call does everything!
# Returns plain text string
OUTPUT=$(dfx canister call spinnet_backend generate_full "(\"$PROMPT\", $TOKENS)")

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)
TPS=$(echo "scale=2; $TOKENS / $DURATION" | bc)

# Clean output (remove quotes/parentheses added by candid print)
CLEAN_OUTPUT=$(echo "$OUTPUT" | sed 's/^("//' | sed 's/")$//' | sed 's/\\n/\n/g')

echo "--------------------------------"
echo "$CLEAN_OUTPUT"
echo "--------------------------------"
echo "Done!"
echo "Generated $TOKENS tokens in $(printf "%.2f" $DURATION)s"
echo "Speed: $TPS tok/s"
