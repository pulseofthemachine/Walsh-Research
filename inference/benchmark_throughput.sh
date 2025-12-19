#!/bin/bash

BATCH_SIZE=2
STEPS=20
SESSION_1_ID=1
SESSION_2_ID=2

# Setup
echo "Initializing Sessions..."
dfx canister call spinnet_backend start_session "(${SESSION_1_ID}:nat32, \"To be or not to \", 100:nat32)" > /dev/null
dfx canister call spinnet_backend start_forward_session "(${SESSION_1_ID}:nat32)" > /dev/null
dfx canister call spinnet_backend process_layers_session "(${SESSION_1_ID}:nat32, 100:nat32)" > /dev/null
dfx canister call spinnet_backend finish_forward_session "(${SESSION_1_ID}:nat32)" > /dev/null

dfx canister call spinnet_backend start_session "(${SESSION_2_ID}:nat32, \"The quick brown \", 100:nat32)" > /dev/null
dfx canister call spinnet_backend start_forward_session "(${SESSION_2_ID}:nat32)" > /dev/null
dfx canister call spinnet_backend process_layers_session "(${SESSION_2_ID}:nat32, 100:nat32)" > /dev/null
dfx canister call spinnet_backend finish_forward_session "(${SESSION_2_ID}:nat32)" > /dev/null

# Batch Generation Measurement
echo "Starting Batch Generation (Batch Size: $BATCH_SIZE, Steps: $STEPS)..."
start_time=$(date +%s.%N)

for ((i=1; i<=STEPS; i++)); do
    # Capture output to avoid terminal scroll lag affecting timing, but verify it works
    # Using > /dev/null for pure speed test of the engine + network overhead
    dfx canister call spinnet_backend generate_batch "(vec {${SESSION_1_ID}:nat32; ${SESSION_2_ID}:nat32})" > /dev/null
    echo -ne "Step $i/$STEPS\r"
done

end_time=$(date +%s.%N)
echo -e "\nDone."

# Calculation
duration=$(echo "$end_time - $start_time" | bc)
total_tokens=$(echo "$STEPS * $BATCH_SIZE" | bc)
throughput=$(echo "scale=2; $total_tokens / $duration" | bc)

echo "---------------------------------------------------"
echo "Total Tokens Generated: $total_tokens"
echo "Total Time: ${duration}s"
echo "Throughput: ${throughput} tok/sec"
echo "---------------------------------------------------"
echo "Baseline (Single Steam): ~0.51 tok/sec"
echo "Speedup Factor: $(echo "scale=2; $throughput / 0.51" | bc)x"
