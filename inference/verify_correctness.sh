#!/bin/bash

SESSION_ID=0

# Setup Session 0
echo "Initializing Session 0..."
# Using the exact prompt from previous successful runs
dfx canister call spinnet_backend start_session "(${SESSION_ID}:nat32, \"To be or not to \", 20:nat32)"
dfx canister call spinnet_backend start_forward_session "(${SESSION_ID}:nat32)"
dfx canister call spinnet_backend process_layers_session "(${SESSION_ID}:nat32, 100:nat32)"
dfx canister call spinnet_backend finish_forward_session "(${SESSION_ID}:nat32)"

# Generate via Batch API (Size 1)
echo "Generating..."
for i in {1..12}; do
    dfx canister call spinnet_backend generate_batch "(vec {${SESSION_ID}:nat32})" > /dev/null
done

# Check Result
echo "Final Result:"
dfx canister call spinnet_backend get_result --query
