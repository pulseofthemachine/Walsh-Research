#!/bin/bash

# Setup
echo "Initializing Session 1..."
dfx canister call spinnet_backend start_session '(1:nat32, "To be or not to ", 20:nat32)'
dfx canister call spinnet_backend start_forward_session '(1:nat32)'
echo "Processing Session 1 Layers..."
dfx canister call spinnet_backend process_layers_session '(1:nat32, 100:nat32)'
dfx canister call spinnet_backend finish_forward_session '(1:nat32)'

echo "Initializing Session 2..."
dfx canister call spinnet_backend start_session '(2:nat32, "The quick brown ", 20:nat32)'
dfx canister call spinnet_backend start_forward_session '(2:nat32)'
echo "Processing Session 2 Layers..."
dfx canister call spinnet_backend process_layers_session '(2:nat32, 100:nat32)'
dfx canister call spinnet_backend finish_forward_session '(2:nat32)'

# Batch Generation
echo "Starting Batch Generation..."
start=$(date +%s%N)
for i in {1..20}; do
    echo "Batch Step $i"
    dfx canister call spinnet_backend generate_batch '(vec {1:nat32; 2:nat32})'
done
end=$(date +%s%N)

# Results
echo "Session 1 Result:"
dfx canister call spinnet_backend get_result --query | grep -o '".*"'
# Need a way to get result for Session 2?
# get_result currently hardcodes session 0.
# I need get_result_session?
# Or start a dummy generation for session 0?
# Actually, I didn't add get_result_session.
# I can rely on the output of generate_batch (it returns vec text).
# But checking final aggregated string is easier.
# I'll rely on the output of generate_batch printing the tokens.
