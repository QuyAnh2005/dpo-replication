#!/bin/bash

# This script tracks the win rate of a DPO model at different temperatures (0.7 vs 1.0).

# --- Configuration ---
BASE_MODEL="quyanh/pythia-2.8b-sft-merged"
DPO_ADAPTER="quyanh/pythia-2.8b-dpo"
DATASET_NAME="Anthropic/hh-rlhf"
PROMPT_NAME="chosen"
SPLIT="test"
MAX_SAMPLES=150 # Keep this low for testing, increase for a full evaluation
# RESULTS=""

# --- DPO Checkpoints (sh-compatible) ---
DPO_CHECKPOINTS="\
100:0ce60cf5642b98bf5b064183ba3912143878ffe8 \
500:d462a1873fe45c73c693cbe72be75cd5433a1d8a \
1000:ecd8106004bed7097b63548a09c74da1f37bf798 \
1500:2e2475660279d1a3d0ec02492c3be7be3d75e9ed \
2000:7c7f6e845779427d70c433e5f972542c032c7499 \
2500:aa2b867b1585137e319244e9e2b8f3b0c3cdf2e7 \
3000:048a9ad84e82dd134660ed1f9cf04521bacf8524 \
3500:066ec451f1f1d47c6612a8ba1b9f512085328263 \
4000:9a35133b09e776179d474a34ac80e45f8a50e0ec \
4500:6b2b64874af2f3b39fdeeaa31b186e2ed47e2866 \
5000:174b737cae379e75758d9b6e66c246a30691c9b0 \
5500:4f71dfdde135be87356c866c6dbf4b9736c3e5e3 \
6000:976726a94c490d7d93d8e74ca06684fe00461970 \
6500:ac56c4f691887f34e2fd8cad079b5dd8ee7e7f81 \
7000:6d1579fc14e4acec05c46e4d2dfae127fbfae697 \
7500:7f7fb804d2edcc43f1658028ecb41e3779646d62 \
8000:88161a7b4d7d4226c33360dac754c31f916909ce
"

# --- Main Script ---
echo "Starting DPO temperature comparison evaluation..."

# Clean up previous results
rm -rf ./inference_results/*

# Loop through DPO versions, generate responses for each temperature, and evaluate
for checkpoint in $DPO_CHECKPOINTS; do
    step=$(echo "$checkpoint" | cut -d':' -f1)
    version=$(echo "$checkpoint" | cut -d':' -f2)
    echo "
--- Processing DPO step $step (Version: $version) ---"

    # --- Generate responses for Temperature 0.7 ---
    RUN_NAME_T07="dpo_step_${step}_temp_0.7"
    python inference.py \
        --base_model "$BASE_MODEL" \
        --adapter "$DPO_ADAPTER" \
        --adapter_version "$version" \
        --dataset_name "$DATASET_NAME" \
        --prompt_name "$PROMPT_NAME" \
        --split "$SPLIT" \
        --run_name "$RUN_NAME_T07" \
        --output_dir "./inference_results" \
        --max_samples "$MAX_SAMPLES" \
        --temperature 0.7 \
        --seed 42
    RESPONSES_T07=$(ls -t ./inference_results/${RUN_NAME_T07}_*.json | head -n 1)

    # --- Generate responses for Temperature 1.0 ---
    RUN_NAME_T10="dpo_step_${step}_temp_1.0"
    python inference.py \
        --base_model "$BASE_MODEL" \
        --adapter "$DPO_ADAPTER" \
        --adapter_version "$version" \
        --dataset_name "$DATASET_NAME" \
        --prompt_name "$PROMPT_NAME" \
        --split "$SPLIT" \
        --run_name "$RUN_NAME_T10" \
        --output_dir "./inference_results" \
        --max_samples "$MAX_SAMPLES" \
        --temperature 1.0 \
        --seed 42
    RESPONSES_T10=$(ls -t ./inference_results/${RUN_NAME_T10}_*.json | head -n 1)
done

