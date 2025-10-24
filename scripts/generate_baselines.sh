#!/bin/bash

# Configuration
TEMPERATURES=(0.25 0.5 0.75 1.0)
MAX_SAMPLES=150
OUTPUT_DIR="./inference_baselines"
DATASET_NAME="Anthropic/hh-rlhf"
PROMPT_NAME="chosen"
SPLIT="test"
SEED=42

# Model configurations
declare -A MODELS=(
    ["base"]="EleutherAI/pythia-2.8b"
    ["sft"]="quyanh/pythia-2.8b-sft-merged"
    ["dpo"]="quyanh/pythia-2.8b-sft-merged"
)

declare -A ADAPTERS=(
    ["base"]=""
    ["sft"]=""
    ["dpo"]="quyanh/pythia-2.8b-dpo"
)

# Function to run inference
run_inference() {
    local model_type=$1
    local temperature=$2
    local run_name="pythia-2.8b-${model_type}-${temperature}"
    
    echo "Running inference: ${run_name}"
    
    local cmd="python inference.py \
        --base_model ${MODELS[$model_type]} \
        --dataset_name $DATASET_NAME \
        --prompt_name $PROMPT_NAME \
        --split $SPLIT \
        --run_name $run_name \
        --output_dir $OUTPUT_DIR \
        --max_samples $MAX_SAMPLES \
        --temperature $temperature \
        --seed $SEED"
    
    # Add adapter if specified
    if [ -n "${ADAPTERS[$model_type]}" ]; then
        cmd="$cmd --adapter ${ADAPTERS[$model_type]}"
    fi
    
    eval $cmd
}

# Main execution
for temperature in "${TEMPERATURES[@]}"; do
    echo "=========================================="
    echo "Temperature: $temperature"
    echo "=========================================="
    
    # Run all model types
    for model_type in base sft dpo; do
        run_inference "$model_type" "$temperature"
    done
done

echo "All inference runs completed!"