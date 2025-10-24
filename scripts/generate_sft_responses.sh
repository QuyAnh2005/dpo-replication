#!/bin/bash

# Generate responses from the SFT model

python inference.py \
    --base_model "EleutherAI/pythia-2.8b" \
    --adapter "quyanh/pythia-2.8b-sft" \
    --dataset_name "Anthropic/hh-rlhf" \
    --prompt_name "chosen" \
    --split "test" \
    --run_name "sft_responses" \
    --output_dir "./inference_results" \
    --max_samples 5 \
    --seed 42

