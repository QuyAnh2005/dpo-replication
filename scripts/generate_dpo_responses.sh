#!/bin/bash

# Generate responses from the DPO model

python inference.py \
    --base_model "quyanh/pythia-2.8b-sft-merged" \
    --adapter "quyanh/pythia-2.8b-dpo" \
    --dataset_name "Anthropic/hh-rlhf" \
    --prompt_name "chosen" \
    --split "test" \
    --run_name "dpo_responses" \
    --output_dir "./inference_results" \
    --max_samples 5 \
    --seed 42

