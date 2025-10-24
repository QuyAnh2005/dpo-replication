#!/bin/bash

# Example training script for modern DPO

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="dpo-replication"

# Login to Hugging Face and Weights & Biases using environment variables
huggingface-cli login --token=$HF_AUTH_TOKEN
wandb login $WANDB_API_KEY

# DPO training for HH dataset
echo "🎯 Starting Direct Preference Optimization (DPO)"
echo "============================================="
python train_dpo.py \
    --model_name_or_path "/workspace/dpo-replication/results/sft_model/merged_model" \
    --dataset_names "hh" \
    --output_dir "./results/dpo_model" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --report_to "wandb" \
    --run_name "dpo-pythia-2.8b-hh-lora" \
    --seed 42 \
    --beta 0.2 \
    --label_smoothing 0.0 \
    --loss_type "sigmoid" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --torch_dtype "float32" \
    --max_length 512 \
    --max_prompt_length 358 \
    --push_to_hub \
    --hub_model_id "quyanh/pythia-2.8b-dpo" \

echo "✅ DPO training completed!"
echo "📁 Model saved to: ./results/dpo_model"
echo "🤗 Model pushed to Hub: quyanh/pythia-2.8b-dpo"



####################################################################
# DPO training for SUM dataset
echo "🎯 Starting Direct Preference Optimization (DPO)"
echo "============================================="
python train_dpo.py \
    --model_name_or_path "CarperAI/openai_summarize_tldr_sft" \
    --dataset_names "sum" \
    --output_dir "./results/dpo_model" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --report_to "wandb" \
    --run_name "dpo-openai_summarize_tldr_sft-hh-lora" \
    --seed 42 \
    --beta 0.2 \
    --label_smoothing 0.0 \
    --loss_type "sigmoid" \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --torch_dtype "float16" \
    --max_length 2304 \
    --max_prompt_length 2048 \
    --push_to_hub \
    --hub_model_id "quyanh/openai_summarize_tldr_sft-dpo" \
    --max_train_samples 20000

echo "✅ DPO training completed!"
echo "📁 Model saved to: ./results/dpo_model"
echo "🤗 Model pushed to Hub: quyanh/openai_summarize_tldr_sft-dpo"
