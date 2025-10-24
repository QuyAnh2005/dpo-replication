#!/bin/bash

# Example SFT training script
# This is the first step in the DPO pipeline

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Login to Hugging Face and W&B using environment variables
huggingface-cli login --token=$HF_TOKEN
wandb login $WANDB_API_KEY

# For hh dataset with EleutherAI/pythia-2.8b model
echo "📚 Starting Supervised Fine-Tuning (SFT)"
echo "========================================"
accelerate launch train_sft.py \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --dataset_names hh \
    --max_length 512 \
    --output_dir ./results/sft_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --save_steps 100 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_strategy steps \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --report_to wandb \
    --run_name "sft-pythia-2.8b-hh-lora" \
    --seed 42 \
    --push_to_hub \
    --hub_model_id "quyanh/pythia-2.8b-sft" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1

echo "✅ SFT training completed!"
echo "📁 Model saved to: ./results/sft_model/final_model"
echo "🎯 Ready for DPO training!"
