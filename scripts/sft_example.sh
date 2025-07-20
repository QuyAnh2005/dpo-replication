#!/bin/bash

# Example SFT training script
# This is the first step in the DPO pipeline

echo "📚 Starting Supervised Fine-Tuning (SFT)"
echo "========================================"

python train_sft.py \
    --model_name_or_path microsoft/DialoGPT-small \
    --dataset_names hh \
    --output_dir ./results/sft_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --report_to wandb \
    --run_name "sft-dialogpt-hh"

echo "✅ SFT training completed!"
echo "📁 Model saved to: ./results/sft_model/final_model"
echo "🎯 Ready for DPO training!"
