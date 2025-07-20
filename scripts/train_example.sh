#!/bin/bash

# Example training script for modern DPO

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="modern-dpo"

# Basic training with CLI arguments
python train_dpo.py \
    --model_name_or_path "microsoft/DialoGPT-medium" \
    --dataset_names "hh" \
    --output_dir "./results/dpo-dialogpt-hh" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --report_to "wandb" \
    --run_name "dpo-dialogpt-hh-example" \
    --seed 42 \
    --beta 0.2 \
    --label_smoothing 0.0 \
    --loss_type "sigmoid" \
    --torch_dtype "float32" \
    --max_length 384 \
    --max_prompt_length 256
    # --use_lora \
    # --lora_r 16 \
    # --lora_alpha 32 \
    # --lora_dropout 0.1 \
    # --torch_dtype "float16" \
    # --max_length 512 \
    # --max_prompt_length 256
