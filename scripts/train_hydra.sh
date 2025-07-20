#!/bin/bash

# Example training script using Hydra configuration

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="modern-dpo"

# Training with Hydra config
python train_dpo.py \
    --config-path config \
    --config-name dpo_config \
    exp_name="dpo-hydra-example" \
    model.model_name_or_path="microsoft/DialoGPT-medium" \
    data.dataset_names=["hh"] \
    training.output_dir="./results/dpo-hydra-hh" \
    training.num_train_epochs=1 \
    training.per_device_train_batch_size=4 \
    training.learning_rate=5e-7 \
    training.beta=0.1 \
    wandb.enabled=true
