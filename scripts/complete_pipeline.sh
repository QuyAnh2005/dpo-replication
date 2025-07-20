#!/bin/bash

# Complete DPO Pipeline: SFT -> DPO -> Inference
# This script demonstrates the full pipeline from the original DPO paper

set -e  # Exit on any error

echo "🚀 Starting Complete DPO Pipeline"
echo "=================================="

# Configuration
MODEL_NAME="microsoft/DialoGPT-small"
DATASET_NAMES="hh"
SFT_OUTPUT_DIR="./results/sft_model"
DPO_OUTPUT_DIR="./results/dpo_model"
BATCH_SIZE=2
EPOCHS=1

echo "📋 Configuration:"
echo "  • Base model: $MODEL_NAME"
echo "  • Dataset: $DATASET_NAMES"
echo "  • SFT output: $SFT_OUTPUT_DIR"
echo "  • DPO output: $DPO_OUTPUT_DIR"
echo "  • Batch size: $BATCH_SIZE"
echo "  • Epochs: $EPOCHS"
echo ""

# Step 1: Supervised Fine-Tuning (SFT)
echo "📚 Step 1: Running Supervised Fine-Tuning (SFT)"
echo "==============================================="
python train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_names $DATASET_NAMES \
    --output_dir $SFT_OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "sft-${MODEL_NAME##*/}-${DATASET_NAMES}"

echo "✅ SFT completed! Model saved to: $SFT_OUTPUT_DIR"
echo ""

# Step 2: Direct Preference Optimization (DPO)
echo "🎯 Step 2: Running Direct Preference Optimization (DPO)"
echo "======================================================="
python train_dpo.py \
    --model_name_or_path $SFT_OUTPUT_DIR/final_model \
    --dataset_names $DATASET_NAMES \
    --output_dir $DPO_OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --max_length 512 \
    --max_prompt_length 256 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 200 \
    --report_to wandb \
    --run_name "dpo-${MODEL_NAME##*/}-${DATASET_NAMES}"

echo "✅ DPO completed! Model saved to: $DPO_OUTPUT_DIR"
echo ""

# Step 3: Model Evaluation
echo "📊 Step 3: Evaluating DPO Model"
echo "==============================="

# Check if vLLM is available for fast evaluation
if python -c "import vllm" 2>/dev/null; then
    echo "🚀 Using vLLM for fast evaluation..."
    python evaluate_vllm.py \
        --model_path $DPO_OUTPUT_DIR \
        --dataset_names $DATASET_NAMES \
        --output_dir ./results/evaluation_vllm \
        --max_examples 200 \
        --generate_samples \
        --num_samples 5
else
    echo "📊 Using standard evaluation..."
    python evaluate.py \
        --model_path $DPO_OUTPUT_DIR \
        --dataset_names $DATASET_NAMES \
        --output_dir ./results/evaluation \
        --batch_size $BATCH_SIZE \
        --max_length 512
fi

echo "✅ Evaluation completed!"
echo ""

# Step 4: Interactive Inference
echo "🤖 Step 4: Testing Interactive Inference"
echo "========================================"
echo "Starting interactive inference with the DPO model..."
echo "You can now chat with your trained model!"
echo ""

# Check if vLLM is available
if python -c "import vllm" 2>/dev/null; then
    echo "🚀 Using vLLM for fast inference..."
    python inference_vllm.py \
        --model_path $DPO_OUTPUT_DIR \
        --mode interactive \
        --temperature 0.7 \
        --max_tokens 256
else
    echo "📝 Using standard HuggingFace inference..."
    python inference.py \
        --model_path $DPO_OUTPUT_DIR \
        --mode interactive \
        --temperature 0.7 \
        --max_length 256
fi

echo ""
echo "🎉 Complete DPO Pipeline Finished!"
echo "=================================="
echo "Your models are ready:"
echo "  • SFT Model: $SFT_OUTPUT_DIR"
echo "  • DPO Model: $DPO_OUTPUT_DIR"
echo "  • Evaluation: ./results/evaluation"
echo ""
echo "Next steps:"
echo "  • Try different hyperparameters"
echo "  • Use larger models (e.g., Llama, Mistral)"
echo "  • Add your own datasets"
echo "  • Deploy with vLLM for production"
