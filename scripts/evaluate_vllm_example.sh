#!/bin/bash

# Fast evaluation using vLLM
# This provides 10-20x speedup over standard HuggingFace evaluation

echo "🚀 Fast Evaluation with vLLM"
echo "============================"

MODEL_PATH="./results/dpo_model"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "💡 Please train a model first or update MODEL_PATH"
    exit 1
fi

echo "📋 Evaluation Options:"
echo "  1. Quick evaluation (100 examples)"
echo "  2. Full evaluation (all examples)"
echo "  3. Evaluation with sample generation"
echo "  4. Multi-GPU evaluation"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "⚡ Running quick evaluation..."
        python evaluate_vllm.py \
            --model_path $MODEL_PATH \
            --dataset_names hh \
            --max_examples 100 \
            --output_dir ./eval_results_quick
        ;;
    2)
        echo "📊 Running full evaluation..."
        python evaluate_vllm.py \
            --model_path $MODEL_PATH \
            --dataset_names hh shp \
            --output_dir ./eval_results_full
        ;;
    3)
        echo "📝 Running evaluation with sample generation..."
        python evaluate_vllm.py \
            --model_path $MODEL_PATH \
            --dataset_names hh \
            --max_examples 500 \
            --generate_samples \
            --num_samples 10 \
            --output_dir ./eval_results_samples
        ;;
    4)
        echo "🔥 Running multi-GPU evaluation..."
        python evaluate_vllm.py \
            --model_path $MODEL_PATH \
            --dataset_names hh shp \
            --tensor_parallel_size 2 \
            --gpu_memory_utilization 0.8 \
            --output_dir ./eval_results_multi_gpu
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎯 vLLM Evaluation Benefits:"
echo "  • 10-20x faster than standard evaluation"
echo "  • Batch processing for efficiency"
echo "  • Multi-GPU support for large models"
echo "  • Detailed scoring with log probabilities"
echo "  • Sample generation for qualitative analysis"
