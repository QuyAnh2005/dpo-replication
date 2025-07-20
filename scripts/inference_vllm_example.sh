#!/bin/bash

# Example vLLM inference scripts

echo "🚀 vLLM Inference Examples"
echo "========================="

MODEL_PATH="./results/dpo_model"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "💡 Please train a model first or update MODEL_PATH"
    exit 1
fi

echo "📋 Available modes:"
echo "  1. Interactive chat"
echo "  2. Batch processing"
echo "  3. Performance benchmark"
echo ""

read -p "Choose mode (1-3): " choice

case $choice in
    1)
        echo "🤖 Starting interactive chat..."
        python inference_vllm.py \
            --model_path $MODEL_PATH \
            --mode interactive \
            --temperature 0.7 \
            --top_p 0.9 \
            --max_tokens 512 \
            --tensor_parallel_size 1
        ;;
    2)
        echo "📂 Starting batch processing..."
        
        # Create sample input file if it doesn't exist
        if [ ! -f "sample_prompts.txt" ]; then
            echo "Creating sample prompts file..."
            cat > sample_prompts.txt << EOF
What is the capital of France?
Explain quantum computing in simple terms.
How do you make a sandwich?
What are the benefits of exercise?
Describe the process of photosynthesis.
EOF
        fi
        
        python inference_vllm.py \
            --model_path $MODEL_PATH \
            --mode batch \
            --input_file sample_prompts.txt \
            --output_file batch_results.jsonl \
            --temperature 0.7 \
            --max_tokens 256 \
            --tensor_parallel_size 1
        
        echo "✅ Results saved to: batch_results.jsonl"
        ;;
    3)
        echo "⏱️ Running performance benchmark..."
        python inference_vllm.py \
            --model_path $MODEL_PATH \
            --mode benchmark \
            --num_prompts 20 \
            --temperature 0.7 \
            --max_tokens 256 \
            --tensor_parallel_size 1
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎯 vLLM Performance Tips:"
echo "  • Use --tensor_parallel_size > 1 for multi-GPU"
echo "  • Adjust --gpu_memory_utilization (default: 0.9)"
echo "  • Use larger batch sizes for better throughput"
echo "  • Consider quantization for memory efficiency"
