"""
Fast inference script using vLLM for accelerated generation.
This provides much faster inference compared to standard HuggingFace transformers.
"""

import argparse
import json
import os
import time
from typing import List, Optional, Dict, Any
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_vllm_model(model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    """Load model with vLLM for fast inference."""
    print(f"Loading model with vLLM: {model_path}")
    
    # Load tokenizer to check chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize vLLM model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Tensor parallel size: {tensor_parallel_size}")
    print(f"✓ GPU memory utilization: {gpu_memory_utilization}")
    
    return llm, tokenizer


def create_sampling_params(
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 512,
    repetition_penalty: float = 1.1,
    stop_tokens: Optional[List[str]] = None
) -> SamplingParams:
    """Create sampling parameters for generation."""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        stop=stop_tokens or [],
    )


def format_prompt(prompt: str, tokenizer, system_message: Optional[str] = None) -> str:
    """Format prompt using chat template if available."""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback to simple format
        if system_message:
            return f"System: {system_message}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            return f"Human: {prompt}\n\nAssistant:"


def interactive_mode(llm: LLM, tokenizer, sampling_params: SamplingParams):
    """Run interactive chat mode."""
    print("\n🚀 Interactive Mode (type 'quit' to exit, 'clear' to clear history)")
    print("=" * 60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared!")
                continue
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append(f"Human: {user_input}")
            
            # Create prompt with history
            prompt = "\n\n".join(conversation_history) + "\n\nAssistant:"
            
            # Generate response
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            generation_time = time.time() - start_time
            
            response = outputs[0].outputs[0].text.strip()
            
            # Add response to history
            conversation_history.append(f"Assistant: {response}")
            
            # Display response with timing
            print(f"\n🤖 Assistant: {response}")
            print(f"⏱️  Generated in {generation_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_mode(llm: LLM, tokenizer, sampling_params: SamplingParams, input_file: str, output_file: str):
    """Run batch inference mode."""
    print(f"📂 Loading prompts from: {input_file}")
    
    # Load prompts
    with open(input_file, 'r') as f:
        if input_file.endswith('.jsonl'):
            prompts_data = [json.loads(line) for line in f]
            prompts = [item.get('prompt', item.get('text', str(item))) for item in prompts_data]
        else:
            prompts = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Loaded {len(prompts)} prompts")
    
    # Format prompts
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = format_prompt(prompt, tokenizer)
        formatted_prompts.append(formatted_prompt)
    
    # Generate responses
    print("🚀 Generating responses...")
    start_time = time.time()
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    total_time = time.time() - start_time
    
    # Process results
    results = []
    for i, output in enumerate(outputs):
        result = {
            "prompt": prompts[i],
            "response": output.outputs[0].text.strip(),
            "tokens_generated": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
        }
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print statistics
    total_tokens = sum(r['tokens_generated'] for r in results)
    avg_tokens_per_second = total_tokens / total_time
    
    print(f"✅ Batch inference completed!")
    print(f"📊 Statistics:")
    print(f"   • Total prompts: {len(prompts)}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Total tokens generated: {total_tokens}")
    print(f"   • Average tokens/second: {avg_tokens_per_second:.2f}")
    print(f"   • Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"📁 Results saved to: {output_file}")


def benchmark_mode(llm: LLM, tokenizer, sampling_params: SamplingParams, num_prompts: int = 10):
    """Run benchmark mode to test performance."""
    print(f"🏃 Running benchmark with {num_prompts} prompts...")
    
    # Create test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "How do you make a sandwich?",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
        "What is machine learning?",
        "How does the internet work?",
        "What are the causes of climate change?",
        "Explain the theory of relativity.",
    ]
    
    # Repeat prompts to reach desired number
    prompts = (test_prompts * ((num_prompts // len(test_prompts)) + 1))[:num_prompts]
    
    # Format prompts
    formatted_prompts = [format_prompt(prompt, tokenizer) for prompt in prompts]
    
    # Benchmark
    print("⏱️  Starting benchmark...")
    start_time = time.time()
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / total_time
    prompts_per_second = len(prompts) / total_time
    
    print(f"\n📊 Benchmark Results:")
    print(f"   • Prompts processed: {len(prompts)}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Total tokens generated: {total_tokens}")
    print(f"   • Tokens per second: {tokens_per_second:.2f}")
    print(f"   • Prompts per second: {prompts_per_second:.2f}")
    print(f"   • Average tokens per prompt: {total_tokens/len(prompts):.1f}")
    print(f"   • Average time per prompt: {total_time/len(prompts):.3f}s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fast inference with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch", "benchmark"], 
                       default="interactive", help="Inference mode")
    parser.add_argument("--input_file", type=str, help="Input file for batch mode")
    parser.add_argument("--output_file", type=str, help="Output file for batch mode")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts for benchmark")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "batch" and (not args.input_file or not args.output_file):
        parser.error("Batch mode requires --input_file and --output_file")
    
    # Load model
    try:
        llm, tokenizer = load_vllm_model(
            args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure vLLM is installed: pip install vllm")
        return
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    
    print(f"🎯 Sampling parameters:")
    print(f"   • Temperature: {args.temperature}")
    print(f"   • Top-p: {args.top_p}")
    print(f"   • Top-k: {args.top_k}")
    print(f"   • Max tokens: {args.max_tokens}")
    print(f"   • Repetition penalty: {args.repetition_penalty}")
    
    # Run inference based on mode
    if args.mode == "interactive":
        interactive_mode(llm, tokenizer, sampling_params)
    elif args.mode == "batch":
        batch_mode(llm, tokenizer, sampling_params, args.input_file, args.output_file)
    elif args.mode == "benchmark":
        benchmark_mode(llm, tokenizer, sampling_params, args.num_prompts)


if __name__ == "__main__":
    main()
