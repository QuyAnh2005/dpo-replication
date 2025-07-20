"""
Inference script for the trained DPO model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
from typing import List, Dict


def load_model_and_tokenizer(model_path: str, use_lora: bool = False):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if use_lora:
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generate a response for the given prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    
    return response


def interactive_chat(model, tokenizer):
    """Interactive chat interface."""
    print("DPO Model Chat Interface")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_input = input("\nHuman: ")
        if user_input.lower() == 'quit':
            break
        
        # Format prompt
        prompt = f"Human: {user_input}\n\nAssistant:"
        
        # Generate response
        response = generate_response(model, tokenizer, prompt)
        
        print(f"Assistant: {response}")


def batch_inference(model, tokenizer, input_file: str, output_file: str):
    """Run batch inference on a file of prompts."""
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="DPO Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model uses LoRA")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="interactive", help="Inference mode")
    parser.add_argument("--input_file", type=str, help="Input file for batch inference")
    parser.add_argument("--output_file", type=str, help="Output file for batch inference")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_lora)
    print("Model loaded successfully!")
    
    if args.mode == "interactive":
        interactive_chat(model, tokenizer)
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file are required for batch mode")
            return
        batch_inference(model, tokenizer, args.input_file, args.output_file)


if __name__ == "__main__":
    main()
