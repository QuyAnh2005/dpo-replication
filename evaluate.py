"""
Evaluation script for DPO models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score

from dataset_utils import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, use_lora: bool = False):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
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


def get_log_probability(model, tokenizer, prompt: str, response: str) -> float:
    """Calculate log probability of response given prompt."""
    # Combine prompt and response
    full_text = prompt + response
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get prompt length for masking
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get log probabilities for the response tokens only
    response_log_probs = []
    for i in range(prompt_length - 1, inputs["input_ids"].shape[1] - 1):
        token_id = inputs["input_ids"][0, i + 1]
        log_prob = log_probs[0, i, token_id].item()
        response_log_probs.append(log_prob)
    
    # Return average log probability
    return np.mean(response_log_probs) if response_log_probs else 0.0


def evaluate_preferences(
    model, 
    tokenizer, 
    eval_dataset, 
    batch_size: int = 8
) -> Dict[str, float]:
    """Evaluate model on preference dataset."""
    correct_predictions = 0
    total_predictions = 0
    
    chosen_log_probs = []
    rejected_log_probs = []
    
    logger.info(f"Evaluating on {len(eval_dataset)} examples...")
    
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        batch = eval_dataset[i:i+batch_size]
        
        for example in batch:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Get log probabilities
            chosen_log_prob = get_log_probability(model, tokenizer, prompt, chosen)
            rejected_log_prob = get_log_probability(model, tokenizer, prompt, rejected)
            
            chosen_log_probs.append(chosen_log_prob)
            rejected_log_probs.append(rejected_log_prob)
            
            # Check if model prefers chosen over rejected
            if chosen_log_prob > rejected_log_prob:
                correct_predictions += 1
            
            total_predictions += 1
    
    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    avg_chosen_log_prob = np.mean(chosen_log_probs)
    avg_rejected_log_prob = np.mean(rejected_log_probs)
    log_prob_diff = avg_chosen_log_prob - avg_rejected_log_prob
    
    return {
        "accuracy": accuracy,
        "avg_chosen_log_prob": avg_chosen_log_prob,
        "avg_rejected_log_prob": avg_rejected_log_prob,
        "log_prob_difference": log_prob_diff,
        "total_examples": total_predictions
    }


def generate_samples(
    model, 
    tokenizer, 
    prompts: List[str], 
    num_samples: int = 1,
    max_length: int = 512,
    temperature: float = 0.7
) -> List[List[str]]:
    """Generate samples for given prompts."""
    all_samples = []
    
    for prompt in tqdm(prompts, desc="Generating samples"):
        samples = []
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            samples.append(response)
        
        all_samples.append(samples)
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model uses LoRA")
    parser.add_argument("--dataset_name", type=str, default="hh", help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--generate_samples", action="store_true", help="Generate sample responses")
    parser.add_argument("--num_generated_samples", type=int, default=3, help="Number of samples to generate per prompt")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_lora)
    logger.info("Model loaded successfully!")
    
    # Load evaluation dataset
    logger.info(f"Loading {args.dataset_name} dataset...")
    eval_dataset = load_dataset(args.dataset_name, args.split)
    
    # Limit number of samples if specified
    if args.num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(args.num_samples))
    
    logger.info(f"Evaluating on {len(eval_dataset)} examples")
    
    # Evaluate preferences
    logger.info("Evaluating preferences...")
    preference_results = evaluate_preferences(model, tokenizer, eval_dataset, args.batch_size)
    
    # Print results
    logger.info("Evaluation Results:")
    for key, value in preference_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    results = {
        "preference_evaluation": preference_results,
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_samples": len(eval_dataset)
    }
    
    # Generate samples if requested
    if args.generate_samples:
        logger.info("Generating sample responses...")
        sample_prompts = [example["prompt"] for example in eval_dataset[:10]]
        generated_samples = generate_samples(
            model, tokenizer, sample_prompts, 
            args.num_generated_samples
        )
        
        results["generated_samples"] = [
            {
                "prompt": prompt,
                "samples": samples
            }
            for prompt, samples in zip(sample_prompts, generated_samples)
        ]
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
