"""
Fast evaluation script using vLLM for accelerated inference.
This provides much faster evaluation compared to standard HuggingFace evaluation.
"""

import argparse
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataset_utils import get_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    preference_accuracy: float
    win_rate: float
    avg_chosen_score: float
    avg_rejected_score: float
    total_examples: int
    evaluation_time: float
    tokens_per_second: float
    examples_per_second: float
    detailed_results: List[Dict[str, Any]]


def load_vllm_model(model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    """Load model with vLLM for fast evaluation."""
    logger.info(f"Loading model with vLLM: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize vLLM model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    
    logger.info(f"✓ Model loaded successfully")
    return llm, tokenizer


def create_evaluation_sampling_params() -> SamplingParams:
    """Create sampling parameters optimized for evaluation."""
    return SamplingParams(
        temperature=0.0,  # Deterministic for evaluation
        top_p=1.0,
        top_k=-1,
        max_tokens=512,
        use_beam_search=False,
        logprobs=1,  # Get log probabilities for scoring
    )


def format_prompt_for_evaluation(prompt: str, response: str, tokenizer) -> str:
    """Format prompt and response for evaluation."""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        return f"Human: {prompt}\n\nAssistant: {response}"


def compute_response_score(llm: LLM, tokenizer, prompt: str, response: str, sampling_params: SamplingParams) -> float:
    """Compute log probability score for a response given a prompt."""
    # Format the full text
    full_text = format_prompt_for_evaluation(prompt, response, tokenizer)
    
    # Generate to get log probabilities
    outputs = llm.generate([full_text], sampling_params)
    
    if outputs[0].outputs[0].logprobs:
        # Sum log probabilities for the response tokens
        logprobs = outputs[0].outputs[0].logprobs
        total_logprob = sum(token_logprob.logprob for token_logprob in logprobs)
        return total_logprob
    else:
        # Fallback: use length-normalized score
        return -len(response.split()) * 0.1


def evaluate_preference_pair(
    llm: LLM, 
    tokenizer, 
    prompt: str, 
    chosen: str, 
    rejected: str, 
    sampling_params: SamplingParams
) -> Dict[str, Any]:
    """Evaluate a single preference pair."""
    
    # Compute scores for both responses
    chosen_score = compute_response_score(llm, tokenizer, prompt, chosen, sampling_params)
    rejected_score = compute_response_score(llm, tokenizer, prompt, rejected, sampling_params)
    
    # Determine preference
    prefers_chosen = chosen_score > rejected_score
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "prefers_chosen": prefers_chosen,
        "score_difference": chosen_score - rejected_score
    }


def evaluate_dataset_vllm(
    llm: LLM,
    tokenizer,
    dataset: List[Dict[str, str]],
    sampling_params: SamplingParams,
    max_examples: int = None
) -> EvaluationResult:
    """Evaluate a dataset using vLLM for fast inference."""
    
    if max_examples:
        dataset = dataset[:max_examples]
    
    logger.info(f"Evaluating {len(dataset)} examples...")
    
    start_time = time.time()
    detailed_results = []
    total_tokens = 0
    
    # Process examples in batches for efficiency
    batch_size = 32  # Adjust based on memory
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i + batch_size]
        
        # Prepare all prompts and responses for batch processing
        chosen_texts = []
        rejected_texts = []
        
        for example in batch:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            chosen_text = format_prompt_for_evaluation(prompt, chosen, tokenizer)
            rejected_text = format_prompt_for_evaluation(prompt, rejected, tokenizer)
            
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
        
        # Batch generate for chosen responses
        chosen_outputs = llm.generate(chosen_texts, sampling_params)
        rejected_outputs = llm.generate(rejected_texts, sampling_params)
        
        # Process results
        for j, example in enumerate(batch):
            chosen_output = chosen_outputs[j]
            rejected_output = rejected_outputs[j]
            
            # Extract scores
            chosen_score = 0.0
            rejected_score = 0.0
            
            if chosen_output.outputs[0].logprobs:
                chosen_score = sum(token_logprob.logprob for token_logprob in chosen_output.outputs[0].logprobs)
            
            if rejected_output.outputs[0].logprobs:
                rejected_score = sum(token_logprob.logprob for token_logprob in rejected_output.outputs[0].logprobs)
            
            # Count tokens
            total_tokens += len(chosen_output.outputs[0].token_ids)
            total_tokens += len(rejected_output.outputs[0].token_ids)
            
            # Store result
            result = {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "prefers_chosen": chosen_score > rejected_score,
                "score_difference": chosen_score - rejected_score
            }
            detailed_results.append(result)
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Compute metrics
    correct_preferences = sum(1 for r in detailed_results if r["prefers_chosen"])
    preference_accuracy = correct_preferences / len(detailed_results)
    win_rate = preference_accuracy  # Same as preference accuracy
    
    avg_chosen_score = np.mean([r["chosen_score"] for r in detailed_results])
    avg_rejected_score = np.mean([r["rejected_score"] for r in detailed_results])
    
    tokens_per_second = total_tokens / evaluation_time
    examples_per_second = len(detailed_results) / evaluation_time
    
    return EvaluationResult(
        preference_accuracy=preference_accuracy,
        win_rate=win_rate,
        avg_chosen_score=avg_chosen_score,
        avg_rejected_score=avg_rejected_score,
        total_examples=len(detailed_results),
        evaluation_time=evaluation_time,
        tokens_per_second=tokens_per_second,
        examples_per_second=examples_per_second,
        detailed_results=detailed_results
    )


def generate_evaluation_samples(
    llm: LLM,
    tokenizer,
    dataset: List[Dict[str, str]],
    num_samples: int = 5
) -> List[Dict[str, str]]:
    """Generate sample responses for qualitative evaluation."""
    
    logger.info(f"Generating {num_samples} sample responses...")
    
    # Select random examples
    import random
    sample_examples = random.sample(dataset, min(num_samples, len(dataset)))
    
    # Create generation sampling params
    gen_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )
    
    # Extract prompts
    prompts = []
    for example in sample_examples:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": example["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"Human: {example['prompt']}\n\nAssistant:"
        prompts.append(prompt)
    
    # Generate responses
    outputs = llm.generate(prompts, gen_sampling_params)
    
    # Format results
    samples = []
    for i, (example, output) in enumerate(zip(sample_examples, outputs)):
        samples.append({
            "prompt": example["prompt"],
            "chosen_reference": example["chosen"],
            "rejected_reference": example["rejected"],
            "model_response": output.outputs[0].text.strip(),
        })
    
    return samples


def save_evaluation_results(results: EvaluationResult, output_dir: str, dataset_name: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        "dataset": dataset_name,
        "preference_accuracy": results.preference_accuracy,
        "win_rate": results.win_rate,
        "avg_chosen_score": results.avg_chosen_score,
        "avg_rejected_score": results.avg_rejected_score,
        "total_examples": results.total_examples,
        "evaluation_time": results.evaluation_time,
        "tokens_per_second": results.tokens_per_second,
        "examples_per_second": results.examples_per_second,
    }
    
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, f"{dataset_name}_detailed.jsonl")
    with open(detailed_path, 'w') as f:
        for result in results.detailed_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Fast evaluation with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_names", nargs="+", default=["hh"], 
                       help="Dataset names to evaluate on")
    parser.add_argument("--output_dir", type=str, default="./eval_results_vllm",
                       help="Output directory for results")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio")
    parser.add_argument("--generate_samples", action="store_true",
                       help="Generate sample responses for qualitative evaluation")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of sample responses to generate")
    
    args = parser.parse_args()
    
    # Load model
    try:
        llm, tokenizer = load_vllm_model(
            args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Make sure vLLM is installed: pip install vllm")
        return
    
    # Create sampling parameters
    sampling_params = create_evaluation_sampling_params()
    
    # Evaluate on each dataset
    for dataset_name in args.dataset_names:
        logger.info(f"Evaluating on dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = get_dataset(dataset_name, split="test", silent=False, cache_dir=None)
            
            # Evaluate
            results = evaluate_dataset_vllm(
                llm, tokenizer, dataset, sampling_params, args.max_examples
            )
            
            # Print results
            print(f"\n📊 Evaluation Results for {dataset_name}:")
            print(f"{'='*50}")
            print(f"Preference Accuracy: {results.preference_accuracy:.3f}")
            print(f"Win Rate: {results.win_rate:.3f}")
            print(f"Avg Chosen Score: {results.avg_chosen_score:.3f}")
            print(f"Avg Rejected Score: {results.avg_rejected_score:.3f}")
            print(f"Total Examples: {results.total_examples}")
            print(f"Evaluation Time: {results.evaluation_time:.2f}s")
            print(f"Tokens/Second: {results.tokens_per_second:.2f}")
            print(f"Examples/Second: {results.examples_per_second:.2f}")
            
            # Save results
            save_evaluation_results(results, args.output_dir, dataset_name)
            
            # Generate samples if requested
            if args.generate_samples:
                logger.info("Generating sample responses...")
                samples = generate_evaluation_samples(
                    llm, tokenizer, dataset, args.num_samples
                )
                
                samples_path = os.path.join(args.output_dir, f"{dataset_name}_samples.json")
                with open(samples_path, 'w') as f:
                    json.dump(samples, f, indent=2)
                
                print(f"\n📝 Sample responses saved to: {samples_path}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate dataset {dataset_name}: {e}")
            continue
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
