"""
Inference script for DPO models with LoRA adapter support.
Generates responses from a dataset using a base model and optional LoRA adapter.
"""

import os
import torch
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from datasets import load_dataset
from peft import PeftModel
from dataset_utils import extract_anthropic_prompt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    GenerationConfig
)
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceArguments:
    """Arguments for inference configuration."""
    base_model: str = field(
        metadata={"help": "Path to the base model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    prompt_name: str = field(
        metadata={"help": "The name of the column in the dataset containing the prompt"}
    )
    run_name: str = field(
        metadata={"help": "A name for the inference run"}
    )
    adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter model"}
    )
    adapter_version: str = field(
        default="main",
        metadata={"help": "Version/revision of the adapter to load (default: main)"}
    )
    split: str = field(
        default="test",
        metadata={"help": "The split of the dataset to use"}
    )
    output_dir: str = field(
        default="./inference_results",
        metadata={"help": "The directory to save inference results"}
    )
    max_samples: int = field(
        default=50,
        metadata={"help": "Maximum number of samples to process"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Generation temperature"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p (nucleus) sampling parameter"}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling parameter"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling for generation"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model loading (float16, bfloat16, float32)"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference"}
    )


def load_model_and_tokenizer(args: InferenceArguments):
    """Load the base model, tokenizer, and apply adapter if specified."""
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    logger.info(f"Loading model from {args.base_model} with dtype {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load and merge LoRA adapter if provided
    if args.adapter:
        logger.info(f"Loading LoRA adapter from {args.adapter} (version: {args.adapter_version})")
        model = PeftModel.from_pretrained(model, args.adapter, revision=args.adapter_version)
        logger.info("Merging LoRA adapter with base model")
        model = model.merge_and_unload()

    return model, tokenizer


# def process_dataset_prompt(sample, dataset_name: str, prompt_name: str):
#     """Process dataset-specific prompts to extract the correct prompt format."""
#     if dataset_name.lower() in ["anthropic/hh-rlhf", "hh"]:
#         # For HH dataset, extract the prompt from chosen/rejected responses
#         if prompt_name in ["chosen", "rejected"]:
#             full_text = sample[prompt_name]
#             # Extract prompt (everything before the last Assistant response)
#             prompt = extract_anthropic_prompt(full_text)
#             return prompt
#         else:
#             # If prompt_name is something else, use it directly
#             return sample[prompt_name]

#     elif dataset_name.lower() in ["quyanh/summarization-dpo", "sum"]:
#         # For summarization dataset, create a proper prompt
#         if "post" in sample:
#             return f"Summarize the following post: {sample['post']}\n\nAssistant:"
#         else:
#             return sample[prompt_name]

#     elif dataset_name.lower() in ["stanfordnlp/shp", "shp"]:
#         # For SHP dataset, create conversational prompt
#         if "history" in sample:
#             return f"Human: {sample['history']}\n\nAssistant:"
#         else:
#             return sample[prompt_name]

#     else:
#         # For other datasets, use the specified column directly
#         return sample[prompt_name]


def process_hh_example(example):
        """Process HH example to DPO format."""
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Extract prompt (everything before the last Assistant response)
        prompt = extract_anthropic_prompt(chosen)
        
        # Extract responses (everything after the prompt)
        chosen_response = chosen[len(prompt):].strip()
        rejected_response = rejected[len(prompt):].strip()
        
        return {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }

def generate_responses(model, tokenizer, dataset, args: InferenceArguments):
    """Generate responses for the dataset."""
    results = []

    # Create generation config
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    logger.info(f"Starting inference on {len(dataset)} samples")

    for i, sample in enumerate(tqdm(dataset, desc="Generating responses")):
        try:
            # Process dataset-specific prompt extraction
            # prompt = process_dataset_prompt(sample, args.dataset_name, args.prompt_name)
            point = process_hh_example(sample) # Only for hh dataset
            prompt = point["prompt"]
            chosen = point["chosen"]
            rejected = point["rejected"]

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Decode the generated text
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Temporary - Truncate generate response to get text before "\n\nHuman:" if have
            generated_text = generated_text.split("\n\nHuman:")[0]

            result = {
                "sample_id": i,
                "prompt": prompt,
                "generated_text": generated_text,
                "chosen": chosen,
                "rejected": rejected
            }

            # # Add other fields from the sample for reference
            # for key, value in sample.items():
            #     if key != args.prompt_name:
            #         result[f"original_{key}"] = value

            results.append(result)

        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            results.append({
                "sample_id": i,
                "prompt": sample.get(args.prompt_name, ""),
                "generated_text": "",
                "error": str(e)
            })

    return results


def save_results(results: List[Dict], args: InferenceArguments):
    """Save inference results and metadata to a single file."""
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combine metadata and results
    output_data = {
        "metadata": {
            "run_name": args.run_name,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "adapter_version": args.adapter_version,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "prompt_name": args.prompt_name,
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "torch_dtype": args.torch_dtype,
            "timestamp": timestamp,
            "total_samples": len(results)
        },
        "results": results
    }

    # Create output filename
    output_file = f"{args.run_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_file)

    # Save to a single JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results and metadata saved to: {output_path}")
    return output_path


def main():
    """Main function to run the inference script."""
    parser = HfArgumentParser((InferenceArguments,))
    args, = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    logger.info(f"Loading dataset {args.dataset_name} (split: {args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    # Limit samples if specified
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples")
        dataset = dataset.select(range(args.max_samples))

    # Generate responses
    results = generate_responses(model, tokenizer, dataset, args)

    # Save results
    save_results(results, args)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()