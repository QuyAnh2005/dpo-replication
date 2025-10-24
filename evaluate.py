import os
import json
import re
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import HfArgumentParser
from prompt import DIALOGUE_WINRATE

# --- Configuration ---
client = OpenAI()

@dataclass
class EvalArguments:
    """Arguments for the evaluation script."""
    model_responses: str = field(
        metadata={"help": "Path to the JSON file with model responses (generated_text) and chosen responses"}
    )
    run_name: str = field(
        metadata={"help": "A name for the evaluation run"}
    )
    output_dir: str = field(
        default="./eval_results",
        metadata={"help": "Directory to save evaluation results"}
    )
    reverse_order: bool = field(
        default=False,
        metadata={"help": "If True, present chosen response as A and model response as B (reduces position bias)"}
    )


def load_model_data(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Loads model responses and metadata from a JSON file.
    
    Returns:
        Tuple of (metadata, list of samples with prompt, generated_text, and chosen)
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    # Extract relevant fields
    samples = []
    for result in results:
        if 'prompt' in result and 'generated_text' in result and 'chosen' in result:
            samples.append({
                'sample_id': result.get('sample_id'),
                'prompt': result['prompt'],
                'generated_text': result['generated_text'],
                'chosen': result['chosen']
            })
    
    return metadata, samples


def get_gpt4_verdict(prompt: str, response_a: str, response_b: str) -> Tuple[str, str]:
    """
    Gets GPT-4's verdict on which response is better.
    
    Returns:
        Tuple of (verdict, raw_content) where verdict is 'A', 'B', 'Tie', or 'Error'
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and unbiased AI assistant that evaluates chatbot responses."
                },
                {
                    "role": "user",
                    "content": DIALOGUE_WINRATE.format(
                        the_user_query=prompt,
                        either_the_test_method_or_baseline=response_a,
                        the_other_response=response_b
                    )
                }
            ],
            temperature=0.0,
            max_tokens=100
        )
        content = response.choices[0].message.content

        # Use regex for more robust parsing
        match = re.search(r"(?:More helpful|Preferred):\s*([AB])", content, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()
        else:
            winner = "Tie"  # Default to Tie if parsing fails

        return winner, content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error", str(e)


def evaluate_sample(
    sample: Dict[str, str],
    reverse_order: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single sample by comparing model response with chosen response.
    
    Args:
        sample: Dictionary with 'prompt', 'generated_text', and 'chosen'
        reverse_order: If True, present chosen as A and generated as B
    
    Returns:
        Dictionary with evaluation results
    """
    prompt = sample['prompt']
    generated = sample['generated_text']
    chosen = sample['chosen']
    
    # Determine order
    if reverse_order:
        response_a = chosen
        response_b = generated
        a_label = "Chosen"
        b_label = "Model"
    else:
        response_a = generated
        response_b = chosen
        a_label = "Model"
        b_label = "Chosen"
    
    verdict, raw_content = get_gpt4_verdict(prompt, response_a, response_b)
    
    # Map verdict back to actual model/chosen labels
    if verdict == "A":
        winner = a_label
    elif verdict == "B":
        winner = b_label
    elif verdict == "Error":
        winner = "Error"
    else:
        winner = "Tie"
    
    return {
        "sample_id": sample.get('sample_id'),
        "prompt": prompt,
        "model_response": generated,
        "chosen_response": chosen,
        "presentation_order": "chosen_first" if reverse_order else "model_first",
        "verdict": winner,
        "raw_gpt4_response": raw_content
    }


def calculate_winrate(win_counts: Dict[str, int]) -> Dict[str, float]:
    """Calculate winrate percentages."""
    total = sum(win_counts.values())
    if total == 0:
        return {"model_winrate": 0.0, "chosen_winrate": 0.0, "tie_rate": 0.0, "error_rate": 0.0}
    
    return {
        "model_winrate": (win_counts["Model"] / total) * 100,
        "chosen_winrate": (win_counts["Chosen"] / total) * 100,
        "tie_rate": (win_counts["Tie"] / total) * 100,
        "error_rate": (win_counts["Error"] / total) * 100
    }


def main():
    parser = HfArgumentParser((EvalArguments,))
    args, = parser.parse_args_into_dataclasses()

    # Load model data
    print(f"Loading responses from: {args.model_responses}")
    metadata, samples = load_model_data(args.model_responses)
    
    print(f"Loaded {len(samples)} samples for evaluation")
    print(f"Model: {metadata.get('base_model', 'Unknown')}")
    print(f"Temperature: {metadata.get('temperature', 'Unknown')}")

    # Evaluate all samples
    results = []
    win_counts = {"Model": 0, "Chosen": 0, "Tie": 0, "Error": 0}

    # samples = samples[:5] # Uncomment for test
    for sample in tqdm(samples, desc="Evaluating responses"):
        result = evaluate_sample(sample, reverse_order=args.reverse_order)
        
        # Update counts
        winner = result['verdict']
        if winner in win_counts:
            win_counts[winner] += 1
        
        results.append(result)

    # Calculate winrates
    winrates = calculate_winrate(win_counts)

    # Prepare output data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "metadata": {
            "run_name": args.run_name,
            "model_responses_file": args.model_responses,
            "source_model": metadata.get('base_model'),
            "adapter": metadata.get('adapter'),
            "temperature": metadata.get('temperature'),
            "dataset": metadata.get('dataset_name'),
            "total_samples_evaluated": len(samples),
            "reverse_order": args.reverse_order,
            "timestamp": timestamp
        },
        "win_counts": win_counts,
        "winrates": winrates,
        "results": results
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.run_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"\nWin Counts:")
    print(f"  Model wins:  {win_counts['Model']:>3} ({winrates['model_winrate']:.2f}%)")
    print(f"  Chosen wins: {win_counts['Chosen']:>3} ({winrates['chosen_winrate']:.2f}%)")
    print(f"  Ties:        {win_counts['Tie']:>3} ({winrates['tie_rate']:.2f}%)")
    print(f"  Errors:      {win_counts['Error']:>3} ({winrates['error_rate']:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()