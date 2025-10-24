"""
Dataset utilities for loading and processing preference datasets.
Modernized version compatible with current Hugging Face datasets library.
"""

import datasets
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Union, Tuple
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from collections import defaultdict
import tqdm


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string: str) -> str:
    """Strip HTML tags from a string, except for <code> tags."""
    soup = BeautifulSoup(html_string, 'html.parser')
    text = []
    
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")
    
    return "\n\n".join(text)


def load_hh_dataset(split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Load Anthropic Helpful-Harmless dataset in DPO format."""
    print(f'Loading HH dataset ({split} split) from Hugging Face...')
    
    dataset = datasets.load_dataset(
        "Anthropic/hh-rlhf", 
        split=split, 
        cache_dir=cache_dir
    )
    
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
    
    processed_dataset = dataset.map(
        process_hh_example,
        remove_columns=dataset.column_names,
        desc="Processing HH dataset"
    )
    
    return processed_dataset


def load_shp_dataset(split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Load Stanford Human Preferences dataset in DPO format."""
    print(f'Loading SHP dataset ({split} split) from Hugging Face...')
    
    dataset = datasets.load_dataset(
        "stanfordnlp/SHP", 
        split=split, 
        cache_dir=cache_dir
    )
    
    # Group by post_id and create preference pairs
    grouped_data = defaultdict(list)
    for example in dataset:
        grouped_data[example['post_id']].append(example)
    
    dpo_examples = []
    for post_id, examples in tqdm.tqdm(grouped_data.items(), desc="Processing SHP dataset"):
        if len(examples) < 2:
            continue
            
        # Sort by score
        examples.sort(key=lambda x: x['score'], reverse=True)
        
        # Create pairs where score ratio is at least 2
        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                if examples[i]['score'] / max(examples[j]['score'], 1) >= 2:
                    dpo_examples.append({
                        'prompt': f"Human: {examples[i]['history']}\n\nAssistant:",
                        'chosen': f" {examples[i]['human_ref_A']}",
                        'rejected': f" {examples[j]['human_ref_A']}"
                    })
    
    return Dataset.from_list(dpo_examples)


def load_se_dataset(split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Load StackExchange dataset in DPO format."""
    print(f'Loading SE dataset ({split} split) from Hugging Face...')
    
    dataset = datasets.load_dataset(
        'HuggingFaceH4/stack-exchange-preferences', 
        cache_dir=cache_dir
    )['train']
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    if split == 'test':
        dataset = dataset.select(range(int(len(dataset) * 0.01)))
    else:
        dataset = dataset.select(range(int(len(dataset) * 0.01), len(dataset)))
    
    def process_se_example(example):
        """Process SE example to DPO format."""
        question = strip_html_tags(example['question'])
        answers = [strip_html_tags(a['text']) for a in example['answers']]
        scores = [a['pm_score'] for a in example['answers']]
        
        if len(answers) < 2:
            return None
            
        # Find best and worst answers
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)
        
        if best_idx == worst_idx:
            return None
            
        return {
            'prompt': f'\n\nHuman: {question}\n\nAssistant:',
            'chosen': f' {answers[best_idx]}',
            'rejected': f' {answers[worst_idx]}'
        }
    
    processed_dataset = dataset.map(
        process_se_example,
        remove_columns=dataset.column_names,
        desc="Processing SE dataset"
    ).filter(lambda x: x is not None)
    
    return processed_dataset


def load_sum_dataset(split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Load SUM dataset in DPO format."""
    print(f'Loading SUM dataset ({split} split) from Hugging Face...')
    
    dataset = datasets.load_dataset(
        "quyanh/summarization-dpo", 
        split=split, 
        cache_dir=cache_dir
    )

    # quyanh/summarization-dpo includes: post, chosen, rejected -> create prompt 
    template = "Summarize the following post: {post}\n\nAssistant:"
    
    return dataset.map(
        lambda x: {"prompt": template.format(post=x["post"]), "chosen": x["chosen"], "rejected": x["rejected"]},
        remove_columns=dataset.column_names,
        desc="Processing SUM dataset"
    )


def get_dataset(name: str, split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Load dataset by name."""
    if name == "hh":
        return load_hh_dataset(split, cache_dir)
    elif name == "shp":
        return load_shp_dataset(split, cache_dir)
    elif name == "se":
        return load_se_dataset(split, cache_dir)
    elif name == "sum":
        return load_sum_dataset(split, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def combine_datasets(dataset_names: List[str], split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """Combine multiple datasets."""
    datasets_list = []
    for name in dataset_names:
        ds = get_dataset(name, split, cache_dir)
        datasets_list.append(ds)
    
    if len(datasets_list) == 1:
        return datasets_list[0]
    
    combined = datasets.concatenate_datasets(datasets_list)
    return combined.shuffle(seed=42)


