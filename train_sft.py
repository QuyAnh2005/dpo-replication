"""
Supervised Fine-Tuning (SFT) script for DPO pipeline.
This is the first step before DPO training - it ensures the model is adapted to the target domain.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from dataset_utils import get_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="microsoft/DialoGPT-small",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer or tokenizer identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_names: List[str] = field(
        default_factory=lambda: ["hh"],
        metadata={"help": "List of dataset names to use (hh, shp, se)"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    train_split_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of data to use for training (rest for validation)"}
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """Training arguments for SFT."""
    output_dir: str = field(
        default="./sft_results",
        metadata={"help": "Output directory for model and logs"}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluate every X updates steps"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end of training"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric to use for best model selection"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether higher metric values are better"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["wandb"],
        metadata={"help": "List of integrations to report to"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the run for wandb"}
    )


def load_model_and_tokenizer(model_args: ModelArguments):
    """Load model and tokenizer."""
    logger.info(f"Loading model and tokenizer from {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Apply LoRA if specified
    if model_args.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_sft_dataset(dataset_names: List[str], tokenizer, max_length: int, train_split_ratio: float):
    """Prepare dataset for SFT training."""
    logger.info(f"Preparing SFT dataset from: {dataset_names}")
    
    # Load and combine datasets
    all_data = []
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name, split="train", silent=True, cache_dir=None)
        all_data.extend(dataset)
    
    logger.info(f"Total examples loaded: {len(all_data)}")
    
    # For SFT, we only use the "chosen" responses as the target
    sft_examples = []
    for example in all_data:
        # Create input-output pairs for SFT
        text = example["prompt"] + example["chosen"]
        sft_examples.append({"text": text})
    
    # Tokenize the data
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Convert to HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_list(sft_examples)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Split into train/eval
    split_idx = int(len(dataset) * train_split_ratio)
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Eval examples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup wandb
    if "wandb" in training_args.report_to:
        wandb.init(
            project="modern-dpo-sft",
            name=training_args.run_name or f"sft-{model_args.model_name_or_path.split('/')[-1]}",
            config={
                **vars(model_args),
                **vars(data_args),
                **vars(training_args),
            }
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_sft_dataset(
        data_args.dataset_names,
        tokenizer,
        data_args.max_length,
        data_args.train_split_ratio
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save model for DPO training
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"SFT training completed! Model saved to: {final_model_path}")
    logger.info("You can now use this model for DPO training with:")
    logger.info(f"python train_dpo.py --model_name_or_path {final_model_path}")


if __name__ == "__main__":
    main()
