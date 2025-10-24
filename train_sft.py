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
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from dataset_utils import get_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HubUploadCallback(TrainerCallback):
    """Custom callback to upload model to Hugging Face Hub after each evaluation."""
    
    def __init__(self, hub_model_id: Optional[str] = None, tokenizer=None):
        self.hub_model_id = hub_model_id
        self.tokenizer = tokenizer
        self.upload_count = 0
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Upload LoRA adapters to Hub after each evaluation."""
        if args.push_to_hub and self.hub_model_id:
            self.upload_count += 1
            logger.info(f"📊 Evaluation {self.upload_count} completed (step {state.global_step})")
            
            if state.log_history:
                latest_logs = state.log_history[-1]
                if 'eval_loss' in latest_logs:
                    logger.info(f"📈 Eval loss: {latest_logs['eval_loss']:.4f}")
            
            try:
                # Create commit message with evaluation info
                commit = f"SFT LoRA checkpoint - step {state.global_step}"
                if state.log_history:
                    latest_logs = state.log_history[-1]
                    if 'eval_loss' in latest_logs:
                        commit += f" (eval_loss: {latest_logs['eval_loss']:.4f})"
                
                # Upload LoRA adapters (not merged model)
                if hasattr(model, 'save_pretrained'):
                    logger.info(f"🚀 Uploading LoRA adapters to {self.hub_model_id}...")
                    model.push_to_hub(
                        repo_id=self.hub_model_id,
                        commit_message=commit,
                    )
                    
                    # Also push tokenizer
                    self.tokenizer.push_to_hub(
                        repo_id=self.hub_model_id,
                        commit_message=f"Tokenizer - {commit}",
                    )
                    
                    logger.info(f"✅ LoRA adapters uploaded successfully to {self.hub_model_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to upload LoRA adapters: {str(e)}")
                logger.info("Training will continue despite upload failure")
                pass


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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use (for debugging)"}
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
        default=0,
        metadata={"help": "Number of warmup steps"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
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
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy"}
    )
    load_best_model_at_end: bool = field(
        default=False,
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
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID for Hugging Face Hub upload"}
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
        torch_dtype=torch.float32,  # Use float32 to avoid numerical instability
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
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_sft_dataset(dataset_names: List[str], tokenizer, max_length: int, train_split_ratio: float, max_train_samples: Optional[int] = None):
    """Prepare dataset for SFT training."""
    logger.info(f"Preparing SFT dataset from: {dataset_names}")
    
    # Load and combine datasets
    all_data = []
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name, split="train", cache_dir=None)
        all_data.extend(dataset)
    
    logger.info(f"Total examples loaded: {len(all_data)}")
    
    # Limit the number of examples if specified
    if max_train_samples is not None and len(all_data) > max_train_samples:
        all_data = all_data[:max_train_samples]
        logger.info(f"Limited to {max_train_samples} examples for debugging")
    
    # For SFT, we only use the "chosen" responses as the target
    sft_examples = []
    for example in all_data:
        # Create input-output pairs for SFT
        prompt = example["prompt"]
        chosen = example["chosen"]
        text = prompt + chosen
        sft_examples.append({
            "text": text,
            "prompt": prompt,
            "chosen": chosen
        })
    
    # Tokenize the data
    def tokenize_function(examples):
        # Tokenize the full text (prompt + chosen)
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Pad to max_length for consistent tensor shapes
            max_length=max_length,
            return_tensors=None,
        )
        
        # Create labels: copy input_ids but set padding tokens to -100
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = input_ids.copy()
            # Set padding tokens to -100 (ignored in loss)
            for j, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    # Convert to HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_list(sft_examples)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "prompt", "chosen"])
    
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
            project="dpo-replication",
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
        data_args.train_split_ratio,
        data_args.max_train_samples
    )
    
    # Data collator - use default since we're already padding during tokenization
    from transformers import default_data_collator
    data_collator = default_data_collator

    # Setup callbacks
    callbacks = []
    if training_args.push_to_hub and training_args.hub_model_id:
        hub_callback = HubUploadCallback(
            hub_model_id=training_args.hub_model_id,
            tokenizer=tokenizer
        )
        callbacks.append(hub_callback)
        logger.info(f"🚀 Hub upload callback enabled for model: {training_args.hub_model_id}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Upload final LoRA adapters to Hub
    if training_args.push_to_hub and training_args.hub_model_id:
        logger.info("Pushing final LoRA adapters to Hugging Face Hub...")
        commit = f"Final SFT LoRA adapters - dataset: {data_args.dataset_names}, epochs: {training_args.num_train_epochs}"
        try:
            # Upload LoRA adapters (not merged model)
            trainer.model.push_to_hub(
                repo_id=training_args.hub_model_id,
                commit_message=commit,
            )
            
            # Also push tokenizer
            tokenizer.push_to_hub(
                repo_id=training_args.hub_model_id,
                commit_message=f"Tokenizer - {commit}",
            )
            
            logger.info(f"✅ Final LoRA adapters successfully uploaded to {training_args.hub_model_id}")
        except Exception as e:
            logger.error(f"❌ Failed to upload final LoRA adapters: {str(e)}")
    
    # Save LoRA adapters locally for DPO training
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save LoRA adapters (not merged model)
    logger.info("Saving LoRA adapters locally for DPO training...")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save merged model at the end of training
    if hasattr(trainer.model, 'merge_and_unload'):
        logger.info("Saving merged model (LoRA weights merged with base model)...")
        merged_model_path = os.path.join(training_args.output_dir, "merged_model")
        os.makedirs(merged_model_path, exist_ok=True)
        
        try:
            # Merge LoRA weights with base model
            merged_model = trainer.model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            logger.info(f"✅ Merged model saved to: {merged_model_path}")
            
            # Clean up memory
            del merged_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to save merged model: {str(e)}")
            logger.info("LoRA adapters are still available for manual merging")
    else:
        logger.info("Model is not a PEFT model, no merging needed")
    
    logger.info(f"SFT training completed! Models saved to:")
    logger.info(f"  - LoRA adapters: {final_model_path}")
    if hasattr(trainer.model, 'merge_and_unload'):
        logger.info(f"  - Merged model: {os.path.join(training_args.output_dir, 'merged_model')}")
    logger.info("You can now use this model for DPO training with:")
    logger.info(f"python train_dpo.py --model_name_or_path {final_model_path}")


if __name__ == "__main__":
    main()
