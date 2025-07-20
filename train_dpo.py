"""
Modern DPO training script using TRL and Hugging Face transformers.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset_utils import combine_datasets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="microsoft/DialoGPT-medium",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
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
        metadata={"help": "List of dataset names to use for training"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache datasets"}
    )


# Use DPOConfig from TRL instead of custom TrainingArguments
# DPOConfig already includes all the necessary parameters


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Setup LoRA if requested
    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_datasets(data_args: DataArguments, tokenizer):
    """Setup training and evaluation datasets."""
    # Load and combine datasets
    train_dataset = combine_datasets(
        data_args.dataset_names,
        split="train",
        cache_dir=data_args.cache_dir
    )
    
    # Try to load eval dataset, fallback to train split if not available
    try:
        eval_dataset = combine_datasets(
            data_args.dataset_names,
            split="test",
            cache_dir=data_args.cache_dir
        )
    except:
        logger.warning("No test split found, using train split for evaluation")
        eval_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse command line arguments
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize wandb if enabled
    if training_args.report_to == ["wandb"]:
        wandb.init(
            project="modern-dpo",
            name=training_args.run_name,
            config={
                **vars(model_args),
                **vars(data_args),
                **vars(training_args)
            }
        )
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup datasets
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = setup_datasets(data_args, tokenizer)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create DPO trainer
    logger.info("Creating DPO trainer...")
    
    # Initialize DPO trainer directly with training_args (which is DPOConfig)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will be loaded automatically
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="config", config_name="dpo_config")
def main_hydra(cfg: DictConfig):
    """Main function for Hydra configuration."""
    # Convert Hydra config to arguments
    model_args = ModelArguments(**cfg.model)
    data_args = DataArguments(**cfg.data)
    training_args = DPOConfig(**cfg.training)
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup datasets
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = setup_datasets(data_args, tokenizer)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will be loaded automatically
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    if "--config-path" in sys.argv or "--config-name" in sys.argv:
        main_hydra()
    else:
        main()
