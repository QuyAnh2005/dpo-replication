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
    TrainerCallback,
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


class HubUploadCallback(TrainerCallback):
    """Custom callback to upload LoRA adapters during evaluation and full merged model at end."""
    
    def __init__(self, hub_model_id: Optional[str] = None, tokenizer=None):
        self.hub_model_id = hub_model_id
        self.tokenizer = tokenizer
        self.upload_count = 0
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Upload LoRA adapters after each evaluation step."""
        if args.push_to_hub and self.hub_model_id and model is not None:
            self.upload_count += 1
            logger.info(f"📊 Evaluation {self.upload_count} completed (step {state.global_step})")
            
            if state.log_history:
                latest_logs = state.log_history[-1]
                if 'eval_loss' in latest_logs:
                    logger.info(f"📈 Eval loss: {latest_logs['eval_loss']:.4f}")
            
            try:
                # Create commit message with metrics
                commit = f"DPO LoRA checkpoint - step {state.global_step}"
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use (for debugging)"}
    )
    train_split_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of data to use for training (rest for validation)"}
    )


@dataclass
class DPOTrainingArguments(DPOConfig):
    """DPO training arguments with additional hub upload parameters."""
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face Hub model ID for uploading the model"}
    )


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer."""
    # Load tokenizer with fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_args.model_name_or_path}: {e}")
        logger.info("Trying fallback tokenizer: gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
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
        low_cpu_mem_usage=True,
        max_memory={0: "13GB"} if torch.cuda.is_available() else None,
    )
    
    # Setup LoRA if requested
    if model_args.use_lora:
        # Auto-detect target modules based on model architecture
        def get_target_modules(model):
            """Auto-detect LoRA target modules based on model architecture."""
            model_type = model.config.model_type.lower()
            
            if model_type == "gptj":
                return ["q_proj", "v_proj", "k_proj", "out_proj"]
            elif model_type in ["gpt2", "gpt_neo"]:
                return ["c_attn", "c_proj", "c_fc"]
            elif model_type == "gpt_neox":
                return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            elif model_type == "llama":
                return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif model_type == "mistral":
                return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # Fallback: try to find common linear layer patterns
                logger.warning(f"Unknown model type {model_type}, using fallback target modules")
                return ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        target_modules = get_target_modules(model)
        logger.info(f"Using LoRA target modules for {model.config.model_type}: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
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
    
    # Limit training samples if specified
    if data_args.max_train_samples is not None:
        logger.info(f"Limiting training samples to {data_args.max_train_samples}")
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    
    # Try to load eval dataset, fallback to train split if not available
    logger.info(f"Train/eval split ratio configured: {data_args.train_split_ratio:.1f}/{1-data_args.train_split_ratio:.1f}")
    try:
        eval_dataset = combine_datasets(
            data_args.dataset_names,
            split="test",
            cache_dir=data_args.cache_dir
        )
        logger.info("✅ Using separate test split for evaluation")
    except:
        logger.warning(f"No test split found, using train split for evaluation with {data_args.train_split_ratio:.1f} split for train, {1-data_args.train_split_ratio:.1f} for eval")
        
        # Split the dataset based on train_split_ratio
        total_samples = len(train_dataset)
        train_size = int(total_samples * data_args.train_split_ratio)
        
        # Create train and eval splits
        train_indices = list(range(train_size))
        eval_indices = list(range(train_size, total_samples))
        
        eval_dataset = train_dataset.select(eval_indices)
        train_dataset = train_dataset.select(train_indices)
        
        logger.info(f"Split dataset: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples")
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOTrainingArguments))
    
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
            project="dpo-replication",
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
    
    # Setup hub upload callback if hub_model_id is provided
    callbacks = []
    if training_args.push_to_hub and training_args.hub_model_id:
        logger.info(f"🚀 Hub upload callback enabled for model: {training_args.hub_model_id}")
        hub_callback = HubUploadCallback(training_args.hub_model_id, tokenizer)
        callbacks.append(hub_callback)
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will be loaded automatically
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        callbacks=callbacks,
    )
    
    # Train the model
    logger.info("Starting DPO training...")
    trainer.train()
    
    # Upload final LoRA adapters to Hub
    if training_args.push_to_hub and training_args.hub_model_id:
        logger.info("Pushing final LoRA adapters to Hugging Face Hub...")
        commit = f"Final DPO LoRA adapters - training completed"
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
    
    # Save LoRA adapters locally for future use
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save LoRA adapters (not merged model)
    logger.info("Saving LoRA adapters locally...")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"DPO training completed! Model saved to: {final_model_path}")


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
    main()
