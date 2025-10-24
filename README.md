# Direct Preference Optimization (DPO) - Paper Replication

## Overview

This project replicates the key results from the original DPO paper: **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by Rafailov et al. (2023), with a focus on the **Anthropic Helpful-Harmless (HH)** dataset.

The implementation uses modern tools (2025 standards) including the latest Hugging Face Transformers, TRL (Transformer Reinforcement Learning) framework, and LoRA for parameter-efficient fine-tuning. This repository provides a complete pipeline from training to evaluation, including GPT-4o-based win-rate evaluation for comprehensive model comparison.


## Model & Dataset

- **Base Model**: [EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)
- **Dataset**: [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (for GPT-4o evaluation)
- Hugging Face account (for model downloads and uploads)
- Weights & Biases account (optional, for experiment tracking)

### Setup

```bash
# Clone the repository
git clone https://github.com/QuyAnh2005/dpo-replication.git
cd dpo-replication

# Install uv (if not already installed)
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
# pip install uv

# Alternative: Install via pipx (recommended for isolation)
# pipx install uv

# Create .env file with required API keys and tokens
cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
EOL

# Load environment variables
source .env  # On Windows: use `set -a; source .env; set +a` or load manually

# Create and activate virtual environment using uv
uv venv dpo-replication
source dpo-replication/bin/activate  # On Windows: dpo-replication\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**Important**: Make sure to replace the placeholder values in `.env` with your actual API keys before proceeding.

## Complete Pipeline

### Step 1: Supervised Fine-Tuning (SFT)

Train the base model using supervised fine-tuning on chosen responses:

```bash
bash scripts/training_sft.sh
```

Or run manually:

```bash
accelerate launch train_sft.py \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --dataset_names hh \
    --max_length 512 \
    --output_dir ./results/sft_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --push_to_hub \
    --hub_model_id "your-username/pythia-2.8b-sft"
```

### Step 2: Direct Preference Optimization (DPO)

Train the DPO model using preference pairs from the HH dataset:

```bash
bash scripts/training_dpo.sh
```

Or run manually:

```bash
python train_dpo.py \
    --model_name_or_path ./results/sft_model/merged_model \
    --dataset_names hh \
    --output_dir ./results/dpo_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --beta 0.2 \
    --loss_type sigmoid \
    --max_length 512 \
    --max_prompt_length 358 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --push_to_hub \
    --hub_model_id "your-username/pythia-2.8b-dpo"
```

### Step 3: Generate Inference Responses

#### 3.1 Generate Baseline Responses

Generate responses from base, SFT, and DPO models at multiple temperatures (0.25, 0.5, 0.75, 1.0):

```bash
bash scripts/generate_baselines.sh
```

This will create inference files in `inference_baselines/` for all three models at each temperature.

#### 3.2 Generate DPO Training Checkpoint Responses

Track DPO model performance across training checkpoints (steps 100-8000) at temperatures 0.7 and 1.0:

```bash
bash scripts/generate_dpo_training.sh
```

This will create inference files in `inference_dpo_training/` for each checkpoint.

### Step 4: Evaluate Win-Rates Using GPT-4o

#### 4.1 Evaluate Baseline Models

Compare model responses against chosen responses using GPT-4o as a judge:

```bash
bash scripts/evaluate_baselines.sh
```

Results are saved to `eval_baselines/`.

#### 4.2 Evaluate DPO Training Progress

Evaluate all DPO training checkpoints:

```bash
bash scripts/evaluate_dpo_training.sh
```

Results are saved to `eval_dpo_training/`.

### Step 5: Visualization and Analysis

Use the provided Jupyter notebook to analyze results and generate plots:

```bash
jupyter notebook nbs/Visualization.ipynb
```

The notebook generates:
- **Win-rate comparison plots** across different models and temperatures
- **Training progression plots** showing DPO improvement over time
- **Statistical tables** for reporting results

## Project Structure

```
dpo-replication/
├── train_sft.py                 # SFT training script
├── train_dpo.py                 # DPO training script
├── inference.py                 # Response generation script
├── evaluate.py                  # GPT-4o evaluation script
├── dataset_utils.py             # Dataset loading utilities
├── prompt.py                    # Evaluation prompts for GPT-4o
├── config/
│   └── dpo_config.yaml         # Configuration file
├── scripts/
│   ├── training_sft.sh         # SFT training script
│   ├── training_dpo.sh         # DPO training script
│   ├── generate_baselines.sh  # Generate baseline responses
│   ├── generate_dpo_training.sh # Generate DPO checkpoint responses
│   ├── evaluate_baselines.sh  # Evaluate baselines
│   └── evaluate_dpo_training.sh # Evaluate DPO training
├── nbs/
│   └── Visualization.ipynb     # Analysis and visualization notebook
├── inference_baselines/        # Generated responses from base/SFT/DPO models
├── inference_dpo_training/     # Generated responses from DPO checkpoints
├── eval_baselines/             # GPT-4o evaluation results for baselines
├── eval_dpo_training/          # GPT-4o evaluation results for DPO training
└── requirements.txt            # Python dependencies
```

## Key Features

### Training
- ✅ Supervised Fine-Tuning (SFT) on chosen responses
- ✅ Direct Preference Optimization with preference pairs
- ✅ LoRA for memory-efficient fine-tuning
- ✅ Multi-GPU support with `accelerate`
- ✅ Weights & Biases integration for experiment tracking
- ✅ Automatic checkpoint saving and model versioning

### Evaluation
- ✅ Automated response generation at multiple temperatures
- ✅ GPT-4o-based win-rate evaluation
- ✅ Tracking training progress across checkpoints
- ✅ Position bias mitigation in pairwise comparisons
- ✅ Comprehensive metrics and visualizations

## Evaluation Methodology

This project uses **GPT-4o as an automated evaluator** to compute win-rates by comparing model responses against reference (chosen) responses. The evaluation prompt follows the original DPO paper's methodology:

1. Present the query and two responses (model vs. chosen)
2. Ask GPT-4o which response is more helpful
3. Calculate win-rate as the percentage of times the model response is preferred

See `prompt.py` for the exact evaluation prompt used.

## Results

The project generates comprehensive results including:
- **Baseline comparisons**: Base model vs. SFT vs. DPO at various temperatures
- **Training dynamics**: DPO performance across 17 checkpoints (100-8000 steps)
- **Temperature analysis**: Impact of sampling temperature on response quality
- **Win-rate metrics**: Model vs. reference chosen responses

All results are visualized in `nbs/Visualization.ipynb`.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - LoRA is already enabled by default

2. **Slow Training**
   - Use multiple GPUs with `accelerate launch`
   - Adjust batch size and accumulation steps for optimal throughput

3. **Training Loss is NaN**
   - Use `float32` instead of `float16` (set in training scripts)
   - Adjust beta parameter (default: 0.2)

4. **API Rate Limits (Evaluation)**
   - The evaluation scripts include rate limiting for GPT-4o API
   - Consider running in batches if you have many samples

## Original Paper

This implementation replicates results from:

**"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**  
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn  
NeurIPS 2023

[Paper](https://arxiv.org/abs/2305.18290) | [Original Code](https://github.com/eric-mitchell/direct-preference-optimization)

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## License

MIT
