# SPIRAL Setup Guide

This guide provides step-by-step instructions to set up the SPIRAL (Self-Play on Zero-Sum Games Incentivizes Reasoning) framework for single A100 GPU usage.

## Prerequisites

- **Hardware**: NVIDIA A100 GPU (40GB VRAM) or equivalent
- **CUDA**: Version 12.x 
- **Python**: 3.10
- **OS**: Ubuntu 20.04+ or similar Linux distribution

## Installation Steps

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -y -n spiral python=3.10
conda activate spiral

# Verify CUDA is available
nvidia-smi
```

### 2. Install Dependencies

```bash
# Clone the repository (if not already done)
git clone https://github.com/spiral-rl/spiral.git
cd spiral

# Install core dependencies in the correct order
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
pip install vllm==0.8.4

# Install OAT framework
pip install oat-llm==0.1.3.post1

# Install remaining requirements
pip install -r requirements.txt

# Fix potential ANTLR version conflicts
pip install antlr4-python3-runtime==4.13.2 --force-reinstall

# Upgrade packaging if needed
pip install --upgrade packaging
```

### 3. Download Pre-trained Model

The Qwen3-4B model will be automatically downloaded on first run, but you can pre-download it:

```bash
python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading Qwen3-4B model...')
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
os.makedirs(cache_dir, exist_ok=True)

model_path = snapshot_download(
    repo_id='Qwen/Qwen3-4B-Base',
    cache_dir=cache_dir,
    resume_download=True
)
print(f'Model downloaded to: {model_path}')
"
```

### 4. Setup Weights & Biases (W&B)

```bash
# Login to W&B with your API key
wandb login YOUR_WANDB_API_KEY

# Verify login
python -c "import wandb; wandb.login(); print('W&B login successful!')"
```

### 5. Configuration for Single GPU

The repository is pre-configured for single A100 usage. Key settings in `run.sh`:

- `--gpus 1` (single GPU)
- `--vllm_gpu_ratio 0.8` (80% GPU memory for vLLM)
- `--max_train 1000` (reduced steps for prototyping)

To modify for different hardware:
- **Multiple GPUs**: Change `--gpus N` where N is number of GPUs
- **Different GPU memory**: Adjust `--vllm_gpu_ratio` (0.3-0.9 range)
- **Longer training**: Increase `--max_train` value

### 6. Verify Installation

Run a quick functionality test:

```bash
python -c "
import spiral
import textarena as ta
import vllm
import torch
import wandb

print('✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'✓ GPU name: {torch.cuda.get_device_name()}')
print('✓ Setup verification complete!')
"
```

## Running Training

### Quick Start

```bash
# Start training with default settings (Kuhn Poker, 1000 steps)
bash run.sh
```

### Monitor Training

- **W&B Dashboard**: Monitor real-time metrics at https://wandb.ai
- **Project Name**: `spiral`
- **Run Name**: `spiral-qwen3-4b-base-kp-4k-self-play`

### Key Metrics to Watch

1. **Game Performance**:
   - Win rate against random opponent
   - Win rate on out-of-domain games
   
2. **Learning Progress**:
   - Policy loss
   - Value loss
   - Reward trends

3. **Math Reasoning**:
   - Accuracy on math benchmarks
   - MATH dataset performance

## Customization

### Different Games

Edit `run.sh` to change the training environment:

```bash
# For TicTacToe
--env_id TicTacToe-v0 \
--use_llm_obs_wrapper false \

# For SimpleNegotiation  
--env_id SimpleNegotiation-v1 \
--use_llm_obs_wrapper true \
```

### Training Duration

Adjust training steps in `run.sh`:

```bash
# For longer training
--max_train 10000 \

# For quick prototyping
--max_train 100 \
```

### Batch Sizes

For different GPU memory constraints:

```bash
# Smaller batches (lower memory)
--rollout_batch_size 64 \
--train_batch_size 64 \

# Larger batches (higher memory, faster training)
--rollout_batch_size 256 \
--train_batch_size 256 \
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `--vllm_gpu_ratio` to 0.6 or lower
   - Decrease batch sizes
   - Reduce `--max_model_len`

2. **ANTLR Version Conflicts**:
   ```bash
   pip install antlr4-python3-runtime==4.13.2 --force-reinstall
   ```

3. **W&B Connection Issues**:
   ```bash
   wandb login --relogin
   ```

4. **Model Download Failures**:
   - Check internet connection
   - Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
   - Re-run model download script

### Performance Optimization

1. **Enable Mixed Precision**: Already enabled via `--gradient-checkpointing`
2. **Optimize GPU Memory**: Adjust `--vllm_gpu_ratio` based on available VRAM
3. **Parallel Data Loading**: Uses optimized data loaders by default

## File Structure

```
spiral/
├── run.sh                 # Main training script
├── train_spiral.py        # Training entry point
├── requirements.txt       # Python dependencies
├── SETUP.md              # This setup guide
├── spiral/               # Core SPIRAL framework
├── data/                 # Training datasets
└── evals/               # Evaluation scripts
```

## Hardware Requirements

### Minimum:
- 1x NVIDIA A100 (40GB) or equivalent
- 32GB System RAM
- 100GB free disk space

### Recommended:
- Multiple A100s for faster training
- 64GB+ System RAM
- 500GB+ NVMe SSD storage

## Support

For issues specific to this setup:
1. Check W&B logs for training metrics
2. Review game state dumps in `./saves/game_state/`
3. Verify GPU memory usage with `nvidia-smi`
4. Check SPIRAL logs for detailed error messages

For framework issues, refer to:
- [SPIRAL Paper](https://arxiv.org/abs/2506.24119)
- [SPIRAL Repository](https://github.com/spiral-rl/spiral)
- [OAT Framework](https://github.com/sail-sg/oat)