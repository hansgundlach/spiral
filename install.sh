#!/bin/bash

# SPIRAL Installation Script
# This script automates the installation of SPIRAL framework for single A100 GPU usage

set -e  # Exit on any error

echo "🚀 Starting SPIRAL installation..."

# Check if we're in the right directory
if [ ! -f "train_spiral.py" ]; then
    echo "❌ Error: Please run this script from the SPIRAL root directory"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$python_version" != "3.10" ]; then
    echo "⚠️  Warning: Python 3.10 is recommended, found $python_version"
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please install NVIDIA drivers and CUDA."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
echo "📦 Installing vLLM..."
pip install vllm==0.8.4

# Install OAT framework  
echo "📦 Installing OAT framework..."
pip install oat-llm==0.1.3.post1

# Install remaining requirements
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt

# Fix ANTLR version conflicts
echo "🔧 Fixing ANTLR version conflicts..."
pip install antlr4-python3-runtime==4.13.2 --force-reinstall

# Upgrade packaging
echo "🔧 Upgrading packaging..."
pip install --upgrade packaging

# Verify installation
echo "🧪 Verifying installation..."
python -c "
import spiral
import textarena as ta
import vllm
import torch
import wandb

print('✅ All imports successful')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    print(f'✅ GPU name: {torch.cuda.get_device_name()}')
else:
    print('❌ CUDA not available')
    exit(1)
"

# Download Qwen3-4B model
echo "📥 Pre-downloading Qwen3-4B model..."
python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading Qwen3-4B model...')
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
os.makedirs(cache_dir, exist_ok=True)

try:
    model_path = snapshot_download(
        repo_id='Qwen/Qwen3-4B-Base',
        cache_dir=cache_dir,
        resume_download=True
    )
    print(f'✅ Model downloaded to: {model_path}')
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
    print('Model will be downloaded automatically on first training run.')
"

echo ""
echo "🎉 SPIRAL installation complete!"
echo ""
echo "Next steps:"
echo "1. Set up Weights & Biases: wandb login YOUR_API_KEY"
echo "2. Start training: bash run.sh"
echo "3. Monitor progress at: https://wandb.ai"
echo ""
echo "For detailed instructions, see SETUP.md"