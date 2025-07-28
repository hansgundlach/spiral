#!/usr/bin/env python3
"""
SPIRAL Setup Validation Script

This script validates that all components of SPIRAL are correctly installed
and configured for single A100 GPU usage.
"""

import sys
import os
import subprocess
import importlib.util

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def check_gpu():
    """Check GPU availability and specifications."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ GPU Count: {gpu_count}")
        
        if "A100" not in gpu_name and gpu_memory < 20:
            print("⚠️  Warning: Recommended GPU is A100 (40GB) or equivalent")
        
        return True
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_model_access():
    """Check if Qwen3-4B model is accessible."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base')
        print("✅ Qwen3-4B model accessible")
        return True
    except Exception as e:
        print(f"❌ Qwen3-4B model access failed: {e}")
        return False

def check_environment():
    """Check TextArena environment creation."""
    try:
        from spiral.envs import make_env
        env = make_env('KuhnPoker-v1', use_llm_obs_wrapper=True)
        env.reset(num_players=2, seed=42)
        player_id, obs = env.get_observation()
        print(f"✅ TextArena environment (KuhnPoker)")
        return True
    except Exception as e:
        print(f"❌ TextArena environment failed: {e}")
        return False

def check_wandb():
    """Check Wandb configuration."""
    try:
        import wandb
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Wandb configured for user: {user.username}")
        return True
    except Exception as e:
        print(f"⚠️  Wandb not configured: {e}")
        print("   Run: wandb login YOUR_API_KEY")
        return False

def check_training_script():
    """Check if training script can be executed."""
    try:
        result = subprocess.run(
            [sys.executable, 'train_spiral.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✅ Training script executable")
            return True
        else:
            print(f"❌ Training script failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Training script check failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔍 SPIRAL Setup Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Core imports
    print("\n📦 Core Dependencies:")
    checks = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('vllm', 'vLLM'),
        ('textarena', 'TextArena'),
        ('spiral', 'SPIRAL'),
        ('wandb', 'Weights & Biases'),
        ('deepspeed', 'DeepSpeed'),
        ('flash_attn', 'Flash Attention'),
    ]
    
    for module, name in checks:
        if not check_import(module, name):
            all_passed = False
    
    # Hardware checks
    print("\n🖥️  Hardware:")
    if not check_gpu():
        all_passed = False
    
    # Model access
    print("\n🤖 Model Access:")
    if not check_model_access():
        all_passed = False
    
    # Environment check  
    print("\n🎮 Game Environment:")
    if not check_environment():
        all_passed = False
    
    # Wandb check
    print("\n📊 Experiment Tracking:")
    if not check_wandb():
        all_passed = False
    
    # Training script check
    print("\n🚀 Training Script:")
    if not check_training_script():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! SPIRAL is ready to run.")
        print("\nNext steps:")
        print("1. If Wandb isn't configured: wandb login YOUR_API_KEY")
        print("2. Start training: bash run.sh")
        print("3. Monitor at: https://wandb.ai")
    else:
        print("❌ Some checks failed. Please review the output above.")
        print("📖 See SETUP.md for detailed installation instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()