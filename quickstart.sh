#!/bin/bash

# SPIRAL Quick Start Script
# Validates setup and starts training immediately

set -e

echo "🚀 SPIRAL Quick Start"
echo "===================="

# Check if we're in the right directory
if [ ! -f "train_spiral.py" ]; then
    echo "❌ Error: Please run this script from the SPIRAL root directory"
    exit 1
fi

# Validate setup
echo "🔍 Validating setup..."
python validate_setup.py

if [ $? -ne 0 ]; then
    echo "❌ Setup validation failed. Please run 'bash install.sh' first."
    exit 1
fi

echo ""
echo "✅ Setup validation passed!"
echo ""

# Check if user wants to proceed
echo "🎯 Ready to start SPIRAL training:"
echo "   - Environment: Kuhn Poker"
echo "   - Model: Qwen3-4B-Base"
echo "   - GPU: Single A100"
echo "   - Steps: 1000 (prototyping)"
echo "   - Wandb: Enabled"
echo ""

read -p "Start training now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting SPIRAL training..."
    echo "📊 Monitor progress at: https://wandb.ai"
    echo ""
    bash run.sh
else
    echo "👋 Training cancelled. Run 'bash run.sh' when ready."
fi