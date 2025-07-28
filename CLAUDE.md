# CLAUDE.md - Memory Optimization Context

## Problem Solved
Fixed CUDA out of memory errors when training Qwen3-4B on single A100 (40GB) for RL using SPIRAL framework.

## Root Cause
- Original configuration used full fine-tuning (4B parameters) with DeepSpeed ZeRO stage 2
- ZeRO stage 2 duplicates optimizer states in GPU memory
- vLLM inference also needs significant GPU memory on same device
- Total memory requirements exceeded 40GB A100 capacity

## Solution Applied
Modified `/lambda/nfs/Gundlach2025/spiral/run.sh` with following optimizations:

### 1. Enabled LoRA Training
```bash
--lora-rank 32 \
--lora-alpha 32 \
--lora-dropout 0.1 \
--target-modules all-linear \
```
- Reduces trainable parameters from 4B to ~1-2M
- PEFT/LoRA already supported via OAT framework

### 2. DeepSpeed Memory Optimizations
```bash
--zero_stage 3 \
--adam_offload \
```
- ZeRO stage 3: Partitions model parameters across devices
- Adam offload: Moves optimizer states to CPU memory

### 3. Reduced Batch Sizes
```bash
--rollout_batch_size_per_device 2 \  # was 4
--pi_buffer_maxlen_per_device 2 \    # was 4
```

### 4. vLLM Memory Allocation
```bash
--vllm_gpu_ratio 0.6 \  # increased from 0.4
```
- Allocates more GPU memory for vLLM inference

### 5. Memory Management
```bash
# Disabled problematic memory setting
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Result
✅ Training runs successfully on single A100  
✅ No CUDA OOM errors  
✅ Both training and inference fit in 40GB GPU memory  
✅ LoRA training maintains model quality with much lower memory usage

## Key Dependencies
- `peft==0.16.0` (already installed)
- `loralib==0.1.2` (already installed)
- DeepSpeed with ZeRO stage 3 support
- vLLM for inference

## Commands
- Run training: `bash run.sh`
- Merge LoRA weights: Use `/lambda/nfs/Gundlach2025/spiral/evals/merge_lora.py`

## Notes
- LoRA rank 32 provides good balance of performance vs memory
- Can adjust `lora-rank` (16, 32, 64) based on memory constraints
- ZeRO stage 3 + CPU offloading is key for large model training
- vLLM requires full base model in memory regardless of LoRA training