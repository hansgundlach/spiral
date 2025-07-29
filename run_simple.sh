# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simplified Training Run - No API Keys Required
# This script removes complex evaluation components and focuses on game metrics

# GPU Memory Check and Cleanup =========
echo "Checking GPU memory usage..."
nvidia-smi

echo "Cleaning up any existing GPU processes..."
# Kill any existing Python training processes
pkill -f "train_spiral.py" || true
pkill -f "python.*train" || true

# Wait a moment for processes to clean up
sleep 2

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "GPU status after cleanup:"
nvidia-smi

# Check available GPU memory
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Available GPU memory: ${FREE_MEM} MiB"

if [ "$FREE_MEM" -lt 8000 ]; then
    echo "Warning: Less than 8GB GPU memory available. Training may fail."
    echo "Consider reducing batch sizes or killing other GPU processes."
fi

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG

# Simplified Training Configuration - GAME METRICS ONLY
# - No external API opponents (only random opponent)
# - Single environment for focused training  
# - NO benchmark evaluations (disabled by default with --eval_data "")
# - Only game metrics logged to wandb for monitoring LLM game improvement
# 
# To enable benchmark evaluations later, change --eval_data "" to --eval_data "./data"

python train_spiral.py \
    --env_id KuhnPoker-v1 \
    --use_llm_obs_wrapper \
    --filter_zero_adv \
    --eval_env_ids KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers True \
    --eval_opponent_names random \
    --eval_split all \
    --gamma 1 \
    --gpus 1 \
    --gradient-checkpointing \
    --zero_stage 3 \
    --adam_offload \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0.0 \
    --target-modules all-linear \
    --num_samples 1 \
    --rollout_batch_size 32 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 2 \
    --pi_buffer_maxlen_per_device 2 \
    --pretrain Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --collocate \
    --vllm-sleep \
    --vllm-gpu-ratio 0.4 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 32 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 4096 \
    --generate_max_length 1024 \
    --max_context_length 8192 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 2 \
    --save_steps -1 \
    --eval_games 2 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 100 \
    --max_save_num 10 \
    --eval_data "" \
    --use-wb \
    --wb-run-name spiral-qwen2.5-1.5b-kuhn-poker \
    --wb_project spiral-simple




