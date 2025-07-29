#!/bin/bash
# Ultra-fast debug configuration - triggers errors ASAP

export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0

python train_spiral.py \
    --env_id KuhnPoker-v1 \
    --use_llm_obs_wrapper \
    --eval_env_ids KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers True \
    --eval_opponent_names random \
    --gamma 1 \
    --gpus 1 \
    --gradient-checkpointing \
    --zero_stage 2 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --target-modules all-linear \
    --num_samples 1 \
    --rollout_batch_size 4 \
    --num_envs 1 \
    --rollout_batch_size_per_device 1 \
    --pi_buffer_maxlen_per_device 1 \
    --pretrain Qwen/Qwen2.5-1.5B-Instruct \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.3 \
    --learning_rate 0.00001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 1 \
    --train_batch_size 4 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 1024 \
    --generate_max_length 256 \
    --max_context_length 2048 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 1 \
    --save_steps -1 \
    --eval_games 1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 512 \
    --max_train 5 \
    --max_save_num 1 \
    --eval_data "" \
    --use-wb \
    --wb-run-name spiral-debug-fast \
    --wb_project spiral-debug