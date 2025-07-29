#!/bin/bash
# Ultra-minimal test to verify training starts and logs metrics quickly

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
    --zero_stage 3 \
    --adam_offload \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --target-modules all-linear \
    --num_samples 1 \
    --rollout_batch_size 8 \
    --num_envs 1 \
    --rollout_batch_size_per_device 1 \
    --pi_buffer_maxlen_per_device 1 \
    --pretrain Qwen/Qwen3-4B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.7 \
    --learning_rate 0.00001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 1 \
    --train_batch_size 8 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 2048 \
    --generate_max_length 512 \
    --max_context_length 4096 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 2 \
    --save_steps -1 \
    --eval_games 1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 1024 \
    --max_train 10 \
    --max_save_num 2 \
    --eval_data "" \
    --use-wb \
    --wb-run-name spiral-test-quick \
    --wb_project spiral-test