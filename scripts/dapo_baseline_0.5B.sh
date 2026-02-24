#!/bin/bash
module load cuda/11.8.0
conda activate trl

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRL_MATH_VERIFY_PARSING_TIMEOUT=30
export TRL_MATH_VERIFY_VERIFY_TIMEOUT=30

WANDB_PROJECT=trl \
accelerate launch \
    --num_processes 1 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/gspo.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --vllm_mode colocate \
    --output_dir /cmlscratch/juzheng/trl/checkpoints/DAPO-Qwen2.5-0.5B-lr-1e-5-tier-1-baseline \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --use_peft \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" "down_proj" "up_proj" "gate_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_generations 8 \
    --steps_per_generation 8 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --beta 0.0 \
    --loss_type dapo \
    --gradient_accumulation_steps 2 \
    --report_to wandb \
    --run_name DAPO-Qwen2.5-0.5B-lr-1e-5-tier-1-baseline \
    --num_completions_to_print 2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 1000 \
    --difficulty_tier '<4' \