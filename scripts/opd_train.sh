#!/bin/bash
module load cuda/11.8.0
conda activate trl

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

WANDB_PROJECT=trl \
accelerate launch \
    --num_processes 1 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/opd.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B \
    --output_dir /cmlscratch/juzheng/trl/checkpoints/opd-Qwen2.5-0.5B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" "down_proj" "up_proj" "gate_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8 \
    --report_to wandb \
    --run_name opd-Qwen2.5-0.5B \
    --num_completions_to_print 2 \
    --difficulty_tier 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 1000 \
    --use_importance_sampling_correction False \
