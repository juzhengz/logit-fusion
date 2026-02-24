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
    --teacher_model_name_or_path Qwen/Qwen2.5-7B \
    --logit_fusion_alpha 0.5 \
    --logit_fusion_alpha_schedule linear \
    --logit_fusion_alpha_decay_steps 1000 \
    --vllm_mode colocate \
    --output_dir /cmlscratch/juzheng/trl/checkpoints/GSPO-Qwen2.5-0.5B-alpha-0.5-steps-1k-lr-1e-5-no-clipping \
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
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --report_to wandb \
    --run_name GSPO-Qwen2.5-0.5B-alpha-0.5-steps-1k-lr-1e-5-no-clipping \
    --num_completions_to_print 2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 1000 \
    --use_importance_sampling_correction False \
    --use_fusion_importance_sampling True \
    --disable_importance_sampling_clipping True \
    --difficulty_tier '<4' \



