#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/logit-fusion-gspo}"
RUN_NAME="${RUN_NAME:-logit-fusion-gspo}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/accelerate_configs/deepspeed_zero3.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ALPHA_INIT="${ALPHA_INIT:-0.5}"
ALPHA_DECAY_STEPS="${ALPHA_DECAY_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
REPORT_TO="${REPORT_TO:-none}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TRL_MATH_VERIFY_PARSING_TIMEOUT="${TRL_MATH_VERIFY_PARSING_TIMEOUT:-30}"
export TRL_MATH_VERIFY_VERIFY_TIMEOUT="${TRL_MATH_VERIFY_VERIFY_TIMEOUT:-30}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --config_file "${ACCELERATE_CONFIG}" \
  examples/scripts/gspo.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --teacher_model_name_or_path "${TEACHER_MODEL_NAME_OR_PATH}" \
  --logit_fusion_alpha "${ALPHA_INIT}" \
  --logit_fusion_alpha_schedule linear \
  --logit_fusion_alpha_decay_steps "${ALPHA_DECAY_STEPS}" \
  --vllm_mode colocate \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type constant \
  --dtype bfloat16 \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --use_peft \
  --lora_target_modules q_proj v_proj k_proj o_proj down_proj up_proj gate_proj \
  --log_completions \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --steps_per_generation "${STEPS_PER_GENERATION}" \
  --loss_type grpo \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --epsilon 3e-4 \
  --epsilon_high 4e-4 \
  --beta 0.0 \
  --use_fusion_importance_sampling True \
  --disable_importance_sampling_clipping True \
  --importance_sampling_level sequence \
  --report_to "${REPORT_TO}" \
  --num_completions_to_print 2 \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 1000 \
  "$@"
