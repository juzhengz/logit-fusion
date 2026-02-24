# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen3-0.6B
pip install num2words==0.5.14

module load cuda/11.8.0

WANDB_PROJECT=trl \
accelerate launch \
    --num_processes 1 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/opd.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --teacher_model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir /cmlscratch/juzheng/trl/checkpoints/opd-Qwen2.5-0.5B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj" "v_proj" \
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
    --run_name Qwen2.5-0.5B-opd \
    --num_completions_to_print 2 \
    --difficulty_tier 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 1000

"""

import os

import torch
from datasets import load_dataset
from dataclasses import dataclass, field

from trl import (
    GRPOConfig,
    ModelConfig,
    OnPolicyDistillationTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward


@dataclass
class TeacherConfig:
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Frozen teacher model used to define logprob advantages."},
    )
    teacher_model_revision: str | None = field(
        default=None, metadata={"help": "Revision of the teacher model to use (branch, tag, or commit hash)."}
    )


@dataclass
class DifficultyConfig:
    difficulty_tier: int = field(
        default=1,
        metadata={"help": "Difficulty tier for training data: 1 (<4), 2 (4-7), 3 (>7)."},
    )


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig, TeacherConfig, DifficultyConfig))
    script_args, training_args, model_args, teacher_args, difficulty_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    if teacher_args.teacher_model_name_or_path is None:
        raise ValueError("teacher_model_name_or_path must be set for on-policy distillation.")

    training_args.teacher_model_init_kwargs = dict(
        revision=teacher_args.teacher_model_revision or model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    if quantization_config is not None:
        training_args.teacher_model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.teacher_model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = load_dataset("juzhengz/DeepMath-103K", split=["train", "test[:1%]"])
    if difficulty_args.difficulty_tier == 1:
        train_dataset = train_dataset.filter(lambda example: example["difficulty"] < 4)
    elif difficulty_args.difficulty_tier == 2:
        train_dataset = train_dataset.filter(lambda example: 4 <= example["difficulty"] <= 7)
    elif difficulty_args.difficulty_tier == 3:
        train_dataset = train_dataset.filter(lambda example: example["difficulty"] > 7)
    else:
        raise ValueError(f"Unsupported difficulty_tier: {difficulty_args.difficulty_tier}. Use 1, 2, or 3.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    SYSTEM_PROMPT = (
        "A conversation between a user and an assistant. The user asks a question, and the assistant solves it.\n"
        "The assistant should reason step by step, then provide the final answer.\n"
        "The reasoning process must be enclosed within <think></think> tags.\n"
        "The final answer must be written in valid LaTeX and MUST be enclosed in \\boxed{...}. "
        "Answers not enclosed in \\boxed{...} will be considered incorrect.\n"
        "Format:\n"
        "<think>\n"
        "step-by-step reasoning here\n"
        "</think>\n"
        "\\boxed{final answer}"
    )

    def make_conversation(example):
        prompt = example["prompt"]
        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, *prompt],
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    ################
    # Training
    ################
    trainer = OnPolicyDistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=teacher_args.teacher_model_name_or_path,
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
