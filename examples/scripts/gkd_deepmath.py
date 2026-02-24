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

import os
from dataclasses import dataclass, field

import torch
import torch._dynamo as dynamo
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.import_utils import is_math_verify_available
from trl.rewards import accuracy_reward


@dataclass
class DifficultyConfig:
    difficulty_tier: int = field(
        default=1,
        metadata={"help": "Difficulty tier for training data: 1 (<4), 2 (4-7), 3 (>7)."},
    )


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


def _build_completion(example: dict) -> str:
    solution = example.get("solution") or example.get("answer") or example.get("final_answer")
    if not solution:
        return ""
    completion = str(solution).strip()
    if completion and "\\boxed" not in completion:
        completion = f"\\boxed{{{completion}}}"
    if completion and "<think>" not in completion:
        completion = f"<think>\n\n</think>\n{completion}"
    return completion


def make_conversation(example: dict) -> dict:
    prompt = example["prompt"]
    completion = _build_completion(example)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *prompt, {"role": "assistant", "content": completion}]
    return {"messages": messages}


if __name__ == "__main__":
    # Enable logging in a Hugging Face Space
    os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
    dynamo.config.disable = True
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    parser = TrlParser((ScriptArguments, GKDConfig, ModelConfig, DifficultyConfig))
    script_args, training_args, model_args, difficulty_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=True,
    )
    if quantization_config is not None:
        teacher_model_kwargs["device_map"] = get_kbit_device_map()
        teacher_model_kwargs["quantization_config"] = quantization_config
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    training_args.remove_unused_columns = False
    if training_args.dataset_kwargs is None:
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    else:
        training_args.dataset_kwargs["skip_prepare_dataset"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    ################
    # Training
    ################
    trainer = GKDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        reward_func=accuracy_reward,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
