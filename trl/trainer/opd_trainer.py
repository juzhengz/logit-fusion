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

from __future__ import annotations

import warnings
from typing import Any

import torch
from transformers import PreTrainedModel

from .grpo_trainer import GRPOTrainer


class OnPolicyDistillationTrainer(GRPOTrainer):
    """On-policy distillation trainer built on GRPO with teacher-student logprob advantages."""

    def __init__(self, *args, teacher_model: str | PreTrainedModel | None = None, **kwargs):
        super().__init__(*args, teacher_model=teacher_model, **kwargs)
        if self.teacher_model is None:
            raise ValueError("OnPolicyDistillationTrainer requires a teacher_model for logprob advantages.")
        if self.logit_fusion_alpha is not None:
            warnings.warn(
                "logit_fusion_alpha is set but OnPolicyDistillationTrainer uses student-only rollouts. "
                "Logit fusion will be disabled.",
                UserWarning,
            )
        self.logit_fusion_alpha = None
        self.logit_fusion_alpha_schedule = None
        self.use_fusion_importance_sampling = False

    def _generate_single_turn(self, prompts: list):
        teacher_model = self.teacher_model
        try:
            self.teacher_model = None
            return super()._generate_single_turn(prompts)
        finally:
            self.teacher_model = teacher_model

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        if self.teacher_model is None:
            raise ValueError("OnPolicyDistillationTrainer requires a teacher_model for logprob advantages.")

        teacher_model = self.teacher_model
        try:
            self.teacher_model = None
            output = super()._generate_and_score_completions(inputs)
        finally:
            self.teacher_model = teacher_model

        prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
        completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        batch_size = (
            self.args.per_device_train_batch_size
            if self.model.training
            else self.args.per_device_eval_batch_size
        )
        forward_kwargs = {
            "pixel_values": output.get("pixel_values"),
            "image_grid_thw": output.get("image_grid_thw"),
            "num_images": output.get("num_images"),
            "pixel_attention_mask": output.get("pixel_attention_mask"),
            "image_sizes": output.get("image_sizes"),
            "token_type_ids": output.get("token_type_ids"),
        }
        forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

        with torch.no_grad():
            teacher_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                teacher_model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                **forward_kwargs,
            )

        student_per_token_logps = output.get("old_per_token_logps")
        if student_per_token_logps is None:
            with torch.no_grad():
                student_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    **forward_kwargs,
                )

        token_counts = completion_mask.sum(-1).clamp(min=1.0)
        teacher_logprob = (teacher_per_token_logps * completion_mask).sum(-1) / token_counts
        student_logprob = (student_per_token_logps * completion_mask).sum(-1) / token_counts
        advantages = teacher_logprob - student_logprob

        output["advantages"] = advantages

        all_advantages = self.accelerator.gather(advantages).detach().cpu().tolist()
        self._logs["advantages"].clear()
        self._logs["advantages"].extend(all_advantages)

        return output
