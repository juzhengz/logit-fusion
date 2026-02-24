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

import random
import textwrap
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available, logging

from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import DataCollatorForChatML, disable_dropout_in_model, empty_cache, get_config_model_id, nanstd
from .gkd_config import GKDConfig


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


logger = logging.get_logger(__name__)
DEFAULT_REWARD_SOLUTION_KEYS = ("solution", "answer", "final_answer")


def _extract_reward_solution(example: dict[str, Any], solution_keys: tuple[str, ...]) -> str | None:
    for key in solution_keys:
        solution = example.get(key)
        if solution is not None:
            return str(solution)
    return None


class _RewardCollatorWrapper:
    def __init__(self, base_collator: DataCollator, solution_keys: tuple[str, ...]):
        self.base_collator = base_collator
        self.solution_keys = solution_keys

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(examples)
        if "reward_solution" not in batch:
            batch["reward_solution"] = [_extract_reward_solution(example, self.solution_keys) for example in examples]
        return batch


class GKDTrainer(SFTTrainer):
    """Trainer for Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`experimental.gkd.GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a [`DataCollatorForChatML`] using the
            `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
        reward_func (`Callable`, *optional*):
            Reward function used to compute rewards for generated completions. If provided, rewards are logged during
            training and evaluation.
        reward_solution_keys (`tuple[str, ...]`, *optional*):
            Keys to search in dataset examples for the ground-truth solution used by the reward function. Defaults to
            ("solution", "answer", "final_answer").
    """

    _tag_names = ["trl", "gkd"]
    _name = "GKD"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
        reward_func: Callable | None = None,
        reward_solution_keys: tuple[str, ...] | None = None,
    ):
        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields (e.g., "messages") remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        reward_solution_keys = reward_solution_keys or DEFAULT_REWARD_SOLUTION_KEYS
        if reward_func is not None:
            data_collator = _RewardCollatorWrapper(data_collator, reward_solution_keys)

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        teacher_model_id = None
        if isinstance(teacher_model, str):
            teacher_model_id = teacher_model
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        teacher_vocab_size = getattr(teacher_model.config, "vocab_size", None)
        model_vocab_size = getattr(self.model.config, "vocab_size", None)
        if teacher_vocab_size is not None and model_vocab_size is not None and teacher_vocab_size != model_vocab_size:
            if isinstance(processing_class, PreTrainedTokenizerBase):
                teacher_tokenizer_id = teacher_model_id or get_config_model_id(teacher_model.config)
                teacher_tokenizer_kwargs = {}
                if isinstance(args.teacher_model_init_kwargs, dict):
                    trust_remote_code = args.teacher_model_init_kwargs.get("trust_remote_code")
                    if trust_remote_code is not None:
                        teacher_tokenizer_kwargs["trust_remote_code"] = trust_remote_code
                try:
                    teacher_tokenizer = AutoTokenizer.from_pretrained(
                        teacher_tokenizer_id, **teacher_tokenizer_kwargs
                    )
                except Exception as exc:
                    raise ValueError(
                        "The teacher model vocab size {} does not match the student model vocab size {} and the "
                        "teacher tokenizer could not be loaded for a vocab check.".format(
                            teacher_vocab_size, model_vocab_size
                        )
                    ) from exc
                if processing_class.get_vocab() != teacher_tokenizer.get_vocab():
                    raise ValueError(
                        "The teacher model vocab size {} does not match the student model vocab size {}.".format(
                            teacher_vocab_size, model_vocab_size
                        )
                    )
            logger.warning(
                "Teacher vocab size %s does not match student vocab size %s; extra teacher logits will be dropped.",
                teacher_vocab_size,
                model_vocab_size,
            )

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.reward_func = reward_func
        self.reward_func_name = reward_func.__name__ if reward_func is not None else None
        self._warned_missing_reward_solution = False

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    def _compute_reward_metrics(self, model, inputs, mode: str) -> None:
        if self.reward_func is None:
            return

        solutions = inputs.get("reward_solution")
        if solutions is None:
            if not self._warned_missing_reward_solution:
                logger.warning("Reward logging is enabled, but no reward solutions were found in the inputs.")
                self._warned_missing_reward_solution = True
            return

        prompt_ids = inputs.get("prompts")
        if prompt_ids is None:
            return

        was_training = model.training
        model.eval()
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model, torch.no_grad():
            generated_outputs = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=inputs.get("prompt_attention_mask"),
                generation_config=self.generation_config,
                return_dict_in_generate=True,
            )
        if was_training:
            model.train()

        generated_tokens = generated_outputs.sequences
        completion_ids = generated_tokens[:, prompt_ids.shape[1] :]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": text}] for text in completions_text]

        valid_indices = [idx for idx, solution in enumerate(solutions) if solution is not None]
        if not valid_indices:
            return

        valid_completions = [completions[idx] for idx in valid_indices]
        valid_solutions = [str(solutions[idx]) for idx in valid_indices]
        try:
            rewards_list = self.reward_func(valid_completions, valid_solutions)
        except Exception as exc:
            logger.warning("Reward function %s failed: %s", self.reward_func_name, exc)
            return

        full_rewards = [float("nan")] * len(completions)
        for idx, reward in zip(valid_indices, rewards_list, strict=True):
            full_rewards[idx] = float("nan") if reward is None else float(reward)

        reward_tensor = torch.tensor(full_rewards, device=prompt_ids.device, dtype=torch.float32)
        gathered_rewards = self.accelerator.gather_for_metrics(reward_tensor)
        non_nan = torch.sum(~torch.isnan(gathered_rewards)).item()
        if non_nan == 0:
            return

        mean_reward = torch.nanmean(gathered_rewards).item()
        std_reward = nanstd(gathered_rewards).item() if non_nan > 1 else 0.0
        reward_key = f"rewards/{self.reward_func_name}/mean"
        self._metrics[mode][reward_key].append(mean_reward)
        self._metrics[mode][f"rewards/{self.reward_func_name}/std"].append(std_reward)
        self._metrics[mode]["reward"].append(mean_reward)
        self._metrics[mode]["reward_std"].append(std_reward)

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_gkd_loss:
            # Forward only through the base models (avoid lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                base_student = unwrapped_student.get_decoder()
            else:
                base_student = getattr(
                    unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                )

            student_outputs = base_student(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
                base_teacher = unwrapped_teacher.get_decoder()
            else:
                base_teacher = getattr(
                    unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
                )
            with torch.no_grad():
                teacher_outputs = base_teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

            # hidden states (shifted)
            student_hidden = student_outputs.last_hidden_state[:, :-1]
            teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

            # Release full outputs to free memory
            del student_outputs, teacher_outputs

            # labels mask and labels (shifted)
            labels_mask = inputs["labels"] != -100
            masked_input_ids = torch.where(
                labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
            )
            true_labels = masked_input_ids[:, 1:].contiguous()

            # Release intermediate tensors
            del labels_mask, masked_input_ids

            # heads
            student_head = unwrapped_student.get_output_embeddings()
            teacher_head = unwrapped_teacher.get_output_embeddings()
            if teacher_head.weight.size(0) != student_head.weight.size(0):
                if teacher_head.weight.size(0) < student_head.weight.size(0):
                    raise ValueError(
                        "Teacher vocab size {} is smaller than student vocab size {}.".format(
                            teacher_head.weight.size(0), student_head.weight.size(0)
                        )
                    )
                teacher_weight = teacher_head.weight[: student_head.weight.size(0)]
                teacher_bias = (
                    teacher_head.bias[: student_head.weight.size(0)] if getattr(teacher_head, "bias", None) else None
                )
            else:
                teacher_weight = teacher_head.weight
                teacher_bias = getattr(teacher_head, "bias", None)

            # liger fused jsd loss
            loss = self.liger_jsd_loss(
                student_input=student_hidden,
                student_weight=student_head.weight,
                teacher_input=teacher_hidden,
                teacher_weight=teacher_weight,
                true_labels=true_labels,
                student_bias=getattr(student_head, "bias", None),
                teacher_bias=teacher_bias,
            )

            # Release hidden states after loss computation
            del student_hidden, teacher_hidden, true_labels
        else:
            # compute student output
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # compute teacher output in eval mode
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            # slice the logits for the generated tokens using the inputs["prompts"] lengths
            prompt_lengths = inputs["prompts"].shape[1]
            shifted_student_logits = student_outputs.logits[:, prompt_lengths-1: -1, :]
            teacher_logits = teacher_outputs.logits
            if teacher_logits.size(-1) != shifted_student_logits.size(-1):
                if teacher_logits.size(-1) < shifted_student_logits.size(-1):
                    raise ValueError(
                        "Teacher vocab size {} is smaller than student vocab size {}.".format(
                            teacher_logits.size(-1), shifted_student_logits.size(-1)
                        )
                    )
                teacher_logits = teacher_logits[..., :shifted_student_logits.size(-1)]
            shifted_teacher_logits = teacher_logits[:, prompt_lengths-1: -1, :]
            shifted_labels = inputs["labels"][:, prompt_lengths:]

            # compute loss
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
            )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        if self.seq_kd:
            use_student = random.random() <= self.lmbda
            rollout_model = model if use_student else self.teacher_model
            with (
                unwrap_model_for_generation(
                    rollout_model, self.accelerator
                ) as unwrapped_model,
                torch.no_grad(),
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        elif random.random() <= self.lmbda:
            with (
                unwrap_model_for_generation(model, self.accelerator) as unwrapped_model,
                torch.no_grad(),
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.reward_func is not None:
            self._compute_reward_metrics(model, inputs, mode="train")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        if self.reward_func is not None:
            prepared_inputs = self._prepare_inputs(inputs)
            self._compute_reward_metrics(model, prepared_inputs, mode="eval")
        return loss, logits, labels
