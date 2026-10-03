"""Core GRPO utilities used by the alignment assignment."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.pytorch_utils import Conv1D


def _patch_gpt2_recursive_initialization() -> None:
    """Work around the recursive GPT-2 initializer shipped by Transformers 4.x."""
    source = inspect.getsource(GPT2PreTrainedModel._init_weights)
    if "for name, p in module.named_parameters()" not in source:
        return

    def _init_weights(self: GPT2PreTrainedModel, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    GPT2PreTrainedModel._init_weights = _init_weights


_patch_gpt2_recursive_initialization()


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt/response pairs and construct next-token labels and masks."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")
    if not prompt_strs:
        raise ValueError("prompt_strs and output_strs must be non-empty")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")

    prompt_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in prompt_strs
    ]
    output_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in output_strs
    ]
    combined_ids = [prompt + output for prompt, output in zip(prompt_ids, output_ids)]
    if any(len(ids) < 2 for ids in combined_ids):
        raise ValueError(
            "each concatenated prompt and output must contain at least two tokens"
        )

    max_length = max(map(len, combined_ids))
    padded = torch.full(
        (len(combined_ids), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    response_mask = torch.zeros(
        (len(combined_ids), max_length - 1),
        dtype=torch.bool,
    )

    for row, (ids, prompt) in enumerate(zip(combined_ids, prompt_ids)):
        padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        # Mask positions are aligned with labels, so the first response token is
        # predicted at prompt_length - 1.
        response_mask[row, max(len(prompt) - 1, 0) : len(ids) - 1] = True

    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:],
        "response_mask": response_mask,
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute categorical entropy without materializing log(softmax(logits))."""
    log_normalizer = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    return log_normalizer - (probabilities * logits).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """Return label log-probabilities (and optionally entropy) at each position."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    log_probs = torch.log_softmax(logits, dim=-1)
    result = {
        "log_probs": torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    """Score rollout responses and summarize each returned reward component."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )
    if not rollout_responses:
        raise ValueError("rollout_responses must be non-empty")

    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    required_keys = {"reward", "format_reward", "answer_reward"}
    if any(not required_keys.issubset(reward) for reward in reward_dicts):
        raise ValueError(f"reward_fn results must contain {sorted(required_keys)}")

    raw_rewards = torch.tensor(
        [reward["reward"] for reward in reward_dicts], dtype=torch.float32
    )
    metadata = {
        key: float(
            sum(float(reward[key]) for reward in reward_dicts) / len(reward_dicts)
        )
        for key in ("reward", "format_reward", "answer_reward")
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    """Apply a per-group baseline and normalization to rollout rewards."""
    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must have shape (rollout_batch_size,)")
    if group_size <= 0 or raw_rewards.numel() % group_size:
        raise ValueError("group_size must evenly divide the rollout batch")
    if baseline not in ("mean", "none"):
        raise ValueError(f"unsupported baseline: {baseline}")
    if advantage_normalizer not in ("std", "none", "mean"):
        raise ValueError(f"unsupported advantage normalizer: {advantage_normalizer}")

    grouped = raw_rewards.reshape(-1, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_means if baseline == "mean" else grouped

    if advantage_normalizer == "std":
        # torch.std's default unbiased estimator is the assignment's convention.
        denominator = grouped.std(dim=1, keepdim=True) + advantage_eps
        advantages = advantages / denominator
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_means + advantage_eps)

    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()) if raw_rewards.numel() > 1 else 0.0,
        "reward_min": float(raw_rewards.min()),
        "reward_max": float(raw_rewards.max()),
    }
    return advantages.reshape_as(raw_rewards), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token negative policy-gradient objective."""
    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape (batch_size, sequence_length)"
        )
    advantages = raw_rewards_or_advantages.reshape(-1, 1).to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("one reward or advantage is required per sequence")

    metadata: dict[str, Tensor] = {}
    if importance_reweighting_method == "none":
        return -(advantages * policy_log_probs), metadata

    if old_log_probs is None:
        raise ValueError("old_log_probs is required for importance reweighting")
    old_log_probs = old_log_probs.to(policy_log_probs.device)
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs must have the same shape as policy_log_probs")

    log_ratios = policy_log_probs - old_log_probs
    if importance_reweighting_method == "noclip":
        ratios = log_ratios.exp()
        return -(advantages * ratios), metadata

    if importance_reweighting_method not in ("grpo", "gspo"):
        raise ValueError(
            f"unsupported importance reweighting method: {importance_reweighting_method}"
        )
    if cliprange is None or cliprange < 0:
        raise ValueError(
            "a non-negative cliprange is required for clipped importance reweighting"
        )

    if importance_reweighting_method == "grpo":
        ratios = log_ratios.exp()
    else:
        if response_mask is None:
            raise ValueError("response_mask is required for GSPO")
        mask = response_mask.to(
            device=policy_log_probs.device, dtype=policy_log_probs.dtype
        )
        token_counts = mask.sum(dim=1, keepdim=True)
        if torch.any(token_counts == 0):
            raise ValueError("every sequence must contain at least one response token")
        ratios = ((log_ratios * mask).sum(dim=1, keepdim=True) / token_counts).exp()

    clipped_ratios = ratios.clamp(1.0 - cliprange, 1.0 + cliprange)
    unclipped_objective = advantages * ratios
    clipped_objective = advantages * clipped_ratios
    loss = -torch.minimum(unclipped_objective, clipped_objective)
    if importance_reweighting_method == "gspo":
        loss = loss.expand_as(policy_log_probs)

    metadata["clip_fraction"] = (
        ((ratios < 1.0 - cliprange) | (ratios > 1.0 + cliprange)).float().mean()
    )
    return loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    """Aggregate masked token losses using sequence or constant normalization."""
    if per_token_policy_gradient_loss.shape != mask.shape:
        raise ValueError("loss and mask must have the same shape")
    mask = mask.to(
        device=per_token_policy_gradient_loss.device,
        dtype=per_token_policy_gradient_loss.dtype,
    )
    masked_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "sequence":
        lengths = mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("every sequence must contain at least one masked token")
        return (masked_loss.sum(dim=1) / lengths).mean()
    if loss_normalization == "constant":
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("a positive normalization_constant is required")
        return masked_loss.sum() / normalization_constant
    raise ValueError(f"unsupported loss normalization: {loss_normalization}")


# The shorter public name is useful outside the assignment adapter.
aggregate_loss = aggregate_loss_across_microbatch


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Run one optimizer update over a rollout batch using microbatching."""
    batch_size = len(repeated_prompts)
    if not (batch_size == len(rollout_responses) == len(repeated_ground_truths)):
        raise ValueError(
            "prompts, responses, and ground truths must have the same length"
        )
    if gradient_accumulation_steps <= 0 or batch_size % gradient_accumulation_steps:
        raise ValueError(
            "gradient_accumulation_steps must evenly divide the rollout batch"
        )

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    # Pad the complete rollout batch before slicing it into microbatches.  If
    # each microbatch is tokenized separately, its sequence length depends on
    # the examples it happens to contain, which changes model outputs (and the
    # alignment of old log-probabilities) across accumulation boundaries.
    tokenized = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )
    microbatch_size = batch_size // gradient_accumulation_steps
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    response_token_count = torch.zeros((), device=device)
    loss_metadata_sums: dict[str, Tensor] = {}
    processed = 0

    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        ids_mb = tokenized["input_ids"][start:end].to(device)
        labels_mb = tokenized["labels"][start:end].to(device)
        mask_mb = tokenized["response_mask"][start:end].to(device)
        advantages_mb = advantages[start:end].to(device)

        scored = get_response_log_probs(
            model,
            ids_mb,
            labels_mb,
            return_token_entropy=True,
        )
        old_mb = (
            None
            if old_log_probs is None
            else old_log_probs[start:end, : ids_mb.shape[1]].to(device)
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            advantages_mb,
            scored["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_mb,
            cliprange=cliprange,
            response_mask=mask_mb,
        )
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_loss,
            mask_mb,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        if loss_normalization == "sequence":
            adjusted_loss = microbatch_loss * ((end - start) / batch_size)
        else:
            adjusted_loss = microbatch_loss
        adjusted_loss.backward()
        total_loss = total_loss + adjusted_loss.detach()

        float_mask = mask_mb.to(scored["token_entropy"].dtype)
        entropy_sum = (
            entropy_sum + (scored["token_entropy"].detach() * float_mask).sum()
        )
        response_token_count = response_token_count + float_mask.sum()
        for key, value in loss_metadata.items():
            weighted = value.detach() * (end - start)
            loss_metadata_sums[key] = (
                loss_metadata_sums.get(key, torch.zeros_like(weighted)) + weighted
            )
        processed += end - start

    if max_grad_norm is None:
        squared_norm = sum(
            parameter.grad.detach().float().norm(2).square()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        grad_norm: Tensor | float = squared_norm.sqrt()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        **reward_metadata,
        **advantage_metadata,
        "loss": total_loss,
        "grad_norm": grad_norm,
        "token_entropy": entropy_sum / response_token_count,
    }
    metadata.update(
        {key: value / processed for key, value in loss_metadata_sums.items()}
    )
    return total_loss, metadata
