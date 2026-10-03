import pytest
import torch

from .adapters import (
    run_aggregate_loss_across_microbatch as aggregate_loss_across_microbatch,
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_compute_policy_gradient_loss as compute_policy_gradient_loss,
    run_compute_rollout_rewards as compute_rollout_rewards,
    run_get_response_log_probs as get_response_log_probs,
    run_grpo_train_step as grpo_train_step,
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
)


def _train_step_snapshot(
    model: torch.nn.Module,
    loss: torch.Tensor,
    metadata: dict[str, torch.Tensor | float],
) -> dict[str, torch.Tensor]:
    del metadata
    output = {
        "loss": loss.detach(),
    }
    for name, param in model.named_parameters():
        output[f"param.{name}"] = param.detach()
    return output


def _train_step_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=1e-3)


def test_tokenize_prompt_and_output(numpy_snapshot, prompt_strs, output_strs, tokenizer):
    output = tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )
    numpy_snapshot.assert_match(output)


def test_get_response_log_probs(numpy_snapshot, model, input_ids, labels):
    output = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True,
    )
    numpy_snapshot.assert_match(output)


def test_compute_rollout_rewards(numpy_snapshot, reward_fn, rollout_responses, repeated_ground_truths):
    raw_rewards, _ = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    numpy_snapshot.assert_match(raw_rewards)


def test_compute_group_normalized_rewards_grpo(numpy_snapshot, advantage_eps):
    raw_rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=2,
        baseline="mean",
        advantage_eps=advantage_eps,
        advantage_normalizer="std",
    )

    numpy_snapshot.assert_match(
        {
            "advantages": advantages,
        }
    )


def test_compute_group_normalized_rewards_drgrpo(numpy_snapshot):
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        group_size=2,
        baseline="mean",
        advantage_normalizer="none",
    )
    no_baseline_advantages, _ = compute_group_normalized_rewards(
        raw_rewards=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        group_size=2,
        baseline="none",
        advantage_normalizer="none",
    )

    numpy_snapshot.assert_match(
        {
            "advantages": advantages,
            "no_baseline_advantages": no_baseline_advantages,
        }
    )


def test_compute_group_normalized_rewards_maxrl(numpy_snapshot):
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        group_size=2,
        baseline="mean",
        advantage_normalizer="mean",
    )

    numpy_snapshot.assert_match(advantages)


def test_compute_policy_gradient_loss_on_policy(numpy_snapshot, raw_rewards_or_advantages, policy_log_probs):
    loss, _ = compute_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="none",
    )

    numpy_snapshot.assert_match(loss)


def test_compute_policy_gradient_loss_off_policy(
    numpy_snapshot,
    raw_rewards_or_advantages,
    policy_log_probs,
    old_log_probs,
):
    noclip_loss, _ = compute_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="noclip",
        old_log_probs=old_log_probs,
    )

    clipped_loss, _ = compute_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="grpo",
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )

    numpy_snapshot.assert_match(
        {
            "noclip_loss": noclip_loss,
            "clipped_loss": clipped_loss,
        }
    )


def test_compute_policy_gradient_loss_off_policy_gspo(
    numpy_snapshot,
    raw_rewards_or_advantages,
    policy_log_probs,
    old_log_probs,
    response_mask,
):
    loss, _ = compute_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="gspo",
        old_log_probs=old_log_probs,
        cliprange=0.1,
        response_mask=response_mask,
    )

    numpy_snapshot.assert_match(loss)


def test_aggregate_loss_across_microbatch_sequence(numpy_snapshot, policy_log_probs, response_mask):
    loss = aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss=policy_log_probs,
        mask=response_mask,
        loss_normalization="sequence",
    )

    numpy_snapshot.assert_match(loss)


def test_aggregate_loss_across_microbatch_constant(numpy_snapshot, policy_log_probs, response_mask):
    loss = aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss=policy_log_probs,
        mask=response_mask,
        loss_normalization="constant",
        normalization_constant=42,
    )

    numpy_snapshot.assert_match(loss)


def _train_step_inputs(tokenizer):
    prompts = ["Hello", "Hello", "This is", "This is"]
    responses = ["world", "test", "a test", "another test"]
    ground_truths = ["42"] * 4
    sequence_length = (
        max(
            len(tokenizer.encode(prompt)) + len(tokenizer.encode(response))
            for prompt, response in zip(prompts, responses)
        )
        - 1
    )
    old_log_probs = torch.zeros((len(prompts), sequence_length), dtype=torch.float32)
    return prompts, responses, ground_truths, old_log_probs


def _train_step_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    del ground_truth
    reward = {
        "world": 1.0,
        "test": 0.0,
        "a test": 0.5,
        "another test": 0.0,
    }[response]
    return {
        "reward": reward,
        "format_reward": float(reward > 0),
        "answer_reward": reward,
    }


def test_grpo_train_step_standard_on_policy(numpy_snapshot, tiny_train_model, tokenizer):
    prompts, responses, ground_truths, _ = _train_step_inputs(tokenizer)
    loss, metadata = grpo_train_step(
        model=tiny_train_model,
        tokenizer=tokenizer,
        optimizer=_train_step_optimizer(tiny_train_model),
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        reward_fn=_train_step_reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=2,
    )

    numpy_snapshot.assert_match(
        _train_step_snapshot(
            tiny_train_model,
            loss,
            metadata,
        ),
        rtol=1e-4,
        atol=1e-6,
    )
    assert all(param.grad is None for param in tiny_train_model.parameters())


@pytest.mark.parametrize(
    "variant_kwargs",
    [
        pytest.param(
            {
                "baseline": "mean",
                "advantage_normalizer": "std",
                "loss_normalization": "constant",
                "normalization_constant": 32,
            },
            id="grpo_constant",
        ),
        pytest.param(
            {
                "baseline": "mean",
                "advantage_normalizer": "none",
                "loss_normalization": "constant",
                "normalization_constant": 32,
            },
            id="dr_grpo",
        ),
        pytest.param(
            {
                "baseline": "none",
                "advantage_normalizer": "none",
                "loss_normalization": "constant",
                "normalization_constant": 32,
            },
            id="rft",
        ),
        pytest.param(
            {
                "baseline": "mean",
                "advantage_normalizer": "mean",
                "loss_normalization": "constant",
                "normalization_constant": 32,
            },
            id="maxrl",
        ),
    ],
)
def test_grpo_train_step_variants_on_policy(numpy_snapshot, tiny_train_model, tokenizer, variant_kwargs):
    prompts, responses, ground_truths, _ = _train_step_inputs(tokenizer)
    loss, metadata = grpo_train_step(
        model=tiny_train_model,
        tokenizer=tokenizer,
        optimizer=_train_step_optimizer(tiny_train_model),
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        reward_fn=_train_step_reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=2,
        **variant_kwargs,
    )

    numpy_snapshot.assert_match(
        _train_step_snapshot(
            tiny_train_model,
            loss,
            metadata,
        ),
        rtol=1e-4,
        atol=1e-6,
    )
    assert all(param.grad is None for param in tiny_train_model.parameters())


@pytest.mark.parametrize(
    "importance_reweighting_method",
    [
        "noclip",
        "grpo",
        "gspo",
    ],
)
def test_grpo_train_step_off_policy(numpy_snapshot, tiny_train_model, tokenizer, importance_reweighting_method):
    prompts, responses, ground_truths, old_log_probs = _train_step_inputs(tokenizer)
    old_log_probs = torch.linspace(-1.5, 0.5, steps=old_log_probs.numel()).reshape_as(old_log_probs)
    loss, metadata = grpo_train_step(
        model=tiny_train_model,
        tokenizer=tokenizer,
        optimizer=_train_step_optimizer(tiny_train_model),
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        reward_fn=_train_step_reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=2,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )

    numpy_snapshot.assert_match(
        _train_step_snapshot(
            tiny_train_model,
            loss,
            metadata,
        ),
        rtol=1e-4,
        atol=1e-6,
    )
    assert all(param.grad is None for param in tiny_train_model.parameters())
