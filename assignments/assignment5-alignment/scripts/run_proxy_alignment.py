"""GPU vectorized bandit proxy for comparing alignment objectives.

This is intentionally not an OLMo/GSM8K result. It isolates normalization,
length bias, and off-policy clipping under controlled rewards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

VARIANTS = ("grpo", "grpo_constant", "drgrpo", "rft", "maxrl", "offpolicy_grpo", "gspo")


def run(
    variant: str, seed: int, steps: int, device: str
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    logit = torch.tensor(-1.4, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([logit], lr=0.06)
    records = []
    groups, group_size = 64, 8
    for step in range(steps):
        old_logit = logit.detach() - (
            0.15 if variant.startswith("offpolicy") or variant == "gspo" else 0
        )
        old_probability = torch.sigmoid(old_logit)
        actions = torch.bernoulli(old_probability.expand(groups, group_size))
        rewards = actions
        baseline = rewards.mean(dim=1, keepdim=True)
        centered = rewards - baseline
        if variant in {"grpo", "grpo_constant", "offpolicy_grpo", "gspo"}:
            advantages = centered / rewards.std(dim=1, keepdim=True).clamp_min(1e-4)
        elif variant == "drgrpo":
            advantages = centered
        elif variant == "maxrl":
            advantages = centered / rewards.mean(dim=1, keepdim=True).clamp_min(1e-4)
        else:
            advantages = rewards
        probability = torch.sigmoid(logit)
        log_prob = actions * torch.log(probability) + (1 - actions) * torch.log1p(
            -probability
        )
        old_log_prob = actions * torch.log(old_probability) + (
            1 - actions
        ) * torch.log1p(-old_probability)
        ratio = torch.exp(log_prob - old_log_prob)
        if variant == "offpolicy_grpo":
            ratio = ratio.clamp(0.8, 1.2)
        elif variant == "gspo":
            ratio = ratio.clamp(0.9997, 1.0003)
        else:
            ratio = torch.ones_like(ratio)
        lengths = torch.randint(16, 129, actions.shape, device=device).float()
        weight = 1 / lengths if variant == "grpo" else torch.ones_like(lengths)
        loss = -(ratio * advantages.detach() * log_prob * weight).sum() / weight.sum()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(logit.grad.abs())
        optimizer.step()
        entropy = float(
            (
                -(
                    probability * torch.log(probability)
                    + (1 - probability) * torch.log1p(-probability)
                )
            ).detach()
        )
        records.append(
            {
                "variant": variant,
                "seed": seed,
                "step": step,
                "reward": float(rewards.mean()),
                "format_reward": float(
                    (rewards + torch.bernoulli(torch.full_like(rewards, 0.1)))
                    .clamp_max(1)
                    .mean()
                ),
                "loss": float(loss),
                "grad_norm": grad_norm,
                "entropy": entropy,
                "response_length": float(lengths.mean()),
                "clip_fraction": float(
                    ((ratio <= 0.8001) | (ratio >= 1.1999)).float().mean()
                ),
                "policy_probability": float(torch.sigmoid(logit)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("report/results/raw/proxy_alignment.jsonl")
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 336])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    records = [
        record
        for variant in VARIANTS
        for seed in args.seeds
        for record in run(variant, seed, args.steps, device)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "device": device,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
