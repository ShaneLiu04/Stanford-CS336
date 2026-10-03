"""Build reproducible Alignment report figures from raw proxy metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("report/results")
FIGURES = RESULTS / "figures"


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [
        json.loads(line)
        for line in (RESULTS / "raw/proxy_alignment.jsonl").read_text().splitlines()
    ]
    frame = pd.DataFrame(rows)
    plt.style.use("seaborn-v0_8-whitegrid")
    curve_specs = [
        ("reward_curves", "reward", "Answer reward", "Proxy reward learning curves"),
        (
            "format_reward_curves",
            "format_reward",
            "Format reward",
            "Format vs answer behavior",
        ),
        ("entropy_curves", "entropy", "Bernoulli entropy", "Exploration entropy"),
        ("gradient_norm", "grad_norm", "Gradient norm", "Gradient stability"),
        ("policy_loss", "loss", "Policy loss", "Surrogate optimization"),
        (
            "response_length",
            "response_length",
            "Mean response length",
            "Length behavior",
        ),
        ("clip_fraction", "clip_fraction", "Clip fraction", "Off-policy clipping"),
        (
            "policy_probability",
            "policy_probability",
            "Correct-action probability",
            "Policy improvement",
        ),
    ]
    for name, metric, ylabel, title in curve_specs:
        fig, axis = plt.subplots(figsize=(8.7, 4.7))
        grouped = frame.groupby(["variant", "step"])[metric]
        mean, std = grouped.mean(), grouped.std().fillna(0)
        for variant in frame["variant"].unique():
            series = mean.loc[variant]
            spread = std.loc[variant]
            axis.plot(series.index, series.values, label=variant)
            axis.fill_between(
                series.index, series - spread, series + spread, alpha=0.12
            )
        axis.set(xlabel="Step", ylabel=ylabel, title=title)
        axis.legend(fontsize=7, ncol=2)
        save(fig, name)

    final = frame.groupby(["variant", "seed"]).tail(1)
    for name, metric, title in (
        ("final_reward", "reward", "Final answer reward by objective"),
        ("final_entropy", "entropy", "Final entropy by objective"),
        ("final_grad_norm", "grad_norm", "Final gradient norm by objective"),
    ):
        stats = final.groupby("variant")[metric].agg(["mean", "std"])
        fig, axis = plt.subplots(figsize=(8.5, 4.5))
        bars = axis.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=4)
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set(ylabel=metric, title=title)
        axis.tick_params(axis="x", rotation=30)
        save(fig, name)

    reward = final.groupby("variant")["reward"].mean()
    fig, axis = plt.subplots(figsize=(7.8, 4.4))
    for k in (1, 2, 4, 8):
        axis.plot(reward.index, 1 - (1 - reward.values) ** k, "o-", label=f"pass@{k}")
    axis.set(ylabel="Estimated pass@k", title="Sampling benefit from group rollouts")
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    save(fig, "pass_at_k")

    advantages = []
    generator = np.random.default_rng(336)
    for difficulty in (0.125, 0.25, 0.5, 0.75):
        rewards = generator.binomial(1, difficulty, size=(500, 8))
        centered = rewards - rewards.mean(axis=1, keepdims=True)
        for method, values in (
            ("GRPO", centered / np.maximum(rewards.std(axis=1, keepdims=True), 1e-4)),
            ("DrGRPO", centered),
            ("MaxRL", centered / np.maximum(rewards.mean(axis=1, keepdims=True), 1e-4)),
        ):
            advantages.append(
                {"difficulty": difficulty, "method": method, "std": float(values.std())}
            )
    advantage = pd.DataFrame(advantages)
    fig, axis = plt.subplots(figsize=(7.8, 4.4))
    for method, series in advantage.groupby("method"):
        axis.plot(series["difficulty"], series["std"], "o-", label=method)
    axis.set(
        xlabel="Group success probability",
        ylabel="Advantage std",
        title="Difficulty reweighting",
    )
    axis.legend()
    save(fig, "advantage_normalization")

    gsm_path = Path("data/gsm8k/train.jsonl")
    if gsm_path.exists():
        gsm = [
            json.loads(line)
            for line in gsm_path.read_text(encoding="utf-8").splitlines()
        ]
        lengths = [len(item["question"].split()) for item in gsm]
        answer_lengths = [len(item["answer"].split()) for item in gsm]
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.1))
        left.hist(lengths, bins=30)
        left.set(xlabel="Question words", title="GSM8K question length")
        right.hist(answer_lengths, bins=30)
        right.set(xlabel="Answer words", title="GSM8K rationale length")
        save(fig, "gsm8k_lengths")

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(["GRPO tests", "Supplement tests"], [20, 6], color=["#2ca02c", "#1f77b4"])
    axis.set(ylabel="Expected test cases", title="Alignment verification coverage")
    save(fig, "test_matrix")

    summary = {
        "records": len(frame),
        "variants": frame["variant"].unique().tolist(),
        "seeds": sorted(frame["seed"].unique().tolist()),
        "steps": int(frame["step"].max() + 1),
        "final_reward": reward.to_dict(),
        "proxy_only": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    final.to_csv(RESULTS / "experiment_log.csv", index=False)
    manifest = RESULTS / "manifest.sha256"
    files = [path for path in RESULTS.rglob("*") if path.is_file() and path != manifest]
    manifest.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(RESULTS).as_posix()}"
            for path in sorted(files)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"built {len(list(FIGURES.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
