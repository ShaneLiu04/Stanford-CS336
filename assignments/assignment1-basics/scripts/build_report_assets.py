from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def run_summary(path: Path) -> dict[str, float | str]:
    records = load_run(path)
    best = min(records, key=lambda item: item["val_loss"])
    return {
        "run": path.name,
        "best_val_loss": float(best["val_loss"]),
        "best_step": int(best["step"]),
        "tokens_seen": int(best["tokens_seen"]),
        "elapsed_seconds": float(best["elapsed_seconds"]),
        "peak_memory_mib": float(max(item.get("peak_memory_mib", 0) for item in records)),
    }


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in (".pdf", ".png"):
        figure.savefig(output.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(figure)


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("metrics.jsonl"))


def median_throughput(records: list[dict[str, float]]) -> float:
    values = [float(item.get("tokens_per_second", 0)) for item in records[1:]]
    return float(np.median(values)) if values else 0.0


def write_manifest(results: Path, figures: Path) -> None:
    manifest = results / "manifest.sha256"
    files = [
        path
        for root in (results, figures)
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    ]
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}"
        for path in sorted(files)
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    experiments = args.raw / "experiments"
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    lr_dirs = sorted((experiments / "tuning").glob("00[0-4]-lr-*")) + sorted(
        (experiments / "lr-fine").glob("*-lr-*")
    )
    for run_dir in lr_dirs:
        records = load_run(run_dir)
        axis.plot(
            [item["tokens_seen"] / 1e6 for item in records],
            [item["val_loss"] for item in records],
            label=run_dir.name.split("-", 1)[1],
            linewidth=1.5,
        )
    axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Learning-rate sweep")
    axis.set_ylim(1.8, 5.5)
    axis.legend(ncol=3, fontsize=8)
    save_figure(figure, args.figures / "learning_rate_sweep")

    batch_dirs = sorted((experiments / "systems-probes").glob("00[0-5]-actual-batch-*"))
    batch_sizes, throughput, memory = [], [], []
    for run_dir in batch_dirs:
        records = load_run(run_dir)
        batch_sizes.append(int(run_dir.name.rsplit("-", 1)[1]))
        throughput.append(float(np.median([item.get("tokens_per_second", 0) for item in records[1:]])))
        memory.append(max(item.get("peak_memory_mib", 0) for item in records) / 1024)
    figure, left = plt.subplots(figsize=(7.6, 4.6))
    right = left.twinx()
    left.plot(batch_sizes, throughput, "o-", label="Throughput", color="#1f77b4")
    right.plot(batch_sizes, memory, "s--", label="Peak VRAM", color="#d62728")
    left.set(xlabel="Actual batch size", ylabel="Tokens / second", title="Batch scaling on RTX 4080 SUPER")
    right.set_ylabel("Peak VRAM (GiB)")
    lines = left.lines + right.lines
    left.legend(lines, [str(line.get_label()) for line in lines], loc="center right")
    save_figure(figure, args.figures / "batch_scaling")

    final_dirs = sorted((experiments / "final-seeds").glob("00[0-2]-baseline-seed-*"))
    seed_records = [load_run(path) for path in final_dirs]
    tokens = np.array([item["tokens_seen"] for item in seed_records[0]]) / 1e6
    values = np.array([[item["val_loss"] for item in records] for records in seed_records])
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for path, records in zip(final_dirs, seed_records):
        axis.plot(tokens, [item["val_loss"] for item in records], alpha=0.45, linewidth=1, label=path.name.rsplit("-", 1)[1])
    axis.plot(tokens, mean, color="black", linewidth=2, label="Mean")
    axis.fill_between(tokens, mean - std, mean + std, color="black", alpha=0.15, label="±1 std")
    axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Three-seed TinyStories training")
    axis.set_ylim(1.3, 3.2)
    axis.legend()
    save_figure(figure, args.figures / "seed_stability")

    baseline = seed_records[0]
    ablation_dirs = sorted((experiments / "extended-ablations").glob("00[0-2]-*"))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        [item["tokens_seen"] / 1e6 for item in baseline],
        [item["val_loss"] for item in baseline],
        linewidth=2,
        label="Pre-norm + RoPE + SwiGLU",
    )
    labels = {"post-norm-full": "Post-norm", "no-rope-full": "NoPE", "silu-ffn-full": "SiLU FFN"}
    for run_dir in ablation_dirs:
        records = load_run(run_dir)
        name = run_dir.name.split("-", 1)[1]
        axis.plot(
            [item["tokens_seen"] / 1e6 for item in records],
            [item["val_loss"] for item in records],
            label=labels[name],
            linewidth=1.5,
        )
    axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Full-budget architecture ablations")
    axis.set_ylim(1.3, 3.2)
    axis.legend()
    save_figure(figure, args.figures / "architecture_ablations")

    edge_dirs = sorted((experiments / "systems-probes").glob("00[6-8]-edge-lr-*"))
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for run_dir in edge_dirs:
        records = load_run(run_dir)
        axis.plot(
            [item["step"] for item in records],
            [item["val_loss"] for item in records],
            marker="o",
            label=run_dir.name.split("edge-lr-", 1)[1],
        )
    axis.set(xlabel="Step", ylabel="Validation loss", title="Edge-of-stability learning rates")
    axis.legend(title="Max LR")
    save_figure(figure, args.figures / "edge_of_stability")

    short_ablation_dirs = sorted((experiments / "ablations").glob("00[0-5]-*"))
    figure, axes_grid = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes_grid.ravel()
    for run_dir in short_ablation_dirs:
        records = load_run(run_dir)
        label = run_dir.name.split("-", 1)[1]
        axes[0].plot([item["step"] for item in records], [item["val_loss"] for item in records], label=label)
        axes[1].plot(
            [item["tokens_seen"] / 1e6 for item in records],
            [item["val_loss"] for item in records],
            label=label,
        )
        axes[2].plot(
            [item["elapsed_seconds"] / 60 for item in records],
            [item["val_loss"] for item in records],
            label=label,
        )
        axes[3].plot(
            [item["tokens_seen"] / 1e6 for item in records],
            [item.get("tokens_per_second", 0) for item in records],
            label=label,
        )
    labels = [
        ("Step", "Validation loss"),
        ("Tokens seen (millions)", "Validation loss"),
        ("Wall-clock (minutes)", "Validation loss"),
        ("Tokens seen (millions)", "Tokens / second"),
    ]
    for axis, (xlabel, ylabel) in zip(axes, labels):
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend(fontsize=7)
    figure.suptitle("40M-token architecture ablations")
    save_figure(figure, args.figures / "ablations_40m")

    comparison_runs = [
        ("Batch 32\nseed mean", float(np.mean([min(item["val_loss"] for item in records) for records in seed_records]))),
        ("Batch 128", min(item["val_loss"] for item in load_run(experiments / "large-batch" / "000-batch-128-full"))),
        ("Batch 256", min(item["val_loss"] for item in load_run(experiments / "batch256" / "000-batch-256-full"))),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar([item[0] for item in comparison_runs], [item[1] for item in comparison_runs])
    axis.set(ylabel="Best validation loss", title="Full-budget TinyStories results")
    axis.set_ylim(1.28, 1.40)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    save_figure(figure, args.figures / "final_batch_comparison")

    tokenizer_path = args.raw / "data" / "tokenizer-experiments.json"
    tokenizer_results = None
    if tokenizer_path.exists():
        tokenizer_results = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        corpora = ["tinystories", "owt"]
        tokenizer_names = ["tinystories_10k", "owt_32k"]
        positions = np.arange(len(corpora))
        width = 0.36
        figure, axis = plt.subplots(figsize=(7.4, 4.6))
        for index, tokenizer_name in enumerate(tokenizer_names):
            values = [
                tokenizer_results[corpus][tokenizer_name]["bytes_per_token"]
                for corpus in corpora
            ]
            bars = axis.bar(
                positions + (index - 0.5) * width,
                values,
                width,
                label=tokenizer_name.replace("_", " "),
            )
            axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set(
            xticks=positions,
            xticklabels=["TinyStories sample", "OpenWebText sample"],
            ylabel="Bytes / token (higher is better compression)",
            title="Cross-corpus tokenizer compression",
        )
        axis.legend()
        save_figure(figure, args.figures / "tokenizer_compression")

    owt_lr_dirs = sorted((experiments / "owt-lr-tuning").glob("[0-9][0-9][0-9]-lr-*"))
    if owt_lr_dirs:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for run_dir in owt_lr_dirs:
            records = load_run(run_dir)
            axis.plot(
                [item["tokens_seen"] / 1e6 for item in records],
                [item["val_loss"] for item in records],
                label=run_dir.name.split("-", 1)[1],
            )
        axis.set(
            xlabel="Tokens seen (millions)",
            ylabel="Validation loss",
            title="OpenWebText 32K-tokenizer learning-rate sweep",
        )
        axis.legend()
        save_figure(figure, args.figures / "owt_learning_rate_sweep")

    no_rmsnorm_dir = experiments / "no-rmsnorm-full"
    no_rmsnorm_summary = run_summary(no_rmsnorm_dir) if (no_rmsnorm_dir / "metrics.jsonl").exists() else None
    tied_pilot_dir = experiments / "tied-embeddings-pilot"
    tied_pilot_summary = run_summary(tied_pilot_dir) if (tied_pilot_dir / "metrics.jsonl").exists() else None
    tied_small_init_dir = experiments / "tied-small-init-pilot"
    tied_small_init_summary = (
        run_summary(tied_small_init_dir)
        if (tied_small_init_dir / "metrics.jsonl").exists()
        else None
    )
    if tied_pilot_summary is not None and tied_small_init_summary is not None:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for run_dir, label in (
            (experiments / "ablations" / "000-baseline", "Untied baseline"),
            (tied_pilot_dir, "Tied, original init"),
            (tied_small_init_dir, "Tied, std=0.02"),
        ):
            records = load_run(run_dir)
            axis.plot(
                [item["tokens_seen"] / 1e6 for item in records],
                [item["val_loss"] for item in records],
                label=label,
            )
        axis.set(
            xlabel="Tokens seen (millions)",
            ylabel="Validation loss",
            title="Weight tying and initialization pilot",
        )
        axis.set_ylim(1.5, 4.5)
        axis.legend()
        save_figure(figure, args.figures / "weight_tying_pilot")
    if no_rmsnorm_summary is not None:
        ablation_bars = [
            ("Baseline", min(item["val_loss"] for item in baseline)),
            ("Post-norm", min(item["val_loss"] for item in load_run(ablation_dirs[0]))),
            ("NoPE", min(item["val_loss"] for item in load_run(ablation_dirs[1]))),
            ("SiLU", min(item["val_loss"] for item in load_run(ablation_dirs[2]))),
            ("No RMSNorm", float(no_rmsnorm_summary["best_val_loss"])),
        ]
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        bars = axis.bar([item[0] for item in ablation_bars], [item[1] for item in ablation_bars])
        axis.set(ylabel="Best validation loss", title="Full-budget ablation outcomes")
        axis.bar_label(bars, fmt="%.3f", padding=3)
        save_figure(figure, args.figures / "full_ablation_outcomes")
    owt_dir = experiments / "owt-final"
    owt_summary = run_summary(owt_dir) if (owt_dir / "metrics.jsonl").exists() else None
    owt_tiny_dir = experiments / "owt-tiny-full"
    owt_tiny_summary = (
        run_summary(owt_tiny_dir)
        if (owt_tiny_dir / "metrics.jsonl").exists()
        else None
    )
    if owt_summary is not None:
        records = load_run(owt_dir)
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        axis.plot(
            [item["tokens_seen"] / 1e6 for item in records],
            [item["val_loss"] for item in records],
            linewidth=1.8,
        )
        axis.set(
            xlabel="Tokens seen (millions)",
            ylabel="Validation loss",
            title="OpenWebText main experiment",
        )
        save_figure(figure, args.figures / "owt_training")
    if owt_summary is not None and owt_tiny_summary is not None:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for run_dir, label in (
            (owt_tiny_dir, "TinyStories tokenizer (10K)"),
            (owt_dir, "OWT tokenizer (32K)"),
        ):
            records = load_run(run_dir)
            axis.plot(
                [item["tokens_seen"] / 1e6 for item in records],
                [item["val_loss"] for item in records],
                label=label,
            )
        axis.set(
            xlabel="Tokens seen (millions)",
            ylabel="Validation loss",
            title="OWT cross-tokenizer training",
        )
        axis.legend()
        save_figure(figure, args.figures / "owt_cross_tokenizer")

    leaderboard_dir = experiments / "owt-leaderboard-tied"
    leaderboard_summary = (
        run_summary(leaderboard_dir)
        if (leaderboard_dir / "metrics.jsonl").exists()
        else None
    )
    if owt_summary is not None and leaderboard_summary is not None:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for run_dir, label in (
            (owt_dir, "Untied OWT main"),
            (leaderboard_dir, "Tied embeddings"),
        ):
            records = load_run(run_dir)
            axis.plot(
                [item["elapsed_seconds"] / 60 for item in records],
                [item["val_loss"] for item in records],
                label=label,
            )
        axis.set(
            xlabel="Wall-clock (minutes)",
            ylabel="Validation loss",
            title="OWT weight-tying modification",
        )
        axis.axvline(45, color="black", linestyle="--", linewidth=1, label="45-minute limit")
        axis.legend()
        save_figure(figure, args.figures / "owt_leaderboard")

    # Existing-data views that were previously described only in prose.
    fine_dirs = sorted((experiments / "lr-fine").glob("*-lr-*"))
    if fine_dirs:
        names = [path.name.split("-lr-", 1)[-1] for path in fine_dirs]
        losses = [min(item["val_loss"] for item in load_run(path)) for path in fine_dirs]
        figure, axis = plt.subplots(figsize=(7.6, 4.5))
        bars = axis.bar(names, losses, color="#4c78a8")
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set(xlabel="Maximum learning rate", ylabel="Best validation loss", title="Fine learning-rate sweep")
        axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "learning_rate_fine")

    if no_rmsnorm_dir.exists() and (no_rmsnorm_dir / "metrics.jsonl").exists():
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        for run_dir, label in (
            (experiments / "final-seeds" / "000-baseline-seed-42", "RMSNorm baseline"),
            (no_rmsnorm_dir, "No RMSNorm"),
        ):
            records = load_run(run_dir)
            axis.plot(
                [item["tokens_seen"] / 1e6 for item in records],
                [item["val_loss"] for item in records],
                marker="o",
                label=label,
            )
        axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Normalization is required for stable optimization")
        axis.set_yscale("log")
        axis.legend()
        axis.grid(True, alpha=0.25)
        save_figure(figure, args.figures / "rmsnorm_stability")

    all_existing_runs = discover_runs(experiments)
    sweep_counts: dict[str, int] = {}
    for path in all_existing_runs:
        relative = path.relative_to(experiments)
        sweep = relative.parts[0] if len(relative.parts) > 1 else "standalone"
        sweep_counts[sweep] = sweep_counts.get(sweep, 0) + 1
    if sweep_counts:
        figure, axis = plt.subplots(figsize=(9.5, 4.8))
        names = list(sweep_counts)
        bars = axis.bar(names, [sweep_counts[name] for name in names], color="#72b7b2")
        axis.bar_label(bars)
        axis.set(ylabel="Archived runs", title="Experiment coverage before RTX 6000D extensions")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "experiment_coverage")

    data_dir = args.raw / "data"
    resource_rows = []
    for corpus in ("tinystories", "owt"):
        metrics_path = data_dir / f"{corpus}-metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            resource_rows.append(
                (
                    corpus,
                    float(metrics.get("bpe_training_seconds") or 0),
                    float(metrics.get("peak_rss_mib") or 0) / 1024,
                    float(metrics.get("train", {}).get("tokens_per_second") or 0) / 1e6,
                )
            )
    if resource_rows:
        figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))
        labels = [item[0] for item in resource_rows]
        values = [
            ("BPE training time (hours)", [item[1] / 3600 for item in resource_rows]),
            ("Peak RSS (GiB)", [item[2] for item in resource_rows]),
            ("Encoding throughput (M tok/s)", [item[3] for item in resource_rows]),
        ]
        for axis, (title, series) in zip(axes, values, strict=True):
            bars = axis.bar(labels, series, color=["#4c78a8", "#f58518"][: len(labels)])
            axis.bar_label(bars, fmt="%.2f")
            axis.set_title(title)
            axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "tokenizer_systems")

    test_rows = []
    for log_path in [args.raw / "pytest.log", args.raw / "logs" / "pytest.log"]:
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        passed = [int(value) for value in re.findall(r"(\d+) passed", text)]
        skipped = [int(value) for value in re.findall(r"(\d+) skipped", text)]
        xpassed = [int(value) for value in re.findall(r"(\d+) xpassed", text)]
        if passed:
            test_rows.append(
                (
                    "Windows" if log_path.parent == args.raw else "Linux archive",
                    passed[-1],
                    skipped[-1] if skipped else 0,
                    xpassed[-1] if xpassed else 0,
                )
            )
    if test_rows:
        figure, axis = plt.subplots(figsize=(7.8, 4.2))
        x = np.arange(len(test_rows))
        passed = np.asarray([item[1] for item in test_rows])
        skipped = np.asarray([item[2] for item in test_rows])
        xpassed = np.asarray([item[3] for item in test_rows])
        axis.bar(x, passed, label="passed", color="#2ca02c")
        axis.bar(x, skipped, bottom=passed, label="skipped", color="#ffbf00")
        axis.bar(x, xpassed, bottom=passed + skipped, label="xpassed", color="#1f77b4")
        axis.set(
            xticks=x,
            xticklabels=[item[0] for item in test_rows],
            ylabel="Test cases",
            title="Verification matrix",
        )
        axis.legend()
        axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "test_matrix")

    # Optional RTX 6000D experiment groups. The report builds before and after
    # these runs, so absent groups simply omit their figures.
    batch_probe = sorted((experiments / "rtx6000d-batch-probe").glob("[0-9][0-9][0-9]-*"))
    if batch_probe:
        batches, speeds, peaks, record_sets = [], [], [], []
        for run_dir in batch_probe:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            records = load_run(run_dir)
            batches.append(int(config["batch_size"]))
            speeds.append(median_throughput(records))
            peaks.append(max(item.get("peak_memory_mib", 0) for item in records) / 1024)
            record_sets.append(records)
        common_tokens = min(max(item["tokens_seen"] for item in records) for records in record_sets)
        losses = [
            float(
                np.interp(
                    common_tokens,
                    [item["tokens_seen"] for item in records],
                    [item["val_loss"] for item in records],
                )
            )
            for records in record_sets
        ]
        figure, (systems_axis, quality_axis) = plt.subplots(1, 2, figsize=(10.8, 4.3))
        memory_axis = systems_axis.twinx()
        systems_axis.plot(batches, speeds, "o-", label="Throughput", color="#1f77b4")
        memory_axis.plot(batches, peaks, "s--", label="Peak VRAM", color="#d62728")
        systems_axis.set(xlabel="Physical batch size", ylabel="Tokens / second", title="RTX 6000D batch scaling")
        memory_axis.set_ylabel("Peak VRAM (GiB)")
        quality_axis.plot(batches, losses, "o-", color="#2ca02c")
        quality_axis.set(
            xlabel="Physical batch size",
            ylabel="Interpolated validation loss",
            title=f"Quality at {common_tokens / 1e6:.1f}M tokens",
        )
        save_figure(figure, args.figures / "rtx6000d_batch_scaling")

    compile_dirs = sorted((experiments / "compile-precision").glob("[0-9][0-9][0-9]-*"))
    if compile_dirs:
        labels, speeds, losses = [], [], []
        for run_dir in compile_dirs:
            labels.append(run_dir.name.split("-", 1)[1])
            records = load_run(run_dir)
            speeds.append(median_throughput(records))
            losses.append(min(item["val_loss"] for item in records))
        figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        bars = left.bar(labels, speeds, color="#4c78a8")
        left.bar_label(bars, fmt="%.0f", rotation=90, padding=2)
        left.set(ylabel="Tokens / second", title="Compile × precision throughput")
        bars = right.bar(labels, losses, color="#f58518")
        right.bar_label(bars, fmt="%.3f", rotation=90, padding=2)
        right.set(ylabel="Best validation loss", title="Compile × precision quality")
        for axis in (left, right):
            axis.tick_params(axis="x", rotation=25)
            axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "compile_precision")

    micro_dirs = sorted((experiments / "microbatch-equivalence").glob("[0-9][0-9][0-9]-*"))
    reference_batch = experiments / "batch256" / "000-batch-256-full"
    if micro_dirs and reference_batch.exists():
        figure, (curve, resources) = plt.subplots(1, 2, figsize=(11, 4.3))
        groups = [(reference_batch, "256×1")] + [
            (path, path.name.split("batch-", 1)[-1]) for path in micro_dirs
        ]
        names, peaks, throughputs = [], [], []
        for run_dir, label in groups:
            records = load_run(run_dir)
            curve.plot(
                [item["tokens_seen"] / 1e6 for item in records],
                [item["val_loss"] for item in records],
                label=label,
            )
            names.append(label)
            peaks.append(max(item.get("peak_memory_mib", 0) for item in records) / 1024)
            throughputs.append(median_throughput(records))
        curve.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Equal-global-batch convergence")
        curve.legend()
        resource_twin = resources.twinx()
        x = np.arange(len(names))
        resources.bar(x - 0.18, peaks, width=0.36, label="VRAM", color="#4c78a8")
        resource_twin.bar(x + 0.18, throughputs, width=0.36, label="tok/s", color="#f58518")
        resources.set(xticks=x, xticklabels=names, ylabel="Peak VRAM (GiB)", title="Microbatch resource trade-off")
        resource_twin.set_ylabel("Tokens / second")
        save_figure(figure, args.figures / "microbatch_equivalence")

    optimizer_dirs = sorted((experiments / "optimizer-gradient").glob("[0-9][0-9][0-9]-*"))
    if optimizer_dirs:
        figure, (norm_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4.4))
        for run_dir in optimizer_dirs:
            records = load_run(run_dir)
            label = run_dir.name.split("-", 1)[1]
            norm_axis.plot([item["step"] for item in records], [item.get("gradient_norm", 0) for item in records], label=label)
            loss_axis.plot([item["tokens_seen"] / 1e6 for item in records], [item["val_loss"] for item in records], label=label)
        norm_axis.set(xlabel="Step", ylabel="Pre-clip gradient norm", title="Optimizer and clipping dynamics")
        norm_axis.set_yscale("log")
        loss_axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Optimizer generalization")
        loss_axis.set_yscale("log")
        norm_axis.legend(fontsize=7)
        loss_axis.legend(fontsize=7)
        save_figure(figure, args.figures / "optimizer_gradient_dynamics")

    full_batch_dirs = sorted((experiments / "batch512-full").glob("[0-9][0-9][0-9]-*"))
    full_batch_dirs += sorted((experiments / "batch512-seeds").glob("[0-9][0-9][0-9]-*"))
    if full_batch_dirs:
        figure, axis = plt.subplots(figsize=(8.4, 4.7))
        for run_dir, label in [
            (reference_batch, "batch 256"),
            *[(path, path.name.split("-", 1)[1]) for path in full_batch_dirs],
        ]:
            records = load_run(run_dir)
            axis.plot([item["tokens_seen"] / 1e6 for item in records], [item["val_loss"] for item in records], label=label)
        axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Full-budget batch 256 vs 512")
        axis.legend()
        save_figure(figure, args.figures / "batch512_full")

    tied_full_dirs = sorted((experiments / "tied-full").glob("[0-9][0-9][0-9]-*"))
    tied_full_dirs += sorted((experiments / "tied-full-seeds").glob("[0-9][0-9][0-9]-*"))
    if tied_full_dirs:
        figure, axis = plt.subplots(figsize=(8.4, 4.7))
        groups = [(experiments / "final-seeds" / "000-baseline-seed-42", "Untied seed 42")]
        groups += [
            (run_dir, f"Tied {run_dir.name.rsplit('-', 1)[-1]}")
            for run_dir in tied_full_dirs
        ]
        for run_dir, label in groups:
            records = load_run(run_dir)
            axis.plot([item["tokens_seen"] / 1e6 for item in records], [item["val_loss"] for item in records], label=label)
        axis.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title="Full-budget weight tying")
        axis.legend()
        save_figure(figure, args.figures / "weight_tying_full")

    generation_path = args.raw / "generation_panel.json"
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))["summary"]
        names = list(generation)
        lengths = [generation[name]["mean_generated_tokens"] for name in names]
        repetition = [generation[name]["mean_trigram_repetition_rate"] * 100 for name in names]
        figure, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.2))
        bars = left.bar(names, lengths, color="#4c78a8")
        left.bar_label(bars, fmt="%.1f")
        left.set(ylabel="Mean generated tokens", title="Generation length")
        bars = right.bar(names, repetition, color="#e45756")
        right.bar_label(bars, fmt="%.1f%%")
        right.set(ylabel="Repeated trigrams (%)", title="Generation degeneration")
        for axis in (left, right):
            axis.tick_params(axis="x", rotation=25)
            axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "generation_quality")

    tokenizer_extended_path = args.raw / "tokenizer-experiments-200doc.json"
    tokenizer_extended = None
    if tokenizer_extended_path.exists():
        tokenizer_extended = json.loads(tokenizer_extended_path.read_text(encoding="utf-8"))
        generator = np.random.default_rng(336)
        labels, means, lower_errors, upper_errors = [], [], [], []
        for corpus, tokenizer_values in tokenizer_extended.items():
            for tokenizer_name, metrics in tokenizer_values.items():
                values = np.asarray(
                    [item["bytes_per_token"] for item in metrics["per_document"]],
                    dtype=float,
                )
                bootstrap = np.asarray(
                    [
                        generator.choice(values, size=len(values), replace=True).mean()
                        for _ in range(2000)
                    ]
                )
                center = values.mean()
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                labels.append(f"{corpus}\n{tokenizer_name}")
                means.append(center)
                lower_errors.append(center - low)
                upper_errors.append(high - center)
        figure, axis = plt.subplots(figsize=(9.4, 4.7))
        x = np.arange(len(labels))
        bars = axis.bar(
            x,
            means,
            yerr=np.asarray([lower_errors, upper_errors]),
            capsize=5,
            color=["#4c78a8", "#72b7b2", "#f58518", "#eeca3b"],
        )
        axis.bar_label(bars, fmt="%.2f", padding=5)
        axis.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Mean bytes / token",
            title="200-document tokenizer comparison (bootstrap 95% CI)",
        )
        axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, args.figures / "tokenizer_uncertainty")

    final_losses = [min(item["val_loss"] for item in records) for records in seed_records]
    experiment_log = []
    for sweep_dir in sorted(path for path in experiments.iterdir() if path.is_dir()):
        if (sweep_dir / "metrics.jsonl").exists():
            summary = run_summary(sweep_dir)
            summary["sweep"] = "standalone"
            experiment_log.append(summary)
            continue
        for run_dir in sorted(path for path in sweep_dir.iterdir() if (path / "metrics.jsonl").exists()):
            summary = run_summary(run_dir)
            summary["sweep"] = sweep_dir.name
            experiment_log.append(summary)
    comparisons = {
        "provenance": {
            "experiment_rows": len(experiment_log),
            "figure_count": len(list(args.figures.glob("*.png"))),
            "raw_root": args.raw.as_posix(),
        },
        "batch32_seed_mean": float(np.mean(final_losses)),
        "batch32_seed_std": float(np.std(final_losses, ddof=1)),
        "batch32_seed_losses": final_losses,
        "batch128_best_val_loss": min(
            item["val_loss"] for item in load_run(experiments / "large-batch" / "000-batch-128-full")
        ),
        "batch256_best_val_loss": min(
            item["val_loss"] for item in load_run(experiments / "batch256" / "000-batch-256-full")
        ),
        "batch_sizes": batch_sizes,
        "batch_throughput": throughput,
        "batch_peak_memory_gib": memory,
        "tokenizer_experiments": tokenizer_results,
        "tokenizer_experiments_200doc": tokenizer_extended,
        "generation_panel": (
            json.loads(generation_path.read_text(encoding="utf-8"))["summary"]
            if generation_path.exists()
            else None
        ),
        "no_rmsnorm_full": no_rmsnorm_summary,
        "tied_embeddings_pilot": tied_pilot_summary,
        "tied_small_init_pilot": tied_small_init_summary,
        "owt_final": owt_summary,
        "owt_tiny_full": owt_tiny_summary,
        "owt_leaderboard_tied": leaderboard_summary,
        "experiment_count": len(experiment_log),
        "experiments": experiment_log,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    with args.summary.with_name("experiment_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        fieldnames = [
            "sweep",
            "run",
            "best_val_loss",
            "best_step",
            "tokens_seen",
            "elapsed_seconds",
            "peak_memory_mib",
        ]
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_log)

    tables = args.summary.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    by_sweep: dict[str, list[dict[str, float | str]]] = {}
    for row in experiment_log:
        by_sweep.setdefault(str(row["sweep"]), []).append(row)
    with (tables / "experiment_budget.tex").open("w", encoding="utf-8") as destination:
        destination.write("\\begin{tabular}{lrrr}\\toprule\n")
        destination.write("Sweep & Runs & Best loss & Max tokens (M)\\\\\\midrule\n")
        for sweep, rows in sorted(by_sweep.items()):
            escaped_sweep = sweep.replace("_", "\\_")
            destination.write(
                f"{escaped_sweep} & {len(rows)} & "
                f"{min(float(row['best_val_loss']) for row in rows):.3f} & "
                f"{max(float(row['tokens_seen']) for row in rows) / 1e6:.1f}\\\\\n"
            )
        destination.write("\\bottomrule\\end{tabular}\n")
    write_manifest(args.summary.parent, args.figures)
    print(
        f"built {len(list(args.figures.glob('*.png')))} figures from "
        f"{len(experiment_log)} runs"
    )


if __name__ == "__main__":
    main()
