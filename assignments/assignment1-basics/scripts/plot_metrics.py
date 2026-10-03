from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes_grid = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes_grid.ravel()
    summaries = []
    for run in args.runs:
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        label = run.name
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
        best = min(records, key=lambda item: item["val_loss"])
        summaries.append({
            "run": label,
            "best_val_loss": best["val_loss"],
            "best_step": best["step"],
            "tokens_seen": best["tokens_seen"],
            "elapsed_seconds": best["elapsed_seconds"],
            "last_tokens_per_second": records[-1].get("tokens_per_second", 0),
            "peak_memory_mib": max(
                item.get("peak_memory_mib", item.get("peak_gpu_memory_bytes", 0) / 2**20)
                for item in records
            ),
        })
    axes[0].set(xlabel="Step", ylabel="Validation loss")
    axes[1].set(xlabel="Tokens seen (millions)", ylabel="Validation loss")
    axes[2].set(xlabel="Wall-clock (minutes)", ylabel="Validation loss")
    axes[3].set(xlabel="Tokens seen (millions)", ylabel="Tokens / second")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()

