"""Create report figures from raw benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_attention(frame: pd.DataFrame, output: Path) -> None:
    data = frame[(frame["experiment"] == "attention") & (frame["status"] == "ok")].copy()
    if data.empty:
        return
    for dtype, subset in data.groupby("dtype"):
        # Never mix CPU fallback and CUDA measurements in one curve. Prefer
        # the report's target GPU whenever both environments are present.
        accelerator_subset = subset[subset["gpu"] != "CPU"]
        if not accelerator_subset.empty:
            subset = accelerator_subset
        has_memory = subset["peak_memory_mib"].notna().any()
        if has_memory:
            fig, (latency_ax, memory_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
        else:
            fig, latency_ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
            memory_ax = None
        for (implementation, head_dimension), series in subset.groupby(
            ["implementation", "head_dimension"]
        ):
            series = series.sort_values("sequence_length")
            label = f"{implementation}, d={head_dimension}"
            latency_ax.plot(series["sequence_length"], series["mean_ms"], marker="o", label=label)
            if memory_ax is not None:
                memory_ax.plot(
                    series["sequence_length"], series["peak_memory_mib"], marker="o", label=label
                )
        axes = (latency_ax, memory_ax) if memory_ax is not None else (latency_ax,)
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("Sequence length")
        latency_ax.set_yscale("log")
        latency_ax.set_ylabel("Latency (ms, log scale)")
        latency_ax.set_title(f"Attention latency ({dtype})")
        if memory_ax is not None:
            memory_ax.set_ylabel("Peak allocated memory (MiB)")
            memory_ax.set_title(f"Attention peak memory ({dtype})")
            memory_ax.legend(fontsize=7, loc="best")
        else:
            latency_ax.legend(fontsize=8, ncol=2, loc="best")
        _save(fig, output / f"attention_{dtype}")


def plot_models(frame: pd.DataFrame, output: Path) -> None:
    data = frame[(frame["experiment"] == "transformer") & (frame["status"] == "ok")].copy()
    if data.empty:
        return
    order = ["small", "medium", "large", "xl"]
    data["model_size"] = pd.Categorical(data["model_size"], categories=order, ordered=True)
    fig, (latency_ax, memory_ax) = plt.subplots(1, 2, figsize=(10, 4.1))
    for dtype, series in data.groupby("dtype", observed=True):
        series = series.sort_values("model_size")
        latency_ax.plot(series["model_size"], series["mean_ms"], marker="o", label=dtype)
        memory_ax.plot(series["model_size"], series["peak_memory_mib"], marker="o", label=dtype)
    latency_ax.set_ylabel("Forward + backward latency (ms)")
    memory_ax.set_ylabel("Peak allocated memory (MiB)")
    for axis in (latency_ax, memory_ax):
        axis.set_xlabel("Model size")
        axis.grid(True, alpha=0.25)
        axis.legend()
    latency_ax.set_title("Transformer step latency")
    memory_ax.set_title("Transformer peak memory")
    _save(fig, output / "transformer_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("report/results/raw/benchmark.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("report/results/figures")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    plot_attention(frame, args.output_dir)
    plot_models(frame, args.output_dir)


if __name__ == "__main__":
    main()
