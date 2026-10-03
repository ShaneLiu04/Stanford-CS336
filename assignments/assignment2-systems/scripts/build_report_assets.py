"""Rebuild all Assignment 2 report figures, tables, summaries, and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "pytorch_sdpa": "PyTorch SDPA",
    "pytorch_flash_tiled": "PyTorch tiled",
    "triton_flash": "Triton",
}
COLORS = {
    "pytorch_sdpa": "#2ca02c",
    "pytorch_flash_tiled": "#d62728",
    "triton_flash": "#1f77b4",
}


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def accelerator_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["gpu"] != "CPU") & (frame["status"] == "ok")].copy()


def plot_attention_mode(
    frame: pd.DataFrame, figures: Path, mode: str, dtype: str, suffix: str
) -> None:
    data = accelerator_rows(frame)
    data = data[
        (data["experiment"] == "attention")
        & (data["mode"] == mode)
        & (data["dtype"] == dtype)
    ]
    if data.empty:
        return
    figure, (latency, memory) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for (implementation, dimension), series in data.groupby(
        ["implementation", "head_dimension"]
    ):
        series = series.sort_values("sequence_length")
        label = f"{LABELS[implementation]}, d={dimension}"
        latency.plot(
            series["sequence_length"],
            series["mean_ms"],
            marker="o",
            label=label,
            color=COLORS[implementation],
            linestyle="-" if dimension == 64 else "--",
        )
        memory.plot(
            series["sequence_length"],
            series["peak_memory_mib"],
            marker="o",
            label=label,
            color=COLORS[implementation],
            linestyle="-" if dimension == 64 else "--",
        )
    for axis in (latency, memory):
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Sequence length")
    latency.set_yscale("log")
    latency.set_ylabel("Latency (ms, log scale)")
    memory.set_yscale("log")
    memory.set_ylabel("Peak allocated memory (MiB, log scale)")
    title_mode = "forward" if mode == "forward" else "forward + backward"
    latency.set_title(f"Attention {title_mode} latency ({dtype})")
    memory.set_title(f"Attention {title_mode} memory ({dtype})")
    memory.legend(fontsize=7, ncol=2)
    save_figure(figure, figures / f"attention_{suffix}_{dtype}")


def plot_attention_tradeoff(frame: pd.DataFrame, figures: Path) -> None:
    data = accelerator_rows(frame)
    data = data[
        (data["experiment"] == "attention")
        & (data["mode"] == "forward_backward")
        & (data["head_dimension"] == 64)
        & (data["sequence_length"] == 4096)
    ]
    if data.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, dtype in zip(axes, ("fp32", "bf16"), strict=True):
        subset = data[data["dtype"] == dtype].sort_values("mean_ms")
        for _, row in subset.iterrows():
            axis.scatter(
                row["mean_ms"],
                row["peak_memory_mib"],
                s=85,
                color=COLORS[row["implementation"]],
                label=LABELS[row["implementation"]],
            )
            axis.annotate(
                LABELS[row["implementation"]],
                (row["mean_ms"], row["peak_memory_mib"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(
            xlabel="Forward + backward latency (ms, log)",
            ylabel="Peak memory (MiB, log)",
            title=f"N=4096, d=64 ({dtype})",
        )
        axis.grid(True, alpha=0.25)
    save_figure(figure, figures / "attention_latency_memory_tradeoff")


def plot_transformer(frame: pd.DataFrame, figures: Path) -> None:
    data = accelerator_rows(frame)
    data = data[data["experiment"] == "transformer"].copy()
    if data.empty:
        return
    order = ["small", "medium", "large", "xl"]
    data["model_size"] = pd.Categorical(data["model_size"], categories=order, ordered=True)
    figure, (latency, memory) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for dtype, series in data.groupby("dtype", observed=True):
        series = series.sort_values("model_size")
        latency.plot(series["model_size"], series["mean_ms"], marker="o", label=dtype)
        memory.plot(
            series["model_size"], series["peak_memory_mib"] / 1024, marker="o", label=dtype
        )
    latency.set_ylabel("Forward + backward latency (ms)")
    memory.set_ylabel("Peak allocated memory (GiB)")
    for axis in (latency, memory):
        axis.set_xlabel("Model size")
        axis.grid(True, alpha=0.25)
        axis.legend()
    latency.set_title("Transformer step latency")
    memory.set_title("Transformer peak memory")
    save_figure(figure, figures / "transformer_scaling")

    pivot = data.pivot_table(index="model_size", columns="dtype", values=["mean_ms", "peak_memory_mib"])
    speedup = pivot["mean_ms"]["fp32"] / pivot["mean_ms"]["bf16"]
    saving = 1 - pivot["peak_memory_mib"]["bf16"] / pivot["peak_memory_mib"]["fp32"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    bars = left.bar(speedup.index.astype(str), speedup.values, color="#1f77b4")
    left.bar_label(bars, fmt="%.2f×")
    left.set(ylabel="FP32 time / BF16 time", title="BF16 speedup")
    bars = right.bar(saving.index.astype(str), saving.values * 100, color="#ff7f0e")
    right.bar_label(bars, fmt="%.1f%%")
    right.set(ylabel="Peak-memory reduction (%)", title="BF16 memory reduction")
    for axis in (left, right):
        axis.grid(True, axis="y", alpha=0.25)
    save_figure(figure, figures / "mixed_precision_benefit")


def plot_correctness(raw: Path, figures: Path) -> dict[str, float]:
    path = raw / "attention_correctness.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    grouped = (
        data.groupby(["implementation", "dtype", "tensor"], as_index=False)["max_abs_error"]
        .max()
        .sort_values("max_abs_error")
    )
    figure, axis = plt.subplots(figsize=(9.2, 4.7))
    labels = [
        f"{LABELS[row.implementation]}\n{row.dtype}/{row.tensor}"
        for row in grouped.itertuples()
    ]
    bars = axis.bar(labels, grouped["max_abs_error"], color="#4c78a8")
    axis.set_yscale("log")
    axis.set(ylabel="Maximum absolute error (log)", title="Attention numerical-error envelope")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.bar_label(bars, fmt="%.1e", fontsize=7, rotation=90, padding=2)
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(figure, figures / "attention_correctness")
    return {
        "max_abs_error": float(data["max_abs_error"].max()),
        "max_p99_relative_error": float(data["p99_relative_error"].max()),
        "cases": int(len(data)),
    }


def plot_checkpointing(raw: Path, figures: Path) -> dict[str, float]:
    path = raw / "checkpointing.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    ok = data[data["status"] == "ok"].copy()
    if ok.empty:
        return {"cases": int(len(data)), "successful_cases": 0}
    figure, (latency, memory) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for sequence, series in ok.groupby("sequence_length"):
        series = series.sort_values("segments")
        label = f"sequence={sequence}"
        latency.plot(series["segments"], series["mean_ms"], marker="o", label=label)
        memory.plot(
            series["segments"], series["peak_memory_mib"] / 1024, marker="o", label=label
        )
    latency.set(ylabel="Step latency (ms)", title="Checkpoint recomputation cost")
    memory.set(ylabel="Peak allocated memory (GiB)", title="Checkpoint memory saving")
    for axis in (latency, memory):
        axis.set_xlabel("Checkpoint segments (0 = disabled)")
        axis.grid(True, alpha=0.25)
        axis.legend()
    save_figure(figure, figures / "activation_checkpointing")
    baseline = ok[ok["segments"] == 0]
    strongest = ok.sort_values("segments").groupby("sequence_length").tail(1)
    merged = baseline.merge(strongest, on="sequence_length", suffixes=("_base", "_checkpoint"))
    return {
        "cases": int(len(data)),
        "successful_cases": int(len(ok)),
        "max_memory_reduction_fraction": float(
            (1 - merged["peak_memory_mib_checkpoint"] / merged["peak_memory_mib_base"]).max()
        )
        if not merged.empty
        else 0.0,
        "max_slowdown": float(
            (merged["mean_ms_checkpoint"] / merged["mean_ms_base"]).max()
        )
        if not merged.empty
        else 0.0,
    }


def plot_tests(raw: Path, figures: Path) -> dict[str, int]:
    path = raw / "test_summary.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(8.8, 4.1))
    positions = np.arange(len(data))
    axis.bar(positions, data["passed"], label="passed", color="#2ca02c")
    axis.bar(
        positions,
        data["skipped"],
        bottom=data["passed"],
        label="skipped",
        color="#ffbf00",
    )
    axis.bar(
        positions,
        data["failed"],
        bottom=data["passed"] + data["skipped"],
        label="failed",
        color="#d62728",
    )
    axis.set(
        xticks=positions,
        xticklabels=data["environment"],
        ylabel="Test cases",
        title="Correctness tests by execution environment",
    )
    axis.tick_params(axis="x", rotation=12)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(figure, figures / "test_matrix")
    return {
        "passed": int(data["passed"].sum()),
        "skipped": int(data["skipped"].sum()),
        "failed": int(data["failed"].sum()),
    }


def write_tables(frame: pd.DataFrame, raw: Path, tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    gpu = accelerator_rows(frame)
    transformer = gpu[gpu["experiment"] == "transformer"][
        ["model_size", "dtype", "mean_ms", "std_ms", "peak_memory_mib"]
    ].sort_values(["model_size", "dtype"])
    transformer.to_latex(
        tables / "transformer_scaling.tex",
        index=False,
        float_format="%.2f",
        caption="RTX 6000D Transformer forward+backward measurements.",
        label="tab:transformer-measured",
    )
    attention = gpu[
        (gpu["experiment"] == "attention")
        & (gpu["sequence_length"] == 4096)
        & (gpu["head_dimension"] == 64)
    ][
        [
            "mode",
            "dtype",
            "implementation",
            "mean_ms",
            "std_ms",
            "peak_memory_mib",
        ]
    ].sort_values(["mode", "dtype", "implementation"])
    attention.to_latex(
        tables / "attention_key_results.tex",
        index=False,
        float_format="%.3f",
        caption="Key attention measurements at sequence length 4096 and head dimension 64.",
        label="tab:attention-measured",
    )
    checkpoint = raw / "checkpointing.csv"
    if checkpoint.exists():
        pd.read_csv(checkpoint).to_latex(
            tables / "checkpointing.tex",
            index=False,
            float_format="%.2f",
            caption="Activation-checkpointing time/memory trade-off.",
            label="tab:checkpoint-measured",
        )


def write_manifest(results: Path) -> None:
    manifest = results / "manifest.sha256"
    lines = []
    for path in sorted(results.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(results).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fused_backward(raw: Path, figures: Path) -> dict[str, float]:
    path = raw / "benchmark_fused_backward.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    data = data[(data["status"] == "ok") & (data["head_dimension"] == 64)]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    for column, dtype in enumerate(("fp32", "bf16")):
        subset = data[data["dtype"] == dtype]
        for implementation, series in subset.groupby("implementation"):
            series = series.sort_values("sequence_length")
            label = LABELS[implementation]
            axes[0, column].plot(series["sequence_length"], series["mean_ms"], "o-", label=label)
            axes[1, column].plot(series["sequence_length"], series["peak_memory_mib"], "o-", label=label)
        axes[0, column].set_title(f"Fused forward + backward ({dtype})")
        axes[0, column].set_ylabel("Latency (ms)")
        axes[1, column].set_ylabel("Peak memory (MiB)")
        for axis in axes[:, column]:
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("Sequence length")
            axis.grid(True, alpha=0.25)
            axis.legend()
    save_figure(figure, figures / "fused_triton_end_to_end")

    old = pd.read_csv(raw / "benchmark.csv")
    old = old[
        (old["gpu"] != "CPU")
        & (old["mode"] == "forward_backward")
        & (old["implementation"] == "triton_flash")
        & (old["head_dimension"] == 64)
    ]
    merged = old.merge(
        data[data["implementation"] == "triton_flash"],
        on=["dtype", "sequence_length", "head_dimension"],
        suffixes=("_fallback", "_fused"),
    )
    merged["speedup"] = merged["mean_ms_fallback"] / merged["mean_ms_fused"]
    figure, axis = plt.subplots(figsize=(7.8, 4.5))
    for dtype, series in merged.groupby("dtype"):
        axis.plot(series["sequence_length"], series["speedup"], "o-", label=dtype)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set(
        xlabel="Sequence length",
        ylabel="PyTorch-backward time / fused-backward time",
        title="Triton backward fusion speedup",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    save_figure(figure, figures / "triton_backward_speedup")
    target = merged[(merged["dtype"] == "bf16") & (merged["sequence_length"] == 4096)]
    return {
        "rows": int(len(data)),
        "bf16_n4096_speedup_vs_fallback": float(target["speedup"].iloc[0]),
        "bf16_n4096_fused_ms": float(target["mean_ms_fused"].iloc[0]),
    }


def plot_model_extensions(raw: Path, figures: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    context_path = raw / "model_context_scaling.csv"
    if context_path.exists():
        data = pd.read_csv(context_path)
        data = data[data["status"] == "ok"]
        figure, (latency, memory) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for model, series in data.groupby("model_size"):
            series = series.sort_values("sequence_length")
            latency.plot(series["sequence_length"], series["mean_ms"], "o-", label=model)
            memory.plot(
                series["sequence_length"],
                series["peak_memory_mib"] / 1024,
                "o-",
                label=model,
            )
        for axis in (latency, memory):
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Sequence length")
            axis.grid(True, alpha=0.25)
            axis.legend()
        latency.set(ylabel="Forward + backward latency (ms)", title="BF16 context scaling")
        memory.set(ylabel="Peak memory (GiB)", title="BF16 context memory")
        save_figure(figure, figures / "model_context_scaling")
        result["context_rows"] = int(len(data))

    compile_path = raw / "model_compile.csv"
    if compile_path.exists():
        compiled = pd.read_csv(compile_path)
        compiled = compiled[compiled["status"] == "ok"].copy()
        eager = pd.read_csv(raw / "benchmark.csv")
        eager = eager[
            (eager["experiment"] == "transformer")
            & (eager["sequence_length"] == 512)
            & eager["model_size"].isin(compiled["model_size"])
            & eager["dtype"].isin(compiled["dtype"])
        ].copy()
        compiled["variant"] = "compiled"
        eager["variant"] = "eager"
        comparison = pd.concat([eager, compiled], ignore_index=True)
        figure, (latency, memory) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        labels = []
        for (model, dtype), series in comparison.groupby(["model_size", "dtype"]):
            series = series.sort_values("variant")
            labels.append(f"{model}/{dtype}")
            latency.bar(
                [f"{model}/{dtype}/{variant}" for variant in series["variant"]],
                series["mean_ms"],
            )
            memory.bar(
                [f"{model}/{dtype}/{variant}" for variant in series["variant"]],
                series["peak_memory_mib"] / 1024,
            )
        latency.set(ylabel="Latency (ms)", title="torch.compile latency")
        memory.set(ylabel="Peak memory (GiB)", title="torch.compile memory")
        for axis in (latency, memory):
            axis.tick_params(axis="x", rotation=35, labelsize=7)
            axis.grid(True, axis="y", alpha=0.25)
        save_figure(figure, figures / "torch_compile_model")
        result["compile_rows"] = int(len(compiled))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    raw = args.results / "raw"
    figures = args.results / "figures"
    tables = args.results / "tables"
    frame = pd.read_csv(raw / "benchmark.csv")
    plt.style.use("seaborn-v0_8-whitegrid")

    for mode, suffix in (("forward", "forward"), ("forward_backward", "end_to_end")):
        for dtype in ("fp32", "bf16"):
            plot_attention_mode(frame, figures, mode, dtype, suffix)
    plot_attention_tradeoff(frame, figures)
    plot_transformer(frame, figures)
    correctness = plot_correctness(raw, figures)
    checkpointing = plot_checkpointing(raw, figures)
    tests = plot_tests(raw, figures)
    fused_backward = plot_fused_backward(raw, figures)
    model_extensions = plot_model_extensions(raw, figures)
    write_tables(frame, raw, tables)

    gpu = accelerator_rows(frame)
    summary = {
        "provenance": {
            "benchmark_rows": int(len(frame)),
            "gpu_rows": int(len(gpu)),
            "gpu": sorted(gpu["gpu"].unique().tolist()),
            "torch_versions": sorted(gpu["torch_version"].unique().tolist()),
        },
        "tests": tests,
        "attention_correctness": correctness,
        "checkpointing": checkpointing,
        "fused_backward": fused_backward,
        "model_extensions": model_extensions,
        "transformer": gpu[gpu["experiment"] == "transformer"].to_dict(orient="records"),
    }
    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frame.to_csv(args.results / "experiment_log.csv", index=False)
    write_manifest(args.results)
    print(
        f"built {len(list(figures.glob('*.png')))} figures and "
        f"{len(list(tables.glob('*.tex')))} tables"
    )


if __name__ == "__main__":
    main()
