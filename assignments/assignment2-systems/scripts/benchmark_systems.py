"""Reproducible single-GPU benchmarks for CS336 Assignment 2.

Results are append-only CSV rows so interrupted AutoDL runs remain useful.
Every timed region is synchronized on CUDA and includes warm-up iterations.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.flash_attention import FlashAttentionPyTorch, TritonFlashAttention


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass
class Result:
    experiment: str
    implementation: str
    model_size: str
    dtype: str
    batch_size: int
    sequence_length: int
    head_dimension: int
    mode: str
    warmup: int
    repetitions: int
    mean_ms: float | None
    std_ms: float | None
    peak_memory_mib: float | None
    status: str
    gpu: str
    torch_version: str
    cuda_version: str
    git_sha: str
    timestamp_utc: str


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, warmup: int, repetitions: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repetitions):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1_000)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
    return mean(samples), stdev(samples) if len(samples) > 1 else 0.0, peak


def _metadata(**kwargs) -> Result:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    return Result(
        mean_ms=None,
        std_ms=None,
        peak_memory_mib=None,
        status="pending",
        gpu=gpu,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        git_sha=_git_sha(),
        timestamp_utc=now,
        **kwargs,
    )


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def benchmark_attention(args: argparse.Namespace) -> list[Result]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    implementations = {
        "pytorch_sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal),
        "pytorch_flash_tiled": lambda q, k, v: FlashAttentionPyTorch.apply(q, k, v, args.causal),
    }
    if device.type == "cuda":
        implementations["triton_flash"] = lambda q, k, v: TritonFlashAttention.apply(q, k, v, args.causal)
    implementations = {name: operation for name, operation in implementations.items() if name in args.implementations}
    if not implementations:
        raise ValueError("none of the requested implementations is available")
    rows: list[Result] = []
    for dtype_name in args.dtypes:
        dtype = _dtype(dtype_name)
        for sequence_length in args.sequence_lengths:
            for head_dimension in args.head_dimensions:
                for implementation, operation in implementations.items():
                    row = _metadata(
                        experiment="attention",
                        implementation=implementation,
                        model_size="operator",
                        dtype=dtype_name,
                        batch_size=args.batch_size,
                        sequence_length=sequence_length,
                        head_dimension=head_dimension,
                        mode=args.mode,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                    )
                    try:
                        q = torch.randn(
                            args.batch_size,
                            sequence_length,
                            head_dimension,
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                        )
                        k = torch.randn_like(q, requires_grad=True)
                        v = torch.randn_like(q, requires_grad=True)

                        def step() -> None:
                            out = operation(q, k, v)
                            if args.mode in {"backward", "forward_backward"}:
                                out.backward(torch.ones_like(out), retain_graph=False)
                                q.grad = k.grad = v.grad = None

                        row.mean_ms, row.std_ms, row.peak_memory_mib = _time(step, args.warmup, args.repetitions)
                        row.status = "ok"
                    except torch.OutOfMemoryError:
                        row.status = "oom"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as exc:  # Preserve sweep progress and exact failure.
                        row.status = f"error:{type(exc).__name__}:{exc}"
                    rows.append(row)
    return rows


def benchmark_model(args: argparse.Namespace) -> list[Result]:
    if not torch.cuda.is_available():
        raise RuntimeError("End-to-end model benchmark requires CUDA")
    rows: list[Result] = []
    device = torch.device("cuda")
    for model_size in args.model_sizes:
        config = MODEL_CONFIGS[model_size]
        for sequence_length in args.sequence_lengths:
            for dtype_name in args.dtypes:
                row = _benchmark_model_case(args, device, model_size, config, sequence_length, dtype_name)
                rows.append(row)
    return rows


def _benchmark_model_case(
    args: argparse.Namespace,
    device: torch.device,
    model_size: str,
    config: dict[str, int],
    sequence_length: int,
    dtype_name: str,
) -> Result:
    row = _metadata(
        experiment="transformer",
        implementation="cs336_basics_compile" if args.compile_model else "cs336_basics_eager",
        model_size=model_size,
        dtype=dtype_name,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        head_dimension=config["d_model"] // config["num_heads"],
        mode=args.mode,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    try:
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=sequence_length,
            **config,
        ).to(device)
        runtime_model = torch.compile(model, mode=args.compile_mode) if args.compile_model else model
        tokens = torch.randint(
            args.vocab_size,
            (args.batch_size, sequence_length),
            device=device,
        )
        targets = torch.randint_like(tokens, high=args.vocab_size)

        def step() -> None:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=_dtype(dtype_name), enabled=dtype_name != "fp32"):
                logits = runtime_model(tokens)
                loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
            if args.mode != "forward":
                loss.backward()

        row.mean_ms, row.std_ms, row.peak_memory_mib = _time(step, args.warmup, args.repetitions)
        row.status = "ok"
    except torch.OutOfMemoryError:
        row.status = "oom"
        torch.cuda.empty_cache()
    except Exception as exc:
        row.status = f"error:{type(exc).__name__}:{exc}"
    return row


def write_rows(rows: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        if not exists:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("attention", "model"))
    parser.add_argument("--output", type=Path, default=Path("report/results/raw/benchmark.csv"))
    parser.add_argument("--model-sizes", nargs="+", choices=MODEL_CONFIGS, default=["small"])
    parser.add_argument("--dtypes", nargs="+", choices=("fp32", "bf16"), default=["fp32", "bf16"])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--head-dimensions", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("pytorch_sdpa", "pytorch_flash_tiled", "triton_flash"),
        default=("pytorch_sdpa", "pytorch_flash_tiled", "triton_flash"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=("forward", "backward", "forward_backward"), default="forward_backward")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if any(value <= 0 for value in (*args.sequence_lengths, *args.head_dimensions)):
        parser.error("sequence lengths and head dimensions must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = benchmark_attention(args) if args.experiment == "attention" else benchmark_model(args)
    if rows:
        write_rows(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output), "host": platform.node()}))


if __name__ == "__main__":
    main()
