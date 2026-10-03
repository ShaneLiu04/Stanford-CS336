"""Measure activation-checkpoint time/memory trade-offs on one GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from cs336_basics.model import BasicsTransformerLM


CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=CONFIGS, default="large")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--segments", nargs="+", type=int, default=[0, 4, 12, 36])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/results/raw/checkpointing.csv"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint benchmark requires CUDA")

    device = torch.device("cuda")
    rows = []
    config = CONFIGS[args.model_size]
    for sequence_length in args.sequence_lengths:
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=sequence_length,
            **config,
        ).to(device)
        tokens = torch.randint(0, 10_000, (args.batch_size, sequence_length), device=device)
        targets = torch.randint_like(tokens, high=10_000)
        for segments in args.segments:
            status = "ok"
            samples: list[float] = []

            def step() -> None:
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if segments == 0:
                        logits = model(tokens)
                    else:
                        hidden = model.token_embeddings(tokens)
                        hidden = checkpoint_sequential(
                            model.layers, segments, hidden, use_reentrant=False
                        )
                        hidden = model.ln_final(hidden)
                        logits = model.lm_head(hidden)
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
                loss.backward()

            try:
                for _ in range(args.warmup):
                    step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.repetitions):
                    start = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1_000)
                peak = torch.cuda.max_memory_allocated() / 2**20
            except torch.OutOfMemoryError:
                status = "oom"
                peak = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.empty_cache()
            rows.append(
                {
                    "model_size": args.model_size,
                    "batch_size": args.batch_size,
                    "sequence_length": sequence_length,
                    "segments": segments,
                    "mean_ms": mean(samples) if samples else "",
                    "std_ms": stdev(samples) if len(samples) > 1 else 0.0 if samples else "",
                    "peak_memory_mib": peak,
                    "status": status,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "gpu": torch.cuda.get_device_name(),
                    "torch_version": torch.__version__,
                }
            )
        model = None
        tokens = targets = None
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
