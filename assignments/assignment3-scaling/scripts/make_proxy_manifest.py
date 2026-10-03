"""Create token-aligned TinyStories IsoFLOP proxy runs for the A1 trainer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODELS = [
    ("n0.8m", 128, 4, 4, 352),
    ("n1.8m", 192, 4, 6, 512),
    ("n3m", 256, 4, 8, 672),
    ("n11m", 384, 6, 12, 1024),
    ("n25m", 512, 8, 16, 1344),
    ("n57m", 768, 8, 24, 2048),
]
COMPUTE_TIERS = [3e14, 1e15, 3e15]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("proxy/proxy_manifest.json")
    )
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    batch, context = 32, 256
    manifest = []
    for compute in COMPUTE_TIERS:
        models = MODELS[:2] if args.small_only else MODELS
        for name, width, layers, heads, feedforward in models:
            parameters = 12 * layers * width**2
            tokens = compute / (6 * parameters)
            steps = max(100, math.ceil(tokens / (batch * context)))
            manifest.append(
                {
                    "name": f"c{compute:.0e}-{name}",
                    "overrides": {
                        "d_model": width,
                        "num_layers": layers,
                        "num_heads": heads,
                        "d_ff": feedforward,
                        "batch_size": batch,
                        "max_steps": steps,
                        "warmup_steps": min(100, max(10, steps // 20)),
                        "eval_interval": max(25, steps // 20),
                        "max_lr": 0.002,
                        "min_lr": 0.0002,
                        "save_checkpoints": False,
                    },
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
