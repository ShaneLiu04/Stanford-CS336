from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cs336_basics.training import TrainConfig, train


def main() -> None:
    data_dir = Path("data/smoke")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(0).integers(0, 128, 8192, dtype=np.uint16).tofile(data_dir / "train.bin")
    np.random.default_rng(1).integers(0, 128, 4096, dtype=np.uint16).tofile(data_dir / "valid.bin")

    print(f"device={torch.cuda.get_device_name()}")
    summary = train(
        TrainConfig(
            train_data=str(data_dir / "train.bin"),
            val_data=str(data_dir / "valid.bin"),
            output_dir="outputs/gpu-smoke",
            vocab_size=128,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            batch_size=8,
            max_steps=3,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=2,
            checkpoint_interval=2,
            device="cuda",
            dtype="bfloat16",
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
