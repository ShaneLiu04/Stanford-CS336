"""Regenerate the deterministic public DDP correctness fixtures."""

from pathlib import Path

import torch


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    destination.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(336)
    torch.save(torch.randn(20, 10, generator=generator), destination / "ddp_test_data.pt")
    torch.save(torch.randn(20, 10, generator=generator), destination / "ddp_test_labels.pt")


if __name__ == "__main__":
    main()
