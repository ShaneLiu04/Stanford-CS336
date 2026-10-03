"""Numerical-error sweep for the assignment attention implementations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_systems.flash_attention import FlashAttentionPyTorch, TritonFlashAttention


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-5)
    return difference.max().item(), relative.quantile(0.99).item()


def run_case(
    implementation: str,
    sequence_length: int,
    head_dimension: int,
    dtype: torch.dtype,
    causal: bool,
    device: torch.device,
) -> list[dict[str, object]]:
    torch.manual_seed(336)
    source = [
        torch.randn(2, sequence_length, head_dimension, device=device, dtype=dtype)
        for _ in range(3)
    ]
    upstream = torch.randn_like(source[0])

    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in source]
    reference = F.scaled_dot_product_attention(
        *reference_inputs, is_causal=causal, scale=head_dimension**-0.5
    )
    reference.backward(upstream)

    test_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in source]
    function = (
        FlashAttentionPyTorch if implementation == "pytorch_flash_tiled" else TritonFlashAttention
    )
    actual = function.apply(*test_inputs, causal)
    actual.backward(upstream)

    rows = []
    tensors = [("output", actual, reference)]
    tensors.extend(
        (name, tensor.grad, reference_tensor.grad)
        for name, tensor, reference_tensor in zip(
            ("dq", "dk", "dv"), test_inputs, reference_inputs, strict=True
        )
    )
    for tensor_name, observed, expected in tensors:
        maximum, p99_relative = error_metrics(observed, expected)
        rows.append(
            {
                "implementation": implementation,
                "dtype": str(dtype).removeprefix("torch."),
                "causal": causal,
                "sequence_length": sequence_length,
                "head_dimension": head_dimension,
                "tensor": tensor_name,
                "max_abs_error": maximum,
                "p99_relative_error": p99_relative,
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                "torch_version": torch.__version__,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/results/raw/attention_correctness.csv"),
    )
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--head-dimensions", nargs="+", type=int, default=[64, 128])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    implementations = ["pytorch_flash_tiled"]
    if device.type == "cuda":
        implementations.append("triton_flash")
    dtypes = [torch.float32]
    if device.type == "cuda":
        dtypes.append(torch.bfloat16)

    rows = []
    for implementation in implementations:
        for dtype in dtypes:
            for causal in (False, True):
                for sequence_length in args.sequence_lengths:
                    for head_dimension in args.head_dimensions:
                        rows.extend(
                            run_case(
                                implementation,
                                sequence_length,
                                head_dimension,
                                dtype,
                                causal,
                                device,
                            )
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
