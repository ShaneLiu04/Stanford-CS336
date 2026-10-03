from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil

try:
    import resource as resource_module
except ImportError:
    resource_module: Any = None

from cs336_basics.tokenizer import Tokenizer, train_bpe

_WORKER_TOKENIZER: Tokenizer | None = None


def _initialize_worker(tokenizer: Tokenizer) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_lines(lines: list[str]) -> list[int]:
    assert _WORKER_TOKENIZER is not None
    return list(_WORKER_TOKENIZER.encode_iterable(lines))


def _line_chunks(lines, chunk_size: int = 128):
    while chunk := list(itertools.islice(lines, chunk_size)):
        yield chunk


def save_tokenizer(output: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "vocab.json").write_text(
        json.dumps({str(index): value.hex() for index, value in vocab.items()}, indent=2),
        encoding="utf-8",
    )
    (output / "merges.json").write_text(
        json.dumps([[left.hex(), right.hex()] for left, right in merges], indent=2),
        encoding="utf-8",
    )


def encode_file(
    tokenizer: Tokenizer,
    source: Path,
    destination: Path,
    workers: int = 1,
) -> dict[str, float | int]:
    started = time.perf_counter()
    count = 0
    with source.open(encoding="utf-8") as lines, destination.open("wb") as output:
        if workers == 1:
            encoded_chunks = (list(tokenizer.encode(line)) for line in lines)
            for encoded in encoded_chunks:
                np.asarray(encoded, dtype=np.uint16).tofile(output)
                count += len(encoded)
        else:
            context = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
            with context.Pool(workers, initializer=_initialize_worker, initargs=(tokenizer,)) as pool:
                for encoded in pool.imap(_encode_lines, _line_chunks(lines), chunksize=4):
                    np.asarray(encoded, dtype=np.uint16).tofile(output)
                    count += len(encoded)
    elapsed = time.perf_counter() - started
    source_bytes = source.stat().st_size
    encoded = np.memmap(destination, dtype=np.uint16, mode="r")
    return {
        "source_bytes": source_bytes,
        "tokens": count,
        "elapsed_seconds": elapsed,
        "tokens_per_second": count / elapsed,
        "bytes_per_token": source_bytes / count,
        "minimum_token_id": int(encoded.min()),
        "maximum_token_id": int(encoded.max()),
        "binary_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--bpe-workers", type=int, default=1)
    parser.add_argument("--bpe-chunks", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    started = time.perf_counter()
    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.json"
    metrics_path = args.output_dir / "metrics.json"
    previous_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    if vocab_path.exists() and merges_path.exists():
        tokenizer = Tokenizer.from_serialized(vocab_path, merges_path, args.special_token)
        vocab, merges = tokenizer.vocab, tokenizer.merges
        previous_bpe_seconds = previous_metrics.get("bpe_training_seconds")
        bpe_seconds = previous_bpe_seconds if previous_bpe_seconds not in (None, 0, 0.0) else None
        bpe_reused = True
    else:
        vocab, merges = train_bpe(
            args.train_text,
            args.vocab_size,
            args.special_token,
            streaming=True,
            pretokenize_workers=args.bpe_workers,
            pretokenize_chunks=args.bpe_chunks,
        )
        tokenizer = Tokenizer(vocab, merges, args.special_token)
        save_tokenizer(args.output_dir, vocab, merges)
        bpe_seconds = time.perf_counter() - started
        bpe_reused = False
    longest = cast(
        bytes,
        max(
            (token for token in vocab.values() if token not in {item.encode() for item in args.special_token}),
            key=len,
        ),
    )
    peak_rss_mib = (
        getattr(resource_module, "getrusage")(getattr(resource_module, "RUSAGE_SELF")).ru_maxrss / 1024
        if resource_module is not None
        else psutil.Process().memory_info().rss / 2**20
    )
    metrics: dict[str, object] = {
        "bpe_training_seconds": bpe_seconds,
        "bpe_reused": bpe_reused,
        "peak_rss_mib": peak_rss_mib,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "longest_token_hex": longest.hex(),
        "longest_token_bytes": len(longest),
        "vocab_sha256": hashlib.sha256((args.output_dir / "vocab.json").read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256((args.output_dir / "merges.json").read_bytes()).hexdigest(),
    }
    metrics["train"] = encode_file(tokenizer, args.train_text, args.output_dir / "train.bin", args.workers)
    metrics["valid"] = encode_file(tokenizer, args.valid_text, args.output_dir / "valid.bin", args.workers)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
