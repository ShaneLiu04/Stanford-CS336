from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


END_OF_TEXT = b"<|endoftext|>"


def sample_documents(path: Path, count: int, seed: int) -> list[str]:
    """Sample documents without reading the full corpus into memory."""
    generator = random.Random(seed)
    documents: list[str] = []
    with path.open("rb") as source:
        size = path.stat().st_size
        while len(documents) < count:
            source.seek(generator.randrange(max(1, size - 1)))
            carry = b""
            while True:
                chunk = source.read(64 * 1024)
                if not chunk:
                    source.seek(0)
                    carry = b""
                    continue
                combined = carry + chunk
                marker = combined.find(END_OF_TEXT)
                if marker >= 0:
                    remainder = combined[marker + len(END_OF_TEXT) :]
                    break
                carry = combined[-(len(END_OF_TEXT) - 1) :]
            payload = bytearray(remainder)
            while True:
                marker = payload.find(END_OF_TEXT)
                if marker >= 0:
                    del payload[marker:]
                    break
                chunk = source.read(64 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
            if payload:
                documents.append(payload.decode("utf-8"))
    return documents


def evaluate(tokenizer: Tokenizer, documents: list[str], repeats: int) -> dict[str, object]:
    source_bytes = sum(len(document.encode("utf-8")) for document in documents)
    per_document = []
    token_count = 0
    for document in documents:
        document_bytes = len(document.encode("utf-8"))
        document_tokens = len(tokenizer.encode(document))
        token_count += document_tokens
        per_document.append(
            {
                "bytes": document_bytes,
                "tokens": document_tokens,
                "bytes_per_token": document_bytes / document_tokens,
            }
        )
    started = time.perf_counter()
    for _ in range(repeats):
        for document in documents:
            tokenizer.encode(document)
    elapsed = time.perf_counter() - started
    return {
        "documents": len(documents),
        "source_bytes": source_bytes,
        "tokens": token_count,
        "bytes_per_token": source_bytes / token_count,
        "bytes_per_second": source_bytes * repeats / elapsed,
        "estimated_pile_hours": 825e9 / (source_bytes * repeats / elapsed) / 3600,
        "per_document": per_document,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the tokenizer comparison deliverables")
    parser.add_argument("--tinystories-text", type=Path, required=True)
    parser.add_argument("--owt-text", type=Path, required=True)
    parser.add_argument("--tinystories-tokenizer", type=Path, required=True)
    parser.add_argument("--owt-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizers = {
        "tinystories_10k": Tokenizer.from_serialized(
            args.tinystories_tokenizer / "vocab.json",
            args.tinystories_tokenizer / "merges.json",
            ["<|endoftext|>"],
        ),
        "owt_32k": Tokenizer.from_serialized(
            args.owt_tokenizer / "vocab.json",
            args.owt_tokenizer / "merges.json",
            ["<|endoftext|>"],
        ),
    }
    corpora = {
        "tinystories": sample_documents(args.tinystories_text, args.documents, args.seed),
        "owt": sample_documents(args.owt_text, args.documents, args.seed + 1),
    }
    results = {
        corpus_name: {
            tokenizer_name: evaluate(tokenizer, documents, args.repeats)
            for tokenizer_name, tokenizer in tokenizers.items()
        }
        for corpus_name, documents in corpora.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
