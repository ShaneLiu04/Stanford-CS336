"""Exact line and MinHash/LSH document deduplication."""

from __future__ import annotations

import hashlib
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mmh3
from xopen import xopen

PathLike = str | os.PathLike[str]
_MAX_HASH = (1 << 64) - 1


def _read_text(path: Path) -> str:
    with xopen(path, mode="rt", encoding="utf-8") as stream:
        return stream.read()


def _write_text(path: Path, text: str) -> None:
    with xopen(path, mode="wt", encoding="utf-8") as stream:
        stream.write(text)


def _line_key(line: str) -> bytes:
    """Return a fixed-size key while treating line endings as separators."""
    content = line.rstrip("\r\n")
    return hashlib.blake2b(content.encode("utf-8"), digest_size=16).digest()


def exact_line_deduplication(input_files: list[PathLike], output_directory: PathLike) -> None:
    """Remove every line that occurs more than once across all input files.

    One output is created for every input, including when all of its lines are
    removed. Output files retain their input basenames.
    """
    paths = [Path(path) for path in input_files]
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies: Counter[bytes] = Counter()
    for path in paths:
        frequencies.update(_line_key(line) for line in _read_text(path).splitlines(keepends=True))

    for path in paths:
        lines = _read_text(path).splitlines(keepends=True)
        retained = "".join(line for line in lines if frequencies[_line_key(line)] == 1)
        _write_text(output_dir / path.name, retained)


def _normalize(text: str) -> str:
    """Apply the RefinedWeb-style normalization used for fuzzy matching."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in without_accents
    )
    return " ".join(without_punctuation.split())


def _word_ngrams(text: str, n: int) -> set[str]:
    words = _normalize(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {"\x1f".join(words)}
    return {"\x1f".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> tuple[int, ...]:
    if not shingles:
        return (_MAX_HASH,) * num_hashes

    encoded = [shingle.encode("utf-8") for shingle in shingles]
    return tuple(
        min(mmh3.hash64(shingle, seed=seed, signed=False)[0] for shingle in encoded) for seed in range(num_hashes)
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Always retain the earliest document, independent of candidate order.
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def minhash_deduplication(
    input_files: list[PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: PathLike,
) -> None:
    """Deduplicate documents using MinHash LSH followed by true Jaccard."""
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")
    if num_bands <= 0 or num_hashes % num_bands:
        raise ValueError("num_bands must be positive and evenly divide num_hashes")
    if ngrams <= 0:
        raise ValueError("ngrams must be positive")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")

    # Canonical path order makes the retained member deterministic even when
    # callers provide files in a different order.
    paths = sorted(
        (Path(path) for path in input_files),
        key=lambda path: os.path.normcase(str(path.resolve())),
    )
    texts = [_read_text(path) for path in paths]
    shingle_sets = [_word_ngrams(text, ngrams) for text in texts]
    signatures = [_minhash_signature(shingles, num_hashes) for shingles in shingle_sets]

    rows_per_band = num_hashes // num_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for document_id, signature in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            for other_id in buckets[key]:
                candidates.add((other_id, document_id))
            buckets[key].append(document_id)

    clusters = _DisjointSet(len(paths))
    for left, right in sorted(candidates):
        if _jaccard(shingle_sets[left], shingle_sets[right]) >= jaccard_threshold:
            clusters.union(left, right)

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, (path, text) in enumerate(zip(paths, texts, strict=True)):
        if clusters.find(document_id) == document_id:
            _write_text(output_dir / path.name, text)
