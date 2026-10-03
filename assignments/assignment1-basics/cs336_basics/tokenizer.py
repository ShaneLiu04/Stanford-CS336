from __future__ import annotations

import json
import os
import heapq
import multiprocessing as mp
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex


GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class _ReversePair:
    pair: tuple[bytes, bytes]

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair


def _chunk_boundaries(path: str | os.PathLike, workers: int, delimiter: bytes) -> list[int]:
    size = os.path.getsize(path)
    boundaries = [0]
    with open(path, "rb") as source:
        for index in range(1, workers):
            source.seek(size * index // workers)
            position = source.tell()
            while block := source.read(1024 * 1024):
                offset = block.find(delimiter)
                if offset >= 0:
                    boundaries.append(position + offset + len(delimiter))
                    break
                position += len(block)
    boundaries.append(size)
    return sorted(set(boundaries))


def _count_pretokens(task: tuple[str, int, int, bytes]) -> Counter[tuple[bytes, ...]]:
    path, start, end, delimiter = task
    with open(path, "rb") as source:
        source.seek(start)
        text = source.read(end - start).decode("utf-8")
    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece in text.split(delimiter.decode("utf-8")):
        for match in regex.finditer(GPT2_PATTERN, piece):
            counts[tuple(bytes([value]) for value in match.group().encode("utf-8"))] += 1
    return counts


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, map(chr, code_points)))


def _special_pattern(tokens: Iterable[str]) -> Any:
    ordered = sorted(set(tokens), key=lambda item: (-len(item), item))
    if not ordered:
        return None
    return regex.compile("(" + "|".join(regex.escape(token) for token in ordered) + ")")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **_kwargs: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a deterministic byte-level BPE vocabulary."""
    if vocab_size < 256 + len(set(special_tokens)):
        raise ValueError("vocab_size is too small for byte and special tokens")

    special_tokens = list(dict.fromkeys(special_tokens))
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    counts: Counter[tuple[bytes, ...]] = Counter()
    splitter = _special_pattern(special_tokens)
    streaming = bool(_kwargs.get("streaming", False))
    worker_value = _kwargs.get("pretokenize_workers", 1)
    if not isinstance(worker_value, int):
        raise TypeError("pretokenize_workers must be an integer")
    pretokenize_workers = worker_value
    chunk_value = _kwargs.get("pretokenize_chunks", pretokenize_workers * 16)
    if not isinstance(chunk_value, int):
        raise TypeError("pretokenize_chunks must be an integer")
    pretokenize_chunks = max(pretokenize_workers, chunk_value)
    if streaming and len(special_tokens) == 1 and pretokenize_workers > 1:
        delimiter = special_tokens[0].encode("utf-8")
        boundaries = _chunk_boundaries(input_path, pretokenize_chunks, delimiter)
        tasks = [
            (os.fspath(input_path), start, end, delimiter)
            for start, end in zip(boundaries, boundaries[1:])
        ]
        with mp.get_context("spawn").Pool(pretokenize_workers) as pool:
            for partial_counts in pool.imap_unordered(_count_pretokens, tasks):
                counts.update(partial_counts)
        pieces = ()
    elif streaming and len(special_tokens) == 1:
        delimiter = special_tokens[0].encode("utf-8")

        def iter_pieces():
            buffer = b""
            with open(input_path, "rb") as source:
                while chunk := source.read(64 * 1024 * 1024):
                    parts = (buffer + chunk).split(delimiter)
                    for part in parts[:-1]:
                        yield part.decode("utf-8")
                    buffer = parts[-1]
            if buffer:
                yield buffer.decode("utf-8")

        pieces = iter_pieces()
    else:
        text = Path(input_path).read_text(encoding="utf-8")
        pieces = splitter.split(text) if splitter else [text]
    for piece in pieces:
        if not piece or piece in special_tokens:
            continue
        for match in regex.finditer(GPT2_PATTERN, piece):
            counts[tuple(bytes([value]) for value in match.group().encode("utf-8"))] += 1

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word, frequency in counts.items():
        occurrences = Counter(zip(word, word[1:]))
        for pair, occurrence_count in occurrences.items():
            pair_counts[pair] += frequency * occurrence_count
            pair_to_words.setdefault(pair, set()).add(word)
    pair_heap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        while pair_heap:
            negative_count, _, best_pair = heapq.heappop(pair_heap)
            best_count = -negative_count
            if best_count > 0 and pair_counts[best_pair] == best_count:
                break
        else:
            break

        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged

        affected_words = list(pair_to_words.get(best_pair, ()))
        replacements: Counter[tuple[bytes, ...]] = Counter()
        first, second = best_pair
        for word in affected_words:
            frequency = counts.pop(word)
            old_occurrences = Counter(zip(word, word[1:]))
            for pair, occurrence_count in old_occurrences.items():
                pair_counts[pair] -= frequency * occurrence_count
                pair_to_words[pair].discard(word)
                if pair_counts[pair] > 0:
                    heapq.heappush(pair_heap, (-pair_counts[pair], _ReversePair(pair), pair))
            result: list[bytes] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and word[index] == first and word[index + 1] == second:
                    result.append(merged)
                    index += 2
                else:
                    result.append(word[index])
                    index += 1
            replacements[tuple(result)] += frequency

        for word, added_frequency in replacements.items():
            counts[word] += added_frequency
            new_occurrences = Counter(zip(word, word[1:]))
            for pair, occurrence_count in new_occurrences.items():
                pair_counts[pair] += added_frequency * occurrence_count
                pair_to_words.setdefault(pair, set()).add(word)
                heapq.heappush(pair_heap, (-pair_counts[pair], _ReversePair(pair), pair))
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.bytes_to_id = {token: index for index, token in self.vocab.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id
            self.special_to_id[token] = self.bytes_to_id[token_bytes]
        self._special_splitter = _special_pattern(self.special_tokens)

    @classmethod
    def from_serialized(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        raw_vocab = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        raw_merges = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        vocab = {int(index): bytes.fromhex(value) for index, value in raw_vocab.items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in raw_merges]
        return cls(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as source:
            serialized_vocab = json.load(source)
        byte_decoder = {character: byte for byte, character in _gpt2_bytes_to_unicode().items()}
        vocab = {
            int(index): bytes(byte_decoder[character] for character in token)
            for token, index in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                left, right = line.split()
                merges.append((
                    bytes(byte_decoder[character] for character in left),
                    bytes(byte_decoder[character] for character in right),
                ))
        return cls(vocab, merges, special_tokens)

    def _encode_bytes(self, data: bytes) -> list[int]:
        pieces = [bytes([value]) for value in data]
        while len(pieces) > 1:
            candidate: tuple[bytes, bytes] | None = None
            candidate_rank = len(self.merge_ranks) + 1
            for pair in zip(pieces, pieces[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < candidate_rank:
                    candidate, candidate_rank = pair, rank
            if candidate is None:
                break
            result: list[bytes] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == candidate:
                    result.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    result.append(pieces[index])
                    index += 1
            pieces = result
        return [self.bytes_to_id[piece] for piece in pieces]

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        pieces = self._special_splitter.split(text) if self._special_splitter else [text]
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_to_id:
                output.append(self.special_to_id[piece])
                continue
            for match in regex.finditer(GPT2_PATTERN, piece):
                output.extend(self._encode_bytes(match.group().encode("utf-8")))
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

