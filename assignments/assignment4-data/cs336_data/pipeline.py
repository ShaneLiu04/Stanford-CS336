"""Composable document filtering and GPT-2 tokenization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tiktoken

from .quality import classify_quality, gopher_quality_filter
from .safety import classify_nsfw, classify_toxic_speech, mask_emails, mask_ips, mask_phone_numbers


@dataclass
class PipelineStats:
    documents_in: int = 0
    documents_out: int = 0
    characters_in: int = 0
    characters_out: int = 0
    emails_masked: int = 0
    phones_masked: int = 0
    ips_masked: int = 0
    rejected_gopher: int = 0
    rejected_quality: int = 0
    rejected_nsfw: int = 0
    rejected_toxic: int = 0


def filter_documents(
    documents: list[str],
    *,
    use_gopher: bool = True,
    quality_threshold: float = 0.5,
    use_safety: bool = True,
) -> tuple[list[str], PipelineStats]:
    stats = PipelineStats()
    output = []
    for document in documents:
        stats.documents_in += 1
        stats.characters_in += len(document)
        text, count = mask_emails(document)
        stats.emails_masked += count
        text, count = mask_phone_numbers(text)
        stats.phones_masked += count
        text, count = mask_ips(text)
        stats.ips_masked += count
        if use_gopher and not gopher_quality_filter(text):
            stats.rejected_gopher += 1
            continue
        quality, score = classify_quality(text)
        if quality != "wiki" and score >= quality_threshold:
            stats.rejected_quality += 1
            continue
        if use_safety:
            if classify_nsfw(text)[0] == "nsfw":
                stats.rejected_nsfw += 1
                continue
            if classify_toxic_speech(text)[0] == "toxic":
                stats.rejected_toxic += 1
                continue
        output.append(text)
        stats.documents_out += 1
        stats.characters_out += len(text)
    return output, stats


def tokenize_documents(documents: list[str], destination: Path) -> dict[str, int]:
    tokenizer = tiktoken.get_encoding("gpt2")
    eos = tokenizer.eot_token
    token_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for document in documents:
            ids = tokenizer.encode_ordinary(document) + [eos]
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            token_count += len(ids)
    return {"documents": len(documents), "tokens": token_count, "bytes": destination.stat().st_size}


def write_stats(stats: PipelineStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
