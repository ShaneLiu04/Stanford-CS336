from __future__ import annotations

import os
from typing import Any

from cs336_data.dedup import exact_line_deduplication, minhash_deduplication
from cs336_data.extract_lang import extract_text_from_html_bytes, identify_language
from cs336_data.quality import classify_quality, gopher_quality_filter
from cs336_data.safety import (
    classify_nsfw,
    classify_toxic_speech,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )
