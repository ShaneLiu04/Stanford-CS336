"""Deterministic document-quality filters and classification.

The rule-based filter follows the subset of Gopher filters specified in the
assignment handout.  The classifier intentionally keeps its features and
weights explicit so that decisions can be inspected and a trained fastText
model can later be supplied without changing the public entry point.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3.0
MAX_MEAN_WORD_LENGTH = 10.0
MAX_ELLIPSIS_LINE_RATIO = 0.30
MIN_ALPHABETIC_WORD_RATIO = 0.80

# A word must contain a Unicode letter or digit.  This excludes stand-alone
# punctuation while preserving numbers, contractions, and hyphenated words.
_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*$")
_CONCATENATED_WORD_RE = re.compile(r"^(.{2,})\1$", flags=re.IGNORECASE)

_BOILERPLATE_MARKERS = (
    "all rights reserved",
    "contact us",
    "cookie policy",
    "copyright",
    "forum index",
    "log in",
    "memberlist",
    "powered by",
    "privacy policy",
    "register",
    "search",
    "sign in",
    "sign up",
    "terms of service",
    "usergroups",
)


class FastTextQualityModel(Protocol):
    """The small part of the fastText model API used by this module."""

    def predict(
        self,
        text: str,
        k: int = 1,
    ) -> tuple[Sequence[str], Sequence[float]]: ...


@dataclass(frozen=True)
class QualityFeatures:
    """Human-readable signals used by the deterministic classifier."""

    word_count: int
    mean_words_per_line: float
    prose_line_ratio: float
    heading_line_ratio: float
    boilerplate_line_ratio: float
    concatenated_word_ratio: float


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def gopher_quality_filter(text: str) -> bool:
    """Return whether *text* passes the assignment's Gopher quality rules."""

    words = _words(text)
    word_count = len(words)
    if not MIN_WORDS <= word_count <= MAX_WORDS:
        return False

    mean_word_length = sum(map(len, words)) / word_count
    if not MIN_MEAN_WORD_LENGTH <= mean_word_length <= MAX_MEAN_WORD_LENGTH:
        return False

    lines = text.splitlines() or [text]
    ellipsis_lines = sum(line.rstrip().endswith("...") for line in lines)
    if ellipsis_lines / len(lines) > MAX_ELLIPSIS_LINE_RATIO:
        return False

    alphabetic_words = sum(any(character.isalpha() for character in word) for word in words)
    return alphabetic_words / word_count >= MIN_ALPHABETIC_WORD_RATIO


def quality_features(text: str) -> QualityFeatures:
    """Extract the deterministic features behind :func:`classify_quality`."""

    words = _words(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text.strip()]

    line_word_counts = [len(_words(line)) for line in lines]
    line_count = len(lines)
    prose_lines = sum(
        word_count >= 12 and bool(_SENTENCE_END_RE.search(line))
        for line, word_count in zip(lines, line_word_counts, strict=True)
    )
    headings = sum(
        0 < word_count <= 12 and not _SENTENCE_END_RE.search(line)
        for line, word_count in zip(lines, line_word_counts, strict=True)
    )
    boilerplate_lines = sum(any(marker in line.casefold() for marker in _BOILERPLATE_MARKERS) for line in lines)
    concatenated_words = sum(bool(_CONCATENATED_WORD_RE.fullmatch(word)) for word in words)

    return QualityFeatures(
        word_count=len(words),
        mean_words_per_line=sum(line_word_counts) / line_count,
        prose_line_ratio=prose_lines / line_count,
        heading_line_ratio=headings / line_count,
        boilerplate_line_ratio=boilerplate_lines / line_count,
        concatenated_word_ratio=concatenated_words / len(words) if words else 0.0,
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _heuristic_quality_logit(features: QualityFeatures) -> float:
    """Combine documented features; positive values indicate wiki-like text."""

    length_signal = _clamp(math.log10(max(features.word_count, 1) / 75) / 2)
    line_depth_signal = _clamp((features.mean_words_per_line - 8) / 32)
    return (
        -1.0
        + 1.4 * length_signal
        + 1.0 * line_depth_signal
        + 0.8 * features.prose_line_ratio
        - 1.8 * features.boilerplate_line_ratio
        - 0.8 * features.heading_line_ratio
        - 2.0 * features.concatenated_word_ratio
    )


def _normalize_model_label(label: str) -> str:
    normalized = label.removeprefix("__label__").casefold()
    if normalized in {"wiki", "high", "high-quality", "high_quality"}:
        return "wiki"
    if normalized in {"cc", "low", "low-quality", "low_quality"}:
        return "cc"
    raise ValueError(f"Unsupported quality-model label: {label!r}")


def classify_quality(
    text: str,
    model: FastTextQualityModel | None = None,
) -> tuple[str, float]:
    """Classify text as ``"cc"`` or ``"wiki"`` and return label confidence.

    With no model, an interpretable fixed-weight classifier is used.  A
    fastText-compatible model can be injected later; its labels may use the
    conventional ``__label__`` prefix and high/low-quality aliases.
    """

    if model is not None:
        labels, probabilities = model.predict(text.replace("\n", " "), k=1)
        if not labels or not probabilities:
            raise ValueError("Quality model returned no predictions")
        return _normalize_model_label(str(labels[0])), float(probabilities[0])

    logit = _heuristic_quality_logit(quality_features(text))
    label = "wiki" if logit >= 0 else "cc"
    confidence = 1.0 / (1.0 + math.exp(-abs(logit)))
    return label, float(confidence)
