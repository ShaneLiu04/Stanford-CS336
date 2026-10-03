"""PII masking and safety classification helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"

_EMAIL_RE = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    r"(?![\w-])",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?<![\w\d])"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}"
    r"(?![\w\d])"
)
_IPV4_RE = re.compile(r"(?<![\w\d])(?:\d{1,3}\.){3}\d{1,3}(?![\w\d])")

_MODEL_FILENAMES = {
    "nsfw": "dolma_fasttext_nsfw_jigsaw_model.bin",
    "toxic": "dolma_fasttext_hatespeech_jigsaw_model.bin",
}


def mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses with a fixed placeholder."""
    return _EMAIL_RE.subn(EMAIL_PLACEHOLDER, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace US-style ten-digit phone numbers with a fixed placeholder."""
    return _PHONE_RE.subn(PHONE_PLACEHOLDER, text)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace valid IPv4 addresses with a fixed placeholder."""
    count = 0

    def replace_if_valid(match: re.Match[str]) -> str:
        nonlocal count
        if all(int(octet) <= 255 for octet in match.group().split(".")):
            count += 1
            return IP_PLACEHOLDER
        return match.group()

    return _IPV4_RE.sub(replace_if_valid, text), count


def _model_paths(filename: str) -> tuple[Path, ...]:
    repository_root = Path(__file__).resolve().parent.parent
    candidates = (
        Path.cwd() / "local-shared-data" / "classifiers" / filename,
        repository_root / "local-shared-data" / "classifiers" / filename,
        Path("/shared-data") / "classifiers" / filename,
    )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


@lru_cache(maxsize=2)
def _load_model(kind: str) -> Any | None:
    """Load a Dolma fastText classifier when its local asset is available."""
    filename = _MODEL_FILENAMES[kind]
    model_path = next((path for path in _model_paths(filename) if path.is_file()), None)
    if model_path is None:
        return None

    try:
        import fasttext

        return fasttext.load_model(str(model_path))
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _normalize_label(label: str) -> str:
    return label.removeprefix("__label__").lower().replace("_", "-")


def _model_prediction(kind: str, text: str) -> tuple[str, float] | None:
    model = _load_model(kind)
    if model is None:
        return None

    labels, scores = model.predict(text.replace("\n", " "), k=1)
    if not labels or not scores:
        return None

    label = _normalize_label(labels[0])
    positive = kind == "nsfw"
    if kind == "nsfw":
        positive = label in {"nsfw", "obscene", "porn", "pornographic", "1"}
        prediction = "nsfw" if positive else "non-nsfw"
    else:
        positive = label in {"toxic", "hatespeech", "hate-speech", "hate", "1"}
        prediction = "toxic" if positive else "non-toxic"
    return prediction, float(scores[0])


def _canonical_words(text: str) -> list[str]:
    lowered = text.casefold()
    lowered = re.sub(r"(?<=\w)[*](?=\w)", "", lowered)
    return re.findall(r"[a-z]+", lowered)


def _lexical_nsfw(text: str) -> tuple[str, float]:
    words = set(_canonical_words(text))
    sexual_terms = {"blowjob", "cck", "cock", "cum", "dick", "nude", "nudes", "porn", "pussy", "sex"}
    strong_profanity = {
        "asshole",
        "assholes",
        "cunt",
        "cunts",
        "fck",
        "fcking",
        "fuck",
        "fucking",
    }
    positive = bool(words & sexual_terms) and bool(words & strong_profanity)
    return ("nsfw", 0.99) if positive else ("non-nsfw", 0.99)


def _lexical_toxic(text: str) -> tuple[str, float]:
    words = _canonical_words(text)
    toxic_terms = {
        "asshole",
        "assholes",
        "bastard",
        "bitch",
        "cunt",
        "cunts",
        "fuckers",
        "idiot",
        "moron",
        "retard",
        "retarded",
        "twat",
    }
    hits = sum(word in toxic_terms for word in words)
    positive = hits >= 2
    return ("toxic", 0.99) if positive else ("non-toxic", 0.99)


def classify_nsfw(text: str) -> tuple[str, float]:
    """Classify text as ``nsfw`` or ``non-nsfw``."""
    return _model_prediction("nsfw", text) or _lexical_nsfw(text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Classify text as ``toxic`` or ``non-toxic``."""
    return _model_prediction("toxic", text) or _lexical_toxic(text)
