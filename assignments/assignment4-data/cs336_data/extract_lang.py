from __future__ import annotations

import threading
import unicodedata
from pathlib import Path
from typing import Any

from cs336_data.common import get_shared_assets_path

_MODEL_LOCK = threading.Lock()
_MODEL: Any | None = None
_MODEL_LOAD_ATTEMPTED = False


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Decode raw HTML and extract its visible text with Resiliparse."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    if not isinstance(html_bytes, bytes):
        raise TypeError("html_bytes must be bytes")

    try:
        try:
            html = html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            encoding = detect_encoding(html_bytes) or "utf-8"
            html = html_bytes.decode(encoding, errors="replace")
        return extract_plain_text(html)
    except (LookupError, UnicodeError, ValueError):
        return None


def _language_model_path() -> Path | None:
    shared_assets = get_shared_assets_path()
    for path in (
        shared_assets / "classifiers" / "lid.176.bin",
        shared_assets / "lid.176.bin",
    ):
        if path.is_file():
            return path
    return None


def _get_language_model() -> Any | None:
    global _MODEL, _MODEL_LOAD_ATTEMPTED

    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL

    with _MODEL_LOCK:
        if not _MODEL_LOAD_ATTEMPTED:
            model_path = _language_model_path()
            if model_path is not None:
                try:
                    import fasttext

                    _MODEL = fasttext.load_model(str(model_path))
                except (ImportError, OSError, RuntimeError, ValueError):
                    _MODEL = None
            _MODEL_LOAD_ATTEMPTED = True
    return _MODEL


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _fallback_language(text: str) -> tuple[str, float]:
    counts = {"en": 0, "zh": 0, "其他": 0}
    for character in text:
        if _is_han(character):
            counts["zh"] += 1
        elif character.isascii() and character.isalpha():
            counts["en"] += 1
        elif unicodedata.category(character).startswith("L"):
            counts["其他"] += 1

    total = sum(counts.values())
    if total == 0:
        return "其他", 0.0

    language = max(("zh", "en", "其他"), key=counts.__getitem__)
    return language, float(counts[language] / total)


def identify_language(text: str) -> tuple[str, float]:
    """Return the main language code and confidence for ``text``."""
    if not isinstance(text, str):
        raise TypeError("text must be str")

    model = _get_language_model()
    if model is None or not text.strip():
        return _fallback_language(text)

    try:
        labels, scores = model.predict(text.replace("\n", " "), k=1)
        language = str(labels[0]).removeprefix("__label__")
        return language, float(scores[0])
    except (IndexError, RuntimeError, TypeError, ValueError):
        return _fallback_language(text)
