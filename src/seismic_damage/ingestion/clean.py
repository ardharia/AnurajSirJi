"""Text cleaning for OCR/PDF extraction artifacts."""

from __future__ import annotations

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_FORM_FEED = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving paragraph breaks."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _FORM_FEED.sub(" ", normalized)
    normalized = _HYPHEN_BREAK.sub(r"\1\2", normalized)
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in normalized.split("\n")]
    joined = "\n".join(lines)
    joined = _MULTI_NL.sub("\n\n", joined)
    return joined.strip()
