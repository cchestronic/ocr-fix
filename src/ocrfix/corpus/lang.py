"""Chunk language from script ratios over prose letters: ko | en | mixed | cjk | other.

``cjk``: Chinese / Japanese (Han and kana) make up at least ``cjk_min`` of the letters. Below
that, Han and kana don't count (Hanja in a Korean text, a Chinese name in an English one), so
they never push a Korean chunk out of ``ko``."""

from __future__ import annotations

import re
import unicodedata

from ocrfix.schema import Span

_TAG = re.compile(r"<[^>]+>")


def is_cjk(ch: str) -> bool:
    """Han (CJK unified ideographs, extension A, compatibility) or kana."""
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
            or 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF)  # fmt: skip


def _script(ch: str) -> str | None:
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        return "hangul_syllable"
    if 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F or 0xA960 <= o <= 0xA97F:
        return "hangul"
    if 0xD7B0 <= o <= 0xD7FF:
        return "hangul"
    if is_cjk(ch):
        return "cjk"
    if not ch.isalpha():
        return None
    if ch.isascii() or unicodedata.name(ch, "").startswith("LATIN"):
        return "latin"
    return "other"


def script_counts(text: str, hangul_weight: float = 2.0) -> dict[str, float]:
    """Weighted letter counts. A Hangul syllable (or a Han character) carries ~2 Latin letters'
    worth of text."""
    counts = {"ko": 0.0, "en": 0.0, "cjk": 0.0, "other": 0.0}
    for ch in text:
        s = _script(ch)
        if s == "hangul_syllable":
            counts["ko"] += hangul_weight
        elif s == "hangul":
            counts["ko"] += 1
        elif s == "cjk":
            counts["cjk"] += hangul_weight
        elif s == "latin":
            counts["en"] += 1
        elif s == "other":
            counts["other"] += 1
    return counts


def detect_lang(
    text: str,
    spans: list[Span],
    *,
    dominant: float = 0.8,
    mixed_min: float = 0.1,
    hangul_weight: float = 2.0,
    cjk_min: float = 0.5,
) -> str:
    """Count prose letters (math never counts); fall back to table text with tags removed."""
    prose = "".join(text[s.start : s.end] for s in spans if s.kind == "prose")
    counts = script_counts(_TAG.sub(" ", prose), hangul_weight)
    if sum(counts.values()) == 0:
        table = " ".join(text[s.start : s.end] for s in spans if s.kind == "table")
        counts = script_counts(_TAG.sub(" ", table), hangul_weight)
    total = sum(counts.values())
    if total == 0:
        return "other"
    if counts["cjk"] / total >= cjk_min:
        return "cjk"
    total -= counts["cjk"]  # Hanja / kana inside Korean or English text don't count
    if total == 0:
        return "other"
    ko, en = counts["ko"] / total, counts["en"] / total
    if ko >= dominant:
        return "ko"
    if en >= dominant:
        return "en"
    if ko >= mixed_min and en >= mixed_min and ko + en >= dominant:
        return "mixed"
    return "other"
