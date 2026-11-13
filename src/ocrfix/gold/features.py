"""Paragraph flags that get fixed sampling quotas (see gold.sample.features in the config)."""

from __future__ import annotations

import re

from ocrfix.corpus.lang import script_counts
from ocrfix.schema import Record

FEATURES = (
    "inline_math",
    "display_math",
    "table",
    "garbled_hangul",
    "mixed_koen",
    "hyphenated",
)

# Inside a math span these mean "real math"; their absence (plus shortness) means the span is
# likely Hangul that MinerU misread as a formula, e.g. `방법$\mathrm{Ol}$` for 방법이.
_OPERATOR_CHARS = set("=+-*/<>^_|!~&,;:") | set("≤≥±×÷·∑∏∫√∞≈≠→←∈∉⊂⊆∪∩∀∃∂∇")
_FORMAT_CMDS = {
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "text", "textrm", "textbf", "textit",
    "rm", "bf", "it", "operatorname", "boldsymbol", "mathcal", "mathscr", "mathfrak",
}  # fmt: skip
_CMD = re.compile(r"\\([A-Za-z]+)")
_HYPHEN_BREAK = re.compile(r"[A-Za-z]-(?:[ \t]*\n[ \t]*| )[a-z]")


def math_inner(raw: str) -> str:
    """Strip one pair of math delimiters; environments are returned unchanged."""
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if raw.startswith(left) and raw.endswith(right) and len(raw) >= len(left) + len(right):
            return raw[len(left) : len(raw) - len(right)]
    return raw


def is_display(raw: str) -> bool:
    return raw.startswith(("$$", "\\[", "\\begin"))


def _is_hangul(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def looks_like_garbled_hangul(raw: str, max_len: int = 8) -> bool:
    if raw.startswith("\\begin"):
        return False
    inner = math_inner(raw)
    if any(c in _OPERATOR_CHARS for c in inner):
        return False
    if any(cmd not in _FORMAT_CMDS for cmd in _CMD.findall(inner)):
        return False
    content = re.sub(r"[\s{}]", "", _CMD.sub("", inner))
    return 0 < len(content) <= max_len and any(c.isalpha() for c in content)


def _hangul_next_to(text: str, start: int, end: int) -> bool:
    """A Hangul char right before or after [start, end), allowing one space in between."""
    for i in (start - 1, start - 2):
        if i >= 0 and (i == start - 1 or text[start - 1] == " ") and _is_hangul(text[i]):
            return True
    for i in (end, end + 1):
        if i < len(text) and (i == end or text[end] == " ") and _is_hangul(text[i]):
            return True
    return False


def paragraph_features(
    rec: Record,
    *,
    garbled_max_len: int = 8,
    mixed_min: float = 0.1,
    hangul_weight: float = 2.0,
) -> tuple[str, ...]:
    text = rec.text
    found: set[str] = set()
    prose = []
    for s in rec.spans:
        raw = text[s.start : s.end]
        if s.kind == "table":
            found.add("table")
        elif s.kind == "math":
            found.add("display_math" if is_display(raw) else "inline_math")
            if looks_like_garbled_hangul(raw, garbled_max_len) and _hangul_next_to(
                text, s.start, s.end
            ):
                found.add("garbled_hangul")
        else:
            prose.append(raw)
    prose_text = "\n".join(prose)
    counts = script_counts(prose_text, hangul_weight)
    total = sum(counts.values())
    if total and min(counts["ko"], counts["en"]) / total >= mixed_min:
        found.add("mixed_koen")
    if _HYPHEN_BREAK.search(prose_text):
        found.add("hyphenated")
    return tuple(f for f in FEATURES if f in found)
