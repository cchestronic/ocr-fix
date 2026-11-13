"""Alignment helpers for `ocrfix eval`: Levenshtein opcodes and region overlap."""

from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:  # the `text` extra is optional
    _lev = None


def opcodes(a, b) -> list[tuple[str, int, int, int, int]]:
    """Levenshtein opcodes, equal blocks included (strings or token lists)."""
    if _lev is not None:
        return [
            (o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end) for o in _lev.opcodes(a, b)
        ]
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def touches(lo: int, hi: int, start: int, end: int) -> bool:
    """Edit region [lo, hi) overlaps span [start, end); empty ones count when they touch."""
    if lo == hi or start == end:
        return start <= hi and lo <= end
    return lo < end and hi > start
