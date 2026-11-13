"""Document texts and position mapping between two versions of a document.

A document is its paragraphs in order, each with blank-line breaks flattened to one space,
joined by one space. So a system that merges two paragraphs (joined by a space) or splits one
(a blank line) produces the same document text as one that leaves the structure alone, and
structure is scored separately (``metrics.structure``).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:  # the `text` extra is optional
    _lev = None

_BREAK = re.compile(r"[ \t]*\n[ \t]*\n\s*")


def flatten(text: str) -> str:
    """Blank-line breaks -> one space; edges stripped."""
    return _BREAK.sub(" ", text).strip()


@dataclass(frozen=True)
class DocText:
    text: str
    ranges: list[tuple[int, int]]  # [start, end) of each piece in text

    @property
    def joints(self) -> list[int]:
        """Index of the space joining piece i and i+1."""
        return [e for _, e in self.ranges[:-1]]


def join(pieces: list[str], *, skip_empty: bool = False) -> DocText:
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    pos = 0
    for raw in pieces:
        p = flatten(raw)
        if skip_empty and not p:
            continue
        if parts:
            pos += 1
        ranges.append((pos, pos + len(p)))
        parts.append(p)
        pos += len(p)
    return DocText(" ".join(parts), ranges)


def opcodes(a, b) -> list[tuple[str, int, int, int, int]]:
    """Levenshtein opcodes, equal blocks included (strings or token lists)."""
    if _lev is not None:
        return [
            (o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end) for o in _lev.opcodes(a, b)
        ]
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


class Aligner:
    """Maps a position in ``a`` to the corresponding position in ``b``.

    Exact inside equal blocks, proportional inside replaced ones. A position where ``b``
    has an insertion maps past it.
    """

    def __init__(self, a: str, b: str) -> None:
        self.len_a, self.len_b = len(a), len(b)
        self.ops = opcodes(a, b)
        self.starts = [o[1] for o in self.ops]

    def map(self, i: int) -> int:
        if not self.ops:
            return 0
        k = max(0, bisect_right(self.starts, i) - 1)
        tag, i1, i2, j1, j2 = self.ops[k]
        if i >= i2:
            return j2
        if tag == "equal":
            return j1 + (i - i1)
        return j1 + round((i - i1) * (j2 - j1) / (i2 - i1))


def touches(lo: int, hi: int, start: int, end: int) -> bool:
    """Edit region [lo, hi) overlaps span [start, end); empty ones count when they touch."""
    if lo == hi or start == end:
        return start <= hi and lo <= end
    return lo < end and hi > start
