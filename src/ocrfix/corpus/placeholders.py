"""Swap math/table spans for numbered placeholders (⟨M1⟩, ⟨T1⟩, ...) and back.

Numbering is 1-based, per kind, in order of appearance. ``mapping`` is ordered by position,
so ``restore`` can demand that the placeholders come back exactly once each, in order.
"""

from __future__ import annotations

import re
from collections import Counter

from ocrfix.schema import MATH_PH, TABLE_PH, Span

PLACEHOLDER = re.compile(r"⟨[MTP]\d+⟩")
_FORMAT = {"math": MATH_PH, "table": TABLE_PH}


class PlaceholderError(ValueError):
    pass


def substitute(text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
    if m := PLACEHOLDER.search(text):
        raise PlaceholderError(f"text already contains placeholder {m.group()!r}")
    parts: list[str] = []
    mapping: dict[str, str] = {}
    counters = Counter[str]()
    pos = 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.kind not in _FORMAT:
            continue
        if s.start < pos:
            raise PlaceholderError(f"overlapping span {s}")
        counters[s.kind] += 1
        ph = _FORMAT[s.kind].format(counters[s.kind])
        parts += [text[pos : s.start], ph]
        mapping[ph] = text[s.start : s.end]
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts), mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    found = PLACEHOLDER.findall(text)
    expected = list(mapping)
    if found != expected:
        raise PlaceholderError(_diagnose(found, expected))
    return PLACEHOLDER.sub(lambda m: mapping[m.group()], text)


def _diagnose(found: list[str], expected: list[str]) -> str:
    counts = Counter(found)
    if missing := [p for p in expected if p not in counts]:
        return f"missing placeholder(s): {' '.join(missing)}"
    if dup := [p for p, n in counts.items() if n > 1]:
        return f"duplicated placeholder(s): {' '.join(dup)}"
    if unknown := [p for p in counts if p not in expected]:
        return f"unknown placeholder(s): {' '.join(unknown)}"
    return f"placeholders out of order: got {' '.join(found)}, expected {' '.join(expected)}"
