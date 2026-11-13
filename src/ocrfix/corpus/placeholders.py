"""Placeholders back to their text (masking is chunks/split.py ``render``).

Numbering is 1-based, per kind, in order of appearance. ``mapping`` is ordered by position,
so ``restore`` can demand that the placeholders come back exactly once each, in order.
"""

from __future__ import annotations

import re
from collections import Counter

PLACEHOLDER = re.compile(r"⟨[MTP]\d+⟩")


class PlaceholderError(ValueError):
    pass


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
