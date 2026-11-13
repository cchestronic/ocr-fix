"""What gets masked: every math span (inline and display) -> ⟨M#⟩, every table -> ⟨T#⟩, and
⟨P#⟩ for the ``chunk.protect`` kinds in prose (URLs, inline code, image links, emails, DOIs,
citation brackets). The chunker masks with it once (chunks/split.py); every later stage reads
that masked text.
"""

from __future__ import annotations

import bisect
import re

from ocrfix.schema import MATH_PH, PROTECTED_PH, TABLE_PH, Span

# Order matters: earlier patterns win where they overlap.
PROTECT_PATTERNS: dict[str, re.Pattern] = {
    "images": re.compile(r"!\[[^\]\n]*\]\([^)\s]+\)"),
    "code": re.compile(r"`[^`\n]+`"),
    "urls": re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"]"),
    "emails": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "dois": re.compile(r"\b10\.\d{4,9}/[^\s]+[^\s.,;:]"),
    "citations": re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]"),
}
_FORMAT = {"math": MATH_PH, "table": TABLE_PH, "protected": PROTECTED_PH}


def protected_spans(text: str, spans: list[Span], kinds: list[str]) -> list[tuple[int, int, str]]:
    """(start, end, kind) of math, tables and protected prose runs, non-overlapping."""
    out = [(s.start, s.end, s.kind) for s in spans if s.kind != "prose"]
    taken = sorted((a, b) for a, b, _ in out)  # non-overlapping, so sorted by both ends
    for s in spans:
        if s.kind != "prose":
            continue
        for name in kinds:
            for m in PROTECT_PATTERNS[name].finditer(text, s.start, s.end):
                a, b = m.span()
                i = bisect.bisect_left(taken, (a, b))
                free = (i == 0 or taken[i - 1][1] <= a) and (i == len(taken) or b <= taken[i][0])
                if free:
                    out.append((a, b, "protected"))
                    taken.insert(i, (a, b))
    return sorted(out)
