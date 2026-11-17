"""Shared types and constants: text spans, placeholder formats, NFC.

Span offsets are Python ``str`` indices (code points) into the NFC-normalized text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

SpanKind = Literal["prose", "math", "table"]
SPAN_KINDS: tuple[str, ...] = ("prose", "math", "table")

# Placeholders (see CLAUDE.md). Note ⟨⟩ are U+27E8/U+27E9.
MATH_PH = "⟨M{}⟩"
TABLE_PH = "⟨T{}⟩"
PROTECTED_PH = "⟨P{}⟩"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: SpanKind

    def __post_init__(self) -> None:
        if self.kind not in SPAN_KINDS:
            raise ValueError(f"span kind must be one of {SPAN_KINDS}, got {self.kind!r}")
        if not 0 <= self.start <= self.end:
            raise ValueError(f"bad span bounds [{self.start}, {self.end})")

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        return cls(start=int(d["start"]), end=int(d["end"]), kind=d["kind"])
