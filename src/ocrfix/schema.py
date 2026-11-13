"""Shared record schema for every JSONL file passed between stages.

Span offsets are Python ``str`` indices (code points) into the NFC-normalized ``text``.
Stage-specific fields ride along in ``extra`` and are flattened into the JSON object.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

SpanKind = Literal["prose", "math", "table"]
SPAN_KINDS: tuple[str, ...] = ("prose", "math", "table")

# Placeholders (see CLAUDE.md). Note ⟨⟩ are U+27E8/U+27E9; 〈〉 are U+3008/U+3009.
MATH_PH = "⟨M{}⟩"
TABLE_PH = "⟨T{}⟩"
PROTECTED_PH = "⟨P{}⟩"
UNCERTAIN = "〈?〉"

CORE_FIELDS = ("doc_id", "para_id", "order", "lang", "text", "spans", "source_path")


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


@dataclass
class Record:
    doc_id: str
    para_id: str
    order: int
    lang: str
    text: str
    spans: list[Span] = field(default_factory=list)
    source_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text != nfc(self.text):
            raise ValueError(f"{self.key}: text is not NFC (normalize before computing spans)")
        for s in self.spans:
            if s.end > len(self.text):
                raise ValueError(f"{self.key}: span {s} exceeds text length {len(self.text)}")

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.para_id)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "doc_id": self.doc_id,
            "para_id": self.para_id,
            "order": self.order,
            "lang": self.lang,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "source_path": self.source_path,
        }
        clash = d.keys() & self.extra.keys()
        if clash:
            raise ValueError(f"extra fields shadow core fields: {sorted(clash)}")
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        """Parse a JSON object. NFC-normalizes text; refuses if that would shift spans."""
        raw = d["text"]
        text = nfc(raw)
        spans = [Span.from_dict(s) for s in d.get("spans", [])]
        if text != raw and spans:
            raise ValueError(
                f"{d['doc_id']}/{d['para_id']}: non-NFC text with spans; offsets would be invalid"
            )
        return cls(
            doc_id=str(d["doc_id"]),
            para_id=str(d["para_id"]),
            order=int(d["order"]),
            lang=d["lang"],
            text=text,
            spans=spans,
            source_path=d.get("source_path", ""),
            extra={k: v for k, v in d.items() if k not in CORE_FIELDS},
        )
