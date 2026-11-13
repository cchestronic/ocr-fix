"""Model output -> one corrected paragraph per marker, placeholders restored.

Per paragraph ``error`` is one of: ``marker_missing``, ``marker_missing_next`` (the next
paragraph's marker is lost, so this body probably swallowed it), ``marker_duplicate``, ``empty``,
``placeholders: <detail>`` (lost, duplicated, renumbered or reordered ⟨M#⟩ ⟨T#⟩ ⟨P#⟩),
or the request's own error (HTTP failure, truncation). A paragraph with an error keeps its
input text downstream. A ``<think>`` block, if any, is dropped first. A one-paragraph chunk
whose answer has no marker at all is accepted as that paragraph (``recovered``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ocrfix.corpus.placeholders import PlaceholderError, restore
from ocrfix.llm.prompt import Chunk
from ocrfix.schema import nfc

MARKER_LINE = re.compile(r"^[ \t]*<<<(\d+)>>>[ \t]*$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class ParaResult:
    doc_id: str
    para_id: str
    chunk_id: str
    ok: bool
    text: str | None = None  # corrected, placeholders restored
    output: str | None = None  # what the model wrote for this paragraph (masked)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def split_markers(text: str) -> tuple[dict[int, list[str]], str, list[int]]:
    """{marker number: [bodies]}, text before the first marker, marker order seen."""
    found: dict[int, list[str]] = {}
    order: list[int] = []
    ms = list(MARKER_LINE.finditer(text))
    if not ms:
        return found, text, order
    pre = text[: ms[0].start()]
    for m, nxt in zip(ms, [*ms[1:], None], strict=True):
        body = text[m.end() : nxt.start() if nxt else len(text)].strip("\n").strip()
        n = int(m.group(1))
        found.setdefault(n, []).append(body)
        order.append(n)
    return found, pre, order


def parse_chunk(
    chunk: Chunk, output: str | None, request_error: str | None = None
) -> list[ParaResult]:
    def fail(i: int, err: str, out: str | None = None) -> ParaResult:
        p = chunk.paras[i]
        return ParaResult(p.doc_id, p.para_id, chunk.chunk_id, False, None, out, err)

    if request_error or output is None:
        return [fail(i, request_error or "no response") for i in range(len(chunk.paras))]
    text = _THINK.sub("", output)
    found, pre, order = split_markers(text)
    notes = []
    if pre.strip():
        if not found and len(chunk.paras) == 1:
            found, notes = {1: [pre.strip()]}, ["recovered: no marker"]
        else:
            notes.append("text before first marker")
    if extra := sorted(set(found) - set(range(1, len(chunk.paras) + 1))):
        notes.append(f"unknown markers {extra}")
    if order != sorted(order):
        notes.append("markers out of order")

    results = []
    for i, p in enumerate(chunk.paras):
        bodies = found.get(i + 1)
        if not bodies:
            results.append(fail(i, "marker_missing"))
            continue
        if len(bodies) > 1:
            results.append(fail(i, "marker_duplicate", bodies[0]))
            continue
        if i + 1 < len(chunk.paras) and not found.get(i + 2):
            results.append(fail(i, "marker_missing_next", bodies[0]))
            continue
        body = nfc(bodies[0])
        if not body.strip():
            results.append(fail(i, "empty", body))
            continue
        try:
            restored = restore(body, p.mapping)
        except PlaceholderError as e:
            results.append(fail(i, f"placeholders: {e}", body))
            continue
        results.append(ParaResult(p.doc_id, p.para_id, chunk.chunk_id, True, restored, body))
    for r in results:
        r.notes = list(notes)
    return results
