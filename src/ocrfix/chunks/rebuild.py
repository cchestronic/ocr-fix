"""`ocrfix reconstruct <chunks> [out_dir]`: chunk JSONL -> markdown files.

``--source original``: byte-exact source files, checked against ``doc_sha256``.
``--source current``: the corrected text. A chunk with no new ``text`` gives its original
bytes; a changed one gives ``prefix + restore(text, placeholders) + suffix``.
Without ``out_dir`` nothing is written: the check alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import groupby, islice
from pathlib import Path, PurePosixPath
from typing import Any

from ocrfix.corpus.placeholders import PlaceholderError, restore
from ocrfix.io import iter_jsonl


def _original(c: dict[str, Any]) -> str:
    if c["raw"] is not None:
        return c["raw"]
    return c["prefix"] + restore(c["source_text"], c["placeholders"]) + c["suffix"]


def _current(c: dict[str, Any]) -> str:
    if c["text"] is None or c["text"] == c["source_text"]:
        return _original(c)
    return c["prefix"] + restore(c["text"], c["placeholders"]) + c["suffix"]


def rebuild_doc(chunks: list[dict[str, Any]], source: str = "original") -> bytes:
    expected = list(range(chunks[0]["n_chunks"]))
    got = [c["chunk_index"] for c in chunks]
    if got != expected:
        raise ValueError(f"expected chunks {len(expected)}, got indexes {got[:5]}...")
    fn = _original if source == "original" else _current
    return "".join(fn(c) for c in chunks).encode("utf-8")


def _safe_target(root: Path, rel: str) -> Path:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe source_path {rel!r}")
    return root / p


def _docs(rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for _, group in groupby(rows, key=lambda c: c["doc_id"]):
        yield list(group)


@dataclass
class RebuildStats:
    docs: int = 0
    bytes: int = 0
    mismatched: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_rebuild(
    chunks: Path, out_dir: Path | None, *, source: str = "original", limit: int | None = None
) -> RebuildStats:
    if source not in ("original", "current"):
        raise ValueError(f"source must be original or current, got {source!r}")
    st = RebuildStats()
    for doc in islice(_docs(iter_jsonl(chunks)), limit):
        rel = doc[0]["source_path"]
        try:
            data = rebuild_doc(doc, source)
            target = _safe_target(out_dir, rel) if out_dir is not None else None
        except (ValueError, PlaceholderError) as e:
            st.errors.append(f"{rel}: {e}")
            continue
        st.docs += 1
        st.bytes += len(data)
        if source == "original" and hashlib.sha256(data).hexdigest() != doc[0]["doc_sha256"]:
            st.mismatched.append(rel)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return st
