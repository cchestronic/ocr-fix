"""`ocrfix reconstruct <chunks> [out_dir]`: chunk JSONL -> markdown files.

``--source original``: byte-exact source files, checked against ``doc_sha256``.
``--source current``: the corrected text. A chunk with no new ``text`` gives its original
bytes; a changed one gives ``prefix + restore(text, placeholders) + suffix``.
Without ``out_dir`` nothing is written: the check alone.

With ``current``, a document that noise removal (`score`: noise words, whole noise chunks)
left nearly empty is dropped: fewer than ``min_letters`` letters/digits in the corrected text,
or fewer than ``min_kept_share`` of the original's. It is not written; ``dropped_docs.jsonl``
in ``out_dir`` lists it (``source_path, doc_id, letters, letters_original, kept_share,
chunks, noise_chunks``). 0 turns either test off. A document with a table (``⟨T#⟩``) is
never dropped: noise removal never removes tables, so a table-only document stays.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import groupby, islice
from pathlib import Path, PurePosixPath
from typing import Any

from ocrfix.corpus.placeholders import PlaceholderError, restore
from ocrfix.io import iter_jsonl
from ocrfix.schema import TABLE_PH

_LETTER = re.compile(r"[^\W_]")  # a letter or digit, any script
_TABLE = TABLE_PH.split("{")[0]  # "⟨T"


def letters(text: str) -> int:
    return len(_LETTER.findall(text))


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
    dropped: int = 0


def drop_reason(doc: list[dict[str, Any]], text: str, min_letters: int,
                min_kept_share: float) -> dict[str, Any] | None:  # fmt: skip
    """The record for dropped_docs.jsonl if ``text`` (the corrected document) is too empty.
    Never for a document with a table."""
    if any(k.startswith(_TABLE) for c in doc for k in c.get("placeholders") or {}):
        return None
    n = letters(text)
    n0 = letters("".join(_original(c) for c in doc))
    share = n / n0 if n0 else 1.0
    if n >= min_letters and share >= min_kept_share:
        return None
    noise = sum(1 for c in doc for h in c.get("history") or () if h.get("stage") == "noise_chunk")
    return {"source_path": doc[0]["source_path"], "doc_id": doc[0]["doc_id"], "letters": n,
            "letters_original": n0, "kept_share": round(share, 4), "chunks": len(doc),
            "noise_chunks": noise}  # fmt: skip


def run_rebuild(
    chunks: Path, out_dir: Path | None, *, source: str = "original", limit: int | None = None,
    min_letters: int = 0, min_kept_share: float = 0.0,
) -> RebuildStats:  # fmt: skip
    """``min_letters`` / ``min_kept_share``: drop nearly empty documents (``current`` only)."""
    if source not in ("original", "current"):
        raise ValueError(f"source must be original or current, got {source!r}")
    st = RebuildStats()
    check = source == "current" and (min_letters > 0 or min_kept_share > 0)
    dropped = None
    if check and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dropped = (out_dir / "dropped_docs.jsonl").open("w", encoding="utf-8")
    try:
        for doc in islice(_docs(iter_jsonl(chunks)), limit):
            _one(doc, out_dir, source, st, min_letters, min_kept_share, check, dropped)
    finally:
        if dropped is not None:
            dropped.close()
    return st


def _one(doc, out_dir, source, st, min_letters, min_kept_share, check, dropped) -> None:
    rel = doc[0]["source_path"]
    try:
        data = rebuild_doc(doc, source)
        target = _safe_target(out_dir, rel) if out_dir is not None else None
    except (ValueError, PlaceholderError) as e:
        st.errors.append(f"{rel}: {e}")
        return
    if check:
        why = drop_reason(doc, data.decode("utf-8"), min_letters, min_kept_share)
        if why is not None:
            st.dropped += 1
            if dropped is not None:
                dropped.write(json.dumps(why, ensure_ascii=False) + "\n")
            if target is not None and target.exists():
                target.unlink()  # from an earlier run
            return
    st.docs += 1
    st.bytes += len(data)
    if source == "original" and hashlib.sha256(data).hexdigest() != doc[0]["doc_sha256"]:
        st.mismatched.append(rel)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
