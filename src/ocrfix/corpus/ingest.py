"""Walk MinerU markdown files and write one Record per paragraph to a JSONL file.

Resumable: records of a document are written contiguously and flushed per document. On
restart every document in the output except the last one is complete, so the last one (and
any torn trailing line) is truncated away and redone. ``*.jsonl.gz`` output cannot be
truncated in place; it is rebuilt through a temp file and renamed when the run finishes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from ocrfix.corpus.lang import detect_lang
from ocrfix.corpus.tagger import tag_spans
from ocrfix.io import iter_jsonl
from ocrfix.schema import Record, Span, nfc

_PARA_BREAK = re.compile(r"\n(?:[ \t]*\n)+")


@dataclass
class IngestStats:
    docs_written: int = 0
    docs_skipped: int = 0
    docs_sampled_out: int = 0
    paragraphs: int = 0


def doc_id_for(rel_path: str) -> str:
    return hashlib.sha256(nfc(rel_path).encode("utf-8")).hexdigest()[:16]


def find_markdown(input_dir: Path, pattern: str = "**/*.md") -> list[Path]:
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def read_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    return nfc(text.replace("\r\n", "\n").replace("\r", "\n"))


def split_paragraphs(text: str, spans: list[Span]) -> Iterator[tuple[str, list[Span]]]:
    """Split on blank lines outside math/table spans; strip edges; re-base span offsets."""
    blocks = [s for s in spans if s.kind != "prose"]
    cuts = [
        m
        for m in _PARA_BREAK.finditer(text)
        if not any(b.start < m.end() and m.start() < b.end for b in blocks)
    ]
    bounds = [0, *(x for m in cuts for x in (m.start(), m.end())), len(text)]
    for a, b in zip(bounds[::2], bounds[1::2], strict=True):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a == b:
            continue
        local = [
            Span(max(s.start, a) - a, min(s.end, b) - a, s.kind)
            for s in spans
            if s.start < b and a < s.end
        ]
        yield text[a:b], local


def doc_records(path: Path, input_dir: Path, lang_opts: dict | None = None) -> list[Record]:
    rel = path.relative_to(input_dir).as_posix()
    doc_id = doc_id_for(rel)
    text = read_markdown(path)
    records = []
    for order, (para, spans) in enumerate(split_paragraphs(text, tag_spans(text))):
        records.append(
            Record(
                doc_id=doc_id,
                para_id=f"p{order:05d}",
                order=order,
                lang=detect_lang(para, spans, **(lang_opts or {})),
                text=para,
                spans=spans,
                source_path=nfc(rel),
            )
        )
    return records


def _resume_plain(out: Path) -> set[str]:
    """Truncate ``out`` to its last complete document; return the doc_ids before it."""
    done: list[str] = []
    keep = 0  # byte offset where the last (possibly incomplete) doc group starts
    last_doc = None
    pos = 0
    with open(out, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                doc_id = json.loads(line)["doc_id"]
            except (ValueError, KeyError):
                break
            if doc_id != last_doc:
                if last_doc is not None:
                    done.append(last_doc)
                last_doc, keep = doc_id, pos
            pos += len(line)
    with open(out, "r+b") as f:
        f.truncate(keep)
    return set(done)


def in_sample(rel_path: str, frac: float, seed: int = 0) -> bool:
    """Deterministic: sha256(seed, NFC relative path) as a fraction in [0, 1) < frac."""
    h = hashlib.sha256(f"{seed}:{nfc(rel_path)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < frac


def ingest(
    input_dir: Path,
    out: Path,
    *,
    pattern: str = "**/*.md",
    limit: int | None = None,
    lang_opts: dict | None = None,
    sample_frac: float | None = None,
    sample_seed: int = 0,
) -> IngestStats:
    """``limit`` caps the number of new documents ingested in this run. ``sample_frac`` keeps
    a deterministic share of documents (by a hash of seed and relative path), so a sample is
    spread over the whole corpus and a rerun picks the same documents."""
    stats = IngestStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    gz = out.name.endswith(".gz")
    tmp = out.with_name(out.name + ".tmp")
    with ExitStack() as stack:
        fh: IO[str]
        if gz:
            fh = stack.enter_context(gzip.open(tmp, "wt", encoding="utf-8"))
            done: set[str] = set()
            if out.exists():
                for d in iter_jsonl(out):
                    done.add(d["doc_id"])
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        else:
            done = _resume_plain(out) if out.exists() else set()
            fh = stack.enter_context(open(out, "a", encoding="utf-8"))
        for path in find_markdown(input_dir, pattern):
            if limit is not None and stats.docs_written >= limit:
                break
            rel = path.relative_to(input_dir).as_posix()
            if sample_frac is not None and not in_sample(rel, sample_frac, sample_seed):
                stats.docs_sampled_out += 1
                continue
            if doc_id_for(rel) in done:
                stats.docs_skipped += 1
                continue
            recs = doc_records(path, input_dir, lang_opts)
            fh.write("".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in recs))
            fh.flush()
            stats.docs_written += 1
            stats.paragraphs += len(recs)
    if gz:
        shutil.move(tmp, out)
    return stats
