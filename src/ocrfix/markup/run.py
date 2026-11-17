"""`ocrfix markup <input_dir> <out_dir>`: repair LaTeX and tables in every markdown file.

Order per document (each step sees the previous one's output):
  1. HTML tables (``tables.fix_html_tables``), then Markdown pipe tables outside them;
  2. math delimiters outside tables (``latex.repair_delimiters``);
  3. every math span the tagger now finds outside tables: ``latex.fix_formula``;
  4. what is still broken is reported: unclosed ``<table>``, a ``$`` left in prose.
Prose is never edited except to add a missing closing delimiter or drop the spaces inside
``$ x $``.

Resumable: a file whose output exists is skipped (outputs are written via ``.tmp`` +
rename, so an existing one is complete), and the report is appended to; delete ``out_dir``
to redo everything (e.g. after changing settings). One process per CPU core by default
(``markup.workers``); each takes files in batches.

Writes ``<out_dir>/<same relative path>`` for every file (unchanged files copied byte for
byte), ``<out_dir>/markup_report.jsonl`` (one row per changed or flagged document: fixes,
issues with a snippet) and ``<out_dir>/markup_stats.json``. A file that isn't UTF-8 is copied
as it is and listed. CRLF line ends and a BOM are kept.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocrfix.corpus.files import find_markdown
from ocrfix.corpus.tagger import _find_html_tables, _find_pipe_tables, tag_spans
from ocrfix.markup.latex import Issue, Result, fix_formula, repair_delimiters, snippet
from ocrfix.markup.tables import fix_html_tables, fix_pipe_tables


def _gaps(spans, n: int) -> list[tuple[int, int]]:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start > pos:
            out.append((pos, s.start))
        pos = max(pos, s.end)
    if pos < n:
        out.append((pos, n))
    return out


def fix_document(text: str, opts: dict[str, Any]) -> Result:
    """One document (LF line ends). ``opts``: the ``markup`` config section."""
    res = Result(text)
    t = text
    if opts["tables"]["html"]:
        r = fix_html_tables(t)
        t, _ = r.text, res.fixes.update(r.fixes)
        res.issues += r.issues
    if opts["tables"]["markdown"]:
        r = fix_pipe_tables(t, [(s.start, s.end) for s in _find_html_tables(t)])
        t, _ = r.text, res.fixes.update(r.fixes)
        res.issues += r.issues
    tables = _find_html_tables(t)
    tables += [p for p in _find_pipe_tables(t) if not any(p.start < h.end and h.start < p.end
                                                          for h in tables)]  # fmt: skip
    if opts["latex"]["delimiters"]:
        r = repair_delimiters(t, _gaps(tables, len(t)))
        t, _ = r.text, res.fixes.update(r.fixes)
        res.issues += r.issues
    if opts["latex"]["formulas"]:
        out, pos = [], 0
        for s in tag_spans(t):
            if s.kind != "math":
                continue
            f = fix_formula(t[s.start : s.end], spacing=opts["latex"]["spacing"],
                            normalize=opts["latex"]["normalize_delimiters"])  # fmt: skip
            res.fixes.update(f.fixes)
            res.issues += [Issue(i.kind, s.start, i.snippet) for i in f.issues]
            out += [t[pos : s.start], f.text]
            pos = s.end
        out.append(t[pos:])
        t = "".join(out)
    # what is left: an unclosed <table>, a lone $ in prose
    for s in tag_spans(t):
        if s.kind != "prose":
            continue
        seg = t[s.start : s.end]
        i = seg.lower().find("<table")
        if i >= 0:
            res.issues.append(Issue("table_unclosed", s.start + i, snippet(seg, i)))
        k = 0
        while (k := seg.find("$", k)) >= 0:
            if k == 0 or seg[k - 1] != "\\":
                res.issues.append(Issue("stray_dollar", s.start + k, snippet(seg, k)))
                break  # one per prose span is enough to find it
            k += 1
    res.text = t if t != text else text
    return res


@dataclass
class MarkupStats:
    docs: int = 0
    skipped: int = 0
    changed: int = 0
    with_issues: int = 0
    not_utf8: list[str] = field(default_factory=list)
    fixes: Counter = field(default_factory=Counter)
    issues: Counter = field(default_factory=Counter)
    bytes: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["fixes"] = dict(self.fixes.most_common())
        d["issues"] = dict(self.issues.most_common())
        return d


def _one(args: tuple[str, str, str, dict]) -> dict[str, Any]:
    src, dst, rel, opts = args
    if Path(dst).is_file():  # written by an earlier run (.tmp + rename: always complete)
        return {"path": rel, "skipped": True, "bytes": 0}
    raw = Path(src).read_bytes()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        shutil.copyfile(src, dst)
        return {"path": rel, "not_utf8": True, "bytes": len(raw)}
    bom = text.startswith("﻿")
    body = text[1:] if bom else text
    crlf = "\r\n" in body and body.count("\r\n") == body.count("\n")
    if crlf:
        body = body.replace("\r\n", "\n")
    r = fix_document(body, opts)
    if r.text is body:
        tmp = Path(dst + ".tmp")
        tmp.write_bytes(raw)
    else:
        new = r.text.replace("\n", "\r\n") if crlf else r.text
        tmp = Path(dst + ".tmp")
        tmp.write_bytes((("﻿" if bom else "") + new).encode("utf-8"))
    os.replace(tmp, dst)
    return {"path": rel, "changed": r.text is not body, "fixes": dict(r.fixes),
            "issues": [i.to_dict() for i in r.issues], "bytes": len(raw)}  # fmt: skip


def run_markup(
    cfg, input_dir: Path, out_dir: Path, *, workers: int | None = None, limit: int | None = None
) -> MarkupStats:
    if out_dir.resolve() == input_dir.resolve():
        raise ValueError("out_dir must differ from input_dir (the corpus is never overwritten)")
    from omegaconf import OmegaConf

    opts = OmegaConf.to_container(cfg.markup, resolve=True)
    files = find_markdown(input_dir, str(cfg.chunk.glob))
    if limit is not None:
        files = files[:limit]
    jobs = [(str(p), str(out_dir / p.relative_to(input_dir)), p.relative_to(input_dir).as_posix(),
             opts) for p in files]  # fmt: skip
    st = MarkupStats()
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = workers if workers is not None else opts.get("workers")
    n = (os.cpu_count() or 1) if n is None else int(n)
    if n <= 0 or len(jobs) < 2:
        results = map(_one, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=n)
        # many small files: hand each process batches, so the queue isn't the bottleneck
        results = pool.map(_one, jobs, chunksize=max(1, min(64, len(jobs) // (n * 8))))
    # appended: a rerun after an interruption adds the rows of the files it still had to do
    with (out_dir / "markup_report.jsonl").open("a", encoding="utf-8") as rep:
        for r in results:
            st.docs += 1
            if r.get("skipped"):
                st.skipped += 1
                continue
            st.bytes += r["bytes"]
            if r.get("not_utf8"):
                st.not_utf8.append(r["path"])
                continue
            st.changed += r["changed"]
            st.fixes.update(r["fixes"])
            st.issues.update(i["kind"] for i in r["issues"])
            st.with_issues += bool(r["issues"])
            if r["changed"] or r["issues"]:
                rep.write(json.dumps(r, ensure_ascii=False) + "\n")
    if n > 0 and len(jobs) >= 2:
        pool.shutdown()
    st.seconds = time.time() - t0
    (out_dir / "markup_stats.json").write_text(
        json.dumps(st.to_dict(), indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    return st
