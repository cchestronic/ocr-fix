"""`ocrfix gold chunks <gold_dir> <out_dir>`: hand-corrected markdown pairs -> chunk gold.

Input: ``<gold_dir>/ocr/**.md`` (the MinerU output, never edited) and ``<gold_dir>/truth/**.md``
(a corrected copy at the same relative path; lost words written as 〈?〉).

Only the OCR side is chunked (`ocrfix chunk`, into ``<out_dir>/chunks``, rebuilt on every run),
exactly as the corpus is, so the gold chunks are the model's real input. Each chunk's gold is
the truth text that a character alignment of the two documents (Levenshtein opcodes) puts
against the chunk's range:
  * a cut in unchanged text maps one to one;
  * text the truth inserts right at a cut belongs to the chunk on that side (at a chunk's
    start to that chunk, at its end to that chunk);
  * OCR text the truth deleted across a cut simply vanishes on both sides;
  * a replacement that crosses a cut can't be split fairly: both chunks get
    ``gold_status: cut_in_edit`` and are left out of training (eval is per document, so
    it doesn't need cuts at all).
The truth piece is masked with the chunk rule, from a tagging of the whole truth document;
its placeholders must equal the OCR chunk's (math, tables, links left as they are), else
``placeholders_changed``. A chunk whose truth piece has no text is ``empty``.
``gold_suffix`` is the truth's whitespace after the chunk (null if text fell there). Where it
differs from the OCR ``suffix`` (``suffix_changed`` in the stats: mostly a line break inside a
sentence that the chunker cut at), the difference sits between chunks, where no correction
reaches it.

Split by document (``gold.split``: eval / dpo shares, the rest sft); ``split.json``
pins every document to its side, so adding gold later never moves a document. Output:
``<out_dir>/{sft,dpo,eval}/part-00000.jsonl``: every chunk of every document of that side
(complete documents, so `score` / `correct` and `reconstruct` work on them), each chunk
row plus ``gold_text``, ``gold_placeholders``, ``gold_status``, ``split``. ``stats.json``
counts documents, chunks per status, edited chunks and 〈?〉.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from ocrfix.chunks.split import atoms_of, normalize, render
from ocrfix.corpus.tagger import tag_spans
from ocrfix.io import iter_jsonl
from ocrfix.schema import UNCERTAIN

SIDES = ("eval", "dpo", "sft")


@dataclass
class _Op:
    tag: str
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int


class TruthMap:
    """OCR offsets -> truth offsets through the opcodes of the two normalized documents."""

    def __init__(self, ocr: str, truth: str) -> None:
        # adjacent non-equal opcodes are one edit (``WXYZ`` -> ``Q`` comes as replace +
        # delete), so a cut between them still counts as crossing a replacement
        self.ops: list[_Op] = []
        for o in Levenshtein.opcodes(ocr, truth):
            last = self.ops[-1] if self.ops else None
            if o.tag != "equal" and last is not None and last.tag != "equal":
                last.src_end, last.dest_end = o.src_end, o.dest_end
            else:
                self.ops.append(_Op(o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end))
        self.starts = [o.src_start for o in self.ops]

    def _touching(self, x: int) -> list:
        k = bisect.bisect_right(self.starts, x) - 1
        out = []
        while k >= 0 and self.ops[k].src_end >= x:
            out.append(self.ops[k])
            k -= 1
        return out

    def at(self, x: int, side: str) -> tuple[int, bool]:
        """(truth offset, crosses a replacement) for a cut at OCR offset ``x``; ``side`` is
        "start" (insertions at x go after the cut) or "end" (they go before it)."""
        cands, crossing = [], False
        for o in self._touching(x):
            if o.tag == "equal":
                cands.append(o.dest_start + x - o.src_start)
            elif o.src_start < x < o.src_end:
                crossing |= o.dest_end > o.dest_start  # a deletion splits cleanly
                cands.append(o.dest_start if side == "start" else o.dest_end)
            else:  # the op starts or ends at x (an insertion does both)
                cands += [o.dest_start] if o.src_start == x else []
                cands += [o.dest_end] if o.src_end == x else []
        if not cands:
            cands = [0]
        return (min(cands) if side == "start" else max(cands)), crossing


def _strip(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def cut_truth(
    ocr: str, truth: str, chunks: list[dict[str, Any]], protect: list[str]
) -> list[dict[str, Any]]:
    """{gold_text, gold_placeholders, gold_status} for each chunk of one document (in order);
    ``ocr`` and ``truth`` normalized as the chunker does."""
    tm = TruthMap(ocr, truth)
    atoms = atoms_of(truth, tag_spans(truth), protect)
    a_starts = [a for a, _, _ in atoms]
    cuts, status = [], []
    prev = 0
    for c in chunks:
        ts, x1 = tm.at(c["char_start"], "start")
        te, x2 = tm.at(c["char_end"], "end")
        ts = max(ts, prev)
        te = max(te, ts)
        cuts.append((ts, te))
        status.append("cut_in_edit" if x1 or x2 else "ok")
        prev = te
    # truth text that fell between two chunks (or before the first / after the last)
    bounds = [0] + [x for ab in cuts for x in ab] + [len(truth)]
    for k in range(len(chunks) + 1):
        if truth[bounds[2 * k] : bounds[2 * k + 1]].strip():
            for j in (k - 1, k):
                if 0 <= j < len(chunks):
                    status[j] = "cut_in_edit"
    out = []
    for k, (c, (ts, te), st) in enumerate(zip(chunks, cuts, status, strict=True)):
        ts, te = _strip(truth, ts, te)
        nxt = _strip(truth, *cuts[k + 1])[0] if k + 1 < len(cuts) else len(truth)
        sep = truth[te:nxt] if truth[te:nxt].isspace() or te == nxt else None
        i = bisect.bisect_left(a_starts, ts) - 1
        straddle = any(
            a < ts < b or a < te < b
            for a, b, _ in atoms[max(i, 0) : bisect.bisect_right(a_starts, te) + 1]
        )
        if straddle:
            out.append({"gold_text": None, "gold_placeholders": None,
                        "gold_status": "placeholders_changed", "gold_suffix": sep})  # fmt: skip
            continue
        gold, mapping = render(truth, atoms, ts, te)
        if st == "ok":
            if list(mapping.items()) != list((c.get("placeholders") or {}).items()):
                st = "placeholders_changed"
            elif not gold.strip():
                st = "empty"
        out.append({"gold_text": gold, "gold_placeholders": mapping, "gold_status": st,
                    "gold_suffix": sep})  # fmt: skip
    return out


def _side(doc_id: str, seed: int, shares: dict[str, float]) -> str:
    h = int(hashlib.sha256(f"{seed}:gold:{doc_id}".encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for s in ("eval", "dpo"):
        acc += float(shares.get(s, 0))
        if h < acc:
            return s
    return "sft"


def build_gold_chunks(
    cfg, gold_dir: Path, out_dir: Path, *, workers: int | None = None
) -> dict[str, Any]:
    """Chunk ``gold_dir/ocr``, cut ``gold_dir/truth`` to match, split, write (module doc)."""
    from ocrfix.chunks.run import run_chunk

    ocr_dir, truth_dir = gold_dir / "ocr", gold_dir / "truth"
    for d in (ocr_dir, truth_dir):
        if not d.is_dir():
            raise ValueError(f"{d}: missing (gold needs ocr/ and truth/ side by side)")
    opts = cfg.gold
    shares = {"eval": float(opts.split.eval), "dpo": float(opts.split.dpo)}
    if sum(shares.values()) >= 1:
        raise ValueError("gold.split: eval + dpo must be < 1 (the rest is sft)")
    seed = int(cfg.seed)
    protect = list(cfg.chunk.protect)

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "chunks"
    if chunk_dir.exists():  # derived output: rebuilt so new or edited gold files are picked up
        shutil.rmtree(chunk_dir)
    cst = run_chunk(cfg, ocr_dir, chunk_dir, workers=workers)

    split_file = out_dir / "split.json"
    pinned: dict[str, str] = {}
    if split_file.is_file():
        for s in SIDES:
            for d in json.loads(split_file.read_text("utf-8")).get(f"{s}_docs", []):
                pinned[d] = s

    st: dict[str, Any] = {
        "docs": Counter(), "chunks": defaultdict(Counter), "edited_chunks": Counter(),
        "uncertain": Counter(), "suffix_changed": Counter(), "no_truth": [],
        "chunk_errors": cst.errors,
    }  # fmt: skip
    sides: dict[str, str] = {}
    tmp = {s: out_dir / s / "part-00000.jsonl.tmp" for s in SIDES}
    for s in SIDES:
        tmp[s].parent.mkdir(exist_ok=True)
        for old in (out_dir / s).glob("part-*"):
            old.unlink()
    handles = {s: tmp[s].open("w", encoding="utf-8") for s in SIDES}
    try:
        for path, rows in groupby(iter_jsonl(chunk_dir), key=lambda r: r["source_path"]):
            rows = sorted(rows, key=lambda r: r["chunk_index"])
            tfile = truth_dir / path
            if not tfile.is_file():
                st["no_truth"].append(path)
                continue
            ocr = normalize((ocr_dir / path).read_bytes().decode("utf-8"))
            truth = normalize(tfile.read_bytes().decode("utf-8"))
            doc_id = rows[0]["doc_id"]
            side = pinned.get(doc_id) or _side(doc_id, seed, shares)
            sides[doc_id] = side
            st["docs"][side] += 1
            for r, g in zip(rows, cut_truth(ocr, truth, rows, protect), strict=True):
                r.update(g, split=side)
                st["chunks"][side][g["gold_status"]] += 1
                # the whitespace after the chunk differs in the truth (e.g. a line break
                # inside a sentence that the chunker cut at): no correction can fix it
                st["suffix_changed"][side] += g["gold_suffix"] != r["suffix"]
                if g["gold_status"] == "ok":
                    st["edited_chunks"][side] += g["gold_text"] != r["source_text"]
                    st["uncertain"][side] += g["gold_text"].count(UNCERTAIN)
                handles[side].write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for h in handles.values():
            h.close()
    for s in SIDES:
        tmp[s].replace(tmp[s].with_suffix(""))
    for d, s in pinned.items():  # documents not in this run keep their side
        sides.setdefault(d, s)
    split = {f"{s}_docs": sorted(d for d, v in sides.items() if v == s) for s in SIDES}
    split_file.write_text(json.dumps(split, indent=1) + "\n", "utf-8")
    st = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}
    st["chunks"] = {s: dict(c) for s, c in st["chunks"].items()}
    (out_dir / "stats.json").write_text(json.dumps(st, indent=1, ensure_ascii=False) + "\n")
    return st
