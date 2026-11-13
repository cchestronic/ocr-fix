"""`ocrfix chunk <input_dir> <out_dir>`: every ``*.md`` under a folder tree -> chunk JSONL.

Files are divided into parts of about ``chunk.shard_bytes``; each worker process takes one
part and writes one shard, ``part-NNNNN.jsonl[.gz]`` via ``.tmp`` + rename. Processes, not
threads: tagging is pure-Python regex work, which threads would run one at a time.

``manifest.json`` records the settings and the parts. A rerun with the same files and
settings skips the finished shards; anything else is refused, so one output folder never
mixes two chunkings. A file that can't be read as UTF-8 is listed in ``stats.json``
(``errors``) and skipped.

A chunk row:
  chunk_id, doc_id, source_path  ids; source_path is relative, exactly as on disk
  chunk_index, n_chunks          position in the document
  doc_sha256                     of the original file bytes (checked by `reconstruct`)
  char_start, char_end           content range in the normalized document
  prefix, suffix                 the whitespace around the content
  source_text                    the masked original
  text                           the masked corrected text; null = unchanged (saves half the
                                 disk: an unchanged chunk would hold the same text twice)
  placeholders                   {"⟨M1⟩": raw text, ...}, in order of appearance
  raw                            the source slice when normalization changed it, else null
  n_tokens, short, lang          short = under chunk.min_tokens (only a document's end)
  status, score, history         "new", null, []; filled by later stages
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ocrfix.chunks.split import Chunk, split_document
from ocrfix.corpus.ingest import doc_id_for, find_markdown, in_sample
from ocrfix.io import open_text

_ENCODER = None
HIST_STEP = 64
MIN_SHARD = 1 << 20  # 1 MB


def _init_worker(tokenizer_path: str) -> None:
    global _ENCODER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from ocrfix.offline import enforce_offline

    enforce_offline()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True).backend_tokenizer

    def encode(text: str) -> tuple[list[int], list[int]]:
        offsets = tok.encode(text, add_special_tokens=False).offsets
        return [a for a, _ in offsets], [b for _, b in offsets]

    _ENCODER = encode


def chunk_rows(
    chunks: Sequence[Chunk], rel: str, doc_sha256: str, min_tokens: int
) -> list[dict[str, Any]]:
    doc_id = doc_id_for(rel)
    return [
        {
            "chunk_id": f"{doc_id}-c{i:05d}",
            "doc_id": doc_id,
            "source_path": rel,
            "chunk_index": i,
            "n_chunks": len(chunks),
            "doc_sha256": doc_sha256,
            "char_start": c.start,
            "char_end": c.end,
            "prefix": c.prefix,
            "suffix": c.suffix,
            "source_text": c.source_text,
            "text": None,
            "placeholders": c.placeholders,
            "raw": c.raw,
            "n_tokens": c.n_tokens,
            "short": c.n_tokens < min_tokens,
            "lang": c.lang,
            "status": "new",
            "score": None,
            "history": [],
        }
        for i, c in enumerate(chunks)
    ]


def _run_part(job: dict[str, Any]) -> dict[str, Any]:
    out = Path(job["out"])
    tmp = out.with_name(out.name + ".tmp")
    root = Path(job["input_dir"])
    st: dict[str, Any] = {"docs": 0, "chunks": 0, "tokens": 0, "short": 0, "mid_short": 0,
                          "with_raw": 0, "bytes": 0, "max_tokens": 0, "hist": Counter(),
                          "errors": []}  # fmt: skip
    with open_text(tmp, "wt") as f:
        for rel in job["files"]:
            try:
                data = (root / rel).read_bytes()
                rel.encode("utf-8")  # a file name JSON can't hold is an error too
                chunks = split_document(
                    data.decode("utf-8"), _ENCODER, min_tokens=job["min_tokens"],
                    max_tokens=job["max_tokens"], protect=job["protect"],
                    lang_opts=job["lang_opts"],
                )  # fmt: skip
            except (OSError, ValueError) as e:
                st["errors"].append({"path": rel.encode("utf-8", "replace").decode(),
                                     "error": f"{type(e).__name__}: {e}"[:300]})  # fmt: skip
                continue
            rows = chunk_rows(chunks, rel, hashlib.sha256(data).hexdigest(), job["min_tokens"])
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            st["docs"] += 1
            st["bytes"] += len(data)
            st["chunks"] += len(rows)
            for r in rows:
                n = r["n_tokens"]
                st["tokens"] += n
                st["short"] += r["short"]
                st["mid_short"] += r["short"] and r["chunk_index"] < r["n_chunks"] - 1
                st["with_raw"] += r["raw"] is not None
                st["max_tokens"] = max(st["max_tokens"], n)
                st["hist"][n // HIST_STEP * HIST_STEP] += 1
    st["hist"] = {str(k): v for k, v in sorted(st["hist"].items())}
    Path(job["side"]).write_text(json.dumps(st, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(out)
    return st


@dataclass
class ChunkStats:
    docs: int = 0
    chunks: int = 0
    tokens: int = 0
    short: int = 0
    mid_short: int = 0  # under min_tokens and not the last chunk of its document
    with_raw: int = 0
    bytes: int = 0
    max_tokens: int = 0
    parts: int = 0
    parts_skipped: int = 0
    seconds: float = 0.0
    hist: Counter = field(default_factory=Counter)
    errors: list[dict[str, str]] = field(default_factory=list)

    def add(self, st: dict[str, Any]) -> None:
        for k in ("docs", "chunks", "tokens", "short", "mid_short", "with_raw", "bytes"):
            setattr(self, k, getattr(self, k) + st[k])
        self.max_tokens = max(self.max_tokens, st["max_tokens"])
        self.hist.update({int(k): v for k, v in st["hist"].items()})
        self.errors += st["errors"]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _plan_parts(files: list[tuple[str, int]], shard_bytes: int) -> list[list[str]]:
    parts: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for rel, n in files:
        if cur and size + n > shard_bytes:
            parts.append(cur)
            cur, size = [], 0
        cur.append(rel)
        size += n
    if cur:
        parts.append(cur)
    return parts


def run_chunk(
    cfg,
    input_dir: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    sample_frac: float | None = None,
    workers: int | None = None,
) -> ChunkStats:
    c = cfg.chunk
    min_tokens, max_tokens = int(c.min_tokens), int(c.max_tokens)
    if not 0 < min_tokens <= max_tokens:
        raise ValueError(
            f"chunk: need 0 < min_tokens <= max_tokens, got {min_tokens}, {max_tokens}"
        )
    tokenizer = str(cfg.paths.tokenizer)
    tok_file = Path(tokenizer) / "tokenizer.json"
    settings = {
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "protect": list(cfg.llm.protect),
        "lang": OmegaConf.to_container(cfg.corpus.lang, resolve=True),
        "tokenizer_sha256": _file_hash(tok_file) if tok_file.is_file() else tokenizer,
    }
    files = []
    for p in find_markdown(input_dir, cfg.corpus.input_glob):
        rel = p.relative_to(input_dir).as_posix()
        if sample_frac is None or in_sample(rel, sample_frac, int(cfg.seed)):
            files.append((rel, p.stat().st_size))
        if limit is not None and len(files) >= limit:
            break
    # Small inputs get smaller shards (at least ~64 parts) so every worker has work. Depends on
    # the files only, never on the worker count, so a resume always plans the same parts.
    total = sum(n for _, n in files)
    shard = min(int(c.shard_bytes), max(MIN_SHARD, total // 64))
    parts = _plan_parts(files, shard)
    manifest = {"version": 1, "input_dir": str(input_dir), "settings": settings, "parts": parts}

    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    if mpath.is_file():
        old = json.loads(mpath.read_text("utf-8"))
        if old.get("settings") != settings or old.get("parts") != parts:
            raise ValueError(
                f"{out_dir} holds a different chunking (other files or settings): "
                "use a new output folder"
            )
    else:
        mpath.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", "utf-8")

    suffix = ".jsonl.gz" if bool(cfg.io.gzip) else ".jsonl"
    stats = ChunkStats(parts=len(parts))
    jobs = []
    for i, part in enumerate(parts):
        out = out_dir / f"part-{i:05d}{suffix}"
        side = out_dir / f".part-{i:05d}.stats.json"
        if out.is_file() and side.is_file():
            stats.add(json.loads(side.read_text("utf-8")))
            stats.parts_skipped += 1
            continue
        jobs.append({"out": str(out), "side": str(side), "input_dir": str(input_dir), "files": part,
                     "min_tokens": min_tokens, "max_tokens": max_tokens,
                     "protect": settings["protect"], "lang_opts": settings["lang"]})  # fmt: skip

    t0 = time.perf_counter()
    n_workers = int(workers if workers is not None else (c.workers or os.cpu_count() or 1))
    if n_workers > 0 and jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(n_workers, len(jobs)), _init_worker, (tokenizer,)) as pool:
            for st in pool.imap_unordered(_run_part, jobs):
                stats.add(st)
    elif jobs:
        _init_worker(tokenizer)
        for st in map(_run_part, jobs):
            stats.add(st)
    stats.seconds = round(time.perf_counter() - t0, 1)
    summary = {k: v for k, v in stats.__dict__.items() if k != "hist"}
    summary["token_histogram"] = {str(k): v for k, v in sorted(stats.hist.items())}
    (out_dir / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    return stats
