"""Streaming JSONL I/O (optionally gzipped) shared by every stage.

A stage output is a JSONL file or a folder of shards ``part-00000.jsonl[.gz]``, ... Shards are
written to ``*.tmp`` and renamed when complete, so a crashed run never leaves a half-written one.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")


def open_text(path: str | Path, mode: str = "rt") -> IO[str]:
    path = Path(path)
    is_gz = path.suffix == ".gz" or path.name.endswith(".gz.tmp")
    if is_gz:
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "") or "rb"), encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def list_shards(path: str | Path) -> list[Path]:
    """A single file, or every *.jsonl / *.jsonl.gz in a directory (sorted, non-recursive)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.name.endswith(JSONL_SUFFIXES))
    return []


def iter_jsonl(paths: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        for shard in list_shards(p):
            with open_text(shard) as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{shard}:{lineno}: {e}") from e


def repair_jsonl_tail(path: str | Path) -> None:
    """Truncate a plain .jsonl after its last complete, parseable line (crash recovery)."""
    keep = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                json.loads(line)
            except ValueError:
                break
            keep += len(line)
    with open(path, "r+b") as f:
        f.truncate(keep)
