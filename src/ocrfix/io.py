"""Streaming, sharded JSONL I/O (optionally gzipped) shared by every stage.

A stage output is a directory of shards: ``part-00000.jsonl[.gz]``, ``part-00001...``.
Shards are written to ``*.tmp`` and renamed on close, so a crashed run never leaves a
half-written shard behind and ``read_ids`` sees only completed work.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from ocrfix.schema import Record

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


def iter_records(
    paths: str | Path | Iterable[str | Path], limit: int | None = None
) -> Iterator[Record]:
    for i, d in enumerate(iter_jsonl(paths)):
        if limit is not None and i >= limit:
            return
        yield Record.from_dict(d)


def read_ids(path: str | Path) -> set[tuple[str, str]]:
    """(doc_id, para_id) of every record already written to ``path``; empty if absent."""
    return {(str(d["doc_id"]), str(d["para_id"])) for d in iter_jsonl(path)}


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


class ShardedWriter:
    """Append records to ``out_dir`` as new shards, continuing after existing ones."""

    def __init__(
        self, out_dir: str | Path, shard_size: int = 50_000, gzip_output: bool = False
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.suffix = ".jsonl.gz" if gzip_output else ".jsonl"
        self._next_index = 1 + max(
            (int(p.name[5:10]) for p in list_shards(self.out_dir) if p.name.startswith("part-")),
            default=-1,
        )
        self._fh: IO[str] | None = None
        self._tmp: Path | None = None
        self._count = 0

    def _open(self) -> None:
        final = self.out_dir / f"part-{self._next_index:05d}{self.suffix}"
        self._tmp = final.with_name(final.name + ".tmp")
        self._fh = open_text(self._tmp, "wt")
        self._count = 0

    def _close_shard(self) -> None:
        if self._fh is None or self._tmp is None:
            return
        self._fh.close()
        self._tmp.rename(self._tmp.with_name(self._tmp.name.removesuffix(".tmp")))
        self._fh = None
        self._tmp = None
        self._next_index += 1

    def write(self, rec: Record | dict[str, Any]) -> None:
        if self._fh is None:
            self._open()
        assert self._fh is not None
        d = rec.to_dict() if isinstance(rec, Record) else rec
        self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._count += 1
        if self._count >= self.shard_size:
            self._close_shard()

    def close(self) -> None:
        self._close_shard()

    def __enter__(self) -> ShardedWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # On error, still commit the partial shard: every line in it is complete,
        # and read_ids lets the rerun skip it.
        self.close()
