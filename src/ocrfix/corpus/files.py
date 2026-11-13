"""Finding the markdown files of a corpus, their stable ids, and the sample by path hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ocrfix.schema import nfc


def doc_id_for(rel_path: str) -> str:
    """First 16 hex of sha256 of the NFC relative POSIX path: stable across reruns."""
    return hashlib.sha256(nfc(rel_path).encode("utf-8")).hexdigest()[:16]


def find_markdown(input_dir: Path, pattern: str = "**/*.md") -> list[Path]:
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def in_sample(rel_path: str, frac: float, seed: int = 0) -> bool:
    """Deterministic: sha256(seed, NFC relative path) as a fraction in [0, 1) < frac."""
    h = hashlib.sha256(f"{seed}:{nfc(rel_path)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < frac
