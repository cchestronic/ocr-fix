"""Character edit distance (Levenshtein). rapidfuzz when installed, else pure Python."""

from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _rf
except ImportError:  # the `text` extra is optional
    _rf = None


def levenshtein(a: str, b: str) -> int:
    if _rf is not None:
        return int(_rf.distance(a, b))
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized(a: str, b: str) -> float:
    """Edit distance / length of the longer string (0 = identical, 1 = nothing shared)."""
    n = max(len(a), len(b))
    return levenshtein(a, b) / n if n else 0.0
