"""Per-paragraph features from per-token surprisal (bits), over its scored tokens.

``n_prose_tokens`` = scored tokens; a paragraph with fewer than ``min_scored_tokens`` gets
``low_confidence: true`` (too few tokens for a stable rate; calibration never flags it).
``max_pos`` and ``n_chars`` refer to the masked text (placeholders in place of math,
tables and protected runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ocrfix.scorer.tokens import TokenizedPara

STAT_KEYS = (
    "n_prose_tokens",
    "n_chars",
    "bpc",
    "mean_bits",
    "p50",
    "p90",
    "p99",
    "max_bits",
    "max_pos",
    "spike_frac",
)


def _r(x: float) -> float:
    return round(float(x), 4)


def spike_key(tau: float) -> str:
    """Column name of spike_frac at another tau: ``spike_frac@8``."""
    return f"spike_frac@{tau:g}"


def prose_stats(
    tp: TokenizedPara, bits: np.ndarray, tau: float, taus: Sequence[float] = ()
) -> dict[str, Any]:
    """Stats over scored tokens (prose, not boundary)."""
    m = tp.scored
    b = bits[m]
    if len(b) == 0:
        empty = dict.fromkeys([*STAT_KEYS, *map(spike_key, taus)], None)
        return empty | {"n_prose_tokens": 0, "n_chars": 0}
    # chars covered by scored tokens (byte-level BPE can split one char over two tokens)
    covered = np.zeros(int(tp.ends.max()), dtype=bool)
    for s, e in zip(tp.starts[m], tp.ends[m], strict=True):
        covered[s:e] = True
    n_chars = int(covered.sum())
    p50, p90, p99 = np.percentile(b, [50, 90, 99])
    k = int(np.argmax(b))
    return {
        "n_prose_tokens": len(b),
        "n_chars": n_chars,
        "bpc": _r(b.sum() / n_chars) if n_chars else None,
        "mean_bits": _r(b.mean()),
        "p50": _r(p50),
        "p90": _r(p90),
        "p99": _r(p99),
        "max_bits": _r(b[k]),
        "max_pos": int(tp.starts[m][k]),
        "spike_frac": _r((b > tau).mean()),
    } | {spike_key(t): _r((b > t).mean()) for t in taus}


def paragraph_features(
    tp: TokenizedPara,
    bits: np.ndarray,
    tau: float,
    taus: Sequence[float] = (),
    min_scored_tokens: int = 10,
) -> dict[str, Any]:
    feats = prose_stats(tp, bits, tau, taus)
    feats["low_confidence"] = feats["n_prose_tokens"] < min_scored_tokens
    if tp.mask_error:
        feats["mask_error"] = True
    return feats
