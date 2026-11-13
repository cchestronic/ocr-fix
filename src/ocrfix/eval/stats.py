"""Paired Wilcoxon signed-rank test (two-sided) and Holm's correction, without scipy.

Zero differences are dropped (Wilcoxon's method) and tied |differences| get average ranks.
Up to ``exact_max_n`` non-zero pairs the p-value is exact: the null distribution of W+ is
enumerated over doubled ranks, so tied (half-integer) ranks are handled. Above that, the
normal approximation with tie and continuity correction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ocrfix.calibrate.fit import rankdata

EPS = 1e-12


@dataclass(frozen=True)
class Wilcoxon:
    n: int  # non-zero differences
    n_zero: int
    w_plus: float  # rank sum of positive differences (x > y)
    w_minus: float
    p: float
    method: str  # exact | normal | none

    def to_dict(self) -> dict:
        return asdict(self)


def _exact_p(ranks2: list[int], w2: int) -> float:
    """Two-sided P under H0 for doubled ranks, observed doubled W+ = w2."""
    total = sum(ranks2)
    dist = [0] * (total + 1)
    dist[0] = 1
    for r in ranks2:
        for s in range(total, r - 1, -1):
            dist[s] += dist[s - r]
    n_all = 2 ** len(ranks2)
    lo = sum(dist[: w2 + 1]) / n_all
    hi = sum(dist[w2:]) / n_all
    return min(1.0, 2 * min(lo, hi))


def wilcoxon(x: Sequence[float], y: Sequence[float], exact_max_n: int = 25) -> Wilcoxon:
    if len(x) != len(y):
        raise ValueError("wilcoxon: x and y differ in length")
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nz = d[np.abs(d) > EPS]
    n, n_zero = len(nz), len(d) - len(nz)
    if n == 0:
        return Wilcoxon(0, n_zero, 0.0, 0.0, 1.0, "none")
    ranks = rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = n * (n + 1) / 2 - w_plus
    if n <= exact_max_n:
        ranks2 = [round(2 * r) for r in ranks]
        p = _exact_p(ranks2, round(2 * w_plus))
        return Wilcoxon(n, n_zero, w_plus, w_minus, p, "exact")
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(nz), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - float((counts**3 - counts).sum()) / 48
    if var <= 0:
        return Wilcoxon(n, n_zero, w_plus, w_minus, 1.0, "normal")
    diff = w_plus - mean
    z = (abs(diff) - 0.5) / math.sqrt(var) if abs(diff) >= 0.5 else 0.0
    return Wilcoxon(n, n_zero, w_plus, w_minus, min(1.0, math.erfc(z / math.sqrt(2))), "normal")


def holm(ps: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[i]))
        out[i] = running
    return out
