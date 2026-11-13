"""Separability metrics, tau choice and flag-rule search on feature arrays.

Everything here works on plain numpy arrays (null features are NaN and never flag), so the
tests can feed synthetic distributions.
"""

from __future__ import annotations

import numpy as np

from ocrfix.calibrate.rules import Rule


def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged (scipy.stats.rankdata 'average')."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2
        i = j
    return ranks


def auroc(neg: np.ndarray, pos: np.ndarray) -> float | None:
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney). NaNs dropped; None if a side is empty."""
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return None
    r = rankdata(np.concatenate([neg, pos]))
    u = r[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(neg) * len(pos)))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def roc_curve(neg: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false-flag rate, recall) for 'flag if x > t' over every distinct t, plus (1, 1)."""
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    ts = np.unique(np.concatenate([neg, pos]))[::-1]
    fpr = np.array([0.0, *((neg[:, None] > ts).mean(0) if len(neg) else 0 * ts), 1.0])
    tpr = np.array([0.0, *((pos[:, None] > ts).mean(0) if len(pos) else 0 * ts), 1.0])
    return fpr, tpr


def _min_threshold(clean: np.ndarray, budget: int) -> float:
    """Smallest t with #(clean > t) <= budget (clean NaNs never count)."""
    vals = np.sort(clean[~np.isnan(clean)])[::-1]
    if budget >= len(vals):
        return -np.inf
    return float(vals[budget])


def _rates(
    ts: float | None,
    tp: float | None,
    s: np.ndarray,
    p: np.ndarray,
    tb: float | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """spike_frac > ts or p90 > tp or bpc > tb; a None threshold is off, NaN never flags."""
    flags = np.zeros(len(s), dtype=bool) if ts is None else s > ts
    if tp is not None:
        flags |= p > tp
    if tb is not None and b is not None:
        flags |= b > tb
    return flags


def _candidates(clean: np.ndarray, lo: float, k: int) -> list[float | None]:
    """None (feature off), then at most k clean values >= lo (thresholds from the clean set)."""
    c = np.unique(clean[~np.isnan(clean)])
    if len(c) > k:
        c = np.unique(np.quantile(c, np.linspace(0, 1, k)))
    c = np.unique(np.concatenate([c, [lo]])) if np.isfinite(lo) else c
    c = c[c >= lo] if np.isfinite(lo) else c
    return [None, *map(float, c)]


def fit_rule(
    clean_s: np.ndarray,
    clean_p: np.ndarray,
    bad_s: np.ndarray,
    bad_p: np.ndarray,
    *,
    clean_b: np.ndarray | None = None,
    bad_b: np.ndarray | None = None,
    target_recall: float,
    max_false_flag: float,
    max_thresholds: int = 200,
) -> Rule:
    """Flag rule ``spike_frac > A or p90 > B or bpc > C`` (any part may be off) with the best
    recall on the bad set within the false-flag budget on the clean set.

    1. Single features at the whole budget: spike_frac, and bpc when given. If the better one
       reaches target_recall, done.
    2. Otherwise a grid over spike_frac and bpc thresholds (each off, or a clean value at or
       above its single-feature threshold); the budget left goes to the lowest p90 threshold.
       Best recall wins, ties go to fewer false flags.

    Thresholds are strict (flag if value > t) and taken from the clean values, so the
    false-flag rate is exact on the clean set. Without bpc arrays this is the old
    spike_frac / p90 search.
    """
    n, n_bad = len(clean_s), len(bad_s)
    budget = int(np.floor(max_false_flag * n + 1e-9))
    use_b = clean_b is not None and bad_b is not None
    cb = clean_b if use_b else np.full(n, np.nan)
    bb = bad_b if use_b else np.full(n_bad, np.nan)

    def rule(ts: float | None, tp: float | None, tb: float | None) -> Rule:
        rc = _rates(ts, tp, clean_s, clean_p, tb, cb)
        rb = _rates(ts, tp, bad_s, bad_p, tb, bb)
        return Rule(spike_frac=ts, p90=tp, bpc=tb,
                    recall=float(rb.mean()) if n_bad else 0.0,
                    false_flag=float(rc.mean()) if n else 0.0)  # fmt: skip

    def better(a: Rule, b: Rule) -> bool:
        ra, rb = a.recall or 0.0, b.recall or 0.0
        fa, fb = a.false_flag or 0.0, b.false_flag or 0.0
        return ra > rb + 1e-12 or (abs(ra - rb) <= 1e-12 and fa < fb)

    with np.errstate(invalid="ignore"):
        t_s = _min_threshold(clean_s, budget)
        best = rule(t_s, None, None)
        t_b = _min_threshold(cb, budget) if use_b else np.inf
        if use_b and better(rule(None, None, t_b), best):
            best = rule(None, None, t_b)
        if (best.recall or 0) >= target_recall or n_bad == 0:
            return _finish(best, target_recall, n, n_bad)

        # with bpc the grid is 2-D: fewer thresholds per axis keep it to a few thousand fits
        k = min(max_thresholds, 60) if use_b else max_thresholds
        for tb in _candidates(cb, t_b, k) if use_b else [None]:
            for ts in _candidates(clean_s, t_s, k):
                used = _rates(ts, None, clean_s, clean_p, tb, cb)
                left = budget - int(used.sum())
                if left < 0:
                    continue
                tp = _min_threshold(np.where(used, np.nan, clean_p), left)
                cand = rule(ts, tp, tb)
                if better(cand, best):
                    best = cand
    return _finish(best, target_recall, n, n_bad)


def _finish(rule: Rule, target: float, n_clean: int, n_bad: int) -> Rule:
    # a -inf threshold flags every row that has the feature; keep the YAML finite
    if rule.spike_frac is not None and not np.isfinite(rule.spike_frac):
        rule.spike_frac = -1.0
    for name in ("p90", "bpc"):
        v = getattr(rule, name)
        setattr(rule, name, None if v is None else (float(v) if np.isfinite(v) else -1e9))
    rule.recall = round(float(rule.recall or 0), 4)
    rule.false_flag = round(float(rule.false_flag or 0), 4)
    rule.spike_frac = None if rule.spike_frac is None else float(rule.spike_frac)
    rule.target_met = bool(rule.recall >= target)
    rule.n_clean, rule.n_corrupted = int(n_clean), int(n_bad)
    return rule


def flags(rule: Rule, s: np.ndarray, p: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return _rates(rule.spike_frac, rule.p90, s, p, rule.bpc, b)


def precision_recall(flagged: np.ndarray, noisy: np.ndarray) -> dict[str, float | int | None]:
    tp = int((flagged & noisy).sum())
    fp = int((flagged & ~noisy).sum())
    fn = int((~flagged & noisy).sum())
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_flag": round(fp / int((~noisy).sum()), 4) if (~noisy).any() else None,
        "n": len(noisy),
        "n_noisy": int(noisy.sum()),
        "n_flagged": int(flagged.sum()),
    }
