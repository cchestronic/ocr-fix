"""Corpus quality at several stages, from `score` / `correct` output (no model needed):

    ocrfix bpc-hist raw=data/real/scored_raw denoised=data/real/scored corrected=data/real/refined

bpc (bits per char: how surprising the base model finds the text; lower = cleaner) per chunk,
per language and over all (``*``), for each stage:

  chunks     chunks with a score (not low_confidence; noise and failed chunks left out)
  noise      chunks dropped as noise (status "noise": nothing left to score)
  mean       mean of the chunks' bpc
  wmean      mean weighted by the scored chars (= bpc of the whole corpus as one text)
  p50 p90 p99 max, and a 0.25-bpc histogram

Read the means together with ``chunks`` / ``noise``: noise removal lowers the mean partly by
cleaning chunks and partly by dropping whole garbage chunks. A chunk `correct` rejected keeps
its old score, so the corrected stage is the corpus as it will be written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrfix.io import iter_jsonl

BIN = 0.25  # bits per char
TOP = 16.0  # higher values go in the last bin
ALL = "*"
SKIP = ("noise", "score_error")


def _hist(v: np.ndarray) -> dict[str, int]:
    idx = np.minimum((v / BIN).astype(int), int(TOP / BIN))
    keys, counts = np.unique(idx, return_counts=True)
    return {f"{i * BIN:.2f}": int(n) for i, n in zip(keys, counts, strict=True)}


def stage_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Per language (and ``*``) bpc statistics of one `score` / `correct` output."""
    vals: dict[str, list[float]] = defaultdict(list)
    wts: dict[str, list[float]] = defaultdict(list)
    noise: Counter = Counter()
    for r in iter_jsonl(path):
        lang = r.get("lang") or "other"
        if r.get("status") == "noise":
            noise[lang] += 1
            noise[ALL] += 1
            continue
        f = r.get("score") or {}
        if r.get("status") in SKIP or f.get("bpc") is None or f.get("low_confidence"):
            continue
        for k in (lang, ALL):
            vals[k].append(float(f["bpc"]))
            wts[k].append(float(f.get("n_chars") or 0))
    out = {}
    for lang in sorted(set(vals) | set(noise), key=lambda k: (k != ALL, -len(vals.get(k, [])))):
        v, w = np.array(vals.get(lang, []), float), np.array(wts.get(lang, []), float)
        s: dict[str, Any] = {"chunks": len(v), "noise": int(noise[lang]), "chars": int(w.sum())}
        if len(v):
            s |= {
                "mean": round(float(v.mean()), 4),
                "wmean": round(float((v * w).sum() / w.sum()), 4) if w.sum() else None,
                **{
                    k: round(float(np.percentile(v, q)), 3)
                    for k, q in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))
                },
                "histogram": _hist(v),
            }
        out[lang] = s
    return out


def compare_stages(stages: list[tuple[str, Path]]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{stage name: stage_stats}`` in the given order."""
    return {name: stage_stats(p) for name, p in stages}


def format_table(res: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    langs = list(dict.fromkeys(lg for st in res.values() for lg in st))
    lines = []
    for lang in langs:
        lines.append(f"{lang}:")
        lines.append(
            f"  {'stage':<11} {'chunks':>9} {'noise':>7} {'mean':>7} {'wmean':>7} "
            f"{'p50':>6} {'p90':>6} {'p99':>6} {'max':>6}"
        )
        for name, st in res.items():
            s = st.get(lang)
            if not s:
                continue
            if not s["chunks"]:
                lines.append(f"  {name:<11} {0:>9} {s['noise']:>7,}")
                continue
            lines.append(
                f"  {name:<11} {s['chunks']:>9,} {s['noise']:>7,} {s['mean']:>7.3f} "
                f"{s['wmean'] if s['wmean'] is not None else float('nan'):>7.3f} "
                f"{s['p50']:>6.2f} {s['p90']:>6.2f} {s['p99']:>6.2f} {s['max']:>6.2f}"
            )
    return lines


def plot_stages(res: dict[str, dict[str, dict[str, Any]]], path: Path) -> Path:
    """One panel per language: each stage's share of chunks per 0.25-bpc bin (log scale),
    overlaid as step lines, with its mean as a dashed line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#d97706", "#15803d", "#9333ea", "#dc2626"]
    ink, muted = "#1f1f1e", "#6b6a63"
    langs = list(dict.fromkeys(lg for st in res.values() for lg in st))
    fig, axes = plt.subplots(len(langs), 1, figsize=(9, 2.6 * len(langs)), squeeze=False)
    hi = BIN
    for ax, lang in zip(axes[:, 0], langs, strict=True):
        ns = len(res)
        low = 1.0
        for k, ((name, st), color) in enumerate(zip(res.items(), colors * 2, strict=False)):
            s = st.get(lang)
            if not s or not s["chunks"]:
                continue
            xs = sorted(float(k) for k in s["histogram"])
            n = s["chunks"]
            edges = [*xs, xs[-1] + BIN]
            ys = [s["histogram"][f"{x:.2f}"] / n for x in xs]
            # earlier stages thicker, so a later stage drawn on top of an equal one stays visible
            ax.stairs(
                ys,
                edges,
                color=color,
                linewidth=1.2 + 1.6 * (ns - 1 - k),
                label=f"{name} (mean {s['mean']:.2f})",
            )
            low = min(low, 1 / n)
            ax.axvline(s["mean"], color=color, linewidth=1, linestyle="--")
            hi = max(hi, edges[-1])
        ax.set_yscale("log")
        ax.set_ylim(min(low, 1e-3) / 2, 1.2)
        ax.set_xlim(0, min(hi, TOP + BIN))
        ax.set_title(lang if lang != ALL else "all languages", loc="left", fontsize=9, color=ink)
        ax.set_ylabel("share of chunks", fontsize=8, color=muted)
        ax.tick_params(labelsize=8, colors=muted)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.5)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("chunk bpc (bits per char; lower = cleaner)", fontsize=9, color=muted)
    fig.suptitle("Chunk bpc by stage", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
