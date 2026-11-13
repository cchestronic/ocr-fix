"""`ocrfix calibrate-noise-chunk <gold_chunks> <scored>`: the whole-chunk noise limit per language.

`score` drops a chunk whose bpc (after word-level noise removal) is above
``calibration.noise_chunk[lang]`` (``"*"`` for other languages). This command picks that limit
from real data:

1. **Gold** (`gold chunks` sft/ and dpo/, ``gold_status: ok``; never eval/): the OCR side of
   every chunk is scored with the base model, noise removal off, as `score` does. A gold chunk
   whose annotator deleted at least ``calibrate.noise_chunk.deleted_share`` of its non-space
   chars (page numbers, junk; `calibrate.noise.noise_labels`) is a **noise chunk**; every other
   one is **real text**. limit = the highest bpc of real-text gold chunks of the language +
   ``calibrate.noise_chunk.margin``. A language with fewer than ``calibrate.min_rows`` real
   gold chunks uses ``"*"`` (all languages together).
2. **Corpus** (the first `score` pass, run with ``score.noise.remove=false
   score.noise.whole_chunks=false``): streamed once. Per language: a bpc histogram, how many
   chunks and chars each candidate limit (gold max + ``candidates``) would drop, and up to
   ``samples`` random chunks in each 0.5-bpc band from 1.0 below the limit to 4.0 above it,
   to read by hand. First-pass bpc is before word-level removal, so it is never lower than the
   bpc `score` later compares: a chunk under the limit here stays under it.
3. **Write** (``--write``): ``noise_chunk`` in the calibration file (`score.calibration`) is
   replaced, the rest kept. Edit the file by hand to use another value.

Outputs in ``out_dir``: ``noise_chunk_report.json``, ``noise_chunk_samples.md`` and
``noise_chunk_hist.png`` (corpus and gold bpc histograms per language, the limit marked).
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrfix.calibrate.noise import REAL, noise_labels
from ocrfix.calibrate.rules import load_calibration, save_calibration
from ocrfix.io import iter_jsonl
from ocrfix.stages.run import Context
from ocrfix.stages.score import tokenize_chunk

ALL = "*"
LANGS = ("en", "ko", "mixed")  # own limits when the gold has enough rows; the rest use "*"
BIN = 0.25  # histogram bin width (bits per char)
TOP = 16.0  # histogram top (higher values go in the last bin)
BAND = 0.5  # sample band width
BAND_LO, BAND_HI = -1.0, 4.0  # sample bands, relative to the limit


def _gold_rows(gold_dir: Path, limit: int | None) -> list[dict]:
    rows = [r for side in ("sft", "dpo") if (gold_dir / side).is_dir()
            for r in iter_jsonl(gold_dir / side) if r.get("gold_status") == "ok"]  # fmt: skip
    rows = rows[:limit] if limit else rows
    if not rows:
        raise ValueError(f"{gold_dir}: no `gold chunks` rows with gold_status ok in sft/ or dpo/")
    return rows


def deleted_share(ocr: str, gold: str) -> float:
    """Share of the OCR chunk's non-space chars the annotator deleted."""
    lab = noise_labels(ocr, gold)
    keep = np.array([not c.isspace() for c in ocr], dtype=bool)
    n = int(keep.sum())
    return float(((lab == REAL) & keep).sum() / n) if n else 0.0


def score_gold(cfg, rows: list[dict], transport=None) -> list[dict | None]:
    """Score features of each row's OCR text (``source_text``), as `score` computes them."""

    async def run() -> list[dict | None]:
        async with Context(cfg, transport) as ctx:
            tps = [tokenize_chunk(r["source_text"], ctx.encode, ctx.boundary) for r in rows]
            bits = await ctx.scorer.bits_batch([tp.ids for tp in tps])
            return [None if isinstance(b, Exception) else ctx.feats(tp, b)
                    for tp, b in zip(tps, bits, strict=True)]  # fmt: skip

    return asyncio.run(run())


def gold_limits(gold: list[dict], margin: float, min_rows: int) -> dict[str, dict[str, Any]]:
    """Per language (and "*"): the real-text bpc maximum and the limit = max + margin; gold noise
    chunks: how many there are and how many the limit catches."""
    out: dict[str, dict[str, Any]] = {}
    for lang in (ALL, *LANGS):
        sel = [g for g in gold if lang == ALL or g["lang"] == lang]
        real = np.array([g["bpc"] for g in sel if not g["noise"]], dtype=float)
        noise = np.array([g["bpc"] for g in sel if g["noise"]], dtype=float)
        if lang != ALL and len(real) < min_rows:
            continue
        if not len(real):
            raise ValueError("no real-text gold chunks with a bpc")
        top = float(real.max())
        limit = round(top + margin, 4)
        out[lang] = {
            "real_chunks": len(real), "noise_chunks": len(noise),
            "real_p50": round(float(np.median(real)), 4),
            "real_p99": round(float(np.percentile(real, 99)), 4),
            "real_max": round(top, 4), "limit": limit,
            "noise_caught": int((noise > limit).sum()),
            "noise_bpc": sorted(round(float(x), 3) for x in noise),
            "real_histogram": _hist(real),
        }  # fmt: skip
    return out


def _hist(values: np.ndarray) -> dict[str, int]:
    """Counts per ``BIN``-wide bin (key = the bin's lower edge), the top bin open."""
    idx = np.minimum((values / BIN).astype(int), int(TOP / BIN))
    keys, counts = np.unique(idx, return_counts=True)
    return {f"{i * BIN:.2f}": int(n) for i, n in zip(keys, counts, strict=True)}


def plot_histograms(report: dict[str, Any], path: Path) -> Path:
    """One panel per language: corpus chunks (bars) and real-text gold chunks (line) by bpc,
    as a share of each set (log scale, so the few high chunks show), gold noise chunks as
    marks, the gold maximum and the limit as vertical lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"
    langs = [g for g in report["gold"] if report["corpus"][g]["chunks"] or g != ALL]
    fig, axes = plt.subplots(len(langs), 1, figsize=(9, 2.6 * len(langs)), squeeze=False)
    for ax, lang in zip(axes[:, 0], langs, strict=True):
        g, c = report["gold"][lang], report["corpus"][lang]
        n = c["chunks"]
        if n:
            x = [float(k) for k in c["histogram"]]
            y = [v / n for v in c["histogram"].values()]
            ax.bar(x, y, width=BIN * 0.9, align="edge", color=blue,
                   label=f"corpus ({n:,} chunks)")  # fmt: skip
        gh = g["real_histogram"]
        xs = np.arange(0, TOP + BIN, BIN)
        gy = np.array([gh.get(f"{v:.2f}", 0) for v in xs]) / max(g["real_chunks"], 1)
        ax.step(xs, np.where(gy > 0, gy, np.nan), where="post", color=orange, linewidth=2,
                label=f"gold real text ({g['real_chunks']:,})")  # fmt: skip
        if g["noise_bpc"]:
            ax.plot(g["noise_bpc"], [0.5] * len(g["noise_bpc"]), "x", color=ink, markersize=8,
                    label=f"gold noise chunks ({g['noise_chunks']})")  # fmt: skip
        ax.axvline(g["real_max"], color=muted, linestyle="--", linewidth=1)
        ax.axvline(g["limit"], color=ink, linewidth=1.5)
        dropped = next(d for d in c["drop"] if d["limit"] == g["limit"])["chunks"]
        top = ax.get_xaxis_transform()  # labels just above the plot, clear of the bars
        ax.text(g["real_max"], 1.01, f"gold max {g['real_max']:.2f} ", color=muted, fontsize=8,
                ha="right", va="bottom", transform=top)  # fmt: skip
        ax.text(g["limit"], 1.01, f" limit {g['limit']:.2f}: {dropped:,} dropped", color=ink,
                fontsize=8, ha="left", va="bottom", transform=top)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(min(1 / max(n, g["real_chunks"], 1), 1e-3) / 2, 1)
        hi = max([g["limit"] + 2, *(float(k) + BIN for k in c["histogram"])], default=6)
        ax.set_xlim(0, min(hi, TOP + BIN))
        ax.set_title(f"{lang}", loc="left", fontsize=10, color=ink)
        ax.set_ylabel("share of chunks", fontsize=8, color=muted)
        ax.tick_params(labelsize=8, colors=muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.8))
    axes[-1, 0].set_xlabel("chunk bpc (bits per char)", fontsize=9, color=muted)
    fig.suptitle("Chunk bpc: corpus (first pass) vs gold", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


class Corpus:
    """One pass over the first-pass scores: histogram, drops per candidate, samples per band."""

    def __init__(self, limits: dict[str, float], candidates: dict[str, list[float]], k: int,
                 seed: int) -> None:  # fmt: skip
        self.limits, self.candidates, self.k = limits, candidates, k
        self.rng = random.Random(seed)
        self.n: dict[str, int] = defaultdict(int)
        self.chars: dict[str, int] = defaultdict(int)
        self.hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(int(TOP / BIN) + 1, int))
        self.drop = {lang: {c: [0, 0] for c in cs} for lang, cs in candidates.items()}
        self.seen: dict[tuple[str, int], int] = defaultdict(int)
        self.samples: dict[tuple[str, int], list[dict]] = defaultdict(list)
        self.skipped = 0

    def key(self, lang: str | None) -> str:
        return lang if lang in self.limits else ALL

    def add(self, r: dict) -> None:
        f = r.get("score") or {}
        b = f.get("bpc")
        if r.get("status") != "scored" or b is None or f.get("low_confidence"):
            self.skipped += 1
            return
        lang = self.key(r.get("lang"))
        text = r.get("text") or r.get("source_text") or ""
        self.n[lang] += 1
        self.chars[lang] += len(text)
        self.hist[lang][min(int(b / BIN), len(self.hist[lang]) - 1)] += 1
        for c, d in self.drop[lang].items():
            if b > c:
                d[0] += 1
                d[1] += len(text)
        rel = b - self.limits[lang]
        if rel >= BAND_LO:  # the last band is open
            band = min(math.floor((rel - BAND_LO) / BAND), int((BAND_HI - BAND_LO) / BAND))
            key = (lang, band)
            self.seen[key] += 1
            s = {"chunk_id": r.get("chunk_id"), "lang": r.get("lang"), "bpc": round(b, 3),
                 "text": text}  # fmt: skip
            if len(self.samples[key]) < self.k:
                self.samples[key].append(s)
            else:  # reservoir: every chunk in the band equally likely
                j = self.rng.randrange(self.seen[key])
                if j < self.k:
                    self.samples[key][j] = s

    def band_range(self, lang: str, band: int) -> tuple[float, float | None]:
        lo = self.limits[lang] + BAND_LO + band * BAND
        last = band == int((BAND_HI - BAND_LO) / BAND)
        return round(lo, 3), None if last else round(lo + BAND, 3)

    def report(self) -> dict[str, Any]:
        out = {}
        for lang in self.limits:
            n, ch = self.n.get(lang, 0), self.chars.get(lang, 0)
            h = self.hist[lang]
            out[lang] = {
                "chunks": n, "chars": ch,
                "histogram": {f"{i * BIN:.2f}": int(v) for i, v in enumerate(h) if v},
                "drop": [{"limit": c, "chunks": d[0], "chars": d[1],
                          "chunk_share": round(d[0] / n, 6) if n else None,
                          "char_share": round(d[1] / ch, 6) if ch else None}
                         for c, d in self.drop[lang].items()],
                "bands": [{"from": self.band_range(lang, b)[0], "to": self.band_range(lang, b)[1],
                           "chunks": self.seen[(lang, b)]}
                          for b in range(int((BAND_HI - BAND_LO) / BAND) + 1)],
            }  # fmt: skip
        return out


def samples_md(c: Corpus, gold: dict[str, dict], chars: int = 600) -> str:
    lines = ["# Whole-chunk noise: corpus samples by bpc", "",
             "Read each band and find where the chunks stop being text. Chunks above the limit "
             "are dropped (their prose removed) by `score`. `⟨M#⟩ ⟨T#⟩ ⟨P#⟩` are masked "
             "math, tables and links.", ""]  # fmt: skip
    for lang in c.limits:
        g = gold[lang]
        lines += [f"## {lang}: limit {c.limits[lang]} (gold real-text max {g['real_max']} + "
                  f"margin)", ""]  # fmt: skip
        for b in range(int((BAND_HI - BAND_LO) / BAND) + 1):
            lo, hi = c.band_range(lang, b)
            n = c.seen[(lang, b)]
            if not n:
                continue
            side = "kept" if hi is not None and hi <= c.limits[lang] else "dropped"
            rng = f"{lo} - {hi}" if hi is not None else f">= {lo}"
            lines += [f"### bpc {rng} ({side}): {n} chunks", ""]
            for s in sorted(c.samples[(lang, b)], key=lambda s: s["bpc"]):
                t = s["text"][:chars].replace("\n", " ⏎ ")
                more = " …" if len(s["text"]) > chars else ""
                lines += [f"- **{s['bpc']}** `{s['chunk_id']}` ({s['lang']}): {t}{more}"]
            lines.append("")
    return "\n".join(lines) + "\n"


def run_calibrate_noise_chunk(cfg, gold_dir: Path, scored: Path, out_dir: Path, *,
                              calibration: Path | None, transport=None,
                              limit: int | None = None) -> dict[str, Any]:  # fmt: skip
    """``calibration``: the file whose ``noise_chunk`` is replaced (None: report only)."""
    o = cfg.calibrate.noise_chunk
    t0 = time.perf_counter()
    rows = _gold_rows(gold_dir, limit)
    feats = score_gold(cfg, rows, transport)
    gold = [{"chunk_id": r["chunk_id"], "lang": r.get("lang"), "bpc": f["bpc"],
             "noise": deleted_share(r["source_text"], r["gold_text"]) >= float(o.deleted_share)}
            for r, f in zip(rows, feats, strict=True)
            if f and f.get("bpc") is not None and not f.get("low_confidence")]  # fmt: skip
    g = gold_limits(gold, float(o.margin), int(cfg.calibrate.min_rows))
    limits = {lang: v["limit"] for lang, v in g.items()}
    cands = {lang: sorted({round(v["real_max"] + float(m), 4) for m in o.candidates} | {v["limit"]})
             for lang, v in g.items()}  # fmt: skip
    c = Corpus(limits, cands, int(o.samples), int(cfg.seed))
    for r in iter_jsonl(scored):
        c.add(r)
    report = {
        "model": str(cfg.model), "margin": float(o.margin),
        "deleted_share": float(o.deleted_share), "limits": limits, "gold": g,
        "gold_scored": len(gold), "gold_rows": len(rows),
        "corpus": c.report(), "corpus_skipped": c.skipped,
        "seconds": round(time.perf_counter() - t0, 1),
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noise_chunk_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    )
    (out_dir / "noise_chunk_samples.md").write_text(samples_md(c, g), encoding="utf-8")
    plot_histograms(report, out_dir / "noise_chunk_hist.png")
    if calibration is not None:
        cal = load_calibration(calibration)
        cal.noise_chunk = dict(limits)
        save_calibration(cal, calibration)
        report["written"] = str(calibration)
    return report


def bpc_histograms(scored: list[Path], limits: dict[str, float] | None = None) -> dict[str, Any]:
    """Chunk bpc per language (other languages under their own name) from `score` output:
    0.25-bin counts, p50 / p99 / p99.9 / max, and with ``limits`` (noise_chunk) the limit and
    the chunks above it. No model needed."""
    vals: dict[str, list[float]] = defaultdict(list)
    for path in scored:
        for r in iter_jsonl(path):
            f = r.get("score") or {}
            if (
                r.get("status") == "scored"
                and f.get("bpc") is not None
                and not f.get("low_confidence")
            ):
                vals[r.get("lang") or "?"].append(float(f["bpc"]))
    out = {}
    for lang in sorted(vals, key=lambda k: -len(vals[k])):
        v = np.array(vals[lang])
        qs = {"p50": 50, "p99": 99, "p99.9": 99.9, "max": 100}
        stats = {k: round(float(np.percentile(v, q)), 3) for k, q in qs.items()}
        lim = (limits or {}).get(lang, (limits or {}).get(ALL))
        above = None if lim is None else int((v > lim).sum())
        out[lang] = {"chunks": len(v), "histogram": _hist(v), **stats, "limit": lim,
                     "above": above}  # fmt: skip
    return out


def plot_bpc(hists: dict[str, Any], path: Path, title: str) -> Path:
    """One panel per language: share of chunks per 0.25-bpc bin (log scale), and the
    noise-chunk limit with the chunks above it (``bpc_histograms(..., limits)``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted = "#2a78d6", "#1f1f1e", "#6b6a63"
    langs = list(hists)
    fig, axes = plt.subplots(len(langs), 1, figsize=(9, 2.3 * len(langs)), squeeze=False)
    for ax, lang in zip(axes[:, 0], langs, strict=True):
        h = hists[lang]
        n = h["chunks"]
        x = [float(k) for k in h["histogram"]]
        ax.bar(x, [v / n for v in h["histogram"].values()], width=BIN * 0.9, align="edge",
               color=blue)  # fmt: skip
        lim = h["limit"]
        hi = max(x) + BIN
        if lim is not None:
            ax.axvline(lim, color=ink, linewidth=1.5)
            ax.text(lim, 1.01, f" limit {lim:.2f}: {h['above']:,} above", color=ink, fontsize=8,
                    ha="left", va="bottom", transform=ax.get_xaxis_transform())  # fmt: skip
            hi = max(hi, lim + 1)
        ax.set_yscale("log")
        ax.set_ylim(min(1 / n, 1e-3) / 2, 1)
        ax.set_xlim(0, min(hi, TOP + BIN))
        ax.set_title(f"{lang}: {n:,} chunks, p50 {h['p50']}, p99 {h['p99']}, max {h['max']}",
                     loc="left", fontsize=9, color=ink)  # fmt: skip
        ax.set_ylabel("share of chunks", fontsize=8, color=muted)
        ax.tick_params(labelsize=8, colors=muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel("chunk bpc (bits per char)", fontsize=9, color=muted)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
