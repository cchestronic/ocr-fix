"""Synthetic OCR damage at measured rates (used by `ocrfix corrupt`, stages/corrupt.py).

Per (chunk, variant), with ``random.Random(sha256(seed, doc_id, chunk_id, variant))``:
  1. Draw a per-paragraph rate (ops per 1000 prose chars) from the paragraph's language
     histogram in ``error_table.json`` (bin by count, uniform inside it; the open last bin
     spans [a, 2a)). With ``corrupt.match_total`` the draw is scaled so its mean equals the
     table's ``total_per_1000``; ``corrupt.rate_scale`` multiplies on top.
  2. Number of ops = rate x prose chars / 1000, rounded stochastically.
  3. Each op's kind is drawn with weight = its measured rate (spacing types split by script,
     ``missing_space.ko`` etc.). A kind with no place in this paragraph is dropped from the
     draw for the rest of the paragraph.
Edits touch prose only and keep a gap from each other, so math/table spans are copied
verbatim and the errors aligner sees each edit separately. Spans of the output text are the
shifted clean spans, plus a math span for every hangul_garbled run wrapped in ``$...$``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from ocrfix.corrupt.ops import GENERATORS, Edit, Paragraph, Resources
from ocrfix.errors.align import SPACING
from ocrfix.schema import Span

ALL = "*"


@dataclass
class LangModel:
    """Everything sampled for one language: rate histogram, op weights, measured pairs."""

    bins: list[tuple[float, float, int]]  # (lo, hi, count)
    scale: float
    kinds: dict[str, float]  # "missing_space.ko" / "char_confusion" -> per-1000 rate
    resources: Resources

    def sample_rate(self, rng: random.Random) -> float:
        lo, hi, _ = rng.choices(self.bins, weights=[b[2] for b in self.bins])[0]
        return rng.uniform(lo, hi) * self.scale


def _parse_bin(label: str) -> tuple[float, float]:
    lo, hi = label.strip("[)").split(",")
    lo_f = float(lo)
    return lo_f, (2 * lo_f if hi == "inf" else float(hi))


def build_models(
    table: dict[str, Any], cmap: dict[str, Any], opts: dict[str, Any]
) -> dict[str, LangModel]:
    groups = {g["lang"]: g for g in table["groups"] if g["stratum"] == ALL}
    models = {}
    for lang, g in groups.items():
        bins = [(*_parse_bin(b), n) for b, n in g["histogram"].items() if n > 0] or [(0.0, 0.0, 1)]
        mean = sum((lo + hi) / 2 * n for lo, hi, n in bins) / sum(n for *_, n in bins)
        scale = float(opts["rate_scale"])
        if opts["match_total"] and mean > 0:
            scale *= g["total_per_1000"] / mean
        kinds: dict[str, float] = {}
        for t, rate in g["per_1000"].items():
            if t in SPACING:
                split = {k: v for k, v in g["spacing_per_1000"].items() if k.split(".")[0] == t}
                kinds.update(split or {f"{t}.other": rate})
            else:
                kinds[t] = rate
        conf = table["char_confusions"].get(lang) or table["char_confusions"].get(ALL, [])
        res = Resources(
            confusions=[(c["src"], c["gold"], c["count"]) for c in conf],
            garbled=cmap.get("hangul_garbled", []),
            opts=opts,
            header_phrases=list(opts["header_phrases"].get(lang, []))
            or [p for ps in opts["header_phrases"].values() for p in ps],
        )
        models[lang] = LangModel(bins, scale, kinds, res)
    return models


def rng_for(seed: int, doc_id: str, para_id: str, variant: int) -> random.Random:
    h = hashlib.sha256(f"{seed}\x00{doc_id}\x00{para_id}\x00{variant}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _round(x: float, rng: random.Random) -> int:
    n = math.floor(x)
    return n + (rng.random() < x - n)


def corrupt_text(
    text: str,
    spans: list[Span],
    model: LangModel,
    rng: random.Random,
    fixed_rate: float | None = None,
) -> tuple[str, list[Span], list[dict[str, Any]], float]:
    """``fixed_rate`` (ops per prose char, e.g. 0.01) replaces the histogram draw."""
    prose = [(s.start, s.end) for s in spans if s.kind == "prose"]
    n_chars = sum(e - s for s, e in prose)
    rate = model.sample_rate(rng) if fixed_rate is None else 1000 * fixed_rate
    n_ops = _round(rate * n_chars / 1000, rng)
    para = Paragraph(text, prose)
    kinds = {k: w for k, w in model.kinds.items() if w > 0}
    edits: list[Edit] = []
    while len(edits) < n_ops and kinds:
        kind = rng.choices(list(kinds), weights=list(kinds.values()))[0]
        base, _, script = kind.partition(".")
        edit = GENERATORS[base](rng, para, model.resources, script or None)
        if edit is None:
            del kinds[kind]
            continue
        para.take(edit)
        edits.append(edit)
    return (*apply(text, spans, edits), rate)


def apply(
    text: str, spans: list[Span], edits: list[Edit]
) -> tuple[str, list[Span], list[dict[str, Any]]]:
    edits = sorted(edits, key=lambda e: e.start)
    out: list[str] = []
    ops: list[dict[str, Any]] = []
    new_math: list[Span] = []
    pos = delta = 0
    shifts: list[tuple[int, int]] = []  # (gold position, cumulative delta after it)
    for e in edits:
        out.append(text[pos : e.start])
        s0 = e.start + delta
        out.append(e.src)
        if e.math:
            new_math.append(Span(s0 + e.math[0], s0 + e.math[1], "math"))
        op = {
            "type": e.type,
            "gold_start": e.start,
            "gold_end": e.end,
            "src_start": s0,
            "src_end": s0 + len(e.src),
            "src": e.src,
            "gold": text[e.start : e.end],
        }
        if e.script:
            op["script"] = e.script
        op.update(e.info)
        ops.append(op)
        delta += len(e.src) - (e.end - e.start)
        shifts.append((e.end, delta))
        pos = e.end
    out.append(text[pos:])
    source = "".join(out)

    def shift(i: int) -> int:
        d = 0
        for at, dd in shifts:
            if at <= i:
                d = dd
        return i + d

    fixed = [Span(shift(s.start), shift(s.end), s.kind) for s in spans if s.kind != "prose"]
    return source, _tile(sorted(fixed + new_math, key=lambda s: s.start), len(source)), ops


def _tile(spans: list[Span], n: int) -> list[Span]:
    out, pos = [], 0
    for s in spans:
        if s.start > pos:
            out.append(Span(pos, s.start, "prose"))
        out.append(s)
        pos = s.end
    if pos < n:
        out.append(Span(pos, n, "prose"))
    return out


def corrupt_opts(cfg) -> dict[str, Any]:
    c = cfg.corrupt
    return {
        "rate_scale": float(c.rate_scale),
        "match_total": bool(c.match_total),
        "line_width": tuple(c.line_width),
        "hyphen_space_prob": float(c.hyphen_space_prob),
        "ligature_glyph_prob": float(c.ligature_glyph_prob),
        "dup_word_prob": float(c.dup_word_prob),
        "garbled_wrap_prob": float(c.garbled_wrap_prob),
        "page_number_prob": float(c.page_number_prob),
        "header_phrases": {k: list(v) for k, v in c.header_phrases.items()},
        "junk": list(c.junk),
    }
