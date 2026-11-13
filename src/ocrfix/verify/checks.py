"""Guardrail checks on one LLM correction: source (what the LLM saw) vs corrected text.

Each check returns ``Check(name, passed, reason)``. ``reason`` explains a failure (and is
informative on pass). The cheap text checks run first; ``scorer`` needs re-scored features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ocrfix.corpus import tag_spans
from ocrfix.corpus.lang import detect_lang, script_counts
from ocrfix.editdist import levenshtein
from ocrfix.schema import UNCERTAIN, Span

CHECKS = ("markers", "length_ratio", "edit_ratio", "language", "gap_fill", "scorer")
_HANGUL_FILL = "[가-힣]{{1,{n}}}"
CONTEXT = 8  # chars on each side of 〈?〉 used to locate the fill
_GAP_RUN = re.compile(f"(?:{re.escape(UNCERTAIN)})+")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "reason": self.reason}


def _blocks(text: str, spans: list[Span]) -> list[str]:
    return [text[s.start : s.end] for s in spans if s.kind != "prose"]


def check_markers(src: str, src_spans: list[Span], dst: str) -> Check:
    """Every math / table span of the source (⟨M#⟩ ⟨T#⟩ before masking) is in the
    correction verbatim, once, in order, and nothing new was tagged as math or a table."""
    want = _blocks(src, src_spans)
    got = _blocks(dst, tag_spans(dst))
    if got == want:
        return Check("markers", True, f"{len(want)} math/table spans intact")
    missing = [b for b in want if b not in got]
    extra = [b for b in got if b not in want]
    dup = sorted({b for b in got if b in want and got.count(b) > want.count(b)})
    if missing:
        reason = f"missing or altered: {_short(missing)}"
    elif dup:
        reason = f"duplicated: {_short(dup)}"
    elif extra:
        reason = f"new math/table: {_short(extra)}"
    else:
        reason = "math/table spans out of order"
    return Check("markers", False, reason)


def _short(items: list[str], n: int = 3, w: int = 30) -> str:
    return ", ".join(repr(x if len(x) <= w else x[:w] + "…") for x in items[:n])


def check_length(src: str, dst: str, lo: float, hi: float) -> Check:
    r = len(dst) / max(1, len(src))
    return Check("length_ratio", lo <= r <= hi, f"{r:.3f} (allowed {lo}–{hi})")


def check_edit(src: str, dst: str, cap: float) -> Check:
    r = levenshtein(src, dst) / max(1, len(src))
    return Check("edit_ratio", r <= cap, f"{r:.3f} (cap {cap})")


def _ko_share(text: str, spans: list[Span], hangul_weight: float) -> float:
    prose = "".join(text[s.start : s.end] for s in spans if s.kind == "prose")
    c = script_counts(prose, hangul_weight)
    total = sum(c.values())
    return c["ko"] / total if total else 0.0


def check_language(
    src: str, src_spans: list[Span], dst: str, max_shift: float, lang_opts: dict
) -> Check:
    dst_spans = tag_spans(dst)
    a, b = detect_lang(src, src_spans, **lang_opts), detect_lang(dst, dst_spans, **lang_opts)
    w = float(lang_opts.get("hangul_weight", 2.0))
    shift = abs(_ko_share(src, src_spans, w) - _ko_share(dst, dst_spans, w))
    if a != b:
        return Check("language", False, f"{a} -> {b}")
    return Check(
        "language", shift <= max_shift, f"{a}; Hangul share moved {shift:.3f} (cap {max_shift})"
    )


def check_gap_fill(src: str, dst: str, max_syllables: int) -> Check:
    """Each 〈?〉 of the source is kept, or filled with 1..max_syllables Hangul syllables.

    The fill is located by the source text around the 〈?〉 (up to CONTEXT chars each side,
    stopping at a neighbouring 〈?〉); if that context is gone, the check fails. Adjacent
    〈?〉〈?〉 count as one gap."""
    if UNCERTAIN not in src:
        return Check("gap_fill", True, "no 〈?〉 in source")
    # adjacent 〈?〉〈?〉 are one gap (one fill)
    src, dst = _GAP_RUN.sub(UNCERTAIN, src), _GAP_RUN.sub(UNCERTAIN, dst)
    fill_ok = re.compile(_HANGUL_FILL.format(n=max_syllables))
    parts = src.split(UNCERTAIN)
    pos, fills = 0, []
    for i in range(len(parts) - 1):
        left = parts[i][-CONTEXT:]
        right = parts[i + 1][:CONTEXT]
        lhs = re.escape(left) if left else (r"\A" if i == 0 else "")
        body = "(.*?)" if right else "(.*)"
        rhs = re.escape(right) if right else r"\Z"
        m = re.compile(lhs + body + rhs, re.DOTALL).search(dst, pos)
        if m is None:
            return Check("gap_fill", False, f"text around 〈?〉 #{i + 1} changed")
        fill = m.group(1)
        if fill != UNCERTAIN and not fill_ok.fullmatch(fill):
            return Check("gap_fill", False, f"〈?〉 #{i + 1} filled with {fill!r}")
        fills.append(fill)
        pos = m.start(1) + len(fill)
    kept = sum(f == UNCERTAIN for f in fills)
    return Check("gap_fill", True, f"{len(fills) - kept} filled, {kept} kept")


def check_scorer(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    min_bpc_gain: float,
    spike_ratio: float,
) -> Check:
    """bpc must drop by >= min_bpc_gain and spike_frac must be <= spike_ratio x before."""
    if before is None:
        return Check("scorer", False, "no score row for the source paragraph")
    if after is None:
        return Check("scorer", False, "corrected text was not scored")
    b_bpc, a_bpc = before.get("bpc"), after.get("bpc")
    b_sp, a_sp = before.get("spike_frac"), after.get("spike_frac")
    if b_bpc is None or a_bpc is None:
        return Check("scorer", True, "no prose to score")
    gain = b_bpc - a_bpc
    reasons = []
    if gain < min_bpc_gain:
        reasons.append(f"bpc {b_bpc:.4f} -> {a_bpc:.4f} (gain {gain:.4f} < {min_bpc_gain})")
    if b_sp is not None and a_sp is not None and a_sp > spike_ratio * b_sp + 1e-12:
        reasons.append(f"spike_frac {b_sp:.4f} -> {a_sp:.4f} (> {spike_ratio} x before)")
    if reasons:
        return Check("scorer", False, "; ".join(reasons))
    return Check("scorer", True, f"bpc gain {gain:.4f}, spike_frac {b_sp} -> {a_sp}")
