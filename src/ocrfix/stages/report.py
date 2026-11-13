"""`ocrfix report`: corpus quality before and after correction, from one output folder.

Every chunk carries three scores of the same base model:
  raw        before noise removal: the ``denoise`` history entry's ``score_before``;
  denoised   before correction: the first ``correct`` entry's ``score_before``;
  final      the current ``score`` (an accepted correction's score).
A missing stage falls back to the next one (no noise found -> raw = denoised; not sent to
the corrector -> denoised = final), so `score` output works too (final = denoised).

Per language (the chunk's ``lang``) and overall, for each stage:
  bpc      bits per char over all chunks: sum(bpc x n_chars) / sum(n_chars), so long chunks
           weigh more and the number is the corpus's, not a mean of chunk rates;
  flagged  share of scored chunks the calibration flags (low_confidence never is), applied
           anew to every stage's score.
With a clean reference (`score` output of clean text, ideally the same domain,
scored with ``score.noise.remove=false``), the excess over clean text, and the share of the
raw excess that noise removal and correction each closed:
  closed_by_denoise = (raw - denoised) / (raw - clean), closed_by_correct = (denoised - final)
  / (raw - clean).
Also: chunks, chars removed as noise, chunks sent to the corrector and accepted.

These are the scoring model's view. On the test corpus they matched the size of the real
gain (document CER 0.58% -> 0.23% after noise removal, -60%; bpc excess closed 44%, flags
above the clean rate -80%), but real damage can differ: check on gold chunks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ocrfix.calibrate.rules import Calibration
from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.io import iter_jsonl

STAGES = ("raw", "denoised", "final")
SENT = ("accepted", "rejected", "unchanged", "error")


def stage_scores(r: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """{raw, denoised, final} score dicts of one chunk (None where it has none)."""
    hist = r.get("history") or []
    final = r.get("score")
    corr = next((h for h in hist if h.get("stage") == "correct" and "score_before" in h), None)
    denoised = corr["score_before"] if corr else final
    dn = next((h for h in hist if h.get("stage") == "denoise"), None)
    raw = dn["score_before"] if dn else denoised
    return {"raw": raw, "denoised": denoised, "final": final}


class _Acc:
    def __init__(self) -> None:
        self.chunks = 0
        self.bits = dict.fromkeys(STAGES, 0.0)
        self.chars = dict.fromkeys(STAGES, 0)
        self.scored = dict.fromkeys(STAGES, 0)
        self.flagged = dict.fromkeys(STAGES, 0)
        self.noise_chars = 0
        self.noise_chunks = 0
        self.whole_chunks = 0  # dropped as noise as a whole
        self.whole_chars = 0
        self.sent = 0
        self.accepted = 0

    def add(self, r: dict[str, Any], cal: Calibration | None, lang: str) -> None:
        self.chunks += 1
        whole = any(h.get("stage") == "noise_chunk" for h in r.get("history") or [])
        if whole:
            self.whole_chunks += 1
            self.whole_chars += sum(not c.isspace() for c in PLACEHOLDER.sub("", r["source_text"]))
        for stage, s in stage_scores(r).items():
            if not s or (whole and stage == "final"):  # dropped: not in the final corpus
                continue
            self.scored[stage] += 1
            if cal is not None and cal.flag(s, lang):
                self.flagged[stage] += 1
            if s.get("bpc") is not None and s.get("n_chars"):
                self.bits[stage] += float(s["bpc"]) * int(s["n_chars"])
                self.chars[stage] += int(s["n_chars"])
        removed = [
            x for h in r.get("history") or [] if h.get("stage") == "denoise"
            for x in h.get("removed", [])
        ]  # fmt: skip
        if removed:
            self.noise_chunks += 1
            self.noise_chars += sum(sum(not c.isspace() for c in x["text"]) for x in removed)
        if r.get("status") in SENT:
            self.sent += 1
            self.accepted += r["status"] == "accepted"

    def to_dict(self, has_cal: bool) -> dict[str, Any]:
        d: dict[str, Any] = {"chunks": self.chunks}
        for st in STAGES:
            d[st] = {
                "bpc": round(self.bits[st] / self.chars[st], 4) if self.chars[st] else None,
                "chars": self.chars[st],
                "scored": self.scored[st],
                "flagged": round(self.flagged[st] / self.scored[st], 4)
                if has_cal and self.scored[st]
                else None,
            }
        d.update(noise_chunks=self.noise_chunks, noise_chars=self.noise_chars,
                 whole_noise_chunks=self.whole_chunks, whole_noise_chars=self.whole_chars,
                 sent=self.sent, accepted=self.accepted)  # fmt: skip
        return d


def _accumulate(path: Path, cal: Calibration | None, reference: bool = False) -> dict:
    acc: dict[str, _Acc] = defaultdict(_Acc)
    for r in iter_jsonl(path):
        lang = r.get("lang") or "other"
        if reference:  # clean text: its current score only
            r = {"score": r.get("score"), "lang": lang}
        acc[lang].add(r, cal, lang)
        acc["all"].add(r, cal, lang)
    return {k: v.to_dict(cal is not None) for k, v in sorted(acc.items())}


def corpus_report(
    path: Path, cal: Calibration | None = None, clean: Path | None = None
) -> dict[str, Any]:
    """Per language and "all": the three stages, noise and corrector counts, and (with a
    clean reference) its bpc and flag rate and the share of the excess each step closed."""
    langs = _accumulate(path, cal)
    ref = _accumulate(clean, cal, reference=True) if clean else {}
    for lang, d in langs.items():
        c = ref.get(lang)
        if not c:
            continue
        cb = c["final"]["bpc"]
        d["clean"] = {"bpc": cb, "flagged": c["final"]["flagged"], "chunks": c["chunks"]}
        raw, den, fin = (d[s]["bpc"] for s in STAGES)
        if None in (cb, raw, den, fin):
            continue
        for s in STAGES:
            d[s]["excess"] = round(d[s]["bpc"] - cb, 4)
        gap = raw - cb
        if gap > 0:
            d["closed_by_denoise"] = round((raw - den) / gap, 4)
            d["closed_by_correct"] = round((den - fin) / gap, 4)
    return {"calibration": cal is not None, "clean": str(clean) if clean else None,
            "langs": langs}  # fmt: skip


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def format_report(rep: dict[str, Any]) -> str:
    langs = rep["langs"]
    clean = rep["clean"] is not None
    head = ["lang", "chunks", "bpc raw", "denoised", "final"]
    if clean:
        head += ["clean", "closed: denoise", "correct"]
    if rep["calibration"]:
        head += ["flagged raw", "denoised", "final"]
        if clean:
            head.append("clean")
    rows = []
    for lang, d in langs.items():
        row = [lang, str(d["chunks"]), *(_num(d[s]["bpc"]) for s in STAGES)]
        if clean:
            c = d.get("clean") or {}
            row += [_num(c.get("bpc")), _pct(d.get("closed_by_denoise")),
                    _pct(d.get("closed_by_correct"))]  # fmt: skip
        if rep["calibration"]:
            row += [_pct(d[s]["flagged"]) for s in STAGES]
            if clean:
                row.append(_pct((d.get("clean") or {}).get("flagged")))
        rows.append(row)
    widths = [max(len(x) for x in col) for col in zip(head, *rows, strict=True)]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    lines = [fmt.format(*head), *(fmt.format(*r) for r in rows), ""]
    for lang, d in langs.items():
        acc = f"{100 * d['accepted'] / d['sent']:.1f}%" if d["sent"] else "-"
        lines.append(
            f"{lang:<6} noise removed in {d['noise_chunks']} chunks ({d['noise_chars']} chars), "
            f"{d['whole_noise_chunks']} chunks dropped whole ({d['whole_noise_chars']} chars); "
            f"sent to the corrector {d['sent']}, accepted {d['accepted']} ({acc})"
        )
    if not rep["calibration"]:
        lines.append("no calibration: flag rates left out (score.calibration or --calibration)")
    return "\n".join(lines)
