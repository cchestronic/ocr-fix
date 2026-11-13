"""`ocrfix verify`: accept or reject each LLM correction; `ocrfix verify-summary`.

Candidates are rows of the `correct` output that the LLM changed (``llm`` in ``fixer``,
no ``llm_error``, text differs from ``llm_source_text``). For each, every check in
``checks.CHECKS`` runs; ``scorer`` compares ``scores_before`` (the scores of the text the
LLM saw) with the corrected text re-scored in the same window context: the rows in file
order, as `score` saw them, with the corrections substituted (``InProcessScorer.rescore``,
or any callable ``(source records, {key: corrected text}) -> {key: features}``). With
``verify.rescore_failed: false`` a candidate that already failed a text check is not
re-scored (``scorer`` = skipped, counted as failed).

Every input row is written to ``out``: candidates get ``text`` = corrected if accepted, else
the source, plus ``original`` (source), ``corrected``, ``accepted``, ``checks``,
``failed_checks`` ([{check, reason}]), ``before`` / ``after`` features; other rows pass
through unchanged.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Any

from ocrfix.corpus import tag_spans
from ocrfix.io import iter_jsonl, iter_records, open_text
from ocrfix.schema import Record, nfc
from ocrfix.verify.checks import (
    CHECKS,
    Check,
    check_edit,
    check_gap_fill,
    check_language,
    check_length,
    check_markers,
    check_scorer,
)

Key = tuple[str, str]
Rescorer = Callable[[list[Record], dict[Key, str]], dict[Key, dict[str, Any]]]
FEATURE_KEYS = ("bpc", "spike_frac", "p90", "p99", "mean_bits", "n_prose_tokens")


@dataclass
class VerifyStats:
    rows: int = 0
    candidates: int = 0
    accepted: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    by_lang: dict[str, Counter[str]] = field(default_factory=dict)


def is_candidate(rec: Record) -> bool:
    src = rec.extra.get("llm_source_text")
    return (
        "llm" in rec.extra.get("fixer", [])
        and not rec.extra.get("llm_error")
        and src is not None
        and nfc(src) != rec.text
    )


def text_checks(src: str, dst: str, v: dict[str, Any], lang_opts: dict[str, Any]) -> list[Check]:
    src_spans = tag_spans(src)
    lo, hi = v["length_ratio"]
    return [
        check_markers(src, src_spans, dst),
        check_length(src, dst, float(lo), float(hi)),
        check_edit(src, dst, float(v["max_edit_ratio"])),
        check_language(src, src_spans, dst, float(v["max_script_shift"]), lang_opts),
        check_gap_fill(src, dst, int(v["gap_fill_max_syllables"])),
    ]


def _features(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {k: d.get(k) for k in FEATURE_KEYS}


def verify(
    corrections: Path,
    scores_before: Path,
    out: Path,
    *,
    rescorer: Rescorer,
    v: dict[str, Any],
    lang_opts: dict[str, Any],
    limit: int | None = None,
) -> VerifyStats:
    stats = VerifyStats()
    before = {(str(d["doc_id"]), str(d["para_id"])): d for d in iter_jsonl(scores_before)}

    rows = list(islice(iter_records(corrections), limit))
    stats.rows = len(rows)
    pending: dict[tuple[str, str], list[Check]] = {}
    for rec in rows:
        if is_candidate(rec):
            src = nfc(rec.extra["llm_source_text"])
            pending[rec.key] = text_checks(src, rec.text, v, lang_opts)

    corrected = {
        rec.key: rec.text for rec in rows if rec.key in pending
        and (v["rescore_failed"] or all(c.passed for c in pending[rec.key]))
    }  # fmt: skip
    # every row as the scorer saw it (the source text), so windows match `score`
    sources = [
        replace(rec, text=src, spans=tag_spans(src))
        if rec.key in pending and (src := nfc(rec.extra["llm_source_text"])) != rec.text
        else rec
        for rec in rows
    ]
    after = rescorer(sources, corrected) if corrected else {}

    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_text(tmp, "wt") as f:
        for rec in rows:
            if rec.key in pending:
                rec = _verdict(
                    rec, pending[rec.key], before.get(rec.key), after.get(rec.key), v, stats
                )
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(out)
    return stats


def _verdict(rec, checks, before, after, v, stats) -> Record:
    if after is None and not v["rescore_failed"] and not all(c.passed for c in checks):
        checks = [*checks, Check("scorer", False, "skipped: a text check already failed")]
    else:
        checks = [*checks, check_scorer(
            before, after, min_bpc_gain=float(v["min_bpc_gain"]),
            spike_ratio=float(v["spike_ratio"]),
        )]  # fmt: skip
    failed = [c for c in checks if not c.passed]
    accepted = not failed
    src = nfc(rec.extra["llm_source_text"])
    stats.candidates += 1
    stats.accepted += accepted
    lang = stats.by_lang.setdefault(rec.lang, Counter())
    lang["candidates"] += 1
    lang["accepted"] += accepted
    for c in failed:
        stats.failures[c.name] += 1
    extra = rec.extra | {
        "original": src,
        "corrected": rec.text,
        "accepted": accepted,
        "checks": {c.name: c.to_dict() for c in checks},
        "failed_checks": [{"check": c.name, "reason": c.reason} for c in failed],
        "before": _features(before),
        "after": _features(after),
    }
    text = rec.text if accepted else src
    if not accepted:
        extra["fixer"] = [x for x in extra.get("fixer", []) if x != "llm"]
    return replace(rec, text=text, spans=tag_spans(text), extra=extra)


def summarize(path: Path) -> dict[str, Any]:
    n = acc = 0
    reasons: Counter[str] = Counter()
    by_lang: dict[str, Counter[str]] = {}
    for d in iter_jsonl(path):
        if "accepted" not in d:
            continue
        n += 1
        acc += bool(d["accepted"])
        lc = by_lang.setdefault(d.get("lang") or "other", Counter())
        lc["candidates"] += 1
        lc["accepted"] += bool(d["accepted"])
        for f in d.get("failed_checks", []):
            reasons[f["check"]] += 1
            lc[f["check"]] += 1
    return {"candidates": n, "accepted": acc, "reasons": reasons, "by_lang": by_lang}


def format_summary(s: dict[str, Any], width: int = 40) -> str:
    n, acc = s["candidates"], s["accepted"]
    lines = [
        f"{n} corrections verified, {acc} accepted ({100 * acc / n:.1f}%)"
        if n
        else "no corrections"
    ]
    for lang, c in sorted(s["by_lang"].items()):
        lines.append(f"  {lang:<6} {c['accepted']}/{c['candidates']} accepted")
    if s["reasons"]:
        top = max(s["reasons"].values())
        lines.append("failure reasons (a correction can fail several checks):")
        for name in CHECKS:
            k = s["reasons"].get(name, 0)
            if k:
                bar = "#" * max(1, round(width * k / top))
                lines.append(f"  {name:<13} {k:>6} {100 * k / n:5.1f}% {bar}")
    return "\n".join(lines)


def run_verify(
    cfg, corrections: Path, scores_before: Path, out: Path, limit=None, rescorer=None
) -> VerifyStats:
    from omegaconf import OmegaConf

    v = OmegaConf.to_container(cfg.verify, resolve=True)
    lang_opts = OmegaConf.to_container(cfg.corpus.lang, resolve=True)
    assert isinstance(v, dict) and isinstance(lang_opts, dict)
    if rescorer is not None:
        return verify(
            corrections,
            scores_before,
            out,
            rescorer=rescorer,
            v=v,
            lang_opts=lang_opts,
            limit=limit,
        )
    from ocrfix.scorer.run import InProcessScorer

    with InProcessScorer(cfg) as scorer:
        return verify(
            corrections, scores_before, out, rescorer=scorer.rescore, v=v, lang_opts=lang_opts,
            limit=limit,
        )  # fmt: skip
