"""`ocrfix calibrate`: choose tau and per-language flag rules from clean vs corrupted scores.

Inputs are `ocrfix score` outputs, or chunk shards from `ocrfix refine score` (features under
``score``), with ``lang``; corrupted rows also carry ``corruption`` and ``error_rate`` (from
`ocrfix corrupt` / `ocrfix refine corrupt`):

  clean_scores       the clean paragraphs
  corrupted_scores   the same paragraphs corrupted, e.g. ``--rate 0.01``, ``0.03``, ``0.1``
                     and table-sampled rates, concatenated
  gold_scores        optional: gold source paragraphs (``ocrfix gold records``) with ``cer``

Steps:
  0. Rows with ``low_confidence`` (too few scored tokens) are left out: the rules never flag
     them, and their noisy rates would only push thresholds up.
  1. For every feature (bpc, p90, p99, spike_frac and spike_frac@tau for each candidate):
     AUROC clean vs corrupted per language (and per corruption label), and Spearman
     correlation with the corruption rate (clean = 0).
  2. tau = the spike_frac column with the highest pooled AUROC.
  3. Per language with enough rows, else ``*``: ``fit.fit_rule`` on the recall set
     (``calibrate.recall_on`` labels) under ``calibrate.max_false_flag``: the rule
     ``spike_frac > A or p90 > B or bpc > C``, any part possibly off.
  4. With gold scores: precision / recall of the rules on real OCR source paragraphs.
Writes calibration.yaml, report.json, summary.md and PNGs to ``out_dir``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ocrfix.calibrate.fit import auroc, fit_rule, flags, precision_recall, spearman
from ocrfix.calibrate.rules import Calibration, save_calibration
from ocrfix.io import iter_jsonl
from ocrfix.scorer.features import spike_key

log = logging.getLogger(__name__)

ALL = "*"
CLEAN = "clean"
BASE_FEATURES = ("bpc", "p90", "p99", "spike_frac")


@dataclass
class Scores:
    """Column store of score rows. Missing / null features are NaN."""

    lang: np.ndarray
    label: np.ndarray
    rate: np.ndarray
    cols: dict[str, np.ndarray]
    noisy: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.lang)

    def take(self, mask: np.ndarray) -> Scores:
        return Scores(
            self.lang[mask],
            self.label[mask],
            self.rate[mask],
            {k: v[mask] for k, v in self.cols.items()},
            None if self.noisy is None else self.noisy[mask],
        )

    def col(self, name: str) -> np.ndarray:
        return self.cols.get(name, np.full(len(self), np.nan))


def _num(v: Any) -> float:
    return np.nan if v is None or isinstance(v, bool) else float(v)


def scores_from_rows(
    rows: list[dict[str, Any]], features: list[str], *, label: str | None = None,
    noisy_cer: float | None = None,
) -> Scores:  # fmt: skip
    rows = [r for r in rows if not r.get("low_confidence")]
    lang = np.array([r.get("lang") or "other" for r in rows], dtype=object)
    if label is not None:
        labels = np.array([label] * len(rows), dtype=object)
        rate = np.zeros(len(rows))
    else:
        labels = np.array([str(r.get("corruption") or "corrupted") for r in rows], dtype=object)
        rate = np.array([_num(r.get("error_rate")) for r in rows])
    cols = {f: np.array([_num(r.get(f)) for r in rows]) for f in features}
    cols = {f: v for f, v in cols.items() if not np.isnan(v).all()}
    noisy = None
    if noisy_cer is not None:
        noisy = np.array(
            [_num(r["cer"]) > noisy_cer if "cer" in r else bool(r.get("noisy")) for r in rows]
        )
    return Scores(lang, labels, rate, cols, noisy)


def flat_scores(row: dict[str, Any]) -> dict[str, Any]:
    """A chunk row from `refine score` keeps its features under ``score``; paragraph score rows
    have them at the top level. Rows that failed to score (``score: null``) give no features."""
    s = row.get("score")
    return {**row, **s} if isinstance(s, dict) else row


def concat(a: Scores, b: Scores) -> Scores:
    keys = set(a.cols) | set(b.cols)
    return Scores(
        np.concatenate([a.lang, b.lang]),
        np.concatenate([a.label, b.label]),
        np.concatenate([a.rate, b.rate]),
        {k: np.concatenate([a.col(k), b.col(k)]) for k in sorted(keys)},
    )


def feature_names(taus: list[float], base_tau: float) -> list[str]:
    return [*BASE_FEATURES, *(spike_key(t) for t in taus if t != base_tau)]


def spike_column(tau: float, base_tau: float, prefix: str = "") -> str:
    return prefix + ("spike_frac" if tau == base_tau else spike_key(tau))


@dataclass
class Report:
    auroc: dict[str, dict[str, float | None]] = field(default_factory=dict)
    auroc_by_label: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)
    spearman: dict[str, dict[str, float | None]] = field(default_factory=dict)
    tau_auroc: dict[str, dict[str, float | None]] = field(default_factory=dict)
    recall_by_label: dict[str, dict[str, float]] = field(default_factory=dict)
    gold: dict[str, Any] | None = None
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    recall_labels: list[str] = field(default_factory=list)  # corrupted labels recall is on


def calibrate_scores(
    data: Scores,
    *,
    taus: list[float],
    base_tau: float,
    target_recall: float,
    max_false_flag: float,
    recall_on: list[str],
    min_rows: int = 50,
    max_thresholds: int = 200,
    gold: Scores | None = None,
) -> tuple[Calibration, Report]:
    rep = Report()
    is_clean = data.label == CLEAN
    langs = [ALL, *sorted(set(data.lang))]
    labels = sorted(set(data.label[~is_clean]))
    features = [f for f in data.cols]

    for lang in langs:
        m = np.ones(len(data), bool) if lang == ALL else data.lang == lang
        c, b = m & is_clean, m & ~is_clean
        rep.counts[lang] = {"clean": int(c.sum())} | {
            lab: int((m & (data.label == lab)).sum()) for lab in labels
        }
        rep.auroc[lang] = {f: _r(auroc(data.cols[f][c], data.cols[f][b])) for f in features}
        rep.auroc_by_label[lang] = {
            lab: {
                f: _r(auroc(data.cols[f][c], data.cols[f][m & (data.label == lab)]))
                for f in features
            }
            for lab in labels
        }
        rate = np.where(is_clean, 0.0, data.rate)
        rep.spearman[lang] = {f: _r(spearman(data.cols[f][m], rate[m])) for f in features}

    # tau: best pooled AUROC of spike_frac
    all_taus = sorted({*taus, base_tau})
    best: tuple[float, float] | None = None
    variant, prefix = "raw", ""
    rep.tau_auroc[variant] = {}
    for t in all_taus:
        col = spike_column(t, base_tau, prefix)
        a = rep.auroc[ALL].get(col)
        rep.tau_auroc[variant][f"{t:g}"] = a
        if a is not None and (best is None or a > best[0] + 1e-12):
            best = (a, t)
    if best is None:
        raise ValueError("no spike_frac column in the scores (score with scorer.tau_candidates)")
    _, tau = best
    s_col, p_col, b_col = spike_column(tau, base_tau, prefix), prefix + "p90", prefix + "bpc"

    in_recall = ~is_clean
    if recall_on and np.isin(data.label, recall_on).any():
        in_recall = np.isin(data.label, recall_on)
    rep.recall_labels = sorted(set(data.label[in_recall]))
    cal = Calibration(
        tau=float(tau), variant=variant, target_recall=target_recall, max_false_flag=max_false_flag
    )
    for lang in langs:
        m = np.ones(len(data), bool) if lang == ALL else data.lang == lang
        c, b = m & is_clean, m & in_recall
        if lang != ALL and (c.sum() < min_rows or b.sum() < min_rows):
            continue
        cal.rules[lang] = fit_rule(
            data.col(s_col)[c], data.col(p_col)[c], data.col(s_col)[b], data.col(p_col)[b],
            clean_b=data.col(b_col)[c], bad_b=data.col(b_col)[b],
            target_recall=target_recall, max_false_flag=max_false_flag,
            max_thresholds=max_thresholds,
        )  # fmt: skip
    for lang in langs:
        m = np.ones(len(data), bool) if lang == ALL else data.lang == lang
        rule = cal.rule(lang)
        f = flags(rule, data.col(s_col), data.col(p_col), data.col(b_col))
        rep.recall_by_label[lang] = {
            lab: round(float(f[m & (data.label == lab)].mean()), 4)
            for lab in [CLEAN, *labels]
            if (m & (data.label == lab)).any()
        }

    if gold is not None and gold.noisy is not None:
        rep.gold = {}
        for lang in [ALL, *sorted(set(gold.lang))]:
            m = np.ones(len(gold), bool) if lang == ALL else gold.lang == lang
            if lang == ALL:
                f = np.zeros(len(gold), bool)
                for lg in set(gold.lang):
                    mm = gold.lang == lg
                    f[mm] = flags(cal.rule(lg), gold.col(s_col)[mm], gold.col(p_col)[mm],
                                  gold.col(b_col)[mm])  # fmt: skip
            else:
                f = flags(cal.rule(lang), gold.col(s_col), gold.col(p_col), gold.col(b_col))
            pr = precision_recall(f[m], gold.noisy[m])
            pr["auroc_spike"] = _r(
                auroc(gold.col(s_col)[m & ~gold.noisy], gold.col(s_col)[m & gold.noisy])
            )
            rep.gold[lang] = pr
    return cal, rep


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def run_calibrate(
    cfg,
    clean: Path,
    corrupted: Path,
    out_dir: Path,
    gold_scores: Path | None = None,
    limit: int | None = None,
    plots: bool = True,
) -> tuple[Calibration, Report]:
    c = cfg.calibrate
    taus = [float(t) for t in cfg.scorer.tau_candidates]
    base_tau = float(cfg.scorer.tau)
    feats = feature_names(taus, base_tau)

    def rows(p: Path) -> list[dict[str, Any]]:
        out = []
        for i, d in enumerate(iter_jsonl(p)):
            if limit is not None and i >= limit:
                break
            out.append(flat_scores(d))
        return out

    data = concat(
        scores_from_rows(rows(clean), feats, label=CLEAN), scores_from_rows(rows(corrupted), feats)
    )
    gold = None
    if gold_scores is not None:
        gold = scores_from_rows(
            rows(gold_scores), feats, label="gold", noisy_cer=float(c.gold_noisy_cer)
        )
    cal, rep = calibrate_scores(
        data,
        taus=taus,
        base_tau=base_tau,
        target_recall=float(c.target_recall),
        max_false_flag=float(c.max_false_flag),
        recall_on=list(c.recall_on or []),
        min_rows=int(c.min_rows),
        max_thresholds=int(c.max_thresholds),
        gold=gold,
    )
    write_outputs(cal, rep, data, out_dir, base_tau=base_tau, plots=plots)
    return cal, rep


def write_outputs(
    cal: Calibration, rep: Report, data: Scores, out_dir: Path, *, base_tau: float, plots: bool
) -> list[Path]:
    from ocrfix.calibrate.report import summary_md, write_plots

    out_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(cal, out_dir / "calibration.yaml")
    (out_dir / "report.json").write_text(
        json.dumps(asdict(rep), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pngs = write_plots(cal, rep, data, out_dir, base_tau=base_tau) if plots else []
    (out_dir / "summary.md").write_text(summary_md(cal, rep, pngs) + "\n", encoding="utf-8")
    return [out_dir / "calibration.yaml", out_dir / "report.json", out_dir / "summary.md", *pngs]
