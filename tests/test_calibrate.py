from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from ocrfix.calibrate import load_calibration
from ocrfix.calibrate.fit import auroc, fit_rule, flags, rankdata, spearman
from ocrfix.calibrate.run import CLEAN, Scores, calibrate_scores, concat, write_outputs
from ocrfix.cli import app

TAUS = [8.0, 10.0, 12.0]
BASE = 10.0


def test_rankdata_auroc_spearman():
    assert rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert auroc(np.array([np.nan]), np.array([1.0])) is None
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0, 1, 300), rng.normal(0.7, 1, 200).round(1)
    brute = (pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean()
    assert auroc(neg, pos) == pytest.approx(brute)
    x = np.arange(20.0)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_fit_rule_spike_only():
    rng = np.random.default_rng(1)
    clean = rng.beta(1, 60, 2000)
    bad = rng.beta(4, 30, 2000)
    rule = fit_rule(
        clean, np.full(2000, 5.0), bad, np.full(2000, 6.0), target_recall=0.8, max_false_flag=0.05
    )
    assert rule.p90 is None and rule.target_met
    assert (clean > rule.spike_frac).mean() <= 0.05
    assert rule.recall == pytest.approx((bad > rule.spike_frac).mean(), abs=1e-4)
    assert rule.false_flag <= 0.05


def test_fit_rule_p90_fallback():
    """A third of the corrupted rows have no spikes but a high p90: spike_frac alone can't
    reach the target, the p90 fallback can, still within the false-flag budget."""
    rng = np.random.default_rng(2)
    n = 3000
    clean_s, clean_p = rng.beta(1, 60, n), rng.normal(6, 1, n)
    bad_s = np.where(np.arange(n) % 3 == 0, 0.0, rng.beta(5, 20, n))
    bad_p = np.where(np.arange(n) % 3 == 0, rng.normal(11, 1, n), rng.normal(7, 1, n))
    # spike-only reaches at most ~2/3
    only = fit_rule(clean_s, np.full(n, np.nan), bad_s, np.full(n, np.nan),
                    target_recall=0.9, max_false_flag=0.05)  # fmt: skip
    assert not only.target_met and only.recall < 0.7
    rule = fit_rule(clean_s, clean_p, bad_s, bad_p, target_recall=0.9, max_false_flag=0.05)
    assert rule.p90 is not None and rule.target_met
    f_clean = flags(rule, clean_s, clean_p)
    assert f_clean.mean() <= 0.05 and rule.false_flag == pytest.approx(f_clean.mean(), abs=1e-4)
    assert flags(rule, bad_s, bad_p).mean() >= 0.9


def test_fit_rule_bpc():
    """Damage that raises the average surprise but makes no spikes: only bpc catches it, and
    the rule still keeps the false-flag budget exactly."""
    rng = np.random.default_rng(3)
    n = 3000
    clean_s, clean_p, clean_b = rng.beta(1, 60, n), rng.normal(6, 1, n), rng.normal(1.0, 0.1, n)
    bad_s, bad_p, bad_b = rng.beta(1, 60, n), rng.normal(6, 1, n), rng.normal(1.4, 0.1, n)
    old = fit_rule(clean_s, clean_p, bad_s, bad_p, target_recall=0.9, max_false_flag=0.05)
    assert not old.target_met and old.bpc is None
    rule = fit_rule(clean_s, clean_p, bad_s, bad_p, clean_b=clean_b, bad_b=bad_b,
                    target_recall=0.9, max_false_flag=0.05)  # fmt: skip
    assert rule.bpc is not None and rule.spike_frac is None and rule.target_met
    f_clean = flags(rule, clean_s, clean_p, clean_b)
    assert f_clean.mean() <= 0.05 and rule.false_flag == pytest.approx(f_clean.mean(), abs=1e-4)
    # mixed damage: half spiky, half high-bpc; the combined rule beats either feature alone
    half = np.arange(n) % 2 == 0
    mix_s = np.where(half, rng.beta(5, 20, n), clean_s)
    mix_b = np.where(half, clean_b, bad_b)
    both = fit_rule(clean_s, clean_p, mix_s, clean_p, clean_b=clean_b, bad_b=mix_b,
                    target_recall=0.99, max_false_flag=0.05)  # fmt: skip
    assert both.spike_frac is not None and both.bpc is not None
    assert both.recall > 0.85 and both.false_flag <= 0.05


def test_rule_without_bpc_loads_and_flags(tmp_path):
    from ocrfix.calibrate.rules import load_calibration

    p = tmp_path / "calibration.yaml"
    p.write_text("version: 1\ntau: 8.0\nrules:\n  '*': {spike_frac: 0.02, p90: 9.0}\n")
    cal = load_calibration(p)
    assert cal.rule("en").bpc is None
    assert cal.flag({"spike_frac": 0.03, "p90": 1.0, "bpc": 9.0}, "en")
    assert not cal.flag({"spike_frac": 0.01, "p90": 1.0, "bpc": 9.0}, "en")
    p.write_text("version: 1\ntau: 8.0\nrules:\n  '*': {spike_frac: null, bpc: 1.2}\n")
    cal = load_calibration(p)
    assert cal.flag({"spike_frac": 0.5, "p90": 1.0, "bpc": 1.3}, "ko")
    assert not cal.flag({"spike_frac": 0.5, "p90": 99.0, "bpc": 1.1}, "ko")
    assert not cal.flag({"spike_frac": 0.5, "bpc": 9.0, "low_confidence": True}, "ko")


def test_fit_rule_nan_never_flags():
    clean = np.array([np.nan, 0.0, 0.01, 0.02] * 50)
    bad = np.array([np.nan, 0.5] * 100)
    rule = fit_rule(clean, clean, bad, bad, target_recall=0.4, max_false_flag=0.0)
    assert not flags(rule, np.array([np.nan]), np.array([np.nan])).any()
    assert rule.false_flag == 0.0 and rule.recall == 0.5


def synthetic(n: int = 600, seed: int = 0, langs=("en", "ko"), small_lang: str | None = None):
    """Clean + corrupted at three rates. spike_frac@12 separates best; flat_ is worse."""
    rng = np.random.default_rng(seed)
    parts = []
    for lang in langs:
        k = 20 if lang == small_lang else n
        for label, rate in ((CLEAN, 0.0), ("r0.01", 10.0), ("r0.1", 100.0), ("table", 30.0)):
            sep = rate / 100  # 0 .. 1
            noise = rng.normal(0, 1, k)
            cols = {
                "spike_frac": np.clip(0.02 + 0.03 * sep + 0.03 * noise, 0, 1),  # tau 10
                "spike_frac@8": np.clip(0.05 + 0.03 * sep + 0.04 * noise, 0, 1),
                "spike_frac@12": np.clip(0.01 + 0.06 * sep + 0.02 * noise, 0, 1),
                "p90": 6 + 3 * sep + rng.normal(0, 1, k),
                "p99": 9 + 3 * sep + rng.normal(0, 1, k),
                "bpc": 1.5 + sep + rng.normal(0, 0.3, k),
            }
            cols |= {f"flat_{c}": v + rng.normal(0, 0.05, k) for c, v in cols.items()}
            parts.append(
                Scores(
                    np.array([lang] * k, dtype=object),
                    np.array([label] * k, dtype=object),
                    np.full(k, rate) + (rng.uniform(-5, 5, k) if label == "table" else 0),
                    cols,
                )
            )
    out = parts[0]
    for p in parts[1:]:
        out = concat(out, p)
    return out


def run(data, **kw):
    opts = dict(
        taus=TAUS, base_tau=BASE, target_recall=0.8, max_false_flag=0.05, recall_on=["table"],
        min_rows=50,
    ) | kw  # fmt: skip
    return calibrate_scores(data, **opts)


def test_calibrate_chooses_tau_and_rules():
    data = synthetic()
    cal, rep = run(data)
    assert cal.tau == 12.0 and cal.variant == "raw"
    assert rep.tau_auroc["raw"]["12"] > rep.tau_auroc["raw"]["10"] > 0.5
    assert set(cal.rules) == {"*", "en", "ko"}
    for lang, rule in cal.rules.items():
        m = np.ones(len(data), bool) if lang == "*" else data.lang == lang
        clean = m & (data.label == CLEAN)
        f = flags(rule, data.col("spike_frac@12"), data.col("p90"), data.col("bpc"))
        assert f[clean].mean() <= 0.05 + 1e-9
        assert rule.recall == pytest.approx(f[m & (data.label == "table")].mean(), abs=1e-4)
    # monotone in corruption rate; the strongest corruption is easiest
    assert rep.spearman["*"]["spike_frac@12"] > 0.5
    by = rep.auroc_by_label["*"]
    assert by["r0.1"]["spike_frac@12"] > by["table"]["spike_frac@12"] > by["r0.01"]["spike_frac@12"]
    assert rep.recall_by_label["*"]["r0.1"] >= rep.recall_by_label["*"]["r0.01"]


def test_small_language_falls_back_to_star():
    data = synthetic(small_lang="ko")
    cal, _ = run(data)
    assert "ko" not in cal.rules and cal.rule("ko") is cal.rules["*"]


def test_gold_precision_recall_and_outputs(tmp_path):
    data = synthetic()
    rng = np.random.default_rng(5)
    k = 400
    noisy = np.arange(k) % 2 == 0
    gold = Scores(
        np.array(["en"] * k, dtype=object),
        np.array(["gold"] * k, dtype=object),
        np.zeros(k),
        {
            "spike_frac@12": np.where(noisy, 0.08, 0.005) + rng.normal(0, 0.01, k),
            "p90": np.full(k, 6.0),
        },
        noisy,
    )
    cal, rep = run(data, gold=gold)
    g = rep.gold["en"]
    assert g["n"] == k and g["n_noisy"] == k // 2
    assert g["precision"] > 0.9 and g["recall"] > 0.9 and g["auroc_spike"] > 0.95
    assert rep.gold["*"]["n_flagged"] == g["n_flagged"]

    paths = write_outputs(cal, rep, data, tmp_path, base_tau=BASE, plots=True)
    names = {p.name for p in paths}
    assert {"calibration.yaml", "report.json", "summary.md", "roc.png", "auroc.png"} <= names
    assert all(p.stat().st_size > 0 for p in paths)
    loaded = load_calibration(tmp_path / "calibration.yaml")
    assert loaded.tau == 12.0 and loaded.rules["en"].spike_frac == cal.rules["en"].spike_frac
    md = (tmp_path / "summary.md").read_text("utf-8")
    assert "tau = **12**" in md and "Real gold source" in md and "![roc](roc.png)" in md

    # the rule as the scorer / verify apply it to one feature row
    r = loaded.rules["en"]
    assert loaded.flag({"spike_frac@12": r.spike_frac + 0.01, "p90": 0.0}, "en")
    assert not loaded.flag({"spike_frac@12": r.spike_frac, "p90": -100.0}, "en")
    # a scorer run with tau=12 writes it as plain spike_frac
    assert loaded.flag({"spike_frac": r.spike_frac + 0.01}, "en")
    assert not loaded.flag({"spike_frac": None, "p90": None}, "zz")


def test_calibrate_cli(tmp_path):
    data = synthetic(n=200)
    clean, bad = tmp_path / "clean.jsonl", tmp_path / "bad.jsonl"
    with open(clean, "w") as fc, open(bad, "w") as fb:
        for i in range(len(data)):
            row = {"doc_id": "d", "para_id": f"p{i}", "lang": data.lang[i]}
            row |= {k: float(v[i]) for k, v in data.cols.items()}
            if data.label[i] == CLEAN:
                fc.write(json.dumps(row) + "\n")
            else:
                row |= {"corruption": data.label[i], "error_rate": float(data.rate[i])}
                fb.write(json.dumps(row) + "\n")
    out = tmp_path / "cal"
    res = CliRunner().invoke(
        app,
        ["calibrate", str(clean), str(bad), str(out), "--no-plots",
         "--set", "scorer.tau_candidates=[8,10,12]", "--set", "calibrate.target_recall=0.8"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "tau=12" in res.output
    assert load_calibration(out / "calibration.yaml").tau == 12.0
