"""`ocrfix calibrate-noise-chunk`: limit from gold real text, corpus report, samples, writing."""

from __future__ import annotations

import json

import httpx
import pytest
from tests.test_calibrate_noise import JunkScorer

from ocrfix.calibrate.noise_chunk import (
    Corpus,
    bpc_histograms,
    deleted_share,
    gold_limits,
    plot_bpc,
    run_calibrate_noise_chunk,
)
from ocrfix.calibrate.rules import Calibration, Rule, load_calibration, save_calibration
from ocrfix.config import load_config


def test_deleted_share():
    assert deleted_share("Results are good here.", "Results are good here.") == 0.0
    assert deleted_share("▯▯▯ ※※¶ ■■■ ok", "ok") > 0.8
    assert deleted_share("Rcsults are gaod.", "Results are good.") == 0.0  # corrected, not deleted


def test_gold_limits_ignore_noise_chunks():
    gold = ([{"lang": "en", "bpc": 1.0 + i / 100, "noise": False} for i in range(60)]
            + [{"lang": "ko", "bpc": 2.0, "noise": False} for _ in range(5)]
            + [{"lang": "en", "bpc": 9.0, "noise": True},
               {"lang": "en", "bpc": 1.5, "noise": True}])  # fmt: skip
    g = gold_limits(gold, margin=1.0, min_rows=50)
    assert set(g) == {"*", "en"}  # ko: too few rows, uses "*"
    assert g["en"]["real_max"] == 1.59 and g["en"]["limit"] == 2.59
    assert g["*"]["real_max"] == 2.0 and g["*"]["limit"] == 3.0
    assert g["en"]["noise_chunks"] == 2 and g["en"]["noise_caught"] == 1


def test_corpus_counts_and_samples():
    c = Corpus({"*": 3.0, "en": 2.0}, {"*": [3.0], "en": [2.0, 4.0]}, k=2, seed=0)
    for i in range(100):
        c.add({"status": "scored", "lang": "en", "chunk_id": f"e{i}", "source_text": "x" * 10,
               "score": {"bpc": i / 10}})  # fmt: skip
    c.add({"status": "scored", "lang": "zh", "source_text": "y", "score": {"bpc": 5.0}})
    c.add({"status": "noise", "lang": "en", "score": {"bpc": 9.0}})
    c.add({"status": "scored", "lang": "en", "score": {"bpc": 9.0, "low_confidence": True}})
    rep = c.report()
    assert rep["en"]["chunks"] == 100 and rep["*"]["chunks"] == 1 and c.skipped == 2
    drops = {d["limit"]: d["chunks"] for d in rep["en"]["drop"]}
    assert drops == {2.0: 79, 4.0: 59}  # bpc > 2.0: 2.1 .. 9.9
    bands = rep["en"]["bands"]
    assert bands[0]["from"] == 1.0 and bands[-1]["to"] is None  # 1.0 below .. open top
    assert sum(b["chunks"] for b in bands) == 90  # bpc 1.0 .. 9.9
    assert all(len(v) <= 2 for v in c.samples.values())


def test_end_to_end(tmp_path, tiny_model_dir):
    gold = tmp_path / "gold" / "sft"
    gold.mkdir(parents=True)
    rows = []
    for i in range(20):
        truth = f"The model {i} reaches good results on every page. The loss decreases steadily."
        rows.append({"chunk_id": f"d{i}-c0", "doc_id": f"d{i}", "lang": "en", "source_text": truth,
                     "gold_text": truth, "gold_status": "ok", "split": "sft"})  # fmt: skip
    junk = "lI1l! I1l;: :lI1;! 1lI:! " * 6  # surprising to JunkScorer
    rows.append({"chunk_id": "j-c0", "doc_id": "j", "lang": "en", "source_text": junk + "page 3",
                 "gold_text": "page 3", "gold_status": "ok", "split": "sft"})  # fmt: skip
    (gold / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    scored = tmp_path / "scored"
    scored.mkdir()
    corpus = [{"chunk_id": f"c{i}", "lang": "en", "status": "scored", "source_text": f"text {i}",
               "score": {"bpc": i / 4}} for i in range(60)]  # fmt: skip
    (scored / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus))
    cal_path = tmp_path / "cal.yaml"
    save_calibration(Calibration(tau=8.0, rules={"*": Rule(spike_frac=0.1)},
                                 noise_chunk={"*": 99.0, "ko": 98.0}), cal_path)  # fmt: skip
    cfg = load_config("rtx5090", [f"model={tiny_model_dir}", "score.backend=vllm",
                                  "score.noise.remove=false", "calibrate.min_rows=10",
                                  "hw.infer.score_urls=[http://127.0.0.1:9001]"])  # fmt: skip
    rep = run_calibrate_noise_chunk(
        cfg, tmp_path / "gold", scored, tmp_path / "out", calibration=cal_path,
        transport=httpx.MockTransport(JunkScorer(tiny_model_dir)),
    )  # fmt: skip
    g = rep["gold"]["en"]
    assert g["real_chunks"] == 20 and g["noise_chunks"] == 1 and g["noise_caught"] == 1
    assert g["limit"] == pytest.approx(g["real_max"] + 1.0, abs=1e-3)
    cal = load_calibration(cal_path)
    assert cal.noise_chunk == rep["limits"] and set(cal.noise_chunk) == {"*", "en"}
    assert cal.rules["*"].spike_frac == 0.1  # the rest of the file is kept
    md = (tmp_path / "out" / "noise_chunk_samples.md").read_text()
    assert "## en: limit" in md and "(dropped)" in md and "(kept)" in md
    assert json.loads((tmp_path / "out" / "noise_chunk_report.json").read_text())["corpus"]["en"]
    assert (tmp_path / "out" / "noise_chunk_hist.png").stat().st_size > 1000
    assert sum(g["real_histogram"].values()) == 20


def test_bpc_histograms_and_plot(tmp_path):
    d = tmp_path / "scored"
    d.mkdir()
    rows = ([{"status": "scored", "lang": "en", "score": {"bpc": 0.5 + i / 100}} for i in range(50)]
            + [{"status": "scored", "lang": "en", "score": {"bpc": 7.0}},
               {"status": "scored", "lang": "zh", "score": {"bpc": 2.0}},
               {"status": "noise", "lang": "en", "score": {"bpc": 9.0}},
               {"status": "scored", "lang": "en",
                "score": {"bpc": 9.0, "low_confidence": True}}])  # fmt: skip
    (d / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    h = bpc_histograms([d], {"en": 3.0, "*": 1.0})
    assert list(h) == ["en", "zh"] and h["en"]["chunks"] == 51 and h["en"]["max"] == 7.0
    assert (h["en"]["limit"], h["en"]["above"]) == (3.0, 1)
    assert (h["zh"]["limit"], h["zh"]["above"]) == (1.0, 1)  # other languages: "*"
    assert sum(h["en"]["histogram"].values()) == 51
    assert bpc_histograms([d])["en"]["limit"] is None
    assert plot_bpc(h, tmp_path / "h.png", "t").stat().st_size > 1000
