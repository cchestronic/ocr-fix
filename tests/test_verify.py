from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.config import load_config
from ocrfix.corpus import tag_spans
from ocrfix.schema import Record
from ocrfix.verify import format_summary, run_verify, summarize
from ocrfix.verify.checks import check_gap_fill

GOOD = {"bpc": 1.0, "spike_frac": 0.01, "p90": 5.0}  # default re-scored features
BEFORE = {"bpc": 2.0, "spike_frac": 0.1, "p90": 9.0}


class FakeScorer:
    """Features per corrected text (default GOOD); records every call."""

    def __init__(self, by_text: dict[str, dict] | None = None):
        self.by_text = by_text or {}
        self.seen: list[str] = []
        self.sources: list[list[str]] = []

    def __call__(self, records: list[Record], corrected: dict) -> dict:
        self.sources.append([r.text for r in records])
        self.seen += list(corrected.values())
        return {k: self.by_text.get(t, GOOD) for k, t in corrected.items()}


def correction(pid: str, src: str, dst: str, lang: str = "en", doc: str = "d", **extra) -> dict:
    r = Record(doc, pid, int(pid[1:]), lang, dst, tag_spans(dst), "x.md")
    return r.to_dict() | {
        "original_text": src, "llm_source_text": src, "fixer": ["unicode", "llm"],
        "llm_chunk_id": "d:abc",
    } | extra  # fmt: skip


def run(tmp_path, rows, *, scorer=None, before=None, overrides=(), limit=None):
    corr = tmp_path / "corrections.jsonl"
    corr.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    sb = tmp_path / "scores_before.jsonl"
    before = before if before is not None else {r["para_id"]: BEFORE for r in rows}
    sb.write_text(
        "".join(
            json.dumps({"doc_id": "d", "para_id": k, "lang": "en", **v}) + "\n"
            for k, v in before.items()
        )
    )
    cfg = load_config("rtx5090", list(overrides))
    out = tmp_path / "verified.jsonl"
    stats = run_verify(cfg, corr, sb, out, rescorer=scorer or FakeScorer(), limit=limit)
    got = {d["para_id"]: d for d in map(json.loads, out.read_text("utf-8").splitlines())}
    return got, stats


def failed(row) -> list[str]:
    return [f["check"] for f in row["failed_checks"]]


EN_SRC = "The rnodel in Tab1e 2 reaches $x_1 = 0.5$ on the benchmark dataset today."
EN_FIX = "The model in Table 2 reaches $x_1 = 0.5$ on the benchmark dataset today."


def test_accepted(tmp_path):
    got, stats = run(tmp_path, [correction("p00000", EN_SRC, EN_FIX)])
    r = got["p00000"]
    assert r["accepted"] and r["failed_checks"] == [] and r["text"] == EN_FIX
    assert r["original"] == EN_SRC and r["corrected"] == EN_FIX
    assert set(r["checks"]) == {
        "markers",
        "length_ratio",
        "edit_ratio",
        "language",
        "gap_fill",
        "scorer",
    }
    assert all(c["passed"] for c in r["checks"].values())
    assert r["before"]["bpc"] == 2.0 and r["after"]["bpc"] == 1.0
    assert r["fixer"] == ["unicode", "llm"]
    assert stats.candidates == 1 and stats.accepted == 1


@pytest.mark.parametrize(
    ("dst", "reason"),
    [
        (EN_FIX.replace("$x_1 = 0.5$", "x1 = 0.5"), "missing or altered"),  # math dropped
        (EN_FIX.replace("$x_1 = 0.5$", "$x_1 = 0.6$"), "missing or altered"),  # math edited
        (EN_FIX.replace("today.", "today $x_1 = 0.5$."), "duplicated"),
        (EN_FIX.replace("today.", "today $y$."), "new math/table"),
    ],
)
def test_markers(tmp_path, dst, reason):
    got, _ = run(tmp_path, [correction("p00000", EN_SRC, dst)])
    r = got["p00000"]
    assert not r["accepted"] and "markers" in failed(r)
    assert r["checks"]["markers"]["reason"].startswith(reason)
    assert r["text"] == EN_SRC  # rejected -> the source text is kept
    assert r["fixer"] == ["unicode"]


def test_markers_order(tmp_path):
    src = "First $a$ then <table><tr><td>1</td></tr></table> end of the paragraph."
    dst = "First <table><tr><td>1</td></tr></table> then $a$ end of the paragraph."
    got, _ = run(tmp_path, [correction("p00000", src, dst)])
    assert got["p00000"]["checks"]["markers"]["reason"] == "math/table spans out of order"


def test_length_and_edit_ratio(tmp_path):
    rows = [
        correction("p00000", EN_SRC, "The model reaches $x_1 = 0.5$."),  # too short
        correction("p00001", EN_SRC, EN_SRC + " It also generalizes well to other sets."),
        # same length, heavily rewritten
        correction(
            "p00002", EN_SRC, "A model in Figure 7 obtains $x_1 = 0.5$ at the bench dataset now!!"
        ),
    ]
    got, _ = run(tmp_path, rows)
    assert "length_ratio" in failed(got["p00000"]) and "length_ratio" in failed(got["p00001"])
    r = got["p00002"]
    assert failed(r)[0] == "edit_ratio" and "length_ratio" not in failed(r)
    assert r["checks"]["edit_ratio"]["reason"].endswith("(cap 0.15)")


KO_SRC = "본 연구에서는 제안한 방법이 기존 방법보다 우수함을 확인하였다."


def test_language(tmp_path):
    rows = [
        correction("p00000", KO_SRC, "In this study we confirmed the method is better.", lang="ko"),
        # still "ko", but much more Latin than before
        # still "ko", but the Latin share grew past the cap
        correction("p00001", KO_SRC, KO_SRC.replace("제안한", "OCR 제안한"), lang="ko"),
    ]
    overrides = [
        "verify.length_ratio=[0.5,2]",
        "verify.max_edit_ratio=1",
        "verify.max_script_shift=0.01",
    ]
    got, _ = run(tmp_path, rows, overrides=overrides)
    assert got["p00000"]["checks"]["language"]["reason"] == "ko -> en"
    assert "language" in failed(got["p00001"])
    assert "Hangul share moved" in got["p00001"]["checks"]["language"]["reason"]


@pytest.mark.parametrize(
    ("dst", "ok", "reason"),
    [
        ("반응 온도는 300 ℃이며 시간은 두시간이었다고 보고하였다.", False, "filled with '300'"),
        ("반응 온도는 삼백 도이며 시간은 두시간이었다고 보고하였다.", False, "text around"),
        ("반응 온도는 〈?〉 ℃이며 시간은 두 시간이었다고 보고하였다.", True, "0 filled, 1 kept"),
        ("반응 온도는 삼백 ℃이며 시간은 두시간이었다고 보고하였다.", True, "1 filled"),
        (
            "반응 온도는 삼백오십도 ℃이며 시간은 두시간이었다고 보고하였다.",
            False,
            "filled with '삼백오십도'",
        ),
        ("반응 온도는 abc ℃이며 시간은 두시간이었다고 보고하였다.", False, "filled with 'abc'"),
    ],
)
def test_gap_fill(tmp_path, dst, ok, reason):
    src = "반응 온도는 〈?〉 ℃이며 시간은 두시간이었다고 보고하였다."
    got, _ = run(tmp_path, [correction("p00000", src, dst, lang="ko")])
    c = got["p00000"]["checks"]["gap_fill"]
    assert c["passed"] is ok and reason in c["reason"]


def test_gap_fill_edges():
    assert check_gap_fill("〈?〉 시작", "가 시작", 4).passed
    assert check_gap_fill("끝 〈?〉", "끝 나다", 4).passed
    assert check_gap_fill("〈?〉〈?〉 둘", "가나 둘", 4).passed  # adjacent gaps share one fill
    assert not check_gap_fill("끝 〈?〉", "끝 xyz", 4).passed
    assert check_gap_fill("no gaps", "anything", 4).reason == "no 〈?〉 in source"


def test_scorer(tmp_path):
    rows = [
        correction("p00000", EN_SRC, EN_FIX),
        correction("p00001", EN_SRC, EN_FIX.replace("today", "now.")[:-1] + " today."),
        correction("p00002", EN_SRC, EN_FIX + " "),  # strips to EN_FIX? no: distinct text below
        correction("p00003", EN_SRC, EN_FIX.replace("dataset", "data set")),
    ]
    rows[2]["text"] = EN_FIX.replace("model", "modle")
    rows[2]["spans"] = [s.to_dict() for s in tag_spans(rows[2]["text"])]
    scorer = FakeScorer(
        {
            rows[1]["text"]: {"bpc": 1.99, "spike_frac": 0.05},  # gain 0.01 < 0.05
            rows[2]["text"]: {"bpc": 1.0, "spike_frac": 0.2},  # spikes doubled
        }
    )
    before = {"p00000": BEFORE, "p00001": BEFORE, "p00002": BEFORE}  # p00003: no row
    got, _ = run(
        tmp_path, rows, scorer=scorer, before=before, overrides=["verify.min_bpc_gain=0.05"]
    )
    assert got["p00000"]["accepted"]
    assert (
        failed(got["p00001"]) == ["scorer"]
        and "gain 0.0100 < 0.05" in got["p00001"]["checks"]["scorer"]["reason"]
    )
    assert (
        failed(got["p00002"]) == ["scorer"]
        and "spike_frac 0.1000 -> 0.2000" in got["p00002"]["checks"]["scorer"]["reason"]
    )
    assert got["p00003"]["checks"]["scorer"]["reason"] == "no score row for the source paragraph"


def test_scorer_no_prose_and_skip(tmp_path):
    rows = [
        correction("p00000", "$x$ $y$", "$x$, $y$"),
        correction("p00001", EN_SRC, "x"),  # fails length: not re-scored
    ]
    scorer = FakeScorer({"$x$, $y$": {"bpc": None, "spike_frac": None}})
    got, _ = run(tmp_path, rows, scorer=scorer)
    assert got["p00000"]["checks"]["scorer"]["reason"] == "no prose to score"
    assert got["p00001"]["checks"]["scorer"]["reason"].startswith("skipped")
    assert scorer.seen == ["$x$, $y$"]
    scorer2 = FakeScorer()
    run(tmp_path, rows, scorer=scorer2, overrides=["verify.rescore_failed=true"])
    assert len(scorer2.seen) == 2


def test_non_candidates_pass_through(tmp_path):
    plain = Record("d", "p00005", 5, "en", "Untouched.", tag_spans("Untouched."), "x.md").to_dict()
    errored = correction("p00006", "Tab1e", "Tab1e", llm_error="marker_missing")
    errored["fixer"] = ["unicode"]
    same = correction("p00007", "Same text.", "Same text.")
    got, stats = run(tmp_path, [plain, errored, same, correction("p00000", EN_SRC, EN_FIX)])
    for pid in ("p00005", "p00006", "p00007"):
        assert "accepted" not in got[pid]
    assert got["p00005"]["text"] == "Untouched." and stats.rows == 4 and stats.candidates == 1


def test_summary_and_cli(tmp_path):
    rows = [
        correction("p00000", EN_SRC, EN_FIX),
        correction("p00001", EN_SRC, "x"),
        correction("p00002", EN_SRC, EN_FIX.replace("$x_1 = 0.5$", "x")),
    ]
    run(tmp_path, rows)
    s = summarize(tmp_path / "verified.jsonl")
    assert (s["candidates"], s["accepted"]) == (3, 1)
    assert s["reasons"]["length_ratio"] == 1 and s["reasons"]["markers"] == 2
    text = format_summary(s)
    assert text.startswith("3 corrections verified, 1 accepted (33.3%)")
    assert "markers" in text and "#" in text
    res = CliRunner().invoke(app, ["verify-summary", str(tmp_path / "verified.jsonl")])
    assert res.exit_code == 0 and "1 accepted" in res.output


def test_in_process_scorer(tiny_model_dir, model_device):
    """The real re-scorer (tiny random model) returns the same feature keys as `score`."""
    import torch

    from ocrfix.scorer.run import InProcessScorer

    cfg = load_config(
        "rtx5090",
        [f"paths.base_model={tiny_model_dir}", "hw.dtype=float32", "hw.scorer.tokenize_workers=0",
         "scorer.calibration=null"],
    )  # fmt: skip
    recs = [
        Record("d", f"p{i}", i, "en", t, tag_spans(t), "")
        for i, t in enumerate(["The model works.", "$x$", "Two\nlines here"])
    ]
    with InProcessScorer(cfg, torch.device(model_device)) as sc:
        feats = sc(recs)
    assert len(feats) == 3 and [f["para_id"] for f in feats] == ["p0", "p1", "p2"]
    assert feats[0]["bpc"] > 0 and feats[1]["bpc"] is None  # "$x$" is only a placeholder
    assert not any(k.startswith("flat_") for k in feats[2])
