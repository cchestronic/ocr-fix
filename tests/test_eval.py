"""`ocrfix eval` on a 10-paragraph gold fixture (3 documents, ko + en).

Gold edits in the fixture: char confusions (Tab1e, rnodel), a hyphen wrap (pro- posed), a
Korean missing space, a running header, a ligature. Five paragraphs are clean, one is
labeled merge-with-next, one split-at:9, and two hold math.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from ocrfix.config import load_config
from ocrfix.eval.metrics import spacing_errors
from ocrfix.eval.run import BASELINE, evaluate, load_gold, parse_system
from ocrfix.eval.stats import holm, wilcoxon

GOLD = Path(__file__).parent / "fixtures" / "eval" / "gold_eval.jsonl"
D1, D2, D3 = "d1en000000000001", "d2ko000000000002", "d3en000000000003"
OPTS = {"baseline": True, "worst_n": 20, "split_tolerance": 3, "exact_max_n": 25}
ERR = {"header_footer_min_chars": 8, "confusion_max_len": 4}


def gold_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]


def rec(doc: str, pid: str, text: str) -> dict:
    return {"doc_id": doc, "para_id": pid, "order": 0, "text": text}


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def perfect_rows() -> list[dict]:
    """Gold text, with the gold structure applied: D1 p2+p3 merged, D2 p2 split."""
    g = {(r["doc_id"], r["para_id"]): r["gold_raw"] for r in gold_rows()}
    return [
        rec(D1, "p00000", g[D1, "p00000"]),
        rec(D1, "p00001", g[D1, "p00001"]),
        rec(D1, "p00002", g[D1, "p00002"] + " " + g[D1, "p00003"]),
        rec(D2, "p00000", g[D2, "p00000"]),
        rec(D2, "p00001", g[D2, "p00001"]),
        rec(D2, "p00002", "3. 실험 설정"),
        rec(D2, "p00002~s1", "모든 모델은 10 에폭 동안 학습하였다."),
        rec(D2, "p00005", g[D2, "p00005"]),
        rec(D3, "p00000", g[D3, "p00000"]),
        rec(D3, "p00001", g[D3, "p00001"]),
    ]


def overfix_rows() -> list[dict]:
    """Fixes D1 p0 only; alters two clean paragraphs (one of them in math); drops D3."""
    s = {(r["doc_id"], r["para_id"]): r["source_raw"] for r in gold_rows()}
    g = {(r["doc_id"], r["para_id"]): r["gold_raw"] for r in gold_rows()}
    return [
        rec(D1, "p00000", g[D1, "p00000"]),
        rec(D1, "p00001", s[D1, "p00001"]),
        rec(D1, "p00002", s[D1, "p00002"]),
        rec(D1, "p00003", "and confirm the trends."),
        rec(D1, "p00009", "a paragraph outside the eval set"),  # ingest id, not gold: dropped
        rec(D2, "p00000", s[D2, "p00000"]),
        rec(D2, "p00001", "실험 결과는 $x^3$ 에 비례한다."),
        rec(D2, "p00002", s[D2, "p00002"]),
        rec(D2, "p00005", s[D2, "p00005"]),
    ]


@pytest.fixture
def run(tmp_path):
    perfect = write(tmp_path / "sft_dpo.jsonl", perfect_rows())
    overfix = write(tmp_path / "base.jsonl", overfix_rows())
    out = tmp_path / "eval"
    m = evaluate(GOLD, [("base", overfix), ("sft_dpo", perfect)], out, opts=OPTS, errors_opts=ERR)
    return m, out


def test_fixture_shape():
    docs, skipped = load_gold(GOLD)
    assert skipped == 0 and len(docs) == 3
    assert sum(len(d.paras) for d in docs) == 10
    assert sum(p.clean for d in docs for p in d.paras) == 5
    assert [d.lang for d in docs] == ["en", "ko", "en"]


def test_perfect_system_despite_merge_and_split(run):
    m, _ = run
    o = m["systems"]["sft_dpo"]["overall"]
    assert o["cer"] == 0 and o["wer"] == 0 and o["ser"] == 0
    assert o["false_fixes"] == 0 and o["identity"]["preservation"] == 1.0
    assert o["recall"]["*"]["recall"] == 1.0
    types = set(o["recall"]) - {"*"}
    assert {
        "char_confusion", "hyphen_wrap", "missing_space.ko", "header_footer_insert", "ligature",
    } <= types  # fmt: skip
    p = o["placeholders"]
    assert (p["expected"], p["matched"], p["extra"], p["integrity"]) == (2, 2, 0, 1.0)
    st = o["structure"]
    assert st["n"] == 10 and st["accuracy"] == 1.0 and st["split_position_accuracy"] == 1.0
    assert st["merge"]["recall"] == 1.0 and st["split"]["precision"] == 1.0


def test_source_baseline(run):
    m, _ = run
    o = m["systems"][BASELINE]["overall"]
    assert o["cer"] > 0 and o["sys_edits"] == 0 and o["false_fix_rate"] == 0
    assert o["recall"]["*"]["fixed"] == 0
    assert o["identity"]["preservation"] == 1.0
    assert o["structure"]["accuracy"] == pytest.approx(0.8)  # merge and split missed
    ko = m["systems"][BASELINE]["by_lang"]["ko"]
    assert ko["ser"] > 0 and ko["docs"] == 1


def test_overfix_system(run):
    m, _ = run
    s = m["systems"]["base"]
    assert s["input"]["missing_docs"] == 1 and s["input"]["non_gold_paragraphs"] == 1
    o = s["overall"]
    assert o["missing_docs"] == 1
    assert o["false_fixes"] == 2 and o["sys_edits"] >= 4
    assert o["identity"]["clean_paragraphs"] == 5 and o["identity"]["altered"] == 2
    assert o["placeholders"]["matched"] == 1 and o["placeholders"]["extra"] == 1
    assert o["recall"]["char_confusion"]["fixed"] == 2  # Tab1e, rnodel
    assert o["recall"]["hyphen_wrap"]["fixed"] == 0
    base_cer = m["systems"][BASELINE]["overall"]["cer"]
    assert 0 < o["cer"] < base_cer + 0.05


def test_comparison_and_outputs(run):
    m, out = run
    pairs = {(c["a"], c["b"]): c for c in m["comparison"]}
    assert set(pairs) == {(BASELINE, "base"), (BASELINE, "sft_dpo"), ("base", "sft_dpo")}
    c = pairs[BASELINE, "sft_dpo"]
    assert c["b_better"] == 3 and c["wilcoxon"]["method"] == "exact"
    assert c["wilcoxon"]["p"] == pytest.approx(0.25)  # 3 docs, all one sign: 2/2^3
    assert c["p_holm"] >= c["wilcoxon"]["p"]
    report = (out / "report.md").read_text()
    for head in (
        "## Overall",
        "## By language",
        "### ko",
        "## Recall by error type",
        "## Structure",
        "## Paired comparison",
        "## Worst 20 paragraphs per system",
    ):
        assert head in report
    assert "### base" in report and "### sft_dpo" in report
    per_doc = [json.loads(x) for x in (out / "per_doc.jsonl").read_text().splitlines()]
    assert len(per_doc) == 9  # 3 systems x 3 docs
    paras = (out / "paragraphs.jsonl").read_text().splitlines()
    assert len(paras) == 30
    assert json.loads((out / "metrics.json").read_text())["gold"]["docs"] == 3


def test_worst_paragraphs_ranked(run):
    _, out = run
    report = (out / "report.md").read_text()
    section = report.split("### base", 1)[1].split("### sft_dpo", 1)[0]
    # the header paragraph left unfixed is the worst one for `base`
    assert section.index("p00005") < section.index("p00001")


def test_sha256_sidecar(tmp_path):
    gold = tmp_path / "gold_eval.jsonl"
    gold.write_bytes(GOLD.read_bytes())
    sys_ = write(tmp_path / "s.jsonl", perfect_rows())
    import hashlib

    good = hashlib.sha256(gold.read_bytes()).hexdigest()
    (tmp_path / "gold_eval.jsonl.sha256").write_text(f"{good}  gold_eval.jsonl\n")
    m = evaluate(gold, [("s", sys_)], tmp_path / "o", opts=OPTS, errors_opts=ERR)
    assert m["gold"]["verified"] is True
    (tmp_path / "gold_eval.jsonl.sha256").write_text("0" * 64 + "  gold_eval.jsonl\n")
    with pytest.raises(ValueError, match="sha256"):
        evaluate(gold, [("s", sys_)], tmp_path / "o", opts=OPTS, errors_opts=ERR)


def test_limit_counts_documents(tmp_path):
    sys_ = write(tmp_path / "s.jsonl", perfect_rows())
    m = evaluate(GOLD, [("s", sys_)], tmp_path / "o", opts=OPTS, errors_opts=ERR, limit=2)
    assert m["gold"]["docs"] == 2


def test_spacing_errors():
    assert spacing_errors("본 연구에서는새로운 방법", "본 연구에서는 새로운 방법") == (1, 3)
    assert spacing_errors("a b", "a b") == (0, 1)
    assert spacing_errors("ab c", "a bc") == (2, 1)


def test_parse_system():
    assert parse_system("sft=runs/x/out.jsonl") == ("sft", Path("runs/x/out.jsonl"))
    assert parse_system("runs/verified.jsonl.gz") == ("verified", Path("runs/verified.jsonl.gz"))


def _brute_p(d: list[float]) -> float:
    """Two-sided exact p by enumerating every sign assignment."""
    import numpy as np

    from ocrfix.calibrate.fit import rankdata

    nz = [x for x in d if x != 0]
    ranks = rankdata(np.abs(np.array(nz)))
    obs = ranks[np.array(nz) > 0].sum()
    sums = [
        sum(r for r, s in zip(ranks, signs, strict=True) if s)
        for signs in itertools.product([0, 1], repeat=len(nz))
    ]
    lo = sum(s <= obs + 1e-9 for s in sums) / len(sums)
    hi = sum(s >= obs - 1e-9 for s in sums) / len(sums)
    return min(1.0, 2 * min(lo, hi))


@pytest.mark.parametrize(
    "d",
    [
        [1, 2, 3, 4, 5, 6],
        [1, 1, -2, 3, 3, 4, 0],  # ties and a zero
        [0.5, -0.25, 0.75, -1, 2, 2, -2, 0.1],
    ],
)
def test_wilcoxon_exact_matches_enumeration(d):
    w = wilcoxon(d, [0] * len(d))
    assert w.method == "exact"
    assert w.p == pytest.approx(_brute_p(d))
    assert w.n_zero == sum(x == 0 for x in d)


def test_wilcoxon_known_values():
    assert wilcoxon([1, 2, 3, 4, 5, 6], [0] * 6).p == pytest.approx(0.03125)
    assert wilcoxon([1, 2], [1, 2]).p == 1.0
    # normal approximation agrees roughly with the exact value
    rng = __import__("random").Random(0)
    d = [rng.gauss(0.3, 1) for _ in range(20)]
    exact = wilcoxon(d, [0] * 20).p
    normal = wilcoxon(d, [0] * 20, exact_max_n=0).p
    assert abs(exact - normal) < 0.02


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cli(tmp_path):
    from typer.testing import CliRunner

    from ocrfix.cli import app

    sys_ = write(tmp_path / "sft.jsonl", perfect_rows())
    r = CliRunner().invoke(
        app, ["eval", str(GOLD), f"sft={sys_}", str(tmp_path / "out"), "--profile", "rtx5090"]
    )
    assert r.exit_code == 0, r.output
    assert "sft: CER 0.0000" in r.output and (tmp_path / "out" / "report.md").is_file()
    r = CliRunner().invoke(app, ["eval", str(GOLD), str(tmp_path / "out")])
    assert r.exit_code != 0


def test_config_has_eval_section():
    cfg = load_config("rtx5090")
    assert cfg.eval.worst_n == 20 and cfg.eval.baseline
