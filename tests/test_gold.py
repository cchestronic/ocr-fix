from __future__ import annotations

import hashlib
import json
import random

import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.corpus import tag_spans
from ocrfix.gold import import_annotations, sample, split_docs, validate_row, write_sample
from ocrfix.gold.features import looks_like_garbled_hangul, paragraph_features
from ocrfix.gold.labelstudio import parse_span_labels, read_export
from ocrfix.schema import Record

# --- validator ---------------------------------------------------------------------------


def row(source, gold, *, lang="en", placeholders=None, labels=None, **kw):
    phs = placeholders if placeholders is not None else {}
    return {
        "doc_id": "d",
        "para_id": "p00000",
        "lang": lang,
        "source_text": source,
        "placeholders": phs,
        "gold_text": gold,
        "span_labels": labels if labels is not None else {k.strip("⟨⟩"): "ok" for k in phs},
        **kw,
    }


def rules(r, **kw):
    return [v.rule for v in validate_row(r, **kw)]


EN = "The energy is ⟨M1⟩ as shown in Tab1e 1 of the rnain paper."
EN_PH = {"⟨M1⟩": "$E = mc^2$"}


def test_valid_row_passes():
    gold = "The energy is ⟨M1⟩ as shown in Table 1 of the main paper."
    assert rules(row(EN, gold, placeholders=EN_PH)) == []
    # unchanged gold (a clean paragraph) is fine too
    assert rules(row(EN, EN, placeholders=EN_PH)) == []


def test_empty_gold():
    assert rules(row(EN, "  \n", placeholders=EN_PH)) == ["empty_gold"]


def test_length_ratio():
    assert "length_ratio" in rules(row(EN, "⟨M1⟩ too short.", placeholders=EN_PH))
    assert "length_ratio" in rules(row(EN, EN + " " + EN[:20], placeholders=EN_PH))
    # short sources skip the ratio check
    assert rules(row("Tab1e", "Table 1 (cont.)"), ratio_min_chars=20) == []
    assert rules(row("Tab1e", "Table 1 (cont.)"), ratio_min_chars=0) == ["length_ratio"]


@pytest.mark.parametrize(
    "gold",
    [
        "The energy is as shown in Table 1 of the main paper, see eq.",  # dropped
        "The energy is ⟨M1⟩ ⟨M1⟩ as shown in Table 1 of the main paper.",  # duplicated
        "The energy is ⟨M2⟩ as shown in Table 1 of the main paper.",  # renumbered
    ],
)
def test_placeholders_must_survive(gold):
    assert rules(row(EN, gold, placeholders=EN_PH)) == ["placeholders"]


def test_placeholder_order():
    src = "Let ⟨M1⟩ and ⟨M2⟩ be the inputs of the network here."
    phs = {"⟨M1⟩": "$x$", "⟨M2⟩": "$y$"}
    assert rules(
        row(src, "Let ⟨M2⟩ and ⟨M1⟩ be the inputs of the network here.", placeholders=phs)
    ) == ["placeholders"]


KO = "제안한 방법 ⟨M1⟩ 기존 방법보다 정확도가 높다는 것을 보였다."
KO_PH = {"⟨M1⟩": "$\\mathrm{Ol}$"}
KO_GOLD = "제안한 방법이 기존 방법보다 정확도가 높다는 것을 보였다."


def test_misclassified_text_may_drop_placeholder():
    r = row(KO, KO_GOLD, lang="ko", placeholders=KO_PH, labels={"M1": "misclassified-text"})
    assert rules(r) == []
    # ...but a misclassified span must not stay a placeholder
    r = row(KO, KO, lang="ko", placeholders=KO_PH, labels={"M1": "misclassified-text"})
    assert rules(r) == ["placeholders"]
    # and dropping it without the label is a violation
    assert rules(row(KO, KO_GOLD, lang="ko", placeholders=KO_PH)) == ["placeholders"]


def test_span_labels():
    gold = "The energy is ⟨M1⟩ as shown in Table 1 of the main paper."
    assert rules(row(EN, gold, placeholders=EN_PH, labels={})) == ["span_label"]
    assert rules(row(EN, gold, placeholders=EN_PH, labels={"M1": "fine"})) == ["span_label"]
    assert rules(row(EN, gold, placeholders=EN_PH, labels={"M1": "ok", "M2": "ok"})) == [
        "span_label"
    ]
    assert rules(row(EN, gold, placeholders=EN_PH, labels="M1=broken\n")) == []
    assert set(rules(row(EN, gold, placeholders=EN_PH, labels="M1 is fine"))) == {"span_label"}
    assert parse_span_labels("⟨M1⟩: ok\nT1=broken") == {"M1": "ok", "T1": "broken"}


def test_no_new_latin_words_in_ko():
    src = "본 연구에서는 rnodel 구조를 사용하여 실험을 수행하였다고 보고한다."
    # an OCR fix of a Latin word that is in the source is allowed
    assert rules(row(src, src.replace("rnodel", "model"), lang="ko")) == []
    # a new word is not (translation, explanation, ...)
    added = src.replace("구조를", "structure 구조를")
    assert rules(row(src, added, lang="ko")) == ["latin_in_ko"]
    # mixed paragraphs are not checked
    assert rules(row(src, added, lang="mixed")) == []
    # Latin from the raw text of a misclassified math span counts as source text
    s2 = "결과는 그림 ⟨M1⟩ 에 나타낸 것과 같이 매우 좋은 성능을 보였다."
    r = row(
        s2,
        "결과는 그림 Fig 에 나타낸 것과 같이 매우 좋은 성능을 보였다.",
        lang="ko",
        placeholders={"⟨M1⟩": "$\\mathrm{Fig}$"},
        labels={"M1": "misclassified-text"},
    )
    assert rules(r) == []


@pytest.mark.parametrize(
    ("label", "ok"),
    [
        ("", True),
        ("none", True),
        ("merge-with-next", True),
        ("split-at:10", True),
        ("split-at:0", False),
        ("split-at:999", False),
        ("split-at:", False),
        ("split", False),
    ],
)
def test_structure_label(label, ok):
    gold = "The energy is ⟨M1⟩ as shown in Table 1 of the main paper."
    r = row(EN, gold, placeholders=EN_PH, structure_label=label)
    assert (rules(r) == []) is ok


def test_unrecoverable_count():
    gold = "The energy is ⟨M1⟩ as shown in Table 〈?〉 of the main paper."
    assert rules(row(EN, gold, placeholders=EN_PH, unrecoverable_count=None)) == []
    assert rules(row(EN, gold, placeholders=EN_PH, unrecoverable_count=1)) == []
    assert rules(row(EN, gold, placeholders=EN_PH, unrecoverable_count=0)) == [
        "unrecoverable_count"
    ]


# --- features ------------------------------------------------------------------------------


def rec(text, doc="d", order=0, lang="en"):
    return Record(doc, f"p{order:05d}", order, lang, text, tag_spans(text), f"{doc}.md")


@pytest.mark.parametrize(
    ("raw", "garbled"),
    [
        ("$\\mathrm{Ol}$", True),
        ("$O$", True),
        ("$x = 1$", False),
        ("$\\alpha$", False),
        ("$x_i$", False),
        ("$12$", False),
        ("$\\mathrm{abcdefghij}$", False),
    ],
)
def test_garbled_hangul_detector(raw, garbled):
    assert looks_like_garbled_hangul(raw) is garbled


def test_paragraph_features():
    assert paragraph_features(rec("제안한 방법$\\mathrm{Ol}$ 좋다.")) == (
        "inline_math",
        "garbled_hangul",
    )
    # same span, but no Hangul next to it
    assert paragraph_features(rec("The value $O$ is small.")) == ("inline_math",)
    assert paragraph_features(rec("$$x^2$$")) == ("display_math",)
    assert paragraph_features(rec("The exam- ple and the counter-\nexample.")) == ("hyphenated",)
    assert paragraph_features(rec("state-of-the-art")) == ()
    assert paragraph_features(rec("<table><tr><td>a</td></tr></table>")) == ("table",)
    assert paragraph_features(rec("Transformer 모델을 사용한 OCR 교정")) == ("mixed_koen",)


# --- sampling ------------------------------------------------------------------------------


def make_corpus(tmp_path, n_docs=12, per_doc=12, seed=0):
    rnd = random.Random(seed)
    recs, scores = [], []
    for d in range(n_docs):
        doc = f"doc{d:03d}"
        for i in range(per_doc):
            kind = i % 4
            if kind == 0:
                text, lang = f"Paragraph {i} of {doc} has inline math $x_{i}$ here.", "en"
            elif kind == 1:
                text, lang = f"문서 {doc}의 {i}번째 문단은 한국어로 작성되었다.", "ko"
            elif kind == 2:
                text, lang = f"Plain English paragraph number {i} in {doc}.", "en"
            else:
                text, lang = (
                    f"이 문단{i}은 표를 포함한다.\n<table><tr><td>{i}</td></tr></table>",
                    "ko",
                )
            recs.append(rec(text, doc, i, lang))
            scores.append({"doc_id": doc, "para_id": f"p{i:05d}", "spike_frac": rnd.random()})
    write_jsonl(tmp_path / "corpus.jsonl", [r.to_dict() for r in recs])
    write_jsonl(tmp_path / "scores.jsonl", scores)
    return recs


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


QUOTAS = dict(
    per_lang={"ko": 10, "en": 10},
    tertiles={"low": 0.4, "mid": 0.3, "high": 0.3},
    features={"inline_math": 5, "table": 4},
)


def test_sample_strata_and_context(tmp_path):
    recs = make_corpus(tmp_path)
    res = sample(
        tmp_path / "scores.jsonl", tmp_path / "corpus.jsonl", seed=1, max_per_doc=3, **QUOTAS
    )
    strata = [r["stratum"] for r in res.rows]
    assert strata.count("feature:inline_math") == 5
    assert strata.count("feature:table") == 4
    assert sum(s.startswith("ko:") for s in strata) == 10
    assert [strata.count(f"en:{t}") for t in ("low", "mid", "high")] == [4, 3, 3]
    assert not res.shortfalls
    keys = [(r["doc_id"], r["para_id"]) for r in res.rows]
    assert len(set(keys)) == len(keys)
    per_doc = {d: sum(k[0] == d for k in keys) for d, _ in keys}
    assert max(per_doc.values()) <= 3

    by_key = {r.key: r for r in recs}
    for r in res.rows:
        order = r["order"]
        prev = by_key.get((r["doc_id"], f"p{order - 1:05d}"))
        nxt = by_key.get((r["doc_id"], f"p{order + 1:05d}"))
        assert r["context_prev"] == (prev.text if prev else None)
        assert r["context_next"] == (nxt.text if nxt else None)
        assert r["gold_text"] == "" and r["annotator_id"] == "" and r["span_labels"] == {}
        assert r["unrecoverable_count"] is None
        if "table" in r["features"]:
            assert "⟨T1⟩" in r["source_text"] and "<table>" in r["placeholders"]["⟨T1⟩"]

    # same data in another order -> same sample; another seed -> a different one
    lines = (tmp_path / "corpus.jsonl").read_text("utf-8").splitlines(keepends=True)
    random.Random(5).shuffle(lines)
    (tmp_path / "shuffled.jsonl").write_text("".join(lines), "utf-8")
    again = sample(
        tmp_path / "scores.jsonl", tmp_path / "shuffled.jsonl", seed=1, max_per_doc=3, **QUOTAS
    )
    assert again.rows == res.rows
    other = sample(
        tmp_path / "scores.jsonl", tmp_path / "corpus.jsonl", seed=2, max_per_doc=3, **QUOTAS
    )
    assert [r["para_id"] for r in other.rows] != [r["para_id"] for r in res.rows]


def test_sample_writes_label_studio(tmp_path):
    make_corpus(tmp_path, n_docs=3)
    res = sample(
        tmp_path / "scores.jsonl",
        tmp_path / "corpus.jsonl",
        seed=0,
        page_image="/pages/{source_stem}.png",
        **QUOTAS,
    )
    paths = write_sample(res, tmp_path / "out" / "sample.jsonl", image_url_prefix="/files?d=")
    assert [p.name for p in paths] == [
        "sample.jsonl",
        "sample.labelstudio.json",
        "sample.labelstudio.xml",
    ]
    tasks = json.loads(paths[1].read_text("utf-8"))
    assert len(tasks) == len(res.rows)
    t = next(t for t in tasks if t["data"]["placeholders"])
    assert t["data"]["page_image"].startswith("/files?d=/pages/doc")
    pred = {x["from_name"]: x["value"] for x in t["predictions"][0]["result"]}
    assert pred["gold_text"]["text"] == [t["data"]["source_text"]]
    assert pred["span_labels"]["text"][0].endswith("=ok")
    assert '<Image name="page"' in paths[2].read_text("utf-8")


# --- import / split ------------------------------------------------------------------------


def annotated(n_docs=10, per_doc=3, annotator="a1"):
    rows = []
    for d in range(n_docs):
        for i in range(per_doc):
            src = f"Paragraph {i} of document {d} cites Tab1e {i} and ⟨M1⟩ here."
            rows.append(
                row(
                    src,
                    src.replace("Tab1e", "Table"),
                    placeholders={"⟨M1⟩": f"$x_{i}$"},
                    doc_id=f"doc{d:03d}",
                    para_id=f"p{i:05d}",
                    order=i,
                    annotator_id=annotator,
                    structure_label="",
                    unrecoverable_count=None,
                )
            )
    return rows


def run_import(tmp_path, rows, name="out", seed=7, **kw):
    src = tmp_path / f"{name}.annotated.jsonl"
    write_jsonl(src, rows)
    out = tmp_path / name
    return import_annotations(src, out, seed=seed, **kw), out


def outputs(out):
    names = ["gold_train.jsonl", "gold_eval.jsonl", "gold_eval.manifest.json", "split.json"]
    return {n: (out / n).read_bytes() for n in names}


def test_split_by_document_is_deterministic(tmp_path):
    rows = annotated()
    rep, out = run_import(tmp_path, rows, "a")
    assert (rep.n_accepted, rep.train_docs, rep.eval_docs) == (30, 5, 5)
    train = [json.loads(x) for x in (out / "gold_train.jsonl").read_text("utf-8").splitlines()]
    evals = [json.loads(x) for x in (out / "gold_eval.jsonl").read_text("utf-8").splitlines()]
    assert not {r["doc_id"] for r in train} & {r["doc_id"] for r in evals}
    assert train[0]["gold_raw"].endswith("and $x_0$ here.") and "Table 0" in train[0]["gold_raw"]
    assert train[0]["span_labels"] == {"M1": "ok"} and train[0]["unrecoverable_count"] == 0

    # input order does not matter; rerunning into a fresh dir gives identical bytes
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)
    _, out_b = run_import(tmp_path, shuffled, "b")
    assert outputs(out) == outputs(out_b)
    # another seed gives another split
    _, out_c = run_import(tmp_path, rows, "c", seed=8)
    assert outputs(out)["split.json"] != outputs(out_c)["split.json"]

    # manifest matches the file
    manifest = json.loads((out / "gold_eval.manifest.json").read_text("utf-8"))
    assert manifest["sha256"] == hashlib.sha256((out / "gold_eval.jsonl").read_bytes()).hexdigest()
    assert manifest["n_records"] == len(evals) == 15
    line = (out / "gold_eval.jsonl").read_text("utf-8").splitlines()[0]
    assert manifest["records"][0]["sha256"] == hashlib.sha256(line.encode()).hexdigest()
    sha_line = (out / "gold_eval.jsonl.sha256").read_text("utf-8")
    assert sha_line == f"{manifest['sha256']}  gold_eval.jsonl\n"


def test_reimport_keeps_eval_docs_in_eval(tmp_path):
    _, out = run_import(tmp_path, annotated(n_docs=10), "x")
    first = json.loads((out / "split.json").read_text("utf-8"))
    rep, _ = run_import(tmp_path, annotated(n_docs=16), "x")
    second = json.loads((out / "split.json").read_text("utf-8"))
    assert set(first["eval_docs"]) <= set(second["eval_docs"])
    assert set(first["train_docs"]) <= set(second["train_docs"])
    assert (rep.train_docs, rep.eval_docs) == (8, 8)


def test_split_docs_balance():
    docs = [f"d{i}" for i in range(9)]
    side = split_docs(docs, seed=0, eval_frac=0.5)
    assert sum(s == "eval" for s in side.values()) == 4
    assert side == split_docs(list(reversed(docs)), seed=0, eval_frac=0.5)


def test_import_reports_violations_and_agreement(tmp_path):
    rows = annotated(n_docs=2, per_doc=2)
    second = annotated(n_docs=2, per_doc=2, annotator="b2")
    second[0]["gold_text"] = second[0]["gold_text"].replace("cites", "cite")  # 1 char apart
    bad = dict(rows[1], gold_text="")  # a1's p00001 of doc000 is empty...
    rows[1] = bad
    dup = dict(rows[2])  # ...and a1 labels doc001/p00000 twice
    rep, out = run_import(tmp_path, [*rows, *second, dup], "v")

    assert rep.violations == {"empty_gold": 1, "duplicate": 1}
    # every paragraph still has one valid annotation (b2's for the bad ones)
    assert rep.n_accepted == 4 and rep.rejected_paragraphs == 0
    viol = [json.loads(x) for x in (out / "violations.jsonl").read_text("utf-8").splitlines()]
    assert {(v["rule"], v["annotator_id"]) for v in viol} == {
        ("empty_gold", "a1"),
        ("duplicate", "a1"),
    }

    agr = rep.agreement
    assert agr["n_pairs"] == 3  # the empty gold is not compared
    assert agr["exact_match"] == pytest.approx(2 / 3, abs=1e-3)
    assert 0 < agr["max_ced"] < 0.05
    report = json.loads((out / "import_report.json").read_text("utf-8"))
    assert report["violations"] == {"duplicate": 1, "empty_gold": 1}


def test_label_studio_export_round_trip(tmp_path):
    ls = [
        {
            "data": {
                "doc_id": "d1",
                "para_id": "p00000",
                "order": 0,
                "stratum": "ko:low",
                "lang": "ko",
                "source_text": KO,
                "placeholders": KO_PH,
            },
            "annotations": [
                {
                    "completed_by": {"id": 3, "email": "ann@example.org"},
                    "result": [
                        {"from_name": "gold_text", "value": {"text": [KO_GOLD]}},
                        {"from_name": "span_labels", "value": {"text": ["M1=misclassified-text"]}},
                        {"from_name": "structure", "value": {"choices": ["split-at"]}},
                        {"from_name": "split_offset", "value": {"number": 7}},
                    ],
                },
                {"completed_by": 4, "was_cancelled": True, "result": []},
            ],
        }
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(ls, ensure_ascii=False), "utf-8")
    (r,) = read_export(path)
    assert r["annotator_id"] == "ann@example.org"
    assert r["structure_label"] == "split-at:7"
    assert validate_row(r) == []
    rep = import_annotations(path, tmp_path / "out", seed=0)
    assert rep.n_accepted == 1 and not rep.violations


def test_gold_cli(tmp_path):
    make_corpus(tmp_path, n_docs=4)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "gold", "sample", str(tmp_path / "scores.jsonl"), str(tmp_path / "corpus.jsonl"),
            str(tmp_path / "sample.jsonl"), "--set", "gold.sample.per_lang.ko=4",
            "--set", "gold.sample.per_lang.en=4", "--set", "gold.sample.per_lang.mixed=0",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    rows = [json.loads(x) for x in (tmp_path / "sample.jsonl").read_text("utf-8").splitlines()]
    assert rows and (tmp_path / "sample.labelstudio.xml").exists()
    for r in rows:  # annotate: keep the text as-is, label every span ok
        r.update(gold_text=r["source_text"], annotator_id="a")
        r["span_labels"] = {k.strip("⟨⟩"): "ok" for k in r["placeholders"]}
    write_jsonl(tmp_path / "annotated.jsonl", rows)
    res = runner.invoke(
        app, ["gold", "import", str(tmp_path / "annotated.jsonl"), str(tmp_path / "gold")]
    )
    assert res.exit_code == 0, res.output
    assert "accepted" in res.output
    assert (tmp_path / "gold" / "gold_train.jsonl").exists()


def test_gold_records(tmp_path):
    from ocrfix.gold.records import gold_records

    rows = annotated(n_docs=2, per_doc=1)
    _, out = run_import(tmp_path, rows, "r")
    n, skipped = gold_records(out / "gold_train.jsonl", tmp_path / "rec.jsonl")
    m, _ = gold_records(out / "gold_eval.jsonl", tmp_path / "rec2.jsonl")
    assert n + m == 2 and skipped == 0
    recs = [json.loads(x) for x in (tmp_path / "rec.jsonl").read_text("utf-8").splitlines()]
    recs += [json.loads(x) for x in (tmp_path / "rec2.jsonl").read_text("utf-8").splitlines()]
    for r in recs:
        assert "Tab1e" in r["text"] and "$x_0$" in r["text"]
        assert [s["kind"] for s in r["spans"]] == ["prose", "math", "prose"]
        assert r["cer"] == round(1 / len(r["gold_raw"]), 5)
