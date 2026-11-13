"""The damage generators (`corrupt/ops.py`, `corrupt_text`): rates, types, spans, seeds.
`ocrfix corrupt` (tests/test_score_correct.py) applies them to chunks."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ocrfix.corpus import tag_spans
from ocrfix.corrupt import build_models, corrupt_text, rng_for
from ocrfix.corrupt.run import ALL
from ocrfix.errors import diff_ops
from ocrfix.schema import Span

OPTS = {
    "rate_scale": 1.0,
    "match_total": True,
    "line_width": (50, 90),
    "hyphen_space_prob": 0.5,
    "ligature_glyph_prob": 0.5,
    "dup_word_prob": 0.25,
    "garbled_wrap_prob": 0.5,
    "page_number_prob": 0.5,
    "header_phrases": {"en": ["Journal of Applied Research"], "ko": ["한국정보과학회 논문지"]},
    "junk": ["~", "·", "|"],
}

EN_WORDS = (
    "the efficient model reflects final figures of the proposed method which outperforms "
    "previous baselines on several benchmarks and the training remains stable across "
    "different settings while the flow of information is well-known in this field"
).split()
KO_WORDS = (
    "본 연구에서는 제안한 방법이 기존 방법보다 우수함을 확인하였다 실험 결과는 표에 "
    "나타내었으며 정확도가 높고 학습이 안정적으로 이루어졌다 데이터의 규모가 작은 경우에도 "
    "성능이 유지되는 것을 보였다"
).split()

# measured rates (ops per 1000 chars); spacing split by script like errors.table writes them
EN_RATES = {
    "char_confusion": 4.0,
    "hyphen_wrap": 2.0,
    "newline_in_sentence": 2.0,
    "missing_space": 1.0,
    "extra_space": 1.0,
    "ligature": 1.0,
    "quote_dash": 1.0,
    "dropped_char": 1.0,
    "duplicated_char": 1.0,
    "header_footer_insert": 0.5,
    "other": 0.5,
}
KO_RATES = {
    "extra_space": 3.0,
    "missing_space": 3.0,
    "newline_in_sentence": 2.0,
    "hangul_garbled": 2.0,
    "char_confusion": 2.0,
    "dropped_char": 1.0,
    "duplicated_char": 1.0,
    "header_footer_insert": 0.5,
    "other": 0.5,
}
TOTAL = 15.0
HIST = {"[0,1)": 0, "[1,2)": 0, "[2,5)": 0, "[5,10)": 0, "[10,20)": 10, "[20,50)": 0}


def group(lang: str, rates: dict[str, float], script: str) -> dict:
    scale = TOTAL / sum(rates.values())
    per = {t: r * scale for t, r in rates.items()}
    spacing = {
        f"{t}.{script}": per[t]
        for t in ("missing_space", "extra_space", "newline_in_sentence")
        if t in per
    }
    return {
        "lang": lang,
        "stratum": "*",
        "total_per_1000": TOTAL,
        "per_1000": per,
        "spacing_per_1000": spacing,
        "histogram": HIST,
    }


TABLE = {
    "groups": [group("en", EN_RATES, "en"), group("ko", KO_RATES, "ko")],
    "char_confusions": {
        "en": [
            {"src": "rn", "gold": "m", "count": 5},
            {"src": "1", "gold": "l", "count": 3},
            {"src": "0", "gold": "o", "count": 1},
        ],
        "ko": [{"src": "걸", "gold": "결", "count": 2}, {"src": "졍", "gold": "정", "count": 1}],
    },
}
CMAP = {
    "hangul_garbled": [
        {
            "source_glyphs": "Ol",
            "gold": "이",
            "gold_jamo": "ㅇㅣ",
            "count": 3,
            "source_raw": {"$\\mathrm{Ol}$": 2, "Ol": 1},
        },
        {"source_glyphs": "E", "gold": "는", "gold_jamo": "ㄴㅡㄴ", "count": 1, "source_raw": {}},
    ]
}


def make_clean(n: int, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        if i % 2:
            words = [rnd.choice(KO_WORDS) for _ in range(rnd.randint(25, 45))]
            mid = len(words) // 2
            text = " ".join(words[:mid]) + " $x_i \\ge 0$ 이고 " + " ".join(words[mid:]) + "."
            if i % 10 == 3:
                text += "\n<table><tr><td>정확도</td><td>98.2</td></tr></table>"
            lang = "ko"
        else:
            words = [rnd.choice(EN_WORDS) for _ in range(rnd.randint(30, 50))]
            mid = len(words) // 2
            text = (
                "“" + " ".join(words[:mid]) + "” — where $E = mc^2$ holds, "
                + " ".join(words[mid:]) + "."
            )  # fmt: skip
            lang = "en"
        rows.append(
            {
                "doc_id": f"doc{i // 5:04d}",
                "para_id": f"p{i % 5:05d}",
                "order": i % 5,
                "lang": lang,
                "text": text,
                "spans": [s.to_dict() for s in tag_spans(text)],
                "source_path": "x.md",
            }
        )
    return rows


def corrupt_rows(rows, seed: int, n_variants: int) -> list[dict]:
    """Each clean row damaged ``n_variants`` times, one random stream per (doc, row, variant)."""
    models = build_models(TABLE, CMAP, OPTS)
    out = []
    for row in rows:
        model = models.get(row["lang"]) or models[ALL]
        spans = [Span.from_dict(x) for x in row["spans"]]
        for v in range(n_variants):
            rng = rng_for(seed, f"{row['doc_id']}\x00table", row["para_id"], v)
            source, new_spans, ops, err = corrupt_text(row["text"], spans, model, rng)
            out.append({"doc_id": row["doc_id"], "src_para_id": row["para_id"], "variant": v,
                        "lang": row["lang"], "source_text": source, "gold_text": row["text"],
                        "spans": [x.to_dict() for x in new_spans], "gold_spans": row["spans"],
                        "error_rate": round(err, 3), "applied_ops": ops})  # fmt: skip
    return out


@pytest.fixture(scope="module")
def corrupted(tmp_path_factory):
    d = tmp_path_factory.mktemp("corrupt")
    rows = corrupt_rows(make_clean(1000), seed=7, n_variants=1)
    assert len(rows) == 1000
    return d, rows


def prose_chars(text: str) -> int:
    return sum(s.end - s.start for s in tag_spans(text) if s.kind == "prose")


@pytest.mark.parametrize(("lang", "rates"), [("en", EN_RATES), ("ko", KO_RATES)])
def test_rates_match_table(corrupted, lang, rates):
    """Re-measure with the errors aligner: total and per-type rates match the table."""
    _, rows = corrupted
    rows = [r for r in rows if r["lang"] == lang]
    counts: Counter[str] = Counter()
    chars = 0
    for r in rows:
        counts.update(op.type for op in diff_ops(r["source_text"], r["gold_text"]))
        chars += prose_chars(r["gold_text"])
    measured = {t: 1000 * n / chars for t, n in counts.items()}
    total = sum(measured.values())
    assert total == pytest.approx(TOTAL, rel=0.1)
    scale = TOTAL / sum(rates.values())
    for t, r in rates.items():
        assert measured.get(t, 0) == pytest.approx(r * scale, rel=0.3, abs=0.3), t
    # nothing the table doesn't have (beyond a little aligner noise)
    stray = sum(v for t, v in measured.items() if t not in rates)
    assert stray < 0.05 * total


def test_applied_ops_agree_with_aligner(corrupted):
    _, rows = corrupted
    applied = measured = 0
    for r in rows:
        applied += len(r["applied_ops"])
        measured += len(diff_ops(r["source_text"], r["gold_text"]))
    assert measured == pytest.approx(applied, rel=0.05)


def test_spans_untouched_and_consistent(corrupted):
    _, rows = corrupted
    wrapped = 0
    for r in rows:
        gold_blocks = [
            r["gold_text"][s["start"] : s["end"]] for s in r["gold_spans"] if s["kind"] != "prose"
        ]
        src_blocks = [
            r["source_text"][s["start"] : s["end"]] for s in r["spans"] if s["kind"] != "prose"
        ]
        garbled = [
            op["src"].lstrip() for op in r["applied_ops"] if op["type"] == "hangul_garbled"
            and op["wrapped"]
        ]  # fmt: skip
        wrapped += len(garbled)
        # clean math/tables verbatim and in order; plus the new $...$ runs
        assert sorted(src_blocks) == sorted(gold_blocks + garbled)
        # spans tile the text and agree with re-tagging the corrupted text
        assert r["spans"][0]["start"] == 0 and r["spans"][-1]["end"] == len(r["source_text"])
        assert [s.to_dict() for s in tag_spans(r["source_text"])] == r["spans"]
        assert r["gold_text"] != r["source_text"] or not r["applied_ops"]
        for op in r["applied_ops"]:
            assert r["gold_text"][op["gold_start"] : op["gold_end"]] == op["gold"]
            assert r["source_text"][op["src_start"] : op["src_end"]] == op["src"]
    assert wrapped > 0


def test_garbled_wrap_probability(corrupted):
    _, rows = corrupted
    ops = [o for r in rows for o in r["applied_ops"] if o["type"] == "hangul_garbled"]
    ol = [o for o in ops if o["gold"] == "이"]
    share = sum(o["wrapped"] for o in ol) / len(ol)
    assert share == pytest.approx(2 / 3, abs=0.15)  # 2 of 3 measured raw forms are $...$
    assert all(o["src"] == "E" for o in ops if o["gold"] == "는" and not o["wrapped"])


def test_seed_reproducibility():
    clean = make_clean(40)
    a = corrupt_rows(clean, seed=1, n_variants=3)
    assert a == corrupt_rows(clean, seed=1, n_variants=3)
    # input order doesn't matter: each (doc, para, variant) has its own stream
    b = corrupt_rows(list(reversed(clean)), seed=1, n_variants=3)
    key = lambda r: (r["doc_id"], r["src_para_id"], r["variant"])  # noqa: E731
    assert sorted(a, key=key) == sorted(b, key=key)
    assert len(a) == 120 and {r["variant"] for r in a} == {0, 1, 2}
    # variants differ, other seeds differ
    by = {key(r): r["source_text"] for r in a}
    assert sum(by[(k[0], k[1], 0)] != by[(k[0], k[1], 1)] for k in by if k[2] == 0) > 30
    c = corrupt_rows(clean, seed=2, n_variants=3)
    assert [r["source_text"] for r in a] != [r["source_text"] for r in c]
