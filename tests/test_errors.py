from __future__ import annotations

import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.errors import TYPES, align, build, diff_ops, to_jamo, write_outputs
from ocrfix.errors.table import compare_forms, glyphs

# 10 hand-written (source, gold) pairs; together they cover every op type.
# Each expectation is the list of (key, src, gold) in order of position.
PAIRS = [
    (  # 1. ko: syllable confusion + OCR-inserted space + missing space
        "실험 걸과는 표 2에 나타 내었으며 결과를보면 좋다.",
        "실험 결과는 표 2에 나타내었으며 결과를 보면 좋다.",
        [
            ("char_confusion", "걸", "결"),
            ("extra_space.ko", " ", ""),
            ("missing_space.ko", "", " "),
        ],
    ),
    (  # 2. en: 1/l, rn/m confusions, a dropped char, an OCR space inside a word
        "Tab1e 3 surnmarizes the modl ac curacy.",
        "Table 3 summarizes the model accuracy.",
        [
            ("char_confusion", "1", "l"),
            ("char_confusion", "rn", "m"),
            ("dropped_char", "", "e"),
            ("extra_space.en", " ", ""),
        ],
    ),
    (  # 3. en: line-break hyphenation, incl. a real hyphen; a line break mid-sentence
        "The pro- posed method out-\nperforms the well- known\nbaseline.",
        "The proposed method outperforms the well-known baseline.",
        [
            ("hyphen_wrap", "- ", ""),
            ("hyphen_wrap", "-\n", ""),
            ("hyphen_wrap", " ", ""),
            ("newline_in_sentence.en", "\n", " "),
        ],
    ),
    (  # 4. ko: Hangul misread as math (misclassified-text span)
        "제안한 방법 ⟨M1⟩ 기존 방법보다 우수하다.",
        "제안한 방법이 기존 방법보다 우수하다.",
        [("hangul_garbled", " $\\mathrm{Ol}$", "이")],
    ),
    (  # 5. en: ligature glyph and a dropped ligature
        "An efﬁcient and ne-grained ow.",
        "An efficient and fine-grained flow.",
        [("ligature", "ﬁ", "fi"), ("ligature", "", "fi"), ("ligature", "", "fl")],
    ),
    (  # 6. en: straight quotes and double hyphen vs typographic
        "He said ''yes'' -- then left.",
        "He said “yes” — then left.",
        [("quote_dash", "''", "“"), ("quote_dash", "''", "”"), ("quote_dash", "--", "—")],
    ),
    (  # 7. en: running header + page number inserted mid-paragraph
        "The model 12 Journal of OCR Research converges quickly.",
        "The model converges quickly.",
        [("header_footer_insert", "12 Journal of OCR Research ", "")],
    ),
    (  # 8. ko + en: duplicated text, and an OCR line break between Hangul words
        "결과과 표에서 the the model\n성능이 높다.",
        "결과 표에서 the model 성능이 높다.",
        [
            ("duplicated_char", "과", ""),
            ("duplicated_char", "the ", ""),
            ("newline_in_sentence.ko", "\n", " "),
        ],
    ),
    (  # 9. ko: unreadable digit -> 〈?〉, and a longer rewrite -> other
        "반응 온도는 3口0 ℃이며 시간은 xq#zz7 이다.",
        "반응 온도는 3〈?〉0 ℃이며 시간은 120min 이다.",
        [("unrecoverable", "口", "〈?〉"), ("other", "xq#zz7", "120min")],
    ),
    (  # 10. en: a short junk insertion (other), placeholders untouched
        "Given ⟨M1⟩ the loss ⟨M2⟩ is ~ minimized.",
        "Given ⟨M1⟩ the loss ⟨M2⟩ is minimized.",
        [("other", "~ ", "")],
    ),
]

ROW4_PLACEHOLDERS = {"⟨M1⟩": "$\\mathrm{Ol}$"}


def gold_row(i: int, lang: str, stratum: str = "ko:low") -> dict:
    src, gold, _ = PAIRS[i]
    row = {
        "doc_id": f"d{i}",
        "para_id": "p00000",
        "lang": lang,
        "stratum": stratum,
        "source_text": src,
        "gold_text": gold,
        "placeholders": {},
        "span_labels": {},
    }
    if i == 3:
        row["placeholders"] = ROW4_PLACEHOLDERS
        row["span_labels"] = {"M1": "misclassified-text"}
    if i == 9:
        row["placeholders"] = {"⟨M1⟩": "$x$", "⟨M2⟩": "$L$"}
        row["span_labels"] = {"M1": "ok", "M2": "ok"}
    return row


def ops_of(i: int):
    src, gold, _ = compare_forms(gold_row(i, "en"))
    return diff_ops(src, gold)


@pytest.mark.parametrize("i", range(len(PAIRS)))
def test_pairs(i):
    expected = PAIRS[i][2]
    assert [(o.key, o.src, o.gold) for o in ops_of(i)] == expected


def test_pairs_cover_every_type():
    seen = {key.split(".")[0] for _, _, exp in PAIRS for key, _, _ in exp}
    assert seen == set(TYPES)


def test_positions_and_alignment():
    src, gold = "Tab1e rnodel", "Table model"
    (a, b) = diff_ops(src, gold)
    assert (a.src_start, a.src_end, a.gold_start, a.gold_end) == (3, 4, 3, 4)
    assert src[b.src_start : b.src_end] == "rn" and gold[b.gold_start : b.gold_end] == "m"
    # adjacent opcodes merge into one region; equal text yields nothing
    assert align("rn", "m") == [(0, 2, 0, 1)]
    assert diff_ops("same text", "same text") == []


def test_compare_forms():
    src, _, kept = compare_forms(gold_row(3, "ko"))
    assert src == "제안한 방법 $\\mathrm{Ol}$ 기존 방법보다 우수하다." and kept == {}


def test_jamo_and_glyphs():
    assert to_jamo("결과") == "ㄱㅕㄹㄱㅘ"
    assert to_jamo("이 a") == "ㅇㅣ a"
    assert glyphs(" $\\mathrm{Ol}$") == "Ol"


def test_tables(tmp_path):
    rows = [gold_row(i, "ko" if i in (0, 3, 7, 8) else "en") for i in range(len(PAIRS))]
    res = build(rows)
    assert res.n_rows == 10
    paths = write_outputs(res, tmp_path, top_k=50)
    assert [p.name for p in paths] == ["error_table.json", "confusion_map.json", "summary.md"]

    table = json.loads(paths[0].read_text("utf-8"))
    groups = {(g["lang"], g["stratum"]): g for g in table["groups"]}
    assert set(groups) == {("ko", "ko:low"), ("ko", "*"), ("en", "ko:low"), ("en", "*"), ("*", "*")}
    total = groups[("*", "*")]
    expected = Counter(key.split(".")[0] for _, _, exp in PAIRS for key, _, _ in exp)
    assert total["counts"] == {t: expected[t] for t in TYPES if expected[t]}
    assert total["n_ops"] == sum(expected.values())
    assert total["per_1000"]["char_confusion"] == round(3 * 1000 / total["n_chars"], 3)
    assert total["spacing_per_1000"]["missing_space.ko"] > 0
    assert sum(total["histogram"].values()) == 10
    # placeholders don't count as chars
    assert total["n_chars"] == sum(len(r["gold_text"]) for r in rows) - 2 * len("⟨M1⟩")
    top = table["char_confusions"]["*"]
    assert {(c["src"], c["gold"]) for c in top} == {("걸", "결"), ("1", "l"), ("rn", "m")}

    cmap = json.loads(paths[1].read_text("utf-8"))["hangul_garbled"]
    assert cmap == [
        {
            "source_glyphs": "Ol",
            "gold": "이",
            "gold_jamo": "ㅇㅣ",
            "count": 1,
            "source_raw": {"$\\mathrm{Ol}$": 1},
        }
    ]
    md = paths[2].read_text("utf-8")
    assert "## Rates per language" in md and "`Ol`" in md


def test_errors_cli(tmp_path):
    rows = [gold_row(i, "en") for i in range(len(PAIRS))]
    gold = tmp_path / "gold_train.jsonl"
    gold.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    res = CliRunner().invoke(
        app,
        [
            "errors",
            "table",
            str(gold),
            str(tmp_path / "out"),
            "--limit",
            "5",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "5 chunks" in res.output
    assert (tmp_path / "out" / "summary.md").exists()
