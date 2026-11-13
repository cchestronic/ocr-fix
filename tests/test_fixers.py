from __future__ import annotations

import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.corpus import tag_spans
from ocrfix.fixers import FIXERS, Fixer, FixerError, Op, build_fixers, discover, register
from ocrfix.fixers.base import rebuild
from ocrfix.fixers.run import fix_document, fix_file
from ocrfix.schema import Record


def rec(text: str, i: int = 0, doc: str = "d", **extra) -> Record:
    return Record(doc, f"p{i:05d}", i, "en", text, tag_spans(text), "x.md", extra)


def fixer(name: str, **opts) -> Fixer:
    return discover()[name](**opts)


def fix_one(name: str, text: str, **opts) -> str | None:
    out, _ = fixer(name, **opts).apply([rec(text)])
    return out[0].text if out else None


def assert_spans_valid(r: Record) -> None:
    assert r.spans[0].start == 0 and r.spans[-1].end == len(r.text)
    assert all(a.end == b.start for a, b in zip(r.spans, r.spans[1:], strict=False))
    assert [s.to_dict() for s in r.spans] == [s.to_dict() for s in tag_spans(r.text)]


# --- unicode ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("An efﬁcient ﬂow of ﬀ", "An efficient flow of ff"),
        ("zero​width﻿ soft­hyphen", "zerowidth softhyphen"),
        ("ctrl\x07char\x1b", "ctrlchar"),
        ("ＡＢＣ１２３，ｘ", "ABC123,x"),
        ("전각　공백 그리고 ﾊﾝ", "전각 공백 그리고 ハン"),
        ("many   spaces\tand  nbsp  \nnext", "many spaces and nbsp\nnext"),
        ("  leading and trailing  ", "leading and trailing"),
        ("money ＄5 and pipe ｜ stay", "money ＄5 and pipe ｜ stay"),  # no new $ or | syntax
        ("math $x  ＋  y$ keeps width", "math $x  ＋  y$ keeps width"),
        ("ligature in math $\\text{ﬁ}$", "ligature in math $\\text{fi}$"),
    ],
)
def test_unicode(before, after):
    out, ops = fixer("unicode").apply([rec(before)])
    assert out[0].text == after
    assert_spans_valid(out[0])
    assert [o.op for o in ops] == ([] if before == after else ["edit"])


def test_unicode_options_and_empty():
    assert fix_one("unicode", "ﬁ  x", ligatures=False) == "ﬁ x"
    assert fix_one("unicode", "ＡＢ", width=False) == "ＡＢ"
    out, ops = fixer("unicode").apply([rec("​​")])
    assert out == [] and ops[0].op == "delete" and ops[0].rule == "unicode.empty"


# --- line_join -------------------------------------------------------------------------------


@pytest.fixture
def wordlist(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("proposed 120\nbenchmark 40\nrare 1\n", encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("The method\nworks well", "The method works well"),
        ("First sentence.\nSecond one", "First sentence.\nSecond one"),
        ("A question?\nyes", "A question?\nyes"),
        ("see (Fig. 2)\nnext", "see (Fig. 2)\nnext"),
        ("ends with colon:\n- item one\n- item two", "ends with colon:\n- item one\n- item two"),
        ("text\n# Heading", "text\n# Heading"),
        ("text\n| a | b |", "text\n| a | b |"),
        ("The pro-\nposed method", "The proposed method"),
        ("a well-\nknown result", "a well-known result"),
        ("COVID-\n19 cases", "COVID-\n19 cases"),
        (
            "본 연구에서는 제안한\n방법을 사용하였다.\n다음 문장",
            "본 연구에서는 제안한 방법을 사용하였다.\n다음 문장",
        ),
        ("inline $x$\nand more", "inline $x$\nand more"),  # math touches the break
        ("before\n$$x^2$$\nafter", "before\n$$x^2$$\nafter"),
        ("spaces  \n   around", "spaces around"),
    ],
)
def test_line_join(before, after, wordlist):
    out, _ = fixer("line_join", wordlist=wordlist).apply([rec(before)])
    assert out[0].text == after
    assert_spans_valid(out[0])


def test_line_join_wordlist_options(wordlist):
    # without a list the hyphen always stays; min_count filters rare words
    assert fix_one("line_join", "pro-\nposed") == "pro-posed"
    assert fix_one("line_join", "ra-\nre", wordlist=wordlist) == "rare"
    assert fix_one("line_join", "ra-\nre", wordlist=wordlist, min_count=5) == "ra-re"
    assert fix_one("line_join", "pro-\nposed", wordlist=wordlist, dehyphenate=False) == "pro-posed"
    with pytest.raises(FixerError, match="no such file"):
        fixer("line_join", wordlist="/nonexistent/words.txt")


# --- layout_junk -----------------------------------------------------------------------------


def test_layout_junk_headers_page_numbers_gutters():
    doc = [rec(f"Journal of Things\nBody {i} text.\n{i + 1}", i) for i in range(4)]
    doc += [
        rec("- 12 -", 4),
        rec("Page 7", 5),
        rec("1 alpha line\n2 beta line\n3 gamma line", 6),
        rec("10\n11\n12\n13", 7),
        rec("Body $x$\nJournal of Things\n<table><tr><td>Journal of Things</td></tr></table>", 8),
        rec("The year 2024 was good.", 9),
    ]
    out, ops = fixer("layout_junk").apply(doc)
    got = {r.para_id: r.text for r in out}
    assert got == {
        "p00000": "Body 0 text.",
        "p00001": "Body 1 text.",
        "p00002": "Body 2 text.",
        "p00003": "Body 3 text.",
        "p00006": "alpha line\nbeta line\ngamma line",
        # header line removed; the same text inside the table is untouched
        "p00008": "Body $x$\n<table><tr><td>Journal of Things</td></tr></table>",
        "p00009": "The year 2024 was good.",
    }
    for r in out:
        assert_spans_valid(r)
    by = {o.src_para_ids[0]: o for o in ops}
    assert {k for k, o in by.items() if o.op == "delete"} == {"p00004", "p00005", "p00007"}
    assert by["p00007"].rule == "layout_junk.gutter"
    assert by["p00000"].rule == "layout_junk.header+page_number"
    assert by["p00000"].evidence["removed"][0] == {"rule": "header", "line": "Journal of Things"}


def test_layout_junk_uses_page_field():
    # repeated in 4 paragraphs, but they are on only 2 pages -> not a running header
    doc = [rec(f"Running Title\nBody {i}.", i, page=i // 2) for i in range(4)]
    out, ops = fixer("layout_junk").apply(doc)
    assert ops == [] and [r.text for r in out] == [r.text for r in doc]
    assert fixer("layout_junk", min_repeats=2).apply(doc)[1]


# --- registry and ops ------------------------------------------------------------------------


@register("test_merge_split")
class MergeSplitFixer(Fixer):
    """Test-only, doc "a": merges p00001 into p00000, splits p00002 at its first '|',
    deletes p00003."""

    def apply(self, paras):
        if paras[0].doc_id != "a":
            return paras, []
        by = {r.para_id: r for r in paras}
        out, ops = [], []
        for r in paras:
            if r.para_id == "p00000" and "p00001" in by:
                merged = rebuild(r, [("prose", r.text + " " + by["p00001"].text)])
                out.append(merged)
                ops.append(Op(r.doc_id, "merge", ["p00000", "p00001"], ["p00000"], "t.merge"))
            elif r.para_id == "p00001":
                continue
            elif r.para_id == "p00002" and "|" in r.text:
                a, b = r.text.split("|", 1)
                out += [
                    rebuild(r, [("prose", a)]),
                    rebuild(replace(r, para_id="p00002.1"), [("prose", b)]),
                ]
                ops.append(Op(r.doc_id, "split", ["p00002"], ["p00002", "p00002.1"], "t.split"))
            elif r.para_id == "p00003":
                ops.append(Op(r.doc_id, "delete", ["p00003"], [], "t.delete"))
            else:
                out.append(r)
        return out, ops


@register("test_sloppy")
class SloppyFixer(Fixer):
    """Test-only: changes text without logging it."""

    def apply(self, paras):
        return [rebuild(paras[0], [("prose", paras[0].text + "!")]), *paras[1:]], []


def test_build_fixers_from_config():
    names = [
        f.name
        for f in build_fixers({"order": ["unicode", "line_join"], "unicode": {"enabled": True}})
    ]
    assert names == ["unicode"]  # line_join has no block -> disabled
    fx = build_fixers({"order": ["test_merge_split"], "test_merge_split": {"enabled": True}})
    assert [f.name for f in fx] == ["test_merge_split"]  # registered outside the package
    with pytest.raises(ValueError, match="unknown fixer"):
        build_fixers({"order": ["nope"]})
    with pytest.raises(ValueError, match="unknown option"):
        build_fixers({"order": ["unicode"], "unicode": {"enabled": True, "bogus": 1}})
    assert {"unicode", "line_join", "layout_junk"} <= set(FIXERS)


def test_default_config_enables_only_unicode():
    from omegaconf import OmegaConf

    from ocrfix.config import load_config

    fix = OmegaConf.to_container(load_config("rtx5090").fix, resolve=True)
    assert [f.name for f in build_fixers(fix)] == ["unicode"]


def test_unlogged_change_is_refused():
    with pytest.raises(FixerError, match="text changed without op"):
        fix_document([rec("a"), rec("b", 1)], [SloppyFixer()])


def _read(path):
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


def test_ops_explain_every_identity_change(tmp_path):
    docs = [
        [
            rec("Head ﬁne\nline", 0, "a"),
            rec("second", 1, "a"),
            rec("left | right", 2, "a"),
            rec("gone", 3, "a"),
            rec("12", 4, "a"),
            rec("untouched.", 5, "a"),
        ],
        [rec(f"Running\nx{i}.", i, "b") for i in range(3)] + [rec("plain.", 3, "b")],
    ]
    src = tmp_path / "in.jsonl"
    src.write_text(
        "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for d in docs for r in d)
    )
    fixers = build_fixers(
        {
            "order": ["unicode", "layout_junk", "test_merge_split", "line_join"],
            "unicode": {"enabled": True},
            "layout_junk": {"enabled": True},
            "test_merge_split": {"enabled": True},
            "line_join": {"enabled": True},
        }
    )
    stats = fix_file(src, tmp_path / "out.jsonl", tmp_path / "ops.jsonl", fixers)
    assert stats.docs == 2

    before = {(r["doc_id"], r["para_id"]): r["text"] for r in _read(src)}
    after = {(r["doc_id"], r["para_id"]): r for r in _read(tmp_path / "out.jsonl")}
    ops = _read(tmp_path / "ops.jsonl")
    assert {o["op"] for o in ops} == {"edit", "delete", "merge", "split"}
    for o in ops:
        assert set(o) == {"doc_id", "op", "src_para_ids", "dst_para_ids", "rule", "evidence"}

    def explained(key, field_names):
        return any(o["doc_id"] == key[0] and key[1] in o[f] for o in ops for f in field_names)

    for key in before.keys() - after.keys():  # disappeared
        assert explained(key, ["src_para_ids"]), key
    for key in after.keys() - before.keys():  # appeared
        assert explained(key, ["dst_para_ids"]), key
    for key in before.keys() & after.keys():  # same id, text changed
        if after[key]["text"] != before[key]:
            assert explained(key, ["dst_para_ids"]), key
            assert after[key]["fixer"]
        else:
            assert after[key]["fixer"] == [] or explained(key, ["dst_para_ids"])
    # concrete outcomes
    a0 = after[("a", "p00000")]
    assert a0["text"] == "Head fine line second"
    assert a0["original_text"] == "Head ﬁne\nline\n\nsecond"
    assert a0["fixer"] == ["line_join", "test_merge_split", "unicode"]
    assert after[("a", "p00002.1")]["text"] == "right"
    assert ("a", "p00004") not in after  # page number paragraph deleted
    assert [after[("b", f"p{i:05d}")]["text"] for i in range(3)] == ["x0.", "x1.", "x2."]
    assert after[("a", "p00005")]["original_text"] == "untouched."


def test_fix_cli(fixtures_dir, tmp_path):
    from ocrfix.corpus import ingest

    paras = tmp_path / "corpus.jsonl"
    ingest(fixtures_dir / "sample_mineru", paras)
    res = CliRunner().invoke(
        app,
        ["fix", str(paras), str(tmp_path / "fixed.jsonl"), "--set", "fix.layout_junk.enabled=true"],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "fixed.ops.jsonl").exists()
    n_in = len(paras.read_text().splitlines())
    rows = _read(tmp_path / "fixed.jsonl")
    assert len(rows) <= n_in and all("original_text" in r and "fixer" in r for r in rows)
    res = CliRunner().invoke(
        app,
        [
            "fix",
            str(paras),
            str(tmp_path / "f2.jsonl"),
            "--ops",
            str(tmp_path / "o.jsonl"),
            "--limit",
            "1",
        ],
    )
    assert res.exit_code == 0 and "1 docs" in res.output and (tmp_path / "o.jsonl").exists()
