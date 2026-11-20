"""`ocrfix markup`: rule-based LaTeX and table repair (MinerU-style damage)."""

from __future__ import annotations

import json

from ocrfix.config import load_config
from ocrfix.corpus.tagger import tag_spans
from ocrfix.markup.latex import fix_formula, normalize_spacing, repair_delimiters
from ocrfix.markup.run import fix_document, run_markup
from ocrfix.markup.tables import fix_html_tables, fix_pipe_tables, split_row

OPTS = {
    "latex": {"delimiters": True, "formulas": True, "spacing": True,
              "normalize_delimiters": True},
    "tables": {"html": True, "markdown": True},
}  # fmt: skip


def _kinds(text: str) -> list[str]:
    return [s.kind for s in tag_spans(text) if s.kind != "prose"]


# --- formulas --------------------------------------------------------------------------------


def test_spacing_mineru_style():
    assert normalize_spacing("x _ { i } ^ { 2 } + y", False) == "x_{i}^{2} + y"
    assert normalize_spacing("\\alpha x", False) == "\\alpha x"  # command + letter: space kept
    assert normalize_spacing("\\text { a  b } _ { 1 }", False) == "\\text{ a  b }_{1}"
    assert normalize_spacing("a \\ b", False) == "a \\ b"  # control space kept
    assert normalize_spacing("a = b \\\\\n  c = d", True) == "a = b \\\\\nc = d"


def test_fix_formula_braces_left_right_envs():
    r = fix_formula("$\\frac { a } { b $")
    assert r.text == "$\\frac{a}{b}$" and r.fixes["latex.brace_added"] == 1
    r = fix_formula("$a } + b$")
    assert r.text == "$a + b$" and r.fixes["latex.brace_dropped"] == 1
    r = fix_formula("$$\\left( x + y$$")
    assert r.text == "$$\\left( x + y \\right.$$"
    r = fix_formula("$$\\begin{array}{cc} a & b \\end{matrix}$$")
    assert "\\end{array}" in r.text and "matrix" not in r.text
    r = fix_formula("$$\\begin{cases} a \\\\ b$$")
    assert r.text.endswith("\\end{cases}$$")
    # \{ \} are not braces
    assert fix_formula("$\\{x\\}$").text == "$\\{x\\}$"


def test_fix_formula_unchanged_is_same_object():
    s = "$x_{i}^{2}$"
    assert fix_formula(s).text is s


def test_normalize_delimiters():
    assert fix_formula("\\( x _ i \\)").text == "$x_i$"
    assert fix_formula("\\[ a = b \\]").text == "$$a = b$$"
    assert fix_formula("\\[ a = b \\]", normalize=False).text == "\\[a = b\\]"


# --- delimiters in a document ----------------------------------------------------------------


def test_unclosed_display_closed_at_paragraph_end():
    text = "앞 문단.\n\n$$\nE = m c ^ { 2 }\n\n다음 문단."
    r = repair_delimiters(text, [(0, len(text))])
    assert r.fixes["latex.display_closed"] == 1
    assert "$$\nE = m c ^ { 2 }\n$$\n\n다음" in r.text
    assert _kinds(r.text) == ["math"]


def test_prose_dollar_never_wrapped():
    text = "가격은 $5 이고 할인은 $ 없다.\n\n끝."
    r = repair_delimiters(text, [(0, len(text))])
    assert r.text is text and not r.fixes


def test_env_end_renamed_and_added():
    text = "\\begin{align*}\na &= b\n\\end{align}\n\n\\begin{equation}\nx = 1\n\n끝."
    r = repair_delimiters(text, [(0, len(text))])
    assert "\\end{align*}" in r.text and "\\end{align}\n" not in r.text
    assert "x = 1\n\\end{equation}\n\n끝." in r.text
    assert r.fixes == {"latex.env_end_renamed": 1, "latex.env_end_added": 1}
    assert _kinds(r.text) == ["math", "math"]


def test_spaced_inline_dollar_tightened():
    text = "여기서 $ x _ i $ 는 입력이고 $ 10 달러 $ 는 아니다."
    r = repair_delimiters(text, [(0, len(text))])
    assert "$x _ i$" in r.text and "$ 10 달러 $" in r.text
    assert r.fixes["latex.inline_tightened"] == 1


def test_orphan_closer_reported_not_guessed():
    text = "a = b \\end{equation}\n\n끝."
    r = repair_delimiters(text, [(0, len(text))])
    assert r.text is text and [i.kind for i in r.issues] == ["orphan_env_end"]


# --- tables ----------------------------------------------------------------------------------


def test_html_table_closed_and_padded():
    text = ("표 1.\n\n<table><tr><td>model<td>CER</tr><tr><td>m0</td></tr>"
            "<tr><td> </td><td></td></tr>\n\n다음.")  # fmt: skip
    r = fix_html_tables(text)
    assert ("<table><tr><td>model</td><td>CER</td></tr><tr><td>m0</td><td></td></tr></table>"
            in r.text)  # fmt: skip
    assert r.fixes["table.html_closed"] == 1 and r.fixes["table.html_row_padded"] == 1
    assert r.fixes["table.html_empty_row_dropped"] == 1
    assert _kinds(r.text) == ["table"]


def test_html_colspan_rowspan_counted():
    t = ("<table><tr><td colspan=\"2\">a</td></tr><tr><td rowspan=\"2\">b</td><td>c</td></tr>"
         "<tr><td>d</td></tr></table>")  # fmt: skip
    assert fix_html_tables(t).text is t  # 2 wide everywhere: nothing to do


def test_html_good_table_untouched_and_cell_math_fixed():
    good = "<table><tr><td>$x_{i}$</td></tr></table>"
    assert fix_html_tables(good).text is good
    r = fix_html_tables("<table><tr><td>$x _ { i$</td></tr></table>")
    assert r.text == "<table><tr><td>$x_{i}$</td></tr></table>"


def test_pipe_table_separator_and_rows():
    text = "| a | b | c |\n| 1 | 2 |\n| 4 | 5 | 6 | |\n\n끝."
    r = fix_pipe_tables(text, [])
    assert r.text.startswith("| a | b | c |\n|---|---|---|\n| 1 | 2 |  |\n| 4 | 5 | 6 |\n")
    assert r.fixes == {"table.md_separator_added": 1, "table.md_row_padded": 1,
                       "table.md_row_trimmed": 1}  # fmt: skip
    assert _kinds(r.text) == ["table"]


def test_pipe_split_keeps_math_code_escapes():
    assert split_row("| $|x|$ | `a|b` | c \\| d |") == [" $|x|$ ", " `a|b` ", " c \\| d "]


def test_good_pipe_table_untouched():
    text = "| a | b |\n|:--|--:|\n| 1 | 2 |\n"
    assert fix_pipe_tables(text, []).text is text


# --- whole documents and the command ---------------------------------------------------------


def test_fix_document_end_to_end():
    doc = ("# 결과\n\n본 연구는 $ \\alpha $ 를 사용한다. 손실은 \\( L _ { ce } \\) 이다.\n\n"
           "\\[ \\frac { 1 } { n } \\sum _ { i } x _ i \\]\n\n"
           "<table><tr><td>m0<td>$O ( n$</tr></table>\n\n| a | b |\n| 1 |\n")  # fmt: skip
    r = fix_document(doc, OPTS)
    t = r.text
    assert "$\\alpha$" in t and "$L_{ce}$" in t
    assert "$$\\frac{1}{n} \\sum_{i} x_i$$" in t
    assert "<table><tr><td>m0</td><td>$O ( n$</td></tr></table>" in t
    assert "| a | b |\n|---|---|\n| 1 |  |" in t
    assert [i.kind for i in r.issues] == []
    assert fix_document(t, OPTS).text is t  # idempotent


def test_run_markup_tree(tmp_path):
    src = tmp_path / "in"
    (src / "v").mkdir(parents=True)
    (src / "v" / "a.md").write_bytes("\ufeff식 $$\r\nx _ i\r\n\r\n끝\r\n".encode())
    (src / "v" / "b.md").write_bytes(b"clean text $x$.\n")
    (src / "v" / "c.md").write_bytes(b"\xff\xfe bad")
    cfg = load_config("cpu", ["markup.workers=0"])
    st = run_markup(cfg, src, tmp_path / "out")
    out = tmp_path / "out" / "v"
    assert (out / "b.md").read_bytes() == (src / "v" / "b.md").read_bytes()
    assert (out / "c.md").read_bytes() == b"\xff\xfe bad" and st.not_utf8 == ["v/c.md"]
    a = (out / "a.md").read_bytes().decode()
    assert a.startswith("\ufeff") and "x_i\r\n$$\r\n\r\n끝" in a
    assert st.docs == 3 and st.changed == 1
    rows = [json.loads(x) for x in (tmp_path / "out" / "markup_report.jsonl").read_text("utf-8")
            .splitlines()]  # fmt: skip
    assert [r["path"] for r in rows] == ["v/a.md"]


# --- a repair that isn't safe keeps the original ---------------------------------------------


def test_formula_kept_original_when_repair_loses_text(monkeypatch):
    import ocrfix.markup.latex as latex

    monkeypatch.setattr(latex, "normalize_spacing", lambda body, display: body.replace("y", ""))
    span = "$x _ { y $"
    r = latex.fix_formula(span)
    assert r.text is span
    assert r.fixes == {"latex.kept_original": 1}
    assert (
        r.issues[0].kind == "formula_kept_original" and "text would be lost" in r.issues[0].snippet
    )


def test_table_kept_original_when_repair_loses_text(monkeypatch):
    import ocrfix.markup.tables as tables

    real = tables._render
    monkeypatch.setattr(tables, "_render", lambda o, items: real(o, items).replace("CER", ""))
    text = "<table><tr><td>model<td>CER</tr></table>"
    r = tables.fix_html_tables(text)
    assert r.text is text and r.fixes == {"table.kept_original": 1}
    assert r.issues[0].kind == "table_kept_original"


def test_table_kept_original_when_not_one_table(monkeypatch):
    import ocrfix.markup.tables as tables

    monkeypatch.setattr(tables, "_render", lambda o, items: "<table><tr><td>model</td>")
    text = "<table><tr><td>model</td></tr>\n\n끝"
    r = tables.fix_html_tables(text)
    assert r.text is text and r.fixes == {"table.kept_original": 1}


def test_pipe_table_kept_original_when_text_lost(monkeypatch):
    import ocrfix.markup.tables as tables

    monkeypatch.setattr(tables, "_render_row", lambda cells, indent: "| |")
    text = "| a | b |\n| 1 |\n"
    r = tables.fix_pipe_tables(text, [])
    assert r.text is text and r.fixes == {"table.kept_original": 1}


def test_run_markup_resumes(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        (src / f"d{i}.md").write_text(f"식 $$\nx _ {i}\n\n끝\n", "utf-8")
    cfg = load_config("cpu", ["markup.workers=0"])
    out = tmp_path / "out"
    st = run_markup(cfg, src, out, limit=2)
    assert st.changed == 2 and st.skipped == 0
    st = run_markup(cfg, src, out)
    assert st.skipped == 2 and st.changed == 1
    rows = (out / "markup_report.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == 3


# --- blank lines inside display math ----------------------------------------------------------


def _math_spans(text: str) -> list[str]:
    return [text[s.start : s.end] for s in tag_spans(text) if s.kind == "math"]


def test_blank_line_inside_display_math_is_joined():
    for opener, closer in (("$$", "$$"), ("\\[", "\\]")):
        body = "a = b \\\\\n\nc = \\text{for all i}"  # a LaTeX line break, then a blank line
        text = f"식은 다음과 같다.\n\n{opener}\n{body}\n{closer}\n\n끝."
        assert _math_spans(text) == []  # the blank line hides the formula from the tagger
        r = fix_document(text, OPTS)
        spans = _math_spans(r.text)
        assert len(spans) == 1 and "\n\n" not in spans[0], r.text
        assert r.fixes["latex.display_blank_lines_joined"] == 1
        assert "식은 다음과 같다." in r.text and r.text.rstrip().endswith("끝.")


def test_prose_between_dollars_is_never_joined():
    # a stray $$, then real paragraphs (English and Korean), then another $$: not one formula
    for middle in ("The model is trained on clean pages.", "모델은 깨끗한 페이지로 학습한다."):
        text = f"$$\na = b\n\n{middle}\n\n$$\nc = d\n$$\n"
        r = fix_document(text, OPTS)
        assert middle in r.text and "latex.display_blank_lines_joined" not in r.fixes
        assert all(middle not in m for m in _math_spans(r.text))
