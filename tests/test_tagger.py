from __future__ import annotations

import pytest

from ocrfix.corpus.lang import detect_lang
from ocrfix.corpus.tagger import tag_spans


def kinds(text: str) -> list[tuple[str, str]]:
    """Non-prose spans as (kind, text); also checks that spans tile the text."""
    spans = tag_spans(text)
    assert [s.start for s in spans[1:]] == [s.end for s in spans[:-1]]
    if text:
        assert spans[0].start == 0 and spans[-1].end == len(text)
    return [(s.kind, text[s.start : s.end]) for s in spans if s.kind != "prose"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # inline $ in prose: currency is not math
        ("The kit costs $5 and the upgrade $10.", []),
        ("costs $5, then $x$ holds", [("math", "$x$")]),
        ("let $a + b$ be", [("math", "$a + b$")]),
        ("a $ b $ c", []),
        # escaped \$ is not a delimiter, \\$ is
        (r"price \$5 and \$6", []),
        (r"$a \$ b$", [("math", r"$a \$ b$")]),
        (r"line\\$x$", [("math", "$x$")]),
        # display math across lines
        ("see\n$$\n\\int_0^1 f\n\\, dx\n$$\nok", [("math", "$$\n\\int_0^1 f\n\\, dx\n$$")]),
        (r"\[ x^2 \] and \(y\)", [("math", r"\[ x^2 \]"), ("math", r"\(y\)")]),
        (r"\\[ not math \\]", []),
        # math delimiters never cross a blank line, environments may
        ("$a\n\nb$", []),
        ("$$a\n\nb$$", []),
        (
            "\\begin{align*}\na &= b\n\nc &= d\n\\end{align*}",
            [("math", "\\begin{align*}\na &= b\n\nc &= d\n\\end{align*}")],
        ),
        ("\\begin{gather}x\\end{gather}", [("math", "\\begin{gather}x\\end{gather}")]),
        ("\\begin{multline*}x\\end{multline*}", [("math", "\\begin{multline*}x\\end{multline*}")]),
        ("\\begin{align}x\\end{align*}", []),  # star must match
        ("\\begin{itemize}$x$\\end{itemize}", [("math", "$x$")]),
        # $ inside an HTML table belongs to the table
        (
            "cost $5 <table><tr><td>$x$</td><td>$9</td></tr></table> and $y$",
            [("table", "<table><tr><td>$x$</td><td>$9</td></tr></table>"), ("math", "$y$")],
        ),
        (
            "<TABLE><tr><td><table><tr><td>$</td></tr></table></td></tr></TABLE>",
            [("table", "<TABLE><tr><td><table><tr><td>$</td></tr></table></td></tr></TABLE>")],
        ),
        ("<table><tr><td>$x$", [("math", "$x$")]),  # unclosed table is prose
        # markdown pipe tables, $ inside them too
        (
            "text\n| a | b |\n|---|:-:|\n| $1 | $x|y$ |\nafter $z$",
            [("table", "| a | b |\n|---|:-:|\n| $1 | $x|y$ |"), ("math", "$z$")],
        ),
        ("a | b\n---\nc", []),  # a horizontal rule is not a delimiter row
    ],
)
def test_tagger(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("The energy is given by $E$.", "en"),
        ("실험 결과는 다음과 같다.", "ko"),
        ("실험 결과는 SNR 이 높다.", "ko"),
        ("본 논문에서는 Transformer encoder 와 attention mechanism 을 사용한다.", "mixed"),
        ("$$x^2$$", "other"),
        ("12", "other"),
        ("これは日本語です", "other"),
        ("<table><tr><td>정확도</td></tr></table>", "ko"),
    ],
)
def test_detect_lang(text, lang):
    assert detect_lang(text, tag_spans(text)) == lang
