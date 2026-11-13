"""Tag text as prose / math / table spans.

Tables are found first (HTML ``<table>`` with nesting, markdown pipe tables); math is then
searched only in the gaps between tables, so a ``$`` inside a table belongs to the table and
spans never overlap. Everything not math or table is prose; the returned spans tile the text.

Math delimiters: ``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]`` (none may cross a blank
line) and ``\\begin{env}...\\end{env}`` for equation/align/gather/multline and starred
variants (may cross blank lines). A delimiter preceded by an odd number of backslashes is
escaped. Inline ``$`` follows pandoc's rule (no space after the opener, no space before the
closer, closer not followed by a digit); a ``$`` that cannot close ends the attempt, so
"costs $5 and $10" stays prose.
"""

from __future__ import annotations

import re

from ocrfix.schema import Span

MATH_ENVS = ("equation", "align", "gather", "multline")

BLANK_LINE = re.compile(r"\n[ \t]*\n")
_BEGIN = re.compile(r"\\begin\{(" + "|".join(MATH_ENVS) + r")(\*?)\}")
_HTML_TABLE = re.compile(r"<table\b[^>]*>|</table\s*>", re.IGNORECASE)
_PIPE_DELIM = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")


def tag_spans(text: str) -> list[Span]:
    tables = _find_tables(text)
    math: list[Span] = []
    pos = 0
    for t in [*tables, Span(len(text), len(text), "table")]:
        math.extend(_find_math(text, pos, t.start))
        pos = t.end
    return _fill_prose(sorted(tables + math, key=lambda s: s.start), len(text))


def _is_escaped(text: str, i: int) -> bool:
    n = 0
    while i - n - 1 >= 0 and text[i - n - 1] == "\\":
        n += 1
    return n % 2 == 1


def _crosses_blank(text: str, start: int, end: int) -> bool:
    return BLANK_LINE.search(text, start, end) is not None


# --- tables -----------------------------------------------------------------------------


def _find_tables(text: str) -> list[Span]:
    spans = _find_html_tables(text)
    spans += [p for p in _find_pipe_tables(text) if not any(_overlaps(p, h) for h in spans)]
    return sorted(spans, key=lambda s: s.start)


def _find_html_tables(text: str) -> list[Span]:
    spans: list[Span] = []
    depth, start = 0, 0
    for m in _HTML_TABLE.finditer(text):
        if m.group().startswith("</"):
            if depth == 0:
                continue  # stray closer
            depth -= 1
            if depth == 0:
                spans.append(Span(start, m.end(), "table"))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return spans  # an unclosed <table> stays prose


def _find_pipe_tables(text: str) -> list[Span]:
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    spans: list[Span] = []
    i = 0
    while i + 1 < len(lines):
        header, delim = lines[i], lines[i + 1]
        if "|" in header and "|" in delim and _PIPE_DELIM.match(delim):
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            start = offsets[i] + len(header) - len(header.lstrip())
            end = offsets[j - 1] + len(lines[j - 1].rstrip())
            spans.append(Span(start, end, "table"))
            i = j
        else:
            i += 1
    return spans


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


# --- math -------------------------------------------------------------------------------


def _find_unescaped(text: str, needle: str, start: int, end: int) -> int:
    while True:
        i = text.find(needle, start, end)
        if i < 0 or not _is_escaped(text, i):
            return i
        start = i + 1


def _find_math(text: str, start: int, end: int) -> list[Span]:
    spans: list[Span] = []
    i = start
    while i < end:
        c = text[i]
        close = -1
        if c == "$" and not _is_escaped(text, i):
            if text.startswith("$$", i):
                j = _find_unescaped(text, "$$", i + 2, end)
                if j > i + 2 and not _crosses_blank(text, i, j):
                    close = j + 2
                else:
                    i += 2
                    continue
            else:
                close = _close_inline_dollar(text, i, end)
        elif c == "\\" and not _is_escaped(text, i):
            nxt = text[i + 1 : i + 2]
            if nxt in ("(", "["):
                closer = "\\)" if nxt == "(" else "\\]"
                j = _find_unescaped(text, closer, i + 2, end)
                if j >= 0 and not _crosses_blank(text, i, j):
                    close = j + 2
            elif m := _BEGIN.match(text, i, end):
                closer = f"\\end{{{m.group(1)}{m.group(2)}}}"
                j = text.find(closer, m.end(), end)
                if j >= 0:
                    close = j + len(closer)
        if close > 0:
            spans.append(Span(i, close, "math"))
            i = close
        else:
            i += 1
    return spans


def _close_inline_dollar(text: str, i: int, end: int) -> int:
    """End offset of the inline math opened at ``i``, or -1."""
    if i + 1 >= end or text[i + 1].isspace():
        return -1
    j = i + 1
    while True:
        j = _find_unescaped(text, "$", j, end)
        if j < 0 or _crosses_blank(text, i, j):
            return -1
        if text.startswith("$$", j):
            return -1
        if text[j - 1].isspace():
            return -1  # this $ can't close; let it try to open instead
        if j + 1 < end and text[j + 1].isdigit():
            j += 1
            continue
        return j + 1


# --- prose ------------------------------------------------------------------------------


def _fill_prose(spans: list[Span], n: int) -> list[Span]:
    out: list[Span] = []
    pos = 0
    for s in spans:
        if s.start > pos:
            out.append(Span(pos, s.start, "prose"))
        out.append(s)
        pos = s.end
    if pos < n:
        out.append(Span(pos, n, "prose"))
    return out
