"""Rule-based repair of the tables MinerU writes (CPU only): HTML ``<table>`` and Markdown
pipe tables. Cell text is left as it is (the corrector's job); math inside cells gets
``latex.fix_formula``.

HTML (``fix_html_tables``):
  - extent: ``<table ..>`` to its matching ``</table>``; without one, to the end of the
    paragraph (MinerU writes a table on one line), and ``</table>`` is added;
  - rebuilt from its tags: missing ``</td>`` / ``</th>`` / ``</tr>`` closed, a cell outside a
    row starts one, text between cells is kept in the cell before it, stray closers dropped;
  - rows shorter than the table (colspan / rowspan counted) padded with empty ``<td></td>``;
    rows with no text in any cell dropped; wider rows reported, never cut;
  - tables holding a nested ``<table>`` are left as they are (reported).
Markdown (``fix_pipe_tables``): blocks of 2+ lines that start and end with ``|``:
  - a missing or malformed ``|---|`` line after the header added / rebuilt (alignment colons
    kept where there were any);
  - rows padded to the header's cell count; extra empty trailing cells dropped, extra cells
    with text reported;
  - ``|`` inside ``$..$``, inline code or escaped ``\\|`` never splits a cell.

A table that needs nothing comes back byte for byte.
"""

from __future__ import annotations

import re
from collections import Counter

from ocrfix.corpus.tagger import _find_math, tag_spans
from ocrfix.markup.latex import Issue, Result, _subseq, fix_formula, snippet

_TAG = re.compile(r"<(/?)(table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col)\b([^>]*)>", re.I)
_SPAN_ATTR = re.compile(r"\b(colspan|rowspan)\s*=\s*[\"']?(\d+)", re.I)
_BLANK = re.compile(r"\n[ \t]*\n")
_SECTIONS = {"thead", "tbody", "tfoot"}


def fix_cell_math(text: str, fixes: Counter, kept: Counter | None = None) -> str:
    """Math in one cell repaired; a formula kept as it was counts in ``kept``, not ``fixes``
    (so it doesn't make the table count as changed)."""
    spans = _find_math(text, 0, len(text))
    if not spans:
        return text
    out, pos = [], 0
    for s in spans:
        r = fix_formula(text[s.start : s.end])
        if "latex.kept_original" in r.fixes:
            if kept is not None:
                kept["latex.kept_original"] += 1
        else:
            fixes.update(r.fixes)
        out += [text[pos : s.start], r.text]
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


# --- HTML ------------------------------------------------------------------------------------


def _html_extent(text: str, start: int) -> tuple[int, bool]:
    """(end, closed) of the table opened at ``start``."""
    depth = 0
    for m in re.finditer(r"<table\b[^>]*>|</table\s*>", text[start:], re.I):
        if m.group().startswith("</"):
            depth -= 1
            if depth == 0:
                return start + m.end(), True
        else:
            depth += 1
    blank = _BLANK.search(text, start)
    end = blank.start() if blank else len(text)
    while end > start and text[end - 1] in " \t\n":
        end -= 1
    return end, False


class _Cell:
    def __init__(self, tag: str, attrs: str) -> None:
        self.tag, self.attrs, self.parts = tag, attrs, []

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def span(self, kind: str) -> int:
        for k, v in _SPAN_ATTR.findall(self.attrs):
            if k.lower() == kind:
                return max(1, int(v))
        return 1


def _parse(table: str) -> tuple[str, list] | None:
    """(opening tag, items) where items are ('sec', '<thead>') / ('row', [cells]) / ('raw', s)."""
    m = re.match(r"<table\b[^>]*>", table, re.I)
    if not m:
        return None
    body = table[m.end() :]
    body = re.sub(r"</table\s*>\s*$", "", body, flags=re.I)
    items: list = []
    row: list[_Cell] | None = None
    cell: _Cell | None = None
    pos = 0
    for t in _TAG.finditer(body):
        txt = body[pos : t.start()]
        pos = t.end()
        if txt:
            if cell is not None:
                cell.parts.append(txt)
            elif txt.strip():
                if row is None:
                    row = []
                    items.append(("row", row))
                cell = _Cell("td", "")
                row.append(cell)
                cell.parts.append(txt)
            elif row is None:
                items.append(("raw", txt))
        close, name, attrs = t.group(1) == "/", t.group(2).lower(), t.group(3)
        if name == "table":
            return None  # nested table
        if name in ("td", "th"):
            cell = None
            if not close:
                if row is None:
                    row = []
                    items.append(("row", row))
                cell = _Cell(name, attrs)
                row.append(cell)
        elif name == "tr":
            cell = None
            row = None
            if not close:
                row = []
                items.append(("row", row))
        elif name in _SECTIONS:
            cell, row = None, None
            items.append(("sec", t.group()))
        else:  # caption / colgroup / col: kept as is, outside rows
            if cell is not None:
                cell.parts.append(t.group())
            else:
                items.append(("raw", t.group()))
    tail = body[pos:]
    if tail:
        if cell is not None:
            cell.parts.append(tail)
        elif tail.strip():
            items.append(("raw", tail))
    return m.group(), items


def _widths(rows: list[list[_Cell]]) -> list[int]:
    """Grid width of each row, counting colspan and cells carried down by rowspan."""
    carry: list[int] = []  # rows left per carried column slot
    out = []
    for cells in rows:
        width = sum(1 for c in carry if c > 0)
        carry = [c - 1 for c in carry if c - 1 > 0]
        for c in cells:
            width += c.span("colspan")
            carry += [c.span("rowspan") - 1] * c.span("colspan") if c.span("rowspan") > 1 else []
        out.append(width)
    return out


def _render(open_tag: str, items: list) -> str:
    out = [open_tag]
    for kind, v in items:
        if kind == "row":
            out.append("<tr>")
            for c in v:
                out.append(f"<{c.tag}{c.attrs}>{c.text}</{c.tag}>")
            out.append("</tr>")
        else:
            out.append(v)
    out.append("</table>")
    return "".join(out)


def fix_html_table(table: str, closed: bool, fixes: Counter, issues: list, base: int) -> str:
    parsed = _parse(table)
    if parsed is None:
        issues.append(Issue("html_table_unparsed", base, snippet(table, 0)))
        return table
    open_tag, items = parsed
    local: Counter = Counter()
    if not closed:
        local["table.html_closed"] += 1
    rows = [v for k, v in items if k == "row"]
    # empty rows out
    keep = []
    for k, v in items:
        empty = k == "row" and not any(re.sub(r"<[^>]+>|&nbsp;", "", c.text).strip() for c in v)
        if empty and not any(c.span("rowspan") > 1 for c in v):
            local["table.html_empty_row_dropped"] += 1
            continue
        keep.append((k, v))
    items = keep
    rows = [v for k, v in items if k == "row"]
    if rows:
        widths = _widths(rows)
        target = max(set(widths), key=lambda w: (widths.count(w), w))
        for cells, w in zip(rows, widths, strict=True):
            if w < target:
                cells.extend(_Cell("td", "") for _ in range(target - w))
                local["table.html_row_padded"] += 1
            elif w > target:
                issues.append(Issue("html_row_too_wide", base, snippet(table, 0)))
    for cells in rows:
        for c in cells:
            t = c.text
            new = fix_cell_math(t, local, fixes)
            if new != t:
                c.parts = [new]
    out = _render(open_tag, items)
    # structure only counts as a fix when the rendering differs beyond whitespace
    if out == table:
        return table
    if re.sub(r"\s+", "", out) != re.sub(r"\s+", "", table) and not local:
        local["table.html_structure"] += 1
    elif not local:
        return table  # only whitespace between tags would change: keep the original
    why = _check_table(_cell_text(table), _cell_text(out), out, local["latex.brace_dropped"])
    if why:  # not safe: the original table stays, reported
        fixes["table.kept_original"] += 1
        issues.append(Issue("table_kept_original", base, f"{why}: {snippet(table, 0)}"))
        return table
    fixes.update(local)
    return out


def _cell_text(html: str) -> str:
    return re.sub(r"<[^>]*>", "", html)


def _check_table(orig_text: str, new_text: str, new: str, dropped: int) -> str:
    """'' if the repaired table is safe to use, else why not: the tagger must find it as one
    whole table, and no text may be lost (only stray ``}`` the math repair dropped)."""
    tags = [s for s in tag_spans(new) if s.kind != "prose"]
    lead = len(new) - len(new.lstrip())
    end = len(new.rstrip())
    if len(tags) != 1 or tags[0].kind != "table" or (tags[0].start, tags[0].end) != (lead, end):
        return "not one table after repair"
    if not _subseq(orig_text, new_text, "}", dropped):
        return "text would be lost"
    return ""


def fix_html_tables(text: str) -> Result:
    res = Result(text)
    out, pos = [], 0
    for m in re.finditer(r"<table\b[^>]*>", text, re.I):
        if m.start() < pos:
            continue  # inside a table already handled
        end, closed = _html_extent(text, m.start())
        table = text[m.start() : end]
        out.append(text[pos : m.start()])
        out.append(fix_html_table(table, closed, res.fixes, res.issues, m.start()))
        pos = end
    if not out:
        return res
    out.append(text[pos:])
    new = "".join(out)
    res.text = new if new != text else text
    return res


# --- Markdown pipe tables --------------------------------------------------------------------

_DELIM_CELL = re.compile(r"^\s*(:?)-{1,}(:?)\s*$")


def split_row(line: str) -> list[str]:
    """Cells of ``| a | b |`` (outer pipes required); ``|`` in math / code / ``\\|`` kept."""
    s = line.strip()
    inner = s[1:-1]
    cells, cur, i = [], [], 0
    in_math = in_code = False
    while i < len(inner):
        c = inner[i]
        if c == "\\" and i + 1 < len(inner):
            cur.append(inner[i : i + 2])
            i += 2
            continue
        if c == "`":
            in_code = not in_code
        elif c == "$" and not in_code:
            in_math = not in_math
        elif c == "|" and not in_math and not in_code:
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    cells.append("".join(cur))
    return cells


def _is_row(line: str) -> bool:
    s = line.strip()
    return len(s) >= 2 and s.startswith("|") and s.endswith("|")


def _render_row(cells: list[str], indent: str) -> str:
    return indent + "| " + " | ".join(c.strip() for c in cells) + " |"


def fix_pipe_block(lines: list[str], fixes: Counter, issues: list, base: int) -> list[str]:
    indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
    header = split_row(lines[0])
    n = len(header)
    out = [lines[0]]
    local: Counter = Counter()
    second = split_row(lines[1])
    if all(_DELIM_CELL.match(c) for c in second):
        if len(second) != n:
            aligns = [_DELIM_CELL.match(c).groups() for c in second][:n]
            aligns += [("", "")] * (n - len(aligns))
            out.append(indent + "|" + "|".join(f"{a}---{b}" for a, b in aligns) + "|")
            local["table.md_separator_fixed"] += 1
        else:
            out.append(lines[1])
        body = lines[2:]
    else:
        out.append(indent + "|" + "|".join("---" for _ in range(n)) + "|")
        local["table.md_separator_added"] += 1
        body = lines[1:]
    for line in body:
        cells = split_row(line)
        if len(cells) < n:
            cells += [""] * (n - len(cells))
            local["table.md_row_padded"] += 1
        elif len(cells) > n:
            extra = cells[n:]
            if all(not c.strip() for c in extra):
                cells = cells[:n]
                local["table.md_row_trimmed"] += 1
            else:
                issues.append(Issue("md_row_too_wide", base, line[:80]))
        new_cells = [fix_cell_math(c, local, fixes) for c in cells]
        changed = len(cells) != len(split_row(line)) or new_cells != cells
        out.append(_render_row(new_cells, indent) if changed else line)
    if not local:
        return lines
    strip = str.maketrans("", "", "|-:")
    nl = chr(10)
    why = _check_table(nl.join(lines).translate(strip), nl.join(out).translate(strip),
                       nl.join(out), local["latex.brace_dropped"])  # fmt: skip
    if why:
        fixes["table.kept_original"] += 1
        issues.append(Issue("table_kept_original", base, f"{why}: {lines[0][:80]}"))
        return lines
    fixes.update(local)
    return out


def fix_pipe_tables(text: str, skip: list[tuple[int, int]]) -> Result:
    """Pipe tables outside ``skip`` (HTML tables)."""
    res = Result(text)
    lines = text.split("\n")
    offs = [0]
    for line in lines:
        offs.append(offs[-1] + len(line) + 1)
    out: list[str] = []
    i = 0
    changed = False
    while i < len(lines):
        if _is_row(lines[i]) and not any(a <= offs[i] < b for a, b in skip):
            j = i
            while j < len(lines) and _is_row(lines[j]):
                j += 1
            if j - i >= 2:
                block = lines[i:j]
                new = fix_pipe_block(block, res.fixes, res.issues, offs[i])
                changed |= new is not block
                out += new
                i = j
                continue
        out.append(lines[i])
        i += 1
    if changed:
        new_text = "\n".join(out)
        res.text = new_text if new_text != text else text
    return res
