"""Rule-based repair of LaTeX math that MinerU's formula OCR left broken (CPU only).

Two levels:

**Delimiters** (``repair_delimiters``, on the whole document outside tables), so that the
tagger (``corpus/tagger.py``) finds every formula as one span:
  - ``\\begin{env}`` of a display environment (equation, align, gather, multline, starred)
    with no ``\\end``: closed at the end of its paragraph; an ``\\end{x}`` that doesn't match
    the open ``\\begin{y}`` is renamed to ``\\end{y}`` (OCR often drops the ``*``).
  - ``$$``, ``\\[`` left open in a paragraph: closed at the paragraph end; ``\\(`` left open:
    closed at the end of its line. Only when the text after the opener looks like math (no
    Hangul, some LaTeX), so prose with a stray ``$`` is never wrapped.
  - ``$ x $`` (spaces inside, so not math by pandoc's rule) whose content looks like math:
    the spaces are dropped so it becomes math.
  - Closers without an opener are reported, never guessed.

**Inside a formula** (``fix_formula``, on each math span the tagger finds):
  - inner environments (``array``, ``cases``, ``matrix`` ...) balanced: a mismatched
    ``\\end`` renamed, a missing one appended;
  - ``\\left`` / ``\\right`` balanced with ``\\right.`` / ``\\left.``;
  - braces balanced: an unmatched ``}`` dropped, missing ``}`` appended (``\\{ \\}`` don't
    count);
  - spacing: runs of spaces collapsed; spaces next to ``_ ^ { }`` removed (MinerU writes
    ``x _ { i } ^ { 2 }``); never inside ``\\text{..}``-like groups, never after a lone
    backslash (``\\ ``), and never between a command and a letter (``\\alpha x``);
  - delimiters normalized: ``\\(..\\)`` -> ``$..$``, ``\\[..\\]`` -> ``$$..$$`` (the inline
    body trimmed so pandoc's rule holds).

Every function returns the text unchanged (same string) when there is nothing to fix, and a
Counter of what it did.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ocrfix.corpus.tagger import MATH_ENVS, _is_escaped, tag_spans

HANGUL = re.compile(r"[가-힣ㄱ-ㅣ]")
_MATHY = re.compile(r"\\[A-Za-z]+|[\^_=]|[{}]")
_SHORT_VAR = re.compile(r"[A-Za-z][A-Za-z0-9']{0,2}")
_ENV_TOKEN = re.compile(r"\\(begin|end)\{([A-Za-z]+\*?)\}")
_TOP_ENV = re.compile(r"(?:" + "|".join(MATH_ENVS) + r")\*?")
_LEFT_RIGHT = re.compile(r"\\(left|right)(?![A-Za-z])")
_TEXT_CMDS = {"text", "mbox", "hbox", "textrm", "textit", "textbf", "texttt", "textsf",
              "textnormal", "textup", "emph"}  # fmt: skip
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\\$|\s+|.", re.DOTALL)
_BLANK = re.compile(r"\n[ \t]*\n")


@dataclass
class Issue:
    kind: str
    offset: int
    snippet: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "offset": self.offset, "snippet": self.snippet}


@dataclass
class Result:
    text: str
    fixes: Counter = field(default_factory=Counter)
    issues: list[Issue] = field(default_factory=list)


def looks_math(s: str) -> bool:
    """Short variable (``x``, ``x_i`` style) or LaTeX-ish, and no Hangul."""
    s = s.strip()
    if not s or HANGUL.search(s):
        return False
    return bool(_MATHY.search(s) or _SHORT_VAR.fullmatch(s))


def snippet(text: str, i: int, w: int = 40) -> str:
    return text[max(0, i - w) : i + w].replace("\n", "\\n")


# --- inside one formula ----------------------------------------------------------------------


def _balance_envs(body: str, fixes: Counter) -> str:
    stack: list[str] = []
    out, pos = [], 0
    for m in _ENV_TOKEN.finditer(body):
        if _is_escaped(body, m.start()):
            continue
        kind, name = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(name)
            continue
        if not stack:
            continue  # an \end with no \begin: left for the reader (reported by the caller)
        top = stack.pop()
        if top != name:
            out += [body[pos : m.start()], f"\\end{{{top}}}"]
            pos = m.end()
            fixes["latex.env_end_renamed"] += 1
    out.append(body[pos:])
    body = "".join(out)
    if stack:
        body = body.rstrip() + "".join(f" \\end{{{n}}}" for n in reversed(stack))
        fixes["latex.env_end_added"] += len(stack)
    return body


def _balance_left_right(body: str, fixes: Counter) -> str:
    depth, orphans = 0, 0
    for m in _LEFT_RIGHT.finditer(body):
        if _is_escaped(body, m.start()):
            continue
        if m.group(1) == "left":
            depth += 1
        elif depth:
            depth -= 1
        else:
            orphans += 1
    if orphans:
        lead = len(body) - len(body.lstrip())
        body = body[:lead] + "\\left. " * orphans + body[lead:]
        fixes["latex.left_added"] += orphans
    if depth:
        body = body.rstrip() + " \\right." * depth
        fixes["latex.right_added"] += depth
    return body


def _balance_braces(body: str, fixes: Counter) -> str:
    depth, drop = 0, []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 2  # \{ \} \\ and any control symbol / the start of a command
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth:
                depth -= 1
            else:
                drop.append(i)
        i += 1
    if drop:
        body = "".join(ch for k, ch in enumerate(body) if k not in set(drop))
        fixes["latex.brace_dropped"] += len(drop)
    if depth:
        body = body.rstrip() + "}" * depth
        fixes["latex.brace_added"] += depth
    return body


def _group_end(body: str, i: int) -> int:
    """``body[i] == '{'``: index after its matching ``}`` (or len(body))."""
    depth = 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(body)


_TIGHT = {"_", "^", "{", "}"}


def normalize_spacing(body: str, display: bool) -> str:
    """MinerU-style spacing out: see the module docstring. Newlines in display math stay."""
    toks: list[str] = []
    i = 0
    while i < len(body):
        m = _TOKEN.match(body, i)
        t = m.group()
        if t[1:] in _TEXT_CMDS and t.startswith("\\"):
            j = m.end()
            while j < len(body) and body[j] in " \t":
                j += 1
            if j < len(body) and body[j] == "{":
                end = _group_end(body, j)
                toks.append(t + body[j:end])  # verbatim text group
                i = end
                continue
        toks.append(t)
        i = m.end()
    out: list[str] = []
    for k, t in enumerate(toks):
        if not t.isspace():
            out.append(t)
            continue
        if display and "\n" in t:
            out.append("\n" * t.count("\n"))
            continue
        prev = toks[k - 1] if k else ""
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if not prev or not nxt:
            continue  # leading / trailing
        if prev == "\\":
            out.append(t)  # a lone backslash before a space: control space, keep
            continue
        if nxt in _TIGHT or prev in ("_", "^", "{"):
            continue  # x _ { i } -> x_{i}; but "} + y" keeps its space
        out.append(" ")
    s = "".join(out)
    if display:
        s = "\n".join(line.strip(" \t") for line in s.split("\n"))
    return s


def split_formula(span: str) -> tuple[str, str, str] | None:
    """(opener, body, closer) of a math span as the tagger finds it."""
    for o, c in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)")):
        if span.startswith(o) and span.endswith(c) and len(span) >= len(o) + len(c):
            return o, span[len(o) : len(span) - len(c)], c
    if span.startswith("$") and span.endswith("$") and len(span) >= 2:
        return "$", span[1:-1], "$"
    m = re.match(r"\\begin\{([A-Za-z]+\*?)\}", span)
    if m:
        closer = f"\\end{{{m.group(1)}}}"
        if span.endswith(closer):
            return m.group(), span[m.end() : len(span) - len(closer)], closer
    return None


def fix_formula(span: str, *, spacing: bool = True, normalize: bool = True) -> Result:
    """One math span (with its delimiters) -> repaired span."""
    parts = split_formula(span)
    if parts is None:
        return Result(span)
    opener, body, closer = parts
    fixes: Counter = Counter()
    new = _balance_braces(_balance_left_right(_balance_envs(body, fixes), fixes), fixes)
    display = opener not in ("$", "\\(")
    if spacing:
        s = normalize_spacing(new, display)
        if not display:
            s = s.strip()
        if s != new:
            fixes["latex.spacing"] += 1
            new = s
    if normalize and opener in ("\\(", "\\[") and "$" not in new:
        opener, closer = ("$", "$") if opener == "\\(" else ("$$", "$$")
        if opener == "$":
            new = new.strip()
        fixes["latex.delimiter_normalized"] += 1
    if not new.strip():
        return Result(span)  # nothing left inside: don't touch
    out = opener + new + closer
    if out == span:
        return Result(span)
    why = check_formula(body, new, out, fixes["latex.brace_dropped"])
    if why:  # the repair isn't safe: the original stays, reported
        return Result(span, Counter({"latex.kept_original": 1}),
                      [Issue("formula_kept_original", 0, f"{why}: {span[:80]}")])  # fmt: skip
    return Result(out, fixes)


def _subseq(orig: str, new: str, may_drop: str, n_drop: int) -> bool:
    a = [c for c in orig if not c.isspace()]
    b = [c for c in new if not c.isspace()]
    j = 0
    for c in a:
        if j < len(b) and b[j] == c:
            j += 1
        elif c in may_drop and n_drop > 0:
            n_drop -= 1
        else:
            # b may have inserted chars (closers): skip ahead in b to find c
            k = j
            while k < len(b) and b[k] != c:
                k += 1
            if k == len(b):
                if c in may_drop and n_drop > 0:
                    n_drop -= 1
                    continue
                return False
            j = k + 1
    return True


def check_formula(body: str, new_body: str, span: str, dropped: int) -> str:
    """'' if the repaired formula is safe to use, else why not."""
    tags = [s for s in tag_spans(span) if s.kind != "prose"]
    if len(tags) != 1 or tags[0].kind != "math" or (tags[0].start, tags[0].end) != (0, len(span)):
        return "not one math span after repair"
    depth, i = 0, 0
    while i < len(new_body):
        c = new_body[i]
        if c == "\\":
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        if depth < 0:
            return "braces still unbalanced"
        i += 1
    if depth:
        return "braces still unbalanced"

    # a mismatched \end{x} renamed to the open env is a fix: env names don't count as text
    def no_env(x: str) -> str:
        return _ENV_TOKEN.sub(lambda m: "\\" + m.group(1), x)

    if not _subseq(no_env(body), no_env(new_body), "}", dropped):
        return "text would be lost"
    return ""


# --- delimiters over a document --------------------------------------------------------------


def _paragraph_end(text: str, i: int, stop: int) -> int:
    """End of the paragraph holding ``i`` (before the blank line), at most ``stop``."""
    m = _BLANK.search(text, i, stop)
    end = m.start() if m else stop
    while end > i and text[end - 1] in " \t\n":
        end -= 1
    return end


def _line_end(text: str, i: int, stop: int) -> int:
    j = text.find("\n", i, stop)
    end = stop if j < 0 else j
    while end > i and text[end - 1] in " \t":
        end -= 1
    return end


def _insertions(text: str, ins: list[tuple[int, str]]) -> str:
    out, pos = [], 0
    for at, s in sorted(ins, key=lambda x: x[0]):
        out += [text[pos:at], s]
        pos = at
    out.append(text[pos:])
    return "".join(out)


def _repair_envs(text: str, a: int, b: int, res: Result) -> str:
    """Top-level display environments in text[a:b]; returns the new text[a:b]."""
    seg = text[a:b]
    stack: list[tuple[str, int]] = []
    ins: list[tuple[int, str]] = []
    renames: list[tuple[int, int, str]] = []
    for m in _ENV_TOKEN.finditer(seg):
        if not _TOP_ENV.fullmatch(m.group(2)) or _is_escaped(seg, m.start()):
            continue
        if m.group(1) == "begin":
            if stack:  # a new top-level env while one is open: close the old one first
                name, at = stack.pop()
                end = _paragraph_end(seg, at, m.start())
                ins.append((end, f"\n\\end{{{name}}}"))
                res.fixes["latex.env_end_added"] += 1
            stack.append((m.group(2), m.end()))
        elif stack:
            name, _ = stack.pop()
            if name != m.group(2):
                renames.append((m.start(), m.end(), f"\\end{{{name}}}"))
                res.fixes["latex.env_end_renamed"] += 1
        else:
            res.issues.append(Issue("orphan_env_end", a + m.start(), snippet(seg, m.start())))
    for name, at in stack:
        end = _paragraph_end(seg, at, len(seg))
        ins.append((end, f"\n\\end{{{name}}}"))
        res.fixes["latex.env_end_added"] += 1
    if not ins and not renames:
        return seg
    out, pos = [], 0
    for s, e, rep in sorted(renames):
        out += [seg[pos:s], rep]
        pos = e
    out.append(seg[pos:])
    seg2 = "".join(out)
    # a rename before an insertion point changes its offset by the length difference
    shifted = [
        (at + sum(len(rep) - (e - st) for st, e, rep in renames if e <= at), s)
        for at, s in sorted(ins)
    ]
    return _insertions(seg2, shifted)


def _unescaped(seg: str, needle: str) -> list[int]:
    out, i = [], 0
    while (j := seg.find(needle, i)) >= 0:
        if not _is_escaped(seg, j):
            out.append(j)
        i = j + len(needle)
    return out


def _repair_paragraph(par: str, base: int, res: Result) -> str:
    """``$$`` / ``\\[`` / ``\\(`` left open in one paragraph (no blank line inside)."""
    ins: list[tuple[int, str]] = []
    # $$ ... (odd count): the last one opens
    dd = _unescaped(par, "$$")
    if len(dd) % 2:
        p = dd[-1]
        rest = par[p + 2 :]
        if looks_math(rest) and rest.strip():
            end = _paragraph_end(par, p + 2, len(par))
            ins.append((end, "\n$$" if "\n" in par[p:end] else "$$"))
            res.fixes["latex.display_closed"] += 1
        else:
            res.issues.append(Issue("unclosed_display", base + p, snippet(par, p)))
    for o, c, where in (("\\[", "\\]", "par"), ("\\(", "\\)", "line")):
        opens, closes = _unescaped(par, o), _unescaped(par, c)
        k = 0
        for p in opens:
            while k < len(closes) and closes[k] < p:
                k += 1
            if k < len(closes):
                k += 1
                continue
            end = (_paragraph_end if where == "par" else _line_end)(par, p + 2, len(par))
            if looks_math(par[p + 2 : end]):
                ins.append((end, c))
                res.fixes["latex.bracket_closed"] += 1
            else:
                res.issues.append(Issue("unclosed_bracket", base + p, snippet(par, p)))
            break  # one repair per kind per paragraph
    return _insertions(par, ins) if ins else par


_SPACED_DOLLAR = re.compile(r"(?<![\\$])\$(?!\$)([ \t]*)([^$\n]{1,120}?)([ \t]*)(?<![\\$])\$(?!\$)")


def tighten_inline(text: str, a: int, b: int, res: Result) -> str:
    """``$ x $`` -> ``$x$`` in text[a:b] (prose) when the content looks like math."""
    seg = text[a:b]

    def rep(m: re.Match) -> str:
        if not (m.group(1) or m.group(3)) or not looks_math(m.group(2)):
            return m.group()
        inner = m.group(2).strip()
        if not inner or inner[-1] == "\\":
            return m.group()
        res.fixes["latex.inline_tightened"] += 1
        return f"${inner}$"

    return _SPACED_DOLLAR.sub(rep, seg)


def repair_delimiters(text: str, regions: list[tuple[int, int]]) -> Result:
    """Delimiter repair in ``regions`` (the non-table parts of ``text``)."""
    res = Result(text)
    out, pos = [], 0
    for a, b in regions:
        out.append(text[pos:a])
        seg = _repair_envs(text, a, b, res)
        # paragraphs of the segment
        pieces, p = [], 0
        for m in _BLANK.finditer(seg):
            pieces.append(_repair_paragraph(seg[p : m.start()], a + p, res))
            pieces.append(m.group())
            p = m.end()
        pieces.append(_repair_paragraph(seg[p:], a + p, res))
        seg = "".join(pieces)
        seg = tighten_inline(seg, 0, len(seg), res)
        out.append(seg)
        pos = b
    out.append(text[pos:])
    new = "".join(out)
    res.text = new if new != text else text
    return res
