"""Op generators: the inverse of each ``errors`` type, applied to clean (gold) text.

Each generator looks at the clean paragraph and proposes one ``Edit`` (replace
``gold[start:end]`` with ``src``) in prose, away from every other edit, or ``None`` when the
paragraph has no place for it. Edits are chosen so that ``errors.diff_ops(source, gold)``
classifies them back as the same type.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ocrfix.errors.align import is_hangul, spacing_script

LIGATURE_GLYPHS = {"ffi": "ﬃ", "ffl": "ﬄ", "ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ"}
_LIGATURE = re.compile("ffi|ffl|ff|fi|fl")
# gold char -> what OCR may print instead
QUOTE_DASH = {
    "“": ['"', "''", "``"],
    "”": ['"', "''"],
    "‘": ["'", "`"],
    "’": ["'"],
    '"': ["“", "”"],
    "'": ["’"],
    "—": ["--", "-", "–"],
    "–": ["-", "--"],
}
_LATIN_WORD = re.compile(r"[A-Za-z]{5,}")
_COMPOUND = re.compile(r"[A-Za-z]{2,}-(?=[A-Za-z]{2,})")
_WORD = re.compile(r"[^\W\d_]{2,}")


@dataclass(frozen=True)
class Edit:
    type: str
    start: int
    end: int
    src: str
    script: str | None = None
    # (start, end) of a new math span inside ``src`` (hangul_garbled wrapped in $...$)
    math: tuple[int, int] | None = None
    info: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


class Paragraph:
    """Clean text + which positions edits may touch (prose, not near another edit)."""

    MARGIN = 2

    def __init__(self, text: str, prose: list[tuple[int, int]]):
        self.text = text
        self.free = [False] * (len(text) + 1)
        for s, e in prose:
            for i in range(s, e):
                self.free[i] = True
        self.prose = prose
        self._matches: dict[str, list[re.Match]] = {}

    def ok(self, start: int, end: int) -> bool:
        """Edit [start, end) (or an insertion at start == end) is allowed."""
        lo, hi = (start - 1, start + 1) if start == end else (start, end)
        return lo >= 0 and hi <= len(self.text) and all(self.free[lo:hi])

    def take(self, e: Edit) -> None:
        lo = max(0, e.start - self.MARGIN)
        hi = min(len(self.free), max(e.end, e.start + 1) + self.MARGIN)
        for i in range(lo, hi):
            self.free[i] = False

    def find(self, pattern: re.Pattern | str) -> list[re.Match]:
        return [m for m in self.matches(pattern) if self.ok(m.start(), m.end())]

    def matches(self, pattern: re.Pattern | str) -> list[re.Match]:
        """Every match in the text, cached: the text never changes, only what is free."""
        key = pattern if isinstance(pattern, str) else pattern.pattern
        if key not in self._matches:
            rx = re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern
            self._matches[key] = list(rx.finditer(self.text))
        return self._matches[key]


def _sample(rng: random.Random, lo: int, hi: int, pred: Callable[[int], bool]) -> int | None:
    """A uniformly random i in [lo, hi) with pred(i): random tries first (cheap when valid
    places are common), then every candidate (exact, and None when there is none). Either way
    each valid place is equally likely."""
    if hi <= lo:
        return None
    for _ in range(64):
        i = rng.randrange(lo, hi)
        if pred(i):
            return i
    return _pick(rng, [i for i in range(lo, hi) if pred(i)])


def _sample_match(rng: random.Random, p: Paragraph, ms: list[re.Match]) -> re.Match | None:
    if not ms:
        return None
    k = _sample(rng, 0, len(ms), lambda j: p.ok(ms[j].start(), ms[j].end()))
    return None if k is None else ms[k]


def _script_of(ch: str) -> str:
    if is_hangul(ch):
        return "ko"
    if ch.isascii() and ch.isalpha():
        return "en"
    return "other" if ch.isalpha() else ""


@dataclass
class Resources:
    """What generators draw on: measured pairs/maps and config knobs."""

    confusions: list[tuple[str, str, int]]  # (src, gold, count)
    garbled: list[dict[str, Any]]  # confusion_map entries
    opts: dict[str, Any]
    header_phrases: list[str]


Gen = Callable[[random.Random, Paragraph, Resources, str | None], Edit | None]


def _pick(rng: random.Random, items: list):
    return items[rng.randrange(len(items))] if items else None


def space_insert(rng, p, res, script):
    """extra_space: OCR splits a word (Korean: between two syllables)."""
    t = p.text
    i = _sample(rng, 1, len(t), lambda i: p.ok(i, i) and _script_of(t[i - 1]) == script
                and _script_of(t[i]) == script and t[i - 1] != "-")  # fmt: skip
    return None if i is None else Edit("extra_space", i, i, " ", script)


def _space(rng: random.Random, p: Paragraph, script: str) -> int | None:
    """A random space between two words of ``script``."""
    t = p.text
    return _sample(rng, 1, len(t) - 1, lambda i: t[i] == " " and p.ok(i, i + 1)
                   and t[i - 1] not in " -\n" and t[i + 1] not in " \n"
                   and spacing_script(t, i, i + 1) == script)  # fmt: skip


def space_delete(rng, p, res, script):
    """missing_space: OCR drops the space between two words."""
    i = _space(rng, p, script)
    return None if i is None else Edit("missing_space", i, i + 1, "", script)


def newline(rng, p, res, script):
    """newline_in_sentence: a line break where the PDF wraps the line."""
    i = _space(rng, p, script)
    return None if i is None else Edit("newline_in_sentence", i, i + 1, "\n", script)


def hyphen_wrap(rng, p, res, script):
    """A word split at a line-length boundary: 'pro-\\nposed' / 'pro- posed'."""
    lo, hi = res.opts["line_width"]
    width = rng.randint(lo, hi)
    sep = " " if rng.random() < res.opts["hyphen_space_prob"] else "\n"
    cands = []
    for b in range(width, len(p.text), width):
        for m in _COMPOUND.finditer(p.text, max(0, b - 30), b + 30):
            if m.start() < b <= m.end() + 2 and p.ok(m.end(), m.end()):
                cands.append(Edit("hyphen_wrap", m.end(), m.end(), sep))
        for m in _LATIN_WORD.finditer(p.text, max(0, b - 30), b + 30):
            if m.start() < b < m.end():
                cut = min(max(b, m.start() + 2), m.end() - 2)
                if p.ok(cut, cut):
                    cands.append(Edit("hyphen_wrap", cut, cut, "-" + sep))
    return _pick(rng, cands)


def char_confusion(rng, p, res, script):
    """Replace a gold string by what OCR read instead (measured pairs, by count)."""
    options = []
    for src, gold, n in res.confusions:
        ms = p.matches(gold)
        if any(p.ok(m.start(), m.end()) for m in ms):
            options.append((n, src, ms))
    if not options:
        return None
    n, src, ms = rng.choices(options, weights=[o[0] for o in options])[0]
    m = _sample_match(rng, p, ms)
    assert m is not None
    return Edit("char_confusion", m.start(), m.end(), src, info={"pair": [src, m.group()]})


def dropped_char(rng, p, res, script):
    t = p.text
    i = _sample(rng, 1, len(t) - 1, lambda i: t[i].isalnum() and t[i - 1].isalnum()
                and p.ok(i, i + 1) and t[i - 1] != t[i])  # fmt: skip
    return None if i is None else Edit("dropped_char", i, i + 1, "")


def duplicated_char(rng, p, res, script):
    t = p.text
    if rng.random() < res.opts["dup_word_prob"]:
        words = [m for m in _WORD.finditer(t) if (m.start() == 0 or t[m.start() - 1] == " ")]
        words = [m for m in words if p.ok(m.start(), m.start()) and m.end() < len(t)]
        m = _pick(rng, [w for w in words if t[w.end()] == " "])
        if m is not None:
            return Edit("duplicated_char", m.start(), m.start(), m.group() + " ")
    i = _sample(rng, 1, len(t), lambda i: t[i - 1].isalpha() and p.ok(i, i))
    return None if i is None else Edit("duplicated_char", i, i, t[i - 1])


def ligature(rng, p, res, script):
    m = _pick(rng, p.find(_LIGATURE))
    if m is None:
        return None
    glyph = rng.random() < res.opts["ligature_glyph_prob"]
    return Edit("ligature", m.start(), m.end(), LIGATURE_GLYPHS[m.group()] if glyph else "")


def quote_dash(rng, p, res, script):
    m = _pick(rng, p.find(re.compile("[" + re.escape("".join(QUOTE_DASH)) + "]")))
    if m is None:
        return None
    return Edit("quote_dash", m.start(), m.end(), rng.choice(QUOTE_DASH[m.group()]))


def hangul_garbled(rng, p, res, script):
    """Hangul -> the glyph run OCR produced for it (confusion map, reversed)."""
    options = [(e, hits) for e in res.garbled if (hits := p.find(e["gold"]))]
    if not options:
        return None
    entry, hits = rng.choices(options, weights=[e["count"] for e, _ in options])[0]
    m = _pick(rng, hits)
    raw = entry.get("source_raw") or {}
    wrapped_forms = {r: n for r, n in raw.items() if r.startswith(("$", "\\("))}
    total = sum(raw.values())
    wrap_p = sum(wrapped_forms.values()) / total if total else res.opts["garbled_wrap_prob"]
    if rng.random() < wrap_p:
        if wrapped_forms:
            form = rng.choices(list(wrapped_forms), weights=list(wrapped_forms.values()))[0]
        else:
            form = f"$\\mathrm{{{entry['source_glyphs']}}}$"
        lead = " " if m.start() > 0 and p.text[m.start() - 1] != " " else ""
        src = lead + form
        return Edit(
            "hangul_garbled", m.start(), m.end(), src, math=(len(lead), len(src)),
            info={"wrapped": True},
        )  # fmt: skip
    return Edit(
        "hangul_garbled", m.start(), m.end(), entry["source_glyphs"], info={"wrapped": False}
    )


def header_footer(rng, p, res, script):
    """A running header / page number inserted mid-paragraph (after a space)."""
    t = p.text
    i = _sample(rng, 2, len(t), lambda i: t[i - 1] == " " and t[i] != " " and p.ok(i, i))
    if i is None:
        return None
    page = str(rng.randint(1, 400))
    if rng.random() < res.opts["page_number_prob"] or not res.header_phrases:
        text = page
    else:
        phrase = rng.choice(res.header_phrases)
        text = f"{page} {phrase}" if rng.random() < 0.5 else f"{phrase} {page}"
    return Edit("header_footer_insert", i, i, text + " ")


def other(rng, p, res, script):
    """Junk symbol inserted at a word start."""
    t = p.text
    i = _sample(rng, 1, len(t), lambda i: t[i - 1] == " " and t[i] != " " and p.ok(i, i))
    if i is None:
        return None
    return Edit("other", i, i, rng.choice(res.opts["junk"]) + " ")


GENERATORS: dict[str, Gen] = {
    "header_footer_insert": header_footer,
    "hyphen_wrap": hyphen_wrap,
    "newline_in_sentence": newline,
    "missing_space": space_delete,
    "extra_space": space_insert,
    "ligature": ligature,
    "quote_dash": quote_dash,
    "hangul_garbled": hangul_garbled,
    "dropped_char": dropped_char,
    "duplicated_char": duplicated_char,
    "char_confusion": char_confusion,
    "other": other,
}
