"""Char-level alignment of OCR source vs gold, and classification of each edit.

``align`` runs Levenshtein opcodes (rapidfuzz; difflib if missing) and merges adjacent
non-equal opcodes into one region, so ``rn`` -> ``m`` is a single edit. ``classify`` gives
each region one type; checks run in the order of ``TYPES``:

  header_footer_insert  OCR text the gold drops: >= 8 non-space chars or a bare page
                        number, on word boundaries (running headers, page numbers)
  hyphen_wrap           "pro- posed", "out-\\nperforms" -> joined; "well- known" -> "well-known"
  newline_in_sentence   a line break the gold replaces by a space or nothing
  missing_space         gold adds whitespace
  extra_space           gold removes whitespace
  ligature              ﬁ ﬂ ... vs fi fl (NFKC-equal), or a dropped fi/fl/ff/ffi/ffl
  quote_dash            both sides only quotes / dashes / apostrophes
  hangul_garbled        a non-Hangul run (often LaTeX) replaced by Hangul
  dropped_char          OCR lost chars (empty source side)
  duplicated_char       OCR repeated the neighbouring chars
  char_confusion        a short substitution (<= confusion_max_len chars each side)
  other                 anything else (long rewrites, junk insertions, mixed edits)

Spacing types (hyphen_wrap excluded) carry ``script``: ``ko`` when a Hangul char is next to
the edit, else ``en`` when a Latin letter or digit is, else ``other``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:  # the `text` extra is optional
    _lev = None

TYPES = (
    "header_footer_insert",
    "hyphen_wrap",
    "newline_in_sentence",
    "missing_space",
    "extra_space",
    "ligature",
    "quote_dash",
    "hangul_garbled",
    "dropped_char",
    "duplicated_char",
    "char_confusion",
    "other",
)
SPACING = ("newline_in_sentence", "missing_space", "extra_space")

LIGATURES = set("ﬀﬁﬂﬃﬄﬅﬆ")
LIGATURE_EXPANSIONS = {"ff", "fi", "fl", "ffi", "ffl"}  # dropped ligature glyphs
QUOTE_DASH = set("'\"`´‘’‚‛“”„‟′″«»‹›-‐‑‒–—―−")
_PAGE_NUMBER = re.compile(r"^(?:[-–—]\s*)?\d{1,4}(?:\s*[-–—])?$")
_HYPHEN_BREAK = re.compile(r"^-[ \t]*\n?[ \t]*$")


@dataclass(frozen=True)
class EditOp:
    type: str
    src_start: int
    src_end: int
    gold_start: int
    gold_end: int
    src: str
    gold: str
    script: str | None = None

    @property
    def key(self) -> str:
        """Type, plus the script for spacing ops: ``missing_space.ko``."""
        return f"{self.type}.{self.script}" if self.script else self.type

    def to_dict(self) -> dict:
        return asdict(self)


def is_hangul(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def has_hangul(s: str) -> bool:
    return any(is_hangul(c) for c in s)


def align(src: str, gold: str) -> list[tuple[int, int, int, int]]:
    """Non-equal regions as (src_start, src_end, gold_start, gold_end), adjacent ones merged."""
    if _lev is not None:
        blocks = [
            (o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end)
            for o in _lev.opcodes(src, gold)
        ]
    else:
        from difflib import SequenceMatcher

        blocks = SequenceMatcher(None, src, gold, autojunk=False).get_opcodes()
    regions: list[list[int]] = []
    for tag, i1, i2, j1, j2 in blocks:
        if tag == "equal":
            continue
        if regions and regions[-1][1] == i1 and regions[-1][3] == j1:
            regions[-1][1], regions[-1][3] = i2, j2
        else:
            regions.append([i1, i2, j1, j2])
    return [tuple(r) for r in _close_gaps(src, gold, regions)]  # type: ignore[misc]


def _close_gaps(src: str, gold: str, regions: list[list[int]], max_gap: int = 3) -> list[list[int]]:
    """Join two pure deletions (or insertions) split by a short equal run, at equal cost.

    Levenshtein ties let a long inserted header "Journal of Applied Research 12" align one
    of its letters with the text around it, leaving two fragments; deleting the whole run
    and keeping that letter on the other side costs the same and is the edit OCR made.
    """
    out: list[list[int]] = []
    for r in regions:
        if out:
            p = out[-1]
            gap_s, gap_g = r[0] - p[1], r[2] - p[3]
            dels = p[2] == p[3] and r[2] == r[3] and gap_s == gap_g
            ins = p[0] == p[1] and r[0] == r[1] and gap_s == gap_g
            if (dels or ins) and 0 < gap_s <= max_gap:
                text = src if dels else gold
                lo, hi = (p[0], r[1]) if dels else (p[2], r[3])
                run, gap = text[lo:hi], text[p[1] : r[0]] if dels else text[p[3] : r[2]]
                # keep the gap text at the end (the edit is a prefix) or at the start
                for keep_end in (True, False):
                    if run.endswith(gap) if keep_end else run.startswith(gap):
                        n = len(gap)
                        if dels:
                            out[-1] = (
                                [lo, hi - n, p[2], p[2]] if keep_end else [lo + n, hi, r[2], r[2]]
                            )
                        else:
                            out[-1] = (
                                [p[0], p[0], lo, hi - n] if keep_end else [r[0], r[0], lo + n, hi]
                            )
                        break
                else:
                    out.append(r)
                continue
        out.append(r)
    return out


def _neighbours(text: str, start: int, end: int) -> tuple[str, str]:
    """Nearest non-space char left of start and right of end ('' at the edges)."""
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    return (left[-1] if left else ""), (right[0] if right else "")


def spacing_script(text: str, start: int, end: int) -> str:
    left, right = _neighbours(text, start, end)
    if has_hangul(left + right):
        return "ko"
    if any(c.isascii() and c.isalnum() for c in left + right):
        return "en"
    return "other"


def _at_word_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return (before.isspace() or text[start].isspace()) and (
        after.isspace() or text[end - 1].isspace()
    )


def classify(
    src: str,
    gold: str,
    region: tuple[int, int, int, int],
    *,
    header_footer_min_chars: int = 8,
    confusion_max_len: int = 4,
) -> EditOp:
    i1, i2, j1, j2 = region
    s, g = src[i1:i2], gold[j1:j2]

    def op(kind: str, script: str | None = None) -> EditOp:
        return EditOp(kind, i1, i2, j1, j2, s, g, script)

    stripped = s.strip()
    if (
        not g.strip()
        and stripped
        and _at_word_boundary(src, i1, i2)
        and (
            len(re.sub(r"\s", "", stripped)) >= header_footer_min_chars
            or _PAGE_NUMBER.match(stripped)
        )
    ):
        return op("header_footer_insert")

    left = src[i1 - 1] if i1 > 0 else ""
    right = src[i2] if i2 < len(src) else ""
    if not g and left.isalpha() and right.isalpha() and _HYPHEN_BREAK.match(s) and len(s) >= 2:
        return op("hyphen_wrap")
    if (
        not g
        and s.isspace()
        and left == "-"
        and i1 >= 2
        and src[i1 - 2].isalpha()
        and right.isalpha()
    ):
        return op("hyphen_wrap")

    if (not s or s.isspace()) and (not g or g.isspace()):
        script = spacing_script(src, i1, i2)
        if "\n" in s and "\n" not in g:
            return op("newline_in_sentence", script)
        if len(g) > len(s):
            return op("missing_space", script)
        if len(s) > len(g):
            return op("extra_space", script)
        return op("other")

    if (LIGATURES & set(s) and unicodedata.normalize("NFKC", s) == g) or (
        not s and g in LIGATURE_EXPANSIONS and (left.isalpha() or right.isalpha())
    ):
        return op("ligature")
    if s and g and set(s) <= QUOTE_DASH and set(g) <= QUOTE_DASH:
        return op("quote_dash")
    if has_hangul(g) and stripped and not has_hangul(s):
        return op("hangul_garbled")
    if not s:
        return op("dropped_char")
    if not g:
        n = len(s)
        if src[max(0, i1 - n) : i1] == s or src[i2 : i2 + n] == s:
            return op("duplicated_char")
        return op("other")
    if (
        len(s) <= confusion_max_len
        and len(g) <= confusion_max_len
        and not any(c.isspace() for c in s + g)
    ):
        return op("char_confusion")
    return op("other")


def diff_ops(src: str, gold: str, **opts) -> list[EditOp]:
    return [classify(src, gold, r, **opts) for r in align(src, gold)]
