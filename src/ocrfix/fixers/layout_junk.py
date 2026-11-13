"""``layout_junk``: page furniture OCR left in the text (disabled by default).

Works on whole prose lines (a line with math or a table in it is never touched):
  - running headers/footers: a first or last line of a paragraph, not ending in sentence
    punctuation, repeated verbatim on >= ``min_repeats`` pages of the same document.
    MinerU markdown has no page numbers, so a record's ``page`` / ``page_idx`` field is
    used when present, else each paragraph counts as its own page.
    Only lines of at most ``max_line_len`` chars with a letter or digit count.
  - bare page numbers: ``12``, ``- 12 -``, ``Page 12``, ``12 / 30``, ``12 of 30``.
  - line-number gutters: a paragraph whose lines (>= 3) all start with increasing numbers
    has the numbers stripped; a paragraph that is only increasing numbers is deleted.
A paragraph left empty is deleted; otherwise the removed lines are an ``edit``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import pairwise
from typing import Any

from ocrfix.fixers.base import Fixer, Op, apply_edits, lines, prose_mask, register, sample
from ocrfix.schema import Record

PAGE_NUMBER = re.compile(
    r"^\s*(?:[-–—]\s*\d{1,4}\s*[-–—]|(?:page|p\.)\s*\d{1,4}|\d{1,4}\s*(?:/|of)\s*\d{1,4}|\d{1,4})\s*$",
    re.IGNORECASE,
)
_GUTTER = re.compile(r"^(\d{1,4})(?:\s+|$)")


def _increasing(nums: list[int]) -> bool:
    return all(b > a for a, b in pairwise(nums)) and (nums[-1] - nums[0] <= 2 * len(nums))


def full_lines(rec: Record) -> list[tuple[int, int]]:
    """Lines that are entirely prose, newline(s) around them included."""
    mask = prose_mask(rec)
    out = []
    for a, b in lines(rec.text):
        lo, hi = max(0, a - 1), min(len(rec.text), b + 1)
        if b > a and all(mask[lo:hi]):
            out.append((a, b))
    return out


def _removal(text: str, a: int, b: int) -> tuple[int, int]:
    """The line plus one adjacent newline."""
    if b < len(text) and text[b] == "\n":
        return a, b + 1
    if a > 0 and text[a - 1] == "\n":
        return a - 1, b
    return a, b


@register("layout_junk")
class LayoutJunkFixer(Fixer):
    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {
            "min_repeats": 3,
            "max_line_len": 100,
            "headers": True,
            "page_numbers": True,
            "gutters": True,
        }

    def repeated(self, paras: list[Record]) -> set[str]:
        pages: dict[str, set[str]] = defaultdict(set)
        for rec in paras:
            page = str(rec.extra.get("page", rec.extra.get("page_idx", rec.para_id)))
            all_lines = lines(rec.text)
            edge = {all_lines[0], all_lines[-1]}
            for a, b in full_lines(rec):
                key = rec.text[a:b].strip()
                if (
                    (a, b) in edge
                    and 0 < len(key) <= self.opts["max_line_len"]
                    and any(c.isalnum() for c in key)
                    and key[-1] not in ".!?:;。"
                ):
                    pages[key].add(page)
        return {k for k, ps in pages.items() if len(ps) >= self.opts["min_repeats"]}

    def apply(self, paras: list[Record]) -> tuple[list[Record], list[Op]]:
        o = self.opts
        junk = self.repeated(paras) if o["headers"] else set()
        out, ops = [], []
        for rec in paras:
            fl = full_lines(rec)
            all_lines = [rec.text[a:b] for a, b in lines(rec.text)]
            edits, ev = [], []
            nums = [m and int(m.group(1)) for m in (_GUTTER.match(x.strip()) for x in all_lines)]
            gutter = (
                o["gutters"] and len(all_lines) >= 3 and len(fl) == len(all_lines)
                and all(n is not None for n in nums) and _increasing(nums)  # type: ignore[arg-type]
            )  # fmt: skip
            if gutter and all(PAGE_NUMBER.match(x) for x in all_lines):
                ops.append(self.delete_op(rec, "gutter", {"lines": sample(all_lines)}))
                continue
            for a, b in fl:
                line = rec.text[a:b]
                if gutter:
                    m = _GUTTER.match(line.lstrip())
                    if m:
                        start = a + (len(line) - len(line.lstrip()))
                        edits.append((start, start + m.end(), ""))
                        ev.append({"rule": "gutter", "number": m.group(1)})
                elif line.strip() in junk:
                    edits.append((*_removal(rec.text, a, b), ""))
                    ev.append({"rule": "header", "line": line.strip()})
                elif o["page_numbers"] and PAGE_NUMBER.match(line):
                    edits.append((*_removal(rec.text, a, b), ""))
                    ev.append({"rule": "page_number", "line": line.strip()})
            if not edits:
                out.append(rec)
                continue
            new = apply_edits(rec, _dedupe(edits))
            rules = sorted({e["rule"] for e in ev})
            if not new.text:
                ops.append(self.delete_op(rec, "+".join(rules), {"lines": sample(ev)}))
            else:
                out.append(new)
                ops.append(self.edit_op(rec, "+".join(rules), {"removed": sample(ev)}))
        return out, ops


def _dedupe(edits: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Consecutive removed lines may both claim the newline between them."""
    out: list[tuple[int, int, str]] = []
    for a, b, r in sorted(edits):
        if out and a < out[-1][1]:
            a = out[-1][1]
        out.append((a, max(a, b), r))
    return out
