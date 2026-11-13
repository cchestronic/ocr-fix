"""``unicode``: character-level cleanup (enabled by default).

Every span: NFC, ligatures (ﬁ -> fi ...), strip zero-width and control chars (Cc except \\n
and \\t, and Cf: ZWSP, ZWJ, BOM, soft hyphen, bidi marks). Prose only: full/half-width
normalization (ＡＢＣ１２３ -> ABC123, ideographic space, halfwidth jamo; ＄ ＼ ｜ are kept so
no new math or table delimiters appear), NBSP -> space, runs of spaces/tabs -> one space,
tabs -> one space, no trailing spaces before a newline.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from ocrfix.fixers.base import Fixer, Op, map_spans, register
from ocrfix.schema import Record

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_KEEP_WIDE = set("＄＼｜")  # would become $ \\ | : math / escape / table syntax
_SPACES = re.compile(r"(?<=\S)(?:[ \t]{2,}|\t)")
_TRAILING = re.compile(r"[ \t]+(?=\n)")


def _width(ch: str) -> str | None:
    o = ord(ch)
    if ch in _KEEP_WIDE:
        return None
    if 0xFF01 <= o <= 0xFF5E:
        return chr(o - 0xFEE0)
    if ch == "　":
        return " "
    if 0xFF61 <= o <= 0xFFDC or 0xFFE0 <= o <= 0xFFEE:
        return unicodedata.normalize("NFKC", ch)
    return None


def _invisible(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return (cat == "Cc" and ch not in "\n\t") or cat == "Cf"


@register("unicode")
class UnicodeFixer(Fixer):
    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {"ligatures": True, "width": True, "strip_invisible": True, "collapse_spaces": True}

    def clean(self, kind: str, text: str, counts: Counter[str]) -> str:
        o = self.opts
        out = []
        for ch in unicodedata.normalize("NFC", text):
            if o["strip_invisible"] and _invisible(ch):
                counts["invisible"] += 1
                continue
            if o["ligatures"] and ch in LIGATURES:
                counts["ligature"] += 1
                out.append(LIGATURES[ch])
                continue
            if kind == "prose" and o["width"] and (w := _width(ch)) is not None:
                counts["width"] += 1
                out.append(w)
                continue
            if kind == "prose" and o["collapse_spaces"] and ch == " ":
                counts["nbsp"] += 1
                out.append(" ")
                continue
            out.append(ch)
        text = "".join(out)
        if kind == "prose" and o["collapse_spaces"]:
            text, n1 = _SPACES.subn(" ", text)
            text, n2 = _TRAILING.subn("", text)
            counts["spaces"] += n1 + n2
        return text

    def apply(self, paras: list[Record]) -> tuple[list[Record], list[Op]]:
        out, ops = [], []
        for rec in paras:
            counts: Counter[str] = Counter()
            new = map_spans(rec, lambda kind, text, c=counts: self.clean(kind, text, c))
            if new.text == rec.text:
                out.append(rec)
            elif not new.text:
                ops.append(self.delete_op(rec, "empty", dict(counts)))
            else:
                out.append(new)
                ops.append(self.edit_op(rec, "chars", dict(counts)))
        return out, ops
