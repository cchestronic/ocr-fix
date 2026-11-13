"""``line_join``: join line breaks inside a paragraph (disabled by default).

A ``\\n`` in prose (with the spaces around it) becomes one space, unless
  - the line before ends with sentence punctuation or a closing bracket,
  - the next line starts a block (list item, heading, quote, table row) or is empty,
  - math or a table touches the break (display math keeps its own line).
``word-\\nword`` (a letter, then lowercase after the break; any other hyphen break is
left alone): the hyphen is dropped when the joined word is in the frequency list
(``wordlist``: one word per line, optionally ``word<TAB or space>count``, at least
``min_count``); otherwise the hyphen stays (``well-\\nknown`` -> ``well-known``).
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

from ocrfix.fixers.base import Fixer, FixerError, Op, apply_edits, prose_mask, register, sample
from ocrfix.schema import Record

END_PUNCT = set(".!?:;。！？…") | set(")]}」』】〉》")
_BREAK = re.compile(r"[ \t]*\n[ \t]*")
_BLOCK_START = re.compile(r"(?:[-*+•]\s|\d{1,3}[.)]\s|#|>|\||\n|$)")
_LEFT_WORD = re.compile(r"([A-Za-z]+)-$")
_RIGHT_WORD = re.compile(r"[a-z]+")


@cache
def load_wordlist(path: str, min_count: int = 1) -> frozenset[str]:
    words = set()
    p = Path(path)
    if not p.is_file():
        raise FixerError(f"line_join.wordlist: no such file {path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) < min_count:
            continue
        words.add(parts[0].casefold())
    return frozenset(words)


@register("line_join")
class LineJoinFixer(Fixer):
    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {"wordlist": None, "min_count": 1, "dehyphenate": True}

    def __init__(self, **opts: Any) -> None:
        super().__init__(**opts)
        wl = self.opts["wordlist"]
        self.words = load_wordlist(str(wl), int(self.opts["min_count"])) if wl else frozenset()

    def edits(self, rec: Record) -> tuple[list[tuple[int, int, str]], list[dict[str, Any]]]:
        t, mask = rec.text, prose_mask(rec)
        edits, ev = [], []
        for m in _BREAK.finditer(t):
            a, b = m.start(), m.end()
            if a == 0 or b >= len(t) or not all(mask[a - 1 : b + 1]):
                continue
            prev = t[a - 1]
            if prev in END_PUNCT or _BLOCK_START.match(t, b):
                continue
            left = _LEFT_WORD.search(t, 0, a)
            right = _RIGHT_WORD.match(t, b)
            if prev == "-":
                if not (left and right and t[a - 2].isalpha()):
                    continue  # "COVID-\n19", "-\n" before a capital: leave the break
                joined = left.group(1) + right.group()
                if self.opts["dehyphenate"] and joined.casefold() in self.words:
                    edits.append((a - 1, b, ""))
                    ev.append({"at": a, "dehyphenated": joined})
                else:
                    edits.append((a, b, ""))
                    ev.append({"at": a, "kept_hyphen": f"{left.group(1)}-{right.group()}"})
                continue
            edits.append((a, b, " "))
            ev.append({"at": a, "joined": t[max(0, a - 12) : a] + "⏎" + t[b : b + 12]})
        return edits, ev

    def apply(self, paras: list[Record]) -> tuple[list[Record], list[Op]]:
        out, ops = [], []
        for rec in paras:
            edits, ev = self.edits(rec)
            if not edits:
                out.append(rec)
                continue
            out.append(apply_edits(rec, edits))
            ops.append(self.edit_op(rec, "join", {"n": len(edits), "examples": sample(ev)}))
        return out, ops
