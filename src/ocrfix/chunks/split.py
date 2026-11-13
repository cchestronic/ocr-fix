"""One markdown document -> chunks of ``min_tokens``..``max_tokens`` masked tokens.

**Masking.** Math ⟨M#⟩ and tables ⟨T#⟩ from the tagger, and ⟨P#⟩ for the ``chunk.protect`` kinds
in prose. A literal placeholder already in the source becomes a ⟨P#⟩ too, so it survives the
round trip. Masked items are atoms: a chunk never cuts one. Numbering is per chunk.

**Tiling.** Chunks cover the whole document. A chunk is ``prefix`` + its content + ``suffix``,
where the two hold the surrounding whitespace, so the model never sees (or changes) the blank
lines between chunks. Original = the concatenation of ``raw`` or, where ``raw`` is null,
``prefix + restore(source_text, placeholders) + suffix``. ``raw`` holds the source slice only
where normalization (BOM, CRLF, NFC) changed it, so reconstruction is byte-exact.

**Cuts**, in order of preference, the last one that fits: blank line, sentence end, line end,
any whitespace, else any token boundary outside a placeholder. Only the last chunk of a
document can be under ``min_tokens`` (``short``), and the cut before it is moved so that
both fit whenever the remainder allows it.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ocrfix.corpus.lang import detect_lang
from ocrfix.corpus.placeholders import PLACEHOLDER, restore
from ocrfix.corpus.protect import protected_spans
from ocrfix.corpus.tagger import tag_spans
from ocrfix.schema import MATH_PH, PROTECTED_PH, TABLE_PH, Span, nfc

_FORMAT = {"math": MATH_PH, "table": TABLE_PH, "protected": PROTECTED_PH}
_BREAKS = (
    re.compile(r"\n[ \t]*\n\s*"),  # blank line
    re.compile(r"[.!?。！？…][\"'”’)\]]*(\s+)"),  # sentence end: the group is the gap
    re.compile(r"\n\s*"),  # line end
    re.compile(r"\s+"),  # any whitespace
)
# A cut before an ASCII character is safe: nothing composes with an ASCII char before it.
_RAW_PIECE = re.compile(r"(?:\r\n|[\x00-\x7f])[^\x00-\x7f]*|[^\x00-\x7f]+")

# text -> (token start offsets, token end offsets)
Encode = Callable[[str], tuple[Sequence[int], Sequence[int]]]


def normalize(raw: str) -> str:
    """What the pipeline reads: no BOM, LF line ends, NFC."""
    return nfc(raw.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n"))


def atoms_of(text: str, spans: list[Span], protect: list[str]) -> list[tuple[int, int, str]]:
    """Everything masked, sorted and non-overlapping: (start, end, kind)."""
    atoms = protected_spans(text, spans, protect)
    taken = [(a, b) for a, b, _ in atoms]  # sorted, non-overlapping
    extra = []
    for s in spans:
        if s.kind == "prose":
            for m in PLACEHOLDER.finditer(text, s.start, s.end):
                a, b = m.span()
                i = bisect.bisect_left(taken, (a, b))
                if (i == 0 or taken[i - 1][1] <= a) and (i == len(taken) or b <= taken[i][0]):
                    extra.append((a, b, "protected"))
    return sorted([*atoms, *extra]) if extra else atoms


def render(
    text: str, atoms: list[tuple[int, int, str]], start: int, end: int
) -> tuple[str, dict[str, str]]:
    """text[start:end] masked, numbered from 1 per kind; atoms must not straddle the range."""
    parts: list[str] = []
    mapping: dict[str, str] = {}
    counts = dict.fromkeys(_FORMAT, 0)
    pos = start
    for k in range(bisect.bisect_left(atoms, (start, -1, "")), len(atoms)):
        a, b, kind = atoms[k]
        if a >= end:
            break
        counts[kind] += 1
        ph = _FORMAT[kind].format(counts[kind])
        parts += [text[pos:a], ph]
        mapping[ph] = text[a:b]
        pos = b
    parts.append(text[pos:end])
    return "".join(parts), mapping


@dataclass
class _Masked:
    """The whole document masked, and where each placeholder sits in it."""

    text: str
    starts: list[int]
    ends: list[int]
    atoms: list[tuple[int, int, str]]

    def to_doc(self, pos: int) -> int:
        """A masked offset outside every placeholder -> document offset."""
        k = bisect.bisect_right(self.starts, pos) - 1
        if k < 0:
            return pos
        return self.atoms[k][0] if pos == self.starts[k] else self.atoms[k][1] + pos - self.ends[k]

    def inside_placeholder(self, pos: int) -> bool:
        k = bisect.bisect_right(self.starts, pos) - 1
        return k >= 0 and self.starts[k] < pos < self.ends[k]


def _mask_doc(text: str, atoms: list[tuple[int, int, str]]) -> _Masked:
    """Every placeholder numbered 1: chunks restart their numbers, so most are one digit, and
    a whole-document number would overcount tokens."""
    parts, starts, ends = [], [], []
    pos = mpos = 0
    for a, b, kind in atoms:
        ph = _FORMAT[kind].format(1)
        parts += [text[pos:a], ph]
        mpos += a - pos
        starts.append(mpos)
        mpos += len(ph)
        ends.append(mpos)
        pos = b
    parts.append(text[pos:])
    return _Masked("".join(parts), starts, ends, atoms)


def plan_cuts(
    text: str,
    atoms: list[tuple[int, int, str]],
    encode: Encode,
    min_tokens: int,
    max_tokens: int,
) -> list[tuple[int, int, int]]:
    """Content ranges (start, end, exact tokens) in document coordinates, stripped of edge
    whitespace.

    Tokens are counted once on the whole masked document to find candidates; each chosen
    chunk is then re-counted exactly, because its placeholders are renumbered from 1 and
    tokens at its edges can differ."""
    doc = _mask_doc(text, atoms)
    masked = doc.text
    t_starts = list(encode(masked)[0])
    breaks = [[m.span(m.lastindex or 0) for m in rx.finditer(masked)] for rx in _BREAKS]

    def before(pos: int) -> int:
        return bisect.bisect_left(t_starts, pos)

    def exact(a: int, b: int) -> int:
        return len(encode(render(text, atoms, doc.to_doc(a), doc.to_doc(b))[0])[0])

    out: list[tuple[int, int, int]] = []
    pos = len(masked) - len(masked.lstrip())
    n_all = len(masked.rstrip())
    while pos < n_all:
        first = before(pos)
        rest = before(n_all) - first
        lo, hi = min_tokens, max_tokens
        if rest <= max_tokens and (n := exact(pos, n_all)) <= max_tokens:
            out.append((doc.to_doc(pos), doc.to_doc(n_all), n))
            break
        # Leave the remainder at least min_tokens when that is possible.
        if rest - hi < min_tokens <= rest - min_tokens:
            hi = rest - min_tokens
        while True:
            end, nxt = _best_cut(doc, breaks, before, t_starts, first, min(lo, hi), hi)
            n = exact(pos, end)
            if n > max_tokens:
                hi -= n - max_tokens
            elif n < min_tokens and lo + min_tokens - n <= hi:
                lo += min_tokens - n
            else:
                break
        out.append((doc.to_doc(pos), doc.to_doc(end), n))
        pos = nxt
    return out


def _best_cut(
    doc: _Masked,
    breaks: list[list[tuple[int, int]]],
    before: Callable[[int], int],
    t_starts: list[int],
    first: int,
    lo: int,
    hi: int,
) -> tuple[int, int]:
    """(content end, next content start) in masked coordinates: the last break of the best
    class with lo <= tokens <= hi, else a token boundary at hi."""
    masked = doc.text
    top = t_starts[min(first + hi, len(t_starts) - 1)]
    for found in breaks:
        for k in range(bisect.bisect_right(found, (top, len(masked))) - 1, -1, -1):
            a, b = found[k]
            n = before(a) - first
            if n < lo:
                break
            if n <= hi:
                return _trim(masked, a, b)
    for t in range(min(first + hi, len(t_starts) - 1), first, -1):
        c = t_starts[t]
        if c > t_starts[t - 1] and not doc.inside_placeholder(c):  # not inside one character
            return _trim(masked, c, c)
    raise ValueError("no token boundary to cut at")


def _trim(masked: str, end: int, nxt: int) -> tuple[int, int]:
    while end > 0 and masked[end - 1].isspace():
        end -= 1
    while nxt < len(masked) and masked[nxt].isspace():
        nxt += 1
    return end, nxt


@dataclass
class Chunk:
    start: int  # content range in the normalized document
    end: int
    prefix: str
    suffix: str
    source_text: str
    placeholders: dict[str, str]
    n_tokens: int
    lang: str
    raw: str | None = None


def raw_slices(raw: str, norm: str, cuts: list[int]) -> list[str]:
    """Split ``raw`` at the places matching ``cuts`` (sorted offsets into ``norm`` =
    ``normalize(raw)``), so ``normalize`` of each slice is the matching slice of ``norm``."""
    bom = 1 if raw.startswith("\ufeff") else 0
    out: list[str] = []
    npos = last = 0
    want = iter(cuts)
    cut = next(want, None)
    for m in _RAW_PIECE.finditer(raw, bom):
        piece = m.group()
        np = normalize(piece)
        while cut is not None and npos < cut < npos + len(np):
            i = _inner_cut(piece, np, cut - npos)
            out.append(raw[last : m.start() + i])
            last = m.start() + i
            cut = next(want, None)
        npos += len(np)
        while cut is not None and cut == npos:
            out.append(raw[last : m.end()])
            last = m.end()
            cut = next(want, None)
    out.append(raw[last:])
    return out


def _inner_cut(piece: str, norm_piece: str, j: int) -> int:
    for i in range(1, len(piece)):
        if normalize(piece[:i]) == norm_piece[:j] and normalize(piece[i:]) == norm_piece[j:]:
            return i
    raise ValueError(f"cannot map a cut into {piece!r}")


def split_document(
    raw: str,
    encode: Encode,
    *,
    min_tokens: int,
    max_tokens: int,
    protect: list[str],
    lang_opts: dict[str, Any] | None = None,
) -> list[Chunk]:
    text = normalize(raw)
    spans = tag_spans(text)
    atoms = atoms_of(text, spans, protect)
    ranges = plan_cuts(text, atoms, encode, min_tokens, max_tokens) or [(len(text), len(text), 0)]
    chunks: list[Chunk] = []
    prev_end = 0
    span_ends = [s.end for s in spans]  # spans tile the text in order
    for k, (a, b, n_tokens) in enumerate(ranges):
        masked, mapping = render(text, atoms, a, b)
        local = []
        for j in range(bisect.bisect_right(span_ends, a), len(spans)):
            s = spans[j]
            if s.start >= b:
                break
            local.append(Span(max(s.start, a) - a, min(s.end, b) - a, s.kind))
        nxt = ranges[k + 1][0] if k + 1 < len(ranges) else len(text)
        chunks.append(
            Chunk(
                start=a,
                end=b,
                prefix=text[prev_end:a] if k == 0 else "",
                suffix=text[b:nxt],
                source_text=masked,
                placeholders=mapping,
                n_tokens=n_tokens,
                lang=detect_lang(text[a:b], local, **(lang_opts or {})),
            )
        )
        prev_end = b
    _attach_raw(raw, text, chunks)
    return chunks


def chunk_original(c: Chunk) -> str:
    return c.prefix + restore(c.source_text, c.placeholders) + c.suffix


def _attach_raw(raw: str, text: str, chunks: list[Chunk]) -> None:
    if raw == text:
        return
    ends = []
    pos = 0
    for c in chunks[:-1]:
        pos += len(chunk_original(c))
        ends.append(pos)
    for c, piece in zip(chunks, raw_slices(raw, text, ends), strict=True):
        if piece != chunk_original(c):
            c.raw = piece
    if "".join(c.raw if c.raw is not None else chunk_original(c) for c in chunks) != raw:
        raise ValueError("reconstruction does not reproduce the source")
