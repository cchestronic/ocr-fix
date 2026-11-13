"""CPU side of scoring: masking, tokenization, per-token masks, and scoring windows.

**Masking.** Each paragraph is scored in the same masked form the LLM sees
(``corpus.protect.mask``): every math span -> ⟨M#⟩, table -> ⟨T#⟩, protected prose run ->
⟨P#⟩. A token is *prose* iff it overlaps no placeholder; the first ``boundary`` prose tokens
after a placeholder are not scored (the model is still adjusting to the unusual symbol).
Scored tokens = prose and not boundary. A text that already contains a literal placeholder
can't be masked; it is scored as it is (``mask_error``).

**Windows.** Consecutive paragraphs of one group (doc_id, plus ``corruption`` and
``variant`` for corrupt rows) with consecutive ``order`` are packed into one sequence,
joined by the tokens of a blank line: a window closes once it holds ``target`` tokens and
never exceeds ``max_tokens``. So a paragraph is scored with the paragraphs before it as
context, and batches are nearly uniform in length. A paragraph longer than ``max_tokens``
is cut at the last sentence end in [target, max_tokens] (else at max_tokens); each piece
starts a window, and its first ``boundary`` tokens are not scored. Features stay per
paragraph, from its own tokens only.

Token arrays live only in memory (and optional .npy dumps), never in JSONL.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocrfix.corpus.placeholders import PLACEHOLDER, PlaceholderError
from ocrfix.corpus.protect import mask
from ocrfix.schema import Record

SEPARATOR = "\n\n"
SENTENCE_END = set(".!?。！？…")


def group_key(rec: Record) -> tuple:
    """Paragraphs of one group can share a window."""
    return (rec.doc_id, rec.extra.get("corruption"), rec.extra.get("variant"))


@dataclass
class TokenizedPara:
    key: tuple[str, str]
    group: tuple
    order: int
    text: str  # masked text; offsets below index into it
    ids: np.ndarray  # int32 [n], no special tokens
    starts: np.ndarray  # int32 [n]
    ends: np.ndarray  # int32 [n]
    prose: np.ndarray  # bool [n] overlaps no placeholder
    boundary: np.ndarray  # bool [n] right after a placeholder or a split
    mask_error: bool = False

    @property
    def scored(self) -> np.ndarray:
        return self.prose & ~self.boundary

    def __len__(self) -> int:
        return len(self.ids)


def mask_text(rec: Record, protect: Sequence[str]) -> tuple[str, list[tuple[int, int]], bool]:
    """(masked text, placeholder char ranges in it, mask_error)."""
    try:
        masked, _ = mask(rec.text, rec.spans, list(protect))
    except PlaceholderError:
        masked = rec.text
        return masked, [m.span() for m in PLACEHOLDER.finditer(masked)], True
    return masked, [m.span() for m in PLACEHOLDER.finditer(masked)], False


def token_masks(
    starts: np.ndarray, ends: np.ndarray, placeholders: list[tuple[int, int]], boundary: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """prose (overlaps no placeholder, non-empty) and boundary masks."""
    n = len(starts)
    hits = np.zeros(n, dtype=bool)
    for a, b in placeholders:
        hits |= (starts < b) & (ends > a)
    prose = ~hits & (ends > starts)
    bmask = np.zeros(n, dtype=bool)
    left = 0
    for i in range(n):
        if hits[i]:
            left = boundary
        elif prose[i] and left > 0:
            bmask[i] = True
            left -= 1
    return prose, bmask


def mark_boundary(tp: TokenizedPara, start: int, boundary: int) -> None:
    """Don't score the first ``boundary`` prose tokens from ``start`` (a split point)."""
    left = boundary
    for i in range(start, len(tp)):
        if left <= 0:
            break
        if tp.prose[i]:
            tp.boundary[i] = True
            left -= 1


# --- tokenization (optionally in worker processes) --------------------------------------

_TOKENIZER = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from ocrfix.offline import enforce_offline

    enforce_offline()
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def _encode(text: str) -> tuple[np.ndarray, np.ndarray]:
    enc = _TOKENIZER(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    offsets = np.asarray(enc["offset_mapping"], dtype=np.int32).reshape(-1, 2)
    return ids, offsets


def _tokenize_one(job: tuple) -> TokenizedPara:
    key, group, order, text, placeholders, boundary, mask_error = job
    ids, offsets = _encode(text)
    starts, ends = offsets[:, 0].copy(), offsets[:, 1].copy()
    prose, bmask = token_masks(starts, ends, placeholders, boundary)
    return TokenizedPara(key, group, order, text, ids, starts, ends, prose, bmask, mask_error)


def _separator_ids(_: object = None) -> np.ndarray:
    return _encode(SEPARATOR)[0]


class Tokenizer:
    """Mask and tokenize Records, in a spawn-based process pool if workers > 0."""

    def __init__(
        self, tokenizer_path: str, workers: int, boundary: int = 3, protect: Sequence[str] = ()
    ):
        self.boundary = boundary
        self.protect = list(protect)
        self._pool = None
        if workers > 0:
            ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA state
            self._pool = ctx.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,))
            self.separator = self._pool.apply(_separator_ids)
        else:
            _init_worker(tokenizer_path)
            self.separator = _separator_ids()

    def __call__(self, records: Iterable[Record]) -> list[TokenizedPara]:
        jobs = []
        for r in records:
            text, phs, err = mask_text(r, self.protect)
            jobs.append((r.key, group_key(r), int(r.order), text, phs, self.boundary, err))
        if self._pool is None:
            return [_tokenize_one(j) for j in jobs]
        return list(self._pool.imap(_tokenize_one, jobs, chunksize=64))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> Tokenizer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- windows ----------------------------------------------------------------------------


@dataclass
class Window:
    ids: np.ndarray  # tokens of every piece, separators between them
    pieces: list[tuple[int, int, int, int]] = field(default_factory=list)  # (para, lo, hi, pos)


def split_points(tp: TokenizedPara, target: int, max_tokens: int) -> list[tuple[int, int]]:
    """Token ranges of a paragraph: one, or cuts at sentence ends when > max_tokens."""
    n = len(tp)
    if n <= max_tokens:
        return [(0, n)]
    out, pos = [], 0
    while n - pos > max_tokens:
        cut = pos + max_tokens
        for i in range(pos + max_tokens - 1, pos + max(1, target) - 1, -1):
            e = int(tp.ends[i])
            if e > 0 and tp.text[e - 1] in SENTENCE_END:
                cut = i + 1
                break
        out.append((pos, cut))
        pos = cut
    out.append((pos, n))
    return out


def plan_windows(
    paras: Sequence[TokenizedPara], sep_len: int, target: int, max_tokens: int, boundary: int
) -> list[list[tuple[int, int, int]]]:
    """Window membership: lists of (para index, lo, hi). Marks split points as boundary."""
    plans: list[list[tuple[int, int, int]]] = []
    cur: list[tuple[int, int, int]] = []
    cur_len = 0
    prev: TokenizedPara | None = None
    for i, tp in enumerate(paras):
        pieces = split_points(tp, target, max_tokens)
        for j, (lo, hi) in enumerate(pieces):
            if lo > 0:
                mark_boundary(tp, lo, boundary)
            n = hi - lo
            follows = (
                j == 0
                and prev is not None
                and prev.group == tp.group
                and tp.order == prev.order + 1
                and len(pieces) == 1
            )
            if cur and follows and cur_len < target and cur_len + sep_len + n <= max_tokens:
                cur.append((i, lo, hi))
                cur_len += sep_len + n
            else:
                if cur:
                    plans.append(cur)
                cur, cur_len = [(i, lo, hi)], n
        prev = tp if len(pieces) == 1 else None  # a split paragraph ends its window
        if len(pieces) > 1:
            plans.append(cur)
            cur, cur_len = [], 0
    if cur:
        plans.append(cur)
    return plans


def build_window(
    paras: Sequence[TokenizedPara], plan: list[tuple[int, int, int]], separator: np.ndarray
) -> Window:
    parts: list[np.ndarray] = []
    w = Window(np.zeros(0, dtype=np.int32))
    pos = 0
    for k, (i, lo, hi) in enumerate(plan):
        if k:
            parts.append(separator)
            pos += len(separator)
        parts.append(paras[i].ids[lo:hi])
        w.pieces.append((i, lo, hi, pos))
        pos += hi - lo
    w.ids = np.concatenate(parts).astype(np.int32) if parts else w.ids
    return w


def para_bits(
    paras: Sequence[TokenizedPara], windows: Sequence[Window], bits: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Per-paragraph token surprisal assembled from the windows."""
    out = [np.full(len(tp), np.nan, dtype=np.float32) for tp in paras]
    for w, b in zip(windows, bits, strict=True):
        for i, lo, hi, pos in w.pieces:
            out[i][lo:hi] = b[pos : pos + hi - lo]
    return out
