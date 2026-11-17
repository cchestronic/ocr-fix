"""Helpers shared by `data sft` (stages/data.py) and `data dpo` (stages/dpo.py).

- ``EMPTY_THINK``: the empty think block of Qwen3.5's non-thinking mode, put before an answer
  when ``sft.empty_think`` is on.
- ``in_val``: the validation split, by document (seeded hash of ``doc_id``).
- ``quotas``: how many examples to take per source and language from the mix.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sized
from typing import Any

EMPTY_THINK = "<think>\n\n</think>\n\n"

_EPS = 1e-9


def in_val(doc_id: str, seed: int | str, val_frac: float) -> bool:
    """True if document ``doc_id`` belongs to the validation side.

    Deterministic per (seed, doc_id), so every chunk of a document lands on the same side and
    reruns give the same split."""
    if val_frac <= 0:
        return False
    h = hashlib.sha256(f"{seed}\x00val\x00{doc_id}".encode()).hexdigest()[:8]
    return int(h, 16) / 16**8 < val_frac


def _norm(weights: Mapping[str, Any], keys: list[str]) -> dict[str, float]:
    """``weights`` restricted to ``keys`` (missing = 0), scaled to sum 1."""
    w = {k: max(float(weights.get(k) or 0.0), 0.0) for k in keys}
    s = sum(w.values())
    if s <= 0:
        raise ValueError(f"weights {dict(weights)} give nothing to {keys}")
    return {k: v / s for k, v in w.items()}


def quotas(
    pools: Mapping[str, Mapping[str, Sized]],
    mix: Mapping[str, Any],
    lang_balance: Mapping[str, Any] | None = None,
    total: int | None = None,
) -> dict[str, dict[str, int]]:
    """Examples to take per source and language: ``{source: {lang: k}}``.

    ``pools``: ``{source: {lang: ids}}`` (anything with a length). ``mix`` gives each source's
    share, renormalized over the sources in ``pools`` (a source missing from ``mix`` gets 0).
    ``lang_balance`` (``{lang: share}``, renormalized; other languages get 0) splits each
    source's share over languages; None = in the proportions the source's pool has.

    The total is the largest whole number for which no cell needs more than its pool holds (no
    reuse), capped by ``total``; each cell is then rounded down."""
    sources = list(pools)
    if not sources:
        return {}
    w = _norm(mix, sources)
    sizes = {s: {lang: len(ids) for lang, ids in pools[s].items()} for s in sources}

    share: dict[str, dict[str, float]] = {}
    for s in sources:
        n_s = sum(sizes[s].values())
        if lang_balance is None:
            share[s] = {
                lang: (w[s] * n / n_s if n_s else 0.0) for lang, n in sizes[s].items()
            }
        else:
            b = _norm(lang_balance, sorted(set(lang_balance) | set(sizes[s])))
            share[s] = {lang: w[s] * b[lang] for lang in b}

    t_max = math.inf
    for s in sources:
        for lang, sh in share[s].items():
            if sh > 0:
                t_max = min(t_max, sizes[s].get(lang, 0) / sh)
    if t_max is math.inf:
        t_max = 0.0
    t = math.floor(t_max + _EPS)
    if total is not None:
        t = min(t, int(total))

    return {
        s: {
            lang: min(sizes[s].get(lang, 0), math.floor(t * sh + _EPS))
            for lang, sh in sorted(share[s].items())
            if sh > 0
        }
        for s in sources
    }
