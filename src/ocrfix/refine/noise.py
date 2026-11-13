"""Find and remove random noise (``ⅡⅢ▯ ¤ʘ…``, ``lIl1 :;. rnvv``, garbled Hangul) in a chunk,
from the scorer's per-token surprisal.

Per-token surprisal does not separate noise from OCR errors: odd symbols are split into 2-3
byte tokens, the later ones predictable, and the model adapts to junk quickly. Bits per
*character* does, because noise packs many tokens into few characters. It is judged per
whitespace-separated word, against a threshold for the word's script. Hangul syllables carry
more bits each than Latin letters, so a chunk-wide threshold either cuts Korean text or keeps
Latin junk.

Measured with vLLM on Qwen3.5-4B, 3,000 test chunks (24-char windows, p99.9 of text far from
noise): Latin 3.7, Hangul / digits / punctuation 5.8. Noise: Latin junk p50 6, symbols 12,
garbled Hangul 14. Text right after noise is high too (the model's context is polluted), so a
window, or a word on a single low threshold, cuts into it; hysteresis does not.

``noise_spans``:
  1. a word's value is the mean bits per char over its scored chars (token bits are spread
     over the token's chars). A word holding a placeholder is never noise and ends a run; a
     word with no scored char (boundary tokens) may join a run only if it has no letter or
     digit of Latin or Hangul;
  2. a word above ``core[script]`` starts a run; neighbouring words above ``extend[script]``
     join it (script = ``hangul`` if at least half its chars are Hangul, else ``other``);
  3. runs with at least ``min_chars`` (config: 20) non-space chars are noise. Shorter ones are
     left for the corrector.
On 3,000 test chunks, vs the old 24-char windows with a per-language threshold: removed text
that is noise 0.79 -> 0.96 at min_chars 12 (noise found 0.86 -> 0.74; the misses are mostly
repeated junk, which the model learns in context). min_chars 20: 0.97 (one real false removal
left by hand), noise found 0.58. Measured too: the gap to the model's top guess
(log2 p(top1) - log2 p(token)) instead of the surprise: about the same trade-off here, and
worse for flagging damaged chunks, so surprise is used for both.
``remove_spans`` deletes the runs and tidies the whitespace.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.scorer.tokens import TokenizedPara

WORD = re.compile(r"\S+")
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
LETTER = re.compile(r"[A-Za-z0-9가-힣ᄀ-ᇿ㄰-㆏]")

STOP, COLD, WARM, HOT = -1, 0, 1, 2


def char_bits(tp: TokenizedPara, bits: np.ndarray) -> np.ndarray:
    """Bits per character (NaN where unknown: placeholders, unscored tokens)."""
    out = np.full(len(tp.text), np.nan)
    for i in np.flatnonzero(tp.scored):
        a, b = int(tp.starts[i]), int(tp.ends[i])
        if b <= a:
            continue
        seg = out[a:b]
        out[a:b] = np.where(np.isnan(seg), 0.0, seg) + float(bits[i]) / (b - a)
    return out


def word_levels(
    text: str, per: np.ndarray, core: Mapping[str, float], extend: Mapping[str, float]
) -> list[tuple[int, int, int, float]]:
    """(start, end, level, bits per char) per word; level STOP / COLD / WARM / HOT."""
    in_ph = np.zeros(len(text), dtype=bool)
    for m in PLACEHOLDER.finditer(text):
        in_ph[m.start() : m.end()] = True
    out = []
    for m in WORD.finditer(text):
        a, e = m.span()
        w = m.group()
        v = per[a:e]
        v = v[~np.isnan(v)]
        if in_ph[a:e].any():
            out.append((a, e, STOP, float("nan")))
        elif len(v) == 0:
            out.append((a, e, STOP if LETTER.search(w) else WARM, float("nan")))
        else:
            script = "hangul" if 2 * len(HANGUL.findall(w)) >= len(w) else "other"
            x = float(v.mean())
            lvl = HOT if x > core[script] else WARM if x > extend[script] else COLD
            out.append((a, e, lvl, x))
    return out


def noise_spans(
    tp: TokenizedPara,
    bits: np.ndarray,
    core: Mapping[str, float],
    extend: Mapping[str, float],
    *,
    min_chars: int = 20,
) -> list[tuple[int, int, float]]:
    """(start, end, bits per char) of noise in ``tp.text``."""
    text = tp.text
    if not text:
        return []
    per = char_bits(tp, bits)
    words = word_levels(text, per, core, extend)
    out, i, n = [], 0, len(words)
    while i < n:
        if words[i][2] != HOT:
            i += 1
            continue
        a = b = i
        while a > 0 and words[a - 1][2] >= WARM:
            a -= 1
        while b + 1 < n and words[b + 1][2] >= WARM:
            b += 1
        start, end = words[a][0], words[b][1]
        if sum(not c.isspace() for c in text[start:end]) >= min_chars:
            seg = per[start:end]
            known = seg[~np.isnan(seg)]
            out.append((start, end, round(float(known.mean()), 2) if len(known) else 0.0))
        i = b + 1
    return out


def remove_spans(text: str, spans: Sequence[tuple[int, int, Any]]) -> str:
    """Delete the spans; the whitespace on both sides of each becomes one separator (a blank
    line if either side had one, a newline, else a space), and the text edges are stripped."""
    parts, pos = [], 0
    for a, b, *_ in sorted(spans):
        parts.append(text[pos:a])
        parts.append("\x00")  # where a span was
        pos = b
    parts.append(text[pos:])
    joined = "".join(parts)

    def gap(m: re.Match) -> str:
        ws = m.group().replace("\x00", "")
        return "\n\n" if "\n\n" in ws else "\n" if "\n" in ws else " "

    joined = re.sub(r"\s*\x00(?:\s*\x00)*\s*", gap, joined)
    return joined.strip()
