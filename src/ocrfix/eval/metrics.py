"""Spacing and word metrics for `ocrfix eval`.

spacing (ser)   spaces present in one of pred, gold but not the other, at aligned
                non-space chars, over gold spaces (the Korean metric)
word edits      Levenshtein over whitespace-separated words (WER numerator)
"""

from __future__ import annotations

from ocrfix.editdist import levenshtein
from ocrfix.eval.align import opcodes


def spacing_errors(pred: str, gold: str) -> tuple[int, int]:
    """(spacing disagreements at aligned non-space chars, spaces in gold)."""

    def squeeze(t: str) -> tuple[str, list[bool]]:
        chars: list[str] = []
        after: list[bool] = []
        for ch in t:
            if ch.isspace():
                if after:
                    after[-1] = True
            else:
                chars.append(ch)
                after.append(False)
        return "".join(chars), after

    pc, ps = squeeze(pred)
    gc, gs = squeeze(gold)
    errors = 0
    for tag, i1, i2, j1, _ in opcodes(pc, gc):
        if tag == "equal":  # the last char's follower differs, so its space isn't comparable
            errors += sum(ps[i1 + k] != gs[j1 + k] for k in range(i2 - i1 - 1))
    return errors, sum(gs)


def word_edits(pred: str, gold: str) -> int:
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(w, len(vocab)) for w in pred.split()]
    b = [vocab.setdefault(w, len(vocab)) for w in gold.split()]
    return levenshtein(a, b)  # type: ignore[arg-type]
