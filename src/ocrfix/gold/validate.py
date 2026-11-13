"""Checks on one annotated gold row. Any violation keeps the row out of the gold split.

Rules (``Violation.rule``):
  empty_gold           gold_text is empty or whitespace
  length_ratio         len(gold) / len(source) outside gold.validate.length_ratio (only when
                       the source has at least gold.validate.ratio_min_chars characters)
  span_label           a placeholder has no label, an unknown label, or a label names a
                       placeholder that is not in the source
  placeholders         gold placeholders differ from the source's, minus those labeled
                       misclassified-text (each exactly once, same order; those must be gone)
  latin_in_ko          lang == ko and gold has a Latin word that is neither in the source
                       (prose or the raw text of misclassified spans) nor an OCR fix of one
                       (edit distance to a source word <= 1 for words of <= 3 letters,
                       else max(2, len // 3); "rnodel" -> "model" is 2)
  structure_label      not "", "merge-with-next" or "split-at:<n>" with 0 < n < len(gold)
  unrecoverable_count  given but != number of 〈?〉 in gold
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ocrfix.corpus.placeholders import PLACEHOLDER, PlaceholderError, restore
from ocrfix.editdist import levenshtein
from ocrfix.gold.labelstudio import parse_span_labels, placeholder_name
from ocrfix.schema import UNCERTAIN, nfc

SPAN_LABELS = ("ok", "broken", "misclassified-text")
MISCLASSIFIED = "misclassified-text"
MERGE = "merge-with-next"
_SPLIT = re.compile(r"^split-at:(\d+)$")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_TEX_CMD = re.compile(r"\\[A-Za-z]+")


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str


def source_placeholders(row: dict[str, Any]) -> list[str]:
    mapping = row.get("placeholders")
    return list(mapping) if mapping else PLACEHOLDER.findall(row.get("source_text") or "")


def latin_words(text: str) -> set[str]:
    return {w.casefold() for w in _LATIN_WORD.findall(PLACEHOLDER.sub(" ", text))}


def _is_ocr_fix(word: str, vocab: set[str]) -> bool:
    # rn -> m costs 2, so 2 edits from 4 letters on; 1 for short words
    budget = 1 if len(word) <= 3 else max(2, len(word) // 3)
    return any(abs(len(word) - len(v)) <= budget and levenshtein(word, v) <= budget for v in vocab)


def validate_row(
    row: dict[str, Any],
    *,
    length_ratio: tuple[float, float] = (0.8, 1.25),
    ratio_min_chars: int = 20,
) -> list[Violation]:
    out: list[Violation] = []
    src = nfc(row.get("source_text") or "")
    gold = nfc(row.get("gold_text") or "")
    if not gold.strip():
        return [Violation("empty_gold", "gold_text is empty")]

    if len(src) >= ratio_min_chars:
        ratio = len(gold) / len(src)
        lo, hi = length_ratio
        if not lo <= ratio <= hi:
            out.append(Violation("length_ratio", f"len(gold)/len(source) = {ratio:.2f}"))

    phs = source_placeholders(row)
    names = {placeholder_name(ph): ph for ph in phs}
    try:
        labels = parse_span_labels(row.get("span_labels"))
    except ValueError as e:
        out.append(Violation("span_label", str(e)))
        labels = {}
    for name in names:
        if name not in labels:
            out.append(Violation("span_label", f"{name} has no label"))
        elif labels[name] not in SPAN_LABELS:
            out.append(Violation("span_label", f"{name}: unknown label {labels[name]!r}"))
    for name in labels.keys() - names.keys():
        out.append(Violation("span_label", f"label for {name}, which is not in the source"))

    misclassified = [ph for ph in phs if labels.get(placeholder_name(ph)) == MISCLASSIFIED]
    in_gold = PLACEHOLDER.findall(gold)
    if left := [ph for ph in misclassified if ph in in_gold]:
        out.append(
            Violation("placeholders", f"{' '.join(left)} labeled {MISCLASSIFIED} but still in gold")
        )
    else:
        try:
            restore(gold, {ph: "" for ph in phs if ph not in misclassified})
        except PlaceholderError as e:
            out.append(Violation("placeholders", str(e)))

    if row.get("lang") == "ko":
        mapping = row.get("placeholders") or {}
        vocab = latin_words(src)
        for ph in misclassified:
            vocab |= latin_words(_TEX_CMD.sub(" ", mapping.get(ph, "")))
        new = sorted(w for w in latin_words(gold) - vocab if not _is_ocr_fix(w, vocab))
        if new:
            out.append(Violation("latin_in_ko", f"new Latin word(s) in ko text: {', '.join(new)}"))

    structure = (row.get("structure_label") or "").strip()
    if structure not in ("", "none", MERGE):
        m = _SPLIT.match(structure)
        if not m:
            out.append(Violation("structure_label", f"unknown structure label {structure!r}"))
        elif not 0 < int(m.group(1)) < len(gold):
            out.append(
                Violation("structure_label", f"{structure}: offset outside gold (len {len(gold)})")
            )

    count = row.get("unrecoverable_count")
    if count not in (None, ""):
        try:
            ok = int(count) == gold.count(UNCERTAIN)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(
                Violation(
                    "unrecoverable_count",
                    f"unrecoverable_count={count!r}, but gold has "
                    f"{gold.count(UNCERTAIN)} {UNCERTAIN}",
                )
            )
    return out
