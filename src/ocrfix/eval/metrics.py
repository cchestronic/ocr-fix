"""Per-document metrics of one system against gold.

Three texts per document, built by ``align.join`` from the gold paragraphs in order:
``source`` (source_raw), ``gold`` (gold_raw) and ``pred`` (the system's records for that
document, in file order, so merges and splits don't change it). Then:

  cer / wer       char / word edit distance pred -> gold, over gold length
  spacing (ser)   spaces present in one of pred, gold but not the other, at aligned
                  non-space chars, over gold spaces (the Korean metric)
  false fixes     system edits (source -> pred regions) that touch no gold edit
                  (source -> gold region): it changed text the annotator left alone
  recall by type  gold edits (``errors.align.diff_ops``, typed as in ``errors table``) with
                  no residual edit (pred -> gold) touching their gold range
  identity        gold paragraphs equal to their source ("clean") that a system edit touches
  placeholders    math/table spans of gold (raw, in order) found verbatim in pred, as the
                  longest common subsequence; ``exact`` when the sequences are equal
  structure       per paragraph: none | merge (with next) | split, gold label vs the system's
                  record boundaries (see ``_structure``)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ocrfix.corpus import tag_spans
from ocrfix.editdist import levenshtein
from ocrfix.errors.align import align, diff_ops
from ocrfix.eval.align import Aligner, DocText, flatten, join, opcodes, touches

MERGE = "merge-with-next"
_SPLIT = re.compile(r"^split-at:(\d+)$")
STRUCT = ("none", "merge", "split")


@dataclass(frozen=True)
class GoldPara:
    doc_id: str
    para_id: str
    order: int
    lang: str
    source: str
    gold: str
    structure: str = ""  # "", merge-with-next, split-at:<n>

    @property
    def clean(self) -> bool:
        return self.source == self.gold

    @property
    def gold_class(self) -> str:
        if self.structure == MERGE:
            return "merge"
        return "split" if _SPLIT.match(self.structure or "") else "none"


@dataclass
class GoldDoc:
    doc_id: str
    paras: list[GoldPara]

    @property
    def lang(self) -> str:
        w: Counter[str] = Counter()
        for p in self.paras:
            w[p.lang] += len(p.gold)
        return w.most_common(1)[0][0] if w else "other"


@dataclass
class ParaResult:
    para_id: str
    lang: str
    gold_len: int
    edits: int
    clean: bool
    altered: bool
    source: str
    pred: str
    gold: str

    @property
    def cer(self) -> float:
        return self.edits / max(1, self.gold_len)


@dataclass
class DocResult:
    doc_id: str
    lang: str
    missing: bool = False
    gold_chars: int = 0
    char_edits: int = 0
    gold_words: int = 0
    word_edits: int = 0
    gold_spaces: int = 0
    space_errors: int = 0
    source_chars: int = 0
    sys_edits: int = 0
    false_fixes: int = 0
    recall: dict[str, list[int]] = field(default_factory=dict)  # key -> [total, fixed]
    paras: list[ParaResult] = field(default_factory=list)
    # structure: (gold_class, pred_class, split position right or None) per scorable paragraph
    structure: list[tuple[str, str, bool | None]] = field(default_factory=list)
    ph_expected: int = 0
    ph_matched: int = 0
    ph_extra: int = 0

    @property
    def cer(self) -> float:
        return self.char_edits / max(1, self.gold_chars)

    @property
    def ph_exact(self) -> bool:
        return self.ph_matched == self.ph_expected and self.ph_extra == 0

    def summary(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "lang": self.lang,
            "missing": self.missing,
            "cer": round(self.cer, 6),
            "wer": round(self.word_edits / max(1, self.gold_words), 6),
            "ser": round(self.space_errors / max(1, self.gold_spaces), 6),
            "gold_chars": self.gold_chars,
            "char_edits": self.char_edits,
            "sys_edits": self.sys_edits,
            "false_fixes": self.false_fixes,
            "placeholders_exact": self.ph_exact,
        }


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


def protected(text: str) -> list[str]:
    return [text[s.start : s.end] for s in tag_spans(text) if s.kind in ("math", "table")]


def lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _structure(
    doc: GoldDoc,
    src: DocText,
    gold: DocText,
    pred: DocText,
    sp: Aligner,
    gp: Aligner,
    tol: int,
) -> list[tuple[str, str, bool | None]]:
    """Gold vs predicted structure per paragraph.

    Predicted ``merge``: the joint after this paragraph (source) maps to no record joint
    in pred (within 1 char), i.e. one record spans both. Only scored when the next gold
    paragraph is the next one in the document (``order + 1``); a merge-with-next whose
    neighbour isn't in the eval set can't be judged. Predicted ``split``: a record joint
    falls strictly inside the paragraph's range in pred; for a gold split its position is
    right if a joint is within ``tol`` chars of the gold offset.
    """
    joints = pred.joints
    out: list[tuple[str, str, bool | None]] = []
    n = len(doc.paras)
    for i, p in enumerate(doc.paras):
        s, e = src.ranges[i]
        ps, pe = sp.map(s), sp.map(e)
        inner = [b for b in joints if ps + 1 < b < pe - 1]
        adjacent = i + 1 < n and doc.paras[i + 1].order == p.order + 1
        merged = False
        if i + 1 < n:
            pj = sp.map(e)
            merged = not any(abs(b - pj) <= 1 for b in joints)
        gold_class = p.gold_class
        if gold_class == "merge" and not adjacent:
            continue  # neighbour not in the eval set
        pred_class = "split" if inner else ("merge" if merged else "none")
        pos_ok: bool | None = None
        if gold_class == "split":
            m = _SPLIT.match(p.structure)
            assert m is not None
            off = len(re.sub(r"[ \t]*\n[ \t]*\n\s*", " ", p.gold[: int(m.group(1))]).lstrip())
            target = gp.map(gold.ranges[i][0] + off)
            pos_ok = any(abs(b - target) <= tol for b in inner)
        out.append((gold_class, pred_class, pos_ok))
    return out


def evaluate_doc(
    doc: GoldDoc,
    pred_texts: list[str] | None,
    *,
    errors_opts: dict[str, Any],
    split_tolerance: int = 3,
) -> DocResult:
    """``pred_texts`` = the system's record texts in order; None = document missing (source
    is used, so paired comparisons keep every document)."""
    res = DocResult(doc.doc_id, doc.lang, missing=pred_texts is None)
    src = join([p.source for p in doc.paras])
    gold = join([p.gold for p in doc.paras])
    sources = [p.source for p in doc.paras]
    pred = join(pred_texts if pred_texts is not None else sources, skip_empty=True)
    S, G, P = src.text, gold.text, pred.text

    res.gold_chars, res.source_chars = len(G), len(S)
    res.char_edits = levenshtein(P, G)
    res.gold_words, res.word_edits = len(G.split()), word_edits(P, G)
    res.space_errors, res.gold_spaces = spacing_errors(P, G)

    gold_regions = align(S, G)
    sys_regions = align(S, P)
    res.sys_edits = len(sys_regions)
    texts = pred_texts if pred_texts is not None else [p.source for p in doc.paras]
    res.false_fixes = sum(
        not any(touches(a1, a2, g1, g2) for g1, g2, _, _ in gold_regions)
        for a1, a2, _, _ in sys_regions
    )

    residual = align(P, G)  # (pred_start, pred_end, gold_start, gold_end)
    rec: dict[str, list[int]] = {}
    for op in diff_ops(S, G, **errors_opts):
        cell = rec.setdefault(op.key, [0, 0])
        cell[0] += 1
        if not any(touches(r[2], r[3], op.gold_start, op.gold_end) for r in residual):
            cell[1] += 1
    res.recall = rec

    sp = Aligner(S, P)
    for i, p in enumerate(doc.paras):
        s, e = src.ranges[i]
        seg = P[sp.map(s) : sp.map(e)].strip()
        g = flatten(p.gold)
        res.paras.append(
            ParaResult(
                para_id=p.para_id,
                lang=p.lang,
                gold_len=len(g),
                edits=levenshtein(seg, g),
                clean=p.clean,
                altered=any(touches(a1, a2, s, e) for a1, a2, _, _ in sys_regions),
                source=p.source,
                pred=seg,
                gold=p.gold,
            )
        )

    res.structure = _structure(doc, src, gold, pred, sp, Aligner(G, P), split_tolerance)

    expected = [x for p in doc.paras for x in protected(p.gold)]
    got = [x for t in texts for x in protected(t)]
    res.ph_expected = len(expected)
    res.ph_matched = lcs(expected, got)
    res.ph_extra = len(got) - res.ph_matched
    return res
