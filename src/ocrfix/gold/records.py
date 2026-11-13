"""`ocrfix gold records`: gold rows -> paragraph Records of the OCR source, for scoring
(``--side gold``: of the gold text, the clean side for `corrupt` and `calibrate`).

Each Record's ``text`` is the source with placeholders restored (``source_raw``), re-tagged,
and carries ``cer`` (char edit distance to ``gold_raw`` / len(gold_raw)), ``stratum`` and
``gold_raw``. Scored with `ocrfix score`, it is the ``--gold-scores`` input of calibrate.
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

from ocrfix.corpus import tag_spans
from ocrfix.editdist import levenshtein
from ocrfix.io import iter_jsonl, open_text
from ocrfix.schema import Record, nfc


def gold_records(
    gold: Path, out: Path, limit: int | None = None, side: str = "source"
) -> tuple[int, int]:
    """Returns (written, skipped: rows without source_raw/gold_raw). ``side="gold"`` writes
    the gold text instead (``cer`` 0): the clean side for `corrupt` and calibration."""
    if side not in ("source", "gold"):
        raise ValueError(f"side must be source or gold, got {side!r}")
    n = skipped = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open_text(tmp, "wt") as f:
        for d in islice(iter_jsonl(gold), limit):
            src, ref = d.get("source_raw"), d.get("gold_raw")
            if src is None or ref is None:
                skipped += 1
                continue
            src, ref = nfc(src), nfc(ref)
            if side == "gold":
                src = ref
            rec = Record(
                doc_id=str(d["doc_id"]),
                para_id=str(d["para_id"]),
                order=int(d.get("order") or 0),
                lang=d.get("lang") or "other",
                text=src,
                spans=tag_spans(src),
                source_path=d.get("source_path", ""),
                extra={
                    "cer": round(levenshtein(src, ref) / max(1, len(ref)), 5),
                    "stratum": d.get("stratum"),
                    "gold_raw": ref,
                },
            )
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out)
    return n, skipped
