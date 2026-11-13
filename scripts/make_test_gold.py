#!/usr/bin/env python3
"""TEST ONLY: a fake gold set of chunk pairs from the error-free test chunks.

    python scripts/make_test_gold.py data/test/chunks/truth data/test/gold --n 5000

Real gold comes from annotators (OCR chunk -> corrected chunk). Here each sampled truth chunk
is damaged with the test generator's noise (`make_test_corpus.corrupt`), so the pair is
(noisy chunk, clean chunk), in the row format `ocrfix errors table` and `calibrate
--gold-scores` read: source_text, gold_text (both masked), placeholders, lang, stratum, cer.
Chunks whose placeholders change under the noise are skipped. Documents are split 50/50
into gold_train.jsonl and gold_eval.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_test_corpus import corrupt

from ocrfix.chunks.split import atoms_of, render
from ocrfix.config import load_config
from ocrfix.corpus.placeholders import restore
from ocrfix.corpus.tagger import tag_spans
from ocrfix.editdist import levenshtein
from ocrfix.io import iter_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("truth_chunks", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--error-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    protect = list(load_config().llm.protect)
    rows = [r for r in iter_jsonl(a.truth_chunks) if not r["short"]]
    r = random.Random(a.seed)
    r.shuffle(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        s: open(a.out_dir / f"gold_{s}.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        for s in ("train", "eval")
    }
    n = skipped = 0
    for row in rows:
        if n >= a.n:
            break
        raw = restore(row["source_text"], row["placeholders"])
        noisy, kinds = corrupt(raw, r, a.error_rate * r.choice([0, 0.5, 1, 2, 4]))
        masked, mapping = render(noisy, atoms_of(noisy, tag_spans(noisy), protect), 0, len(noisy))
        if mapping != row["placeholders"]:
            skipped += 1
            continue
        gold = row["source_text"]
        side = "eval" if hashlib.sha256(row["doc_id"].encode()).digest()[0] % 2 else "train"
        files[side].write(json.dumps({
            "doc_id": row["doc_id"], "para_id": row["chunk_id"], "chunk_id": row["chunk_id"],
            "order": row["chunk_index"], "lang": row["lang"], "stratum": f"{row['lang']}:test",
            "source_path": row["source_path"], "source_text": masked, "gold_text": gold,
            "placeholders": mapping, "span_labels": {}, "structure_label": "",
            "cer": round(levenshtein(masked, gold) / max(1, len(gold)), 5),
            "injected": dict(kinds),
        }, ensure_ascii=False) + "\n")  # fmt: skip
        n += 1
    for f in files.values():
        f.close()
    print(f"{a.out_dir}: {n} gold chunk pairs ({skipped} skipped: placeholders changed)")


if __name__ == "__main__":
    main()
