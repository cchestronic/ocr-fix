"""`ocrfix corrupt`: clean chunks -> the same chunks with synthetic OCR damage, for
calibration (clean vs corrupted scores) and training data.

Per chunk and variant: restore the placeholders, damage the prose with `corrupt.corrupt_text`
at the rates measured on the gold set (`errors table` output; or a fixed ``--rate``), mask the
result again with the same rule as `chunk`. Math, tables and protected runs are never
touched, so the placeholders come back unchanged, except that a garbled-Hangul op wrapping
Hangul in ``$..$`` adds a math span and renumbers them (``gold_placeholders`` keeps the
clean map), and damage next to a protected run can change where its pattern ends.

Output rows are chunk rows that `score` accepts: ``chunk_id`` =
``<chunk_id>~<r0.01|table>~v<n>``, ``source_text`` (corrupted, masked), ``placeholders``,
``text: null``, plus ``corruption``, ``variant``, ``error_rate`` (ops per 1000 prose chars),
``gold_text`` (the clean masked text), ``gold_placeholders``, ``src_chunk_id`` and
``applied_ops``. They don't tile a document, so they can't be reconstructed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from ocrfix.chunks.split import atoms_of, render
from ocrfix.corpus.placeholders import restore
from ocrfix.corpus.tagger import tag_spans
from ocrfix.corrupt.run import ALL, build_models, corrupt_opts, corrupt_text, rng_for
from ocrfix.io import iter_jsonl, open_text
from ocrfix.schema import Span


def _masked_as_blocks(raw: str, protect: list[str]) -> list[Span]:
    """Spans where everything the chunker masks (math, tables, and the protected runs such as
    URLs and citations), plus the whitespace on each side, is a non-prose block, so the
    damage lands in prose only and the placeholders survive."""
    spans, pos = [], 0
    for a, b, _ in atoms_of(raw, tag_spans(raw), protect):
        # the whitespace right around a masked item stays too: deleting it would glue a word
        # onto a URL or citation, and its pattern would then swallow that word
        a = a - 1 if a > pos and raw[a - 1].isspace() else a
        b = b + 1 if b < len(raw) and raw[b].isspace() else b
        if a > pos:
            spans.append(Span(pos, a, "prose"))
        spans.append(Span(a, b, "table"))
        pos = b
    if pos < len(raw):
        spans.append(Span(pos, len(raw), "prose"))
    return spans


def corrupt_chunks(
    rows: Iterable[dict[str, Any]],
    models: dict,
    *,
    seed: int,
    n_variants: int,
    rate: float | None,
    protect: list[str],
) -> Iterable[dict[str, Any]]:
    tag = "table" if rate is None else f"r{rate:g}"
    for row in rows:
        clean = row["text"] if row.get("text") is not None else row["source_text"]
        raw = restore(clean, row["placeholders"])
        spans = _masked_as_blocks(raw, protect)
        model = models.get(row.get("lang") or ALL) or models[ALL]
        for v in range(n_variants):
            rng = rng_for(seed, f"{row['doc_id']}\x00{tag}", row["chunk_id"], v)
            noisy, _, ops, err = corrupt_text(raw, spans, model, rng, fixed_rate=rate)
            masked, mapping = render(
                noisy, atoms_of(noisy, tag_spans(noisy), protect), 0, len(noisy)
            )
            yield {
                "chunk_id": f"{row['chunk_id']}~{tag}~v{v}",
                "doc_id": row["doc_id"],
                "source_path": row.get("source_path", ""),
                "src_chunk_id": row["chunk_id"],
                "lang": row.get("lang"),
                "source_text": masked,
                "text": None,
                "placeholders": mapping,
                "gold_text": clean,
                "gold_placeholders": row["placeholders"],
                "corruption": tag,
                "variant": v,
                "error_rate": round(err, 3),
                "applied_ops": ops,
                "status": "new",
                "score": None,
                "history": [],
            }


def run_corrupt(
    cfg, chunks: Path, error_table: Path, confusion_map: Path, out: Path, *,
    rate: float | None = None, n_variants: int | None = None, limit: int | None = None,
) -> int:  # fmt: skip
    """Writes one JSONL file (``.tmp`` + rename); returns the number of rows."""
    models = build_models(
        json.loads(error_table.read_text("utf-8")),
        json.loads(confusion_map.read_text("utf-8")),
        corrupt_opts(cfg),
    )
    n_var = int(cfg.corrupt.variants if n_variants is None else n_variants)
    rows = corrupt_chunks(
        islice(iter_jsonl(chunks), limit), models, seed=int(cfg.seed), n_variants=n_var,
        rate=rate, protect=list(cfg.chunk.protect),
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    with open_text(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out)
    return n
