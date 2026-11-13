"""`ocrfix fix`: run the enabled fixers over each document, log every change.

Input is paragraph Records grouped by document (as ingest writes them; a doc_id that
reappears later is an error). For each document the fixers run in ``fix.order``; after each
one, ``check_ops`` verifies its ops explain every para_id that appeared, disappeared or
changed. Output Records carry ``original_text`` (sources joined by a blank line for a merge)
and ``fixer`` (names of the fixers that touched them), in document order. Ops go to
``--ops``. Both files are written via ``.tmp`` + rename; a rerun regenerates them.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from ocrfix.fixers.base import Fixer, FixerError, Op, build_fixers, check_ops
from ocrfix.io import iter_records, open_text
from ocrfix.schema import Record

log = logging.getLogger(__name__)


@dataclass
class FixStats:
    docs: int = 0
    paragraphs_in: int = 0
    paragraphs_out: int = 0
    ops: Counter[str] = field(default_factory=Counter)  # "<rule>:<op>" -> n


def documents(path: Path, limit: int | None = None) -> Iterator[list[Record]]:
    seen: set[str] = set()
    doc: list[Record] = []
    n = 0
    for rec in iter_records(path):
        if doc and rec.doc_id != doc[0].doc_id:
            yield sorted(doc, key=lambda r: r.order)
            n += 1
            if limit is not None and n >= limit:
                return
            doc = []
        if not doc:
            if rec.doc_id in seen:
                raise FixerError(f"{path}: document {rec.doc_id} is not contiguous")
            seen.add(rec.doc_id)
        doc.append(rec)
    if doc and (limit is None or n < limit):
        yield sorted(doc, key=lambda r: r.order)


def fix_document(paras: list[Record], fixers: list[Fixer]) -> tuple[list[Record], list[Op]]:
    original = {r.para_id: r.text for r in paras}
    touched: dict[str, list[str]] = {}
    all_ops: list[Op] = []
    for fx in fixers:
        before = paras
        paras, ops = fx.apply(paras)
        check_ops(before[0].doc_id if before else "", before, paras, ops, fx.name)
        for op in ops:
            for pid in op.dst_para_ids:
                srcs = [touched.get(s, []) for s in op.src_para_ids]
                names = [n for ns in srcs for n in ns]
                touched[pid] = sorted({*names, fx.name})
                if pid not in original:
                    original[pid] = "\n\n".join(original.get(s, "") for s in op.src_para_ids)
            if op.op == "merge" and op.dst_para_ids:
                original[op.dst_para_ids[0]] = "\n\n".join(
                    original.get(s, "") for s in op.src_para_ids
                )
        all_ops += ops
        if not paras:
            break
    out = [
        replace(
            r,
            extra=r.extra
            | {"original_text": original[r.para_id], "fixer": touched.get(r.para_id, [])},
        )
        for r in paras
    ]
    return out, all_ops


def fix_file(
    paragraphs: Path, out: Path, ops_path: Path, fixers: list[Fixer], limit: int | None = None
) -> FixStats:
    stats = FixStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    ops_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.name + ".tmp")
    tmp_ops = ops_path.with_name(ops_path.name + ".tmp")
    with open_text(tmp_out, "wt") as fo, open_text(tmp_ops, "wt") as fops:
        for doc in documents(paragraphs, limit):
            stats.docs += 1
            stats.paragraphs_in += len(doc)
            fixed, ops = fix_document(doc, fixers)
            stats.paragraphs_out += len(fixed)
            for r in fixed:
                fo.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            for op in ops:
                fops.write(json.dumps(op.to_dict(), ensure_ascii=False) + "\n")
                stats.ops[f"{op.rule}:{op.op}"] += 1
    tmp_out.replace(out)
    tmp_ops.replace(ops_path)
    return stats


def run_fix(cfg, paragraphs: Path, out: Path, ops_path: Path, limit: int | None = None) -> FixStats:
    from omegaconf import OmegaConf

    fix_cfg = OmegaConf.to_container(cfg.fix, resolve=True)
    assert isinstance(fix_cfg, dict)
    fixers = build_fixers(fix_cfg)
    log.info("fixers: %s", ", ".join(f.name for f in fixers) or "(none)")
    return fix_file(paragraphs, out, ops_path, fixers, limit)
