"""`ocrfix gold import`: validate annotations, measure agreement, split by document.

Input is the sample JSONL with the annotation fields filled in, or a Label Studio JSON export
(``*.json``). Outputs in ``out_dir``:

  gold_train.jsonl / gold_eval.jsonl   accepted rows, one per paragraph, sorted by (doc, order)
  gold_eval.manifest.json              sha256 of the eval file and of every eval row
  gold_eval.jsonl.sha256               ``sha256sum -c`` format
  split.json                           seed + doc ids per side
  violations.jsonl                     one line per rule broken
  import_report.json                   counts, violations per rule, inter-annotator distance

Split: docs already listed in an existing ``split.json`` keep their side (so a re-import
after more annotation never moves an eval doc into train); the new docs are shuffled with
``random.Random(seed)`` in sorted order and fill eval up to ``eval_frac`` of all docs.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from ocrfix.corpus.placeholders import restore
from ocrfix.editdist import normalized
from ocrfix.gold.labelstudio import parse_span_labels, placeholder_name, read_export
from ocrfix.gold.validate import MISCLASSIFIED, Violation, source_placeholders, validate_row
from ocrfix.io import iter_jsonl
from ocrfix.schema import UNCERTAIN, nfc

TEXT_FIELDS = ("source_text", "gold_text", "context_prev", "context_next", "notes")
Key = tuple[str, str]


@dataclass
class ImportReport:
    n_rows: int = 0
    n_paragraphs: int = 0
    n_accepted: int = 0
    n_train: int = 0
    n_eval: int = 0
    train_docs: int = 0
    eval_docs: int = 0
    violations: Counter[str] = field(default_factory=Counter)
    rejected_paragraphs: int = 0
    agreement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["violations"] = dict(sorted(self.violations.items()))
        return d


def load_annotations(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_export(path) if path.suffix == ".json" else list(iter_jsonl(path))
    rows = rows[:limit] if limit is not None else rows
    for r in rows:
        for k in TEXT_FIELDS:
            if isinstance(r.get(k), str):
                r[k] = nfc(r[k])
        if isinstance(r.get("placeholders"), dict):
            r["placeholders"] = {k: nfc(v) for k, v in r["placeholders"].items()}
        r["doc_id"], r["para_id"] = str(r["doc_id"]), str(r["para_id"])
        r["annotator_id"] = str(r.get("annotator_id") or "")
    return rows


def agreement(groups: dict[Key, list[dict[str, Any]]]) -> dict[str, Any]:
    """Normalized char edit distance between gold texts of different annotators."""
    pairs = []
    for key, rows in sorted(groups.items()):
        by_ann = {r["annotator_id"]: r for r in rows if r["annotator_id"] and r.get("gold_text")}
        for a, b in combinations(sorted(by_ann), 2):
            d = normalized(by_ann[a]["gold_text"], by_ann[b]["gold_text"])
            pairs.append({"doc_id": key[0], "para_id": key[1], "a": a, "b": b, "ced": round(d, 4)})
    if not pairs:
        return {"n_pairs": 0}
    ds = [p["ced"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        "n_paragraphs": len({(p["doc_id"], p["para_id"]) for p in pairs}),
        "mean_ced": round(statistics.fmean(ds), 4),
        "median_ced": round(statistics.median(ds), 4),
        "max_ced": max(ds),
        "exact_match": round(sum(d == 0 for d in ds) / len(ds), 4),
        "pairs": pairs,
    }


def finalize(row: dict[str, Any], n_annotations: int) -> dict[str, Any]:
    """Normalized fields plus raw (placeholders restored) source and gold."""
    out = dict(row)
    labels = parse_span_labels(row.get("span_labels"))
    out["span_labels"] = labels
    structure = row.get("structure_label")
    out["structure_label"] = "" if structure in (None, "none") else structure
    out["unrecoverable_count"] = row["gold_text"].count(UNCERTAIN)
    out["n_annotations"] = n_annotations
    mapping = row.get("placeholders")
    if mapping:
        kept = {
            ph: raw
            for ph, raw in mapping.items()
            if labels.get(placeholder_name(ph)) != MISCLASSIFIED
        }
        out["source_raw"] = restore(row["source_text"], mapping)
        out["gold_raw"] = restore(row["gold_text"], kept)
    elif not source_placeholders(row):
        out["source_raw"], out["gold_raw"] = row["source_text"], row["gold_text"]
    return out


def split_docs(
    docs: list[str], seed: int, eval_frac: float, pinned: dict[str, str] | None = None
) -> dict[str, str]:
    """doc_id -> "train" | "eval". Pinned docs keep their side."""
    unique = set(docs)
    pinned = {d: s for d, s in (pinned or {}).items() if d in unique}
    new = sorted(unique - pinned.keys())
    random.Random(seed).shuffle(new)
    n_eval = max(0, int(len(unique) * eval_frac) - sum(s == "eval" for s in pinned.values()))
    side = dict(pinned)
    side.update({d: "eval" if i < n_eval else "train" for i, d in enumerate(new)})
    return side


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def import_annotations(
    annotated: Path,
    out_dir: Path,
    *,
    seed: int,
    eval_frac: float = 0.5,
    length_ratio: tuple[float, float] = (0.8, 1.25),
    ratio_min_chars: int = 20,
    limit: int | None = None,
) -> ImportReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_annotations(annotated, limit)
    report = ImportReport(n_rows=len(rows))

    groups: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["doc_id"], r["para_id"])].append(r)
    report.n_paragraphs = len(groups)
    report.agreement = agreement(groups)

    violations: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        seen: set[str] = set()
        valid = []
        for r in group:
            found = validate_row(r, length_ratio=length_ratio, ratio_min_chars=ratio_min_chars)
            if r["annotator_id"] in seen:
                found.insert(0, Violation("duplicate", "same annotator labeled this twice"))
            seen.add(r["annotator_id"])
            for v in found:
                violations.append(
                    {
                        "doc_id": key[0],
                        "para_id": key[1],
                        "annotator_id": r["annotator_id"],
                        "rule": v.rule,
                        "message": v.message,
                    }
                )
                report.violations[v.rule] += 1
            if not found:
                valid.append(r)
        if valid:
            # several valid annotations: keep the first annotator id (stable, documented)
            best = min(valid, key=lambda r: r["annotator_id"])
            accepted.append(finalize(best, len(group)))
        else:
            report.rejected_paragraphs += 1

    split_path = out_dir / "split.json"
    pinned: dict[str, str] = {}
    if split_path.exists():
        prev = json.loads(split_path.read_text(encoding="utf-8"))
        pinned = {d: "train" for d in prev["train_docs"]} | {d: "eval" for d in prev["eval_docs"]}
    side = split_docs([r["doc_id"] for r in accepted], seed, eval_frac, pinned)
    # docs from an earlier import that have no accepted rows now still keep their side
    for d, s in pinned.items():
        side.setdefault(d, s)

    accepted.sort(key=lambda r: (r["doc_id"], int(r.get("order") or 0), r["para_id"]))
    train = [r for r in accepted if side[r["doc_id"]] == "train"]
    evals = [r for r in accepted if side[r["doc_id"]] == "eval"]
    eval_text = "".join(_canonical(r) + "\n" for r in evals)
    _write(out_dir / "gold_train.jsonl", "".join(_canonical(r) + "\n" for r in train))
    _write(out_dir / "gold_eval.jsonl", eval_text)
    digest = hashlib.sha256(eval_text.encode("utf-8")).hexdigest()
    _write(out_dir / "gold_eval.jsonl.sha256", f"{digest}  gold_eval.jsonl\n")
    manifest = {
        "file": "gold_eval.jsonl",
        "sha256": digest,
        "n_records": len(evals),
        "n_docs": len({r["doc_id"] for r in evals}),
        "seed": seed,
        "records": [
            {
                "doc_id": r["doc_id"],
                "para_id": r["para_id"],
                "sha256": hashlib.sha256(_canonical(r).encode("utf-8")).hexdigest(),
            }
            for r in evals
        ],
    }
    _write(out_dir / "gold_eval.manifest.json", json.dumps(manifest, indent=1) + "\n")
    _write(
        split_path,
        json.dumps(
            {
                "seed": seed,
                "eval_frac": eval_frac,
                "train_docs": sorted(d for d, s in side.items() if s == "train"),
                "eval_docs": sorted(d for d, s in side.items() if s == "eval"),
            },
            indent=1,
        )
        + "\n",
    )
    _write(
        out_dir / "violations.jsonl",
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in violations),
    )

    report.n_accepted = len(accepted)
    report.n_train, report.n_eval = len(train), len(evals)
    report.train_docs = len({r["doc_id"] for r in train})
    report.eval_docs = len({r["doc_id"] for r in evals})
    _write(
        out_dir / "import_report.json",
        json.dumps(report.to_dict(), ensure_ascii=False, indent=1) + "\n",
    )
    return report


def run_import(cfg, annotated: Path, out_dir: Path, limit: int | None = None) -> ImportReport:
    v = cfg.gold.validate
    return import_annotations(
        annotated,
        out_dir,
        seed=int(cfg.gold.split.seed),
        eval_frac=float(cfg.gold.split.eval_frac),
        length_ratio=(float(v.length_ratio[0]), float(v.length_ratio[1])),
        ratio_min_chars=int(v.ratio_min_chars),
        limit=limit,
    )
