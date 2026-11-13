"""`ocrfix eval`: score system outputs against the gold eval set, and compare systems.

Inputs: ``gold_eval.jsonl`` (from ``ocrfix gold import``) and one or more system outputs:
paragraph records (``correct``, ``verify`` or ``fix`` output) produced from the gold eval
sources (``ocrfix gold records gold_eval.jsonl``), so their doc_ids and para_ids match. Name
a system with ``name=path``, else it is named after the file. A document a system lacks is
scored as the unchanged source and counted in ``missing_docs``.

Only records of gold documents are used. Within those, records with an ingest-style
para_id (``p00012``) that isn't a gold paragraph are dropped (the system ran on more of the
document); other ids are kept, as they come from splits.

With ``eval.baseline`` the unchanged source is scored as system ``source``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from ocrfix.eval.metrics import STRUCT, DocResult, GoldDoc, GoldPara, evaluate_doc
from ocrfix.eval.stats import holm, wilcoxon
from ocrfix.io import iter_jsonl
from ocrfix.schema import nfc

BASELINE = "source"
ALL = "*"
_INGEST_ID = re.compile(r"^p\d{5}$")


# ---------------------------------------------------------------- inputs


def gold_digest(path: Path) -> dict[str, Any]:
    """sha256 of the gold file, checked against ``<gold>.sha256`` when it exists."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    side = path.with_name(path.name + ".sha256")
    verified = None
    if digest and side.is_file():
        want = side.read_text().split()[0]
        if want != digest:
            raise ValueError(f"{path}: sha256 {digest} does not match {side} ({want})")
        verified = True
    return {"path": str(path), "sha256": digest, "verified": verified}


def load_gold(path: Path, limit: int | None = None) -> tuple[list[GoldDoc], int]:
    """Gold documents (paragraphs sorted by order), at most ``limit`` docs; rows skipped."""
    docs: dict[str, list[GoldPara]] = defaultdict(list)
    skipped = 0
    for d in iter_jsonl(path):
        src, ref = d.get("source_raw"), d.get("gold_raw")
        if src is None or ref is None:
            skipped += 1
            continue
        doc_id = str(d["doc_id"])
        if doc_id not in docs and limit is not None and len(docs) >= limit:
            continue
        docs[doc_id].append(
            GoldPara(
                doc_id=doc_id,
                para_id=str(d["para_id"]),
                order=int(d.get("order") or 0),
                lang=d.get("lang") or "other",
                source=nfc(src),
                gold=nfc(ref),
                structure=(d.get("structure_label") or "").strip(),
            )
        )
    out = [
        GoldDoc(k, sorted(v, key=lambda p: (p.order, p.para_id))) for k, v in sorted(docs.items())
    ]
    return out, skipped


@dataclass
class SystemInput:
    name: str
    path: Path | None
    texts: dict[str, list[str]] = field(default_factory=dict)  # doc_id -> record texts in order
    stats: Counter[str] = field(default_factory=Counter)


def parse_system(arg: str) -> tuple[str, Path]:
    name, sep, rest = arg.partition("=")
    if sep and name and "/" not in name:
        return name, Path(rest)
    p = Path(arg)
    stem = p.name
    for suf in (".gz", ".jsonl", ".json"):
        stem = stem.removesuffix(suf)
    return stem, p


def load_system(name: str, path: Path, gold: list[GoldDoc]) -> SystemInput:
    ids = {d.doc_id: {p.para_id for p in d.paras} for d in gold}
    sysin = SystemInput(name, path)
    seen: set[tuple[str, str]] = set()
    for d in iter_jsonl(path):
        sysin.stats["records"] += 1
        doc_id, para_id = str(d["doc_id"]), str(d["para_id"])
        if doc_id not in ids:
            sysin.stats["other_docs"] += 1
            continue
        if (doc_id, para_id) in seen:
            sysin.stats["duplicates"] += 1
            continue
        seen.add((doc_id, para_id))
        if para_id not in ids[doc_id] and _INGEST_ID.match(para_id):
            sysin.stats["non_gold_paragraphs"] += 1
            continue
        sysin.texts.setdefault(doc_id, []).append(nfc(d.get("text") or ""))
        sysin.stats["used"] += 1
    sysin.stats["missing_docs"] = sum(d.doc_id not in sysin.texts for d in gold)
    return sysin


# ---------------------------------------------------------------- aggregation


def _rate(num: float, den: float, empty: float | None = None) -> float | None:
    return round(num / den, 6) if den else empty


def aggregate(results: Iterable[DocResult]) -> dict[str, Any]:
    rs = list(results)
    s = Counter()
    rec: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    struct = Counter()
    clean = altered = 0
    for r in rs:
        for k in (
            "gold_chars", "char_edits", "gold_words", "word_edits", "gold_spaces",
            "space_errors", "source_chars", "sys_edits", "false_fixes",
            "ph_expected", "ph_matched", "ph_extra",
        ):  # fmt: skip
            s[k] += getattr(r, k)
        s["missing"] += r.missing
        s["ph_exact"] += r.ph_exact
        for key, (tot, fixed) in r.recall.items():
            rec[key][0] += tot
            rec[key][1] += fixed
        for gc, pc, pos in r.structure:
            struct["n"] += 1
            struct["correct"] += gc == pc
            struct[f"gold:{gc}"] += 1
            struct[f"pred:{pc}"] += 1
            struct[f"tp:{gc}"] += gc == pc
            if pos is not None:
                struct["split_pos_n"] += 1
                struct["split_pos_ok"] += pos
        for p in r.paras:
            if p.clean:
                clean += 1
                altered += p.altered
    total = [sum(v[0] for v in rec.values()), sum(v[1] for v in rec.values())]
    return {
        "docs": len(rs),
        "missing_docs": s["missing"],
        "paragraphs": sum(len(r.paras) for r in rs),
        "cer": _rate(s["char_edits"], s["gold_chars"], 0.0),
        "cer_doc_mean": round(sum(r.cer for r in rs) / len(rs), 6) if rs else None,
        "wer": _rate(s["word_edits"], s["gold_words"], 0.0),
        "ser": _rate(s["space_errors"], s["gold_spaces"]),
        "char_edits": s["char_edits"],
        "gold_chars": s["gold_chars"],
        "sys_edits": s["sys_edits"],
        "false_fixes": s["false_fixes"],
        "false_fix_rate": _rate(s["false_fixes"], s["sys_edits"], 0.0),
        "false_fixes_per_1000": _rate(1000 * s["false_fixes"], s["source_chars"], 0.0),
        "recall": {
            k: {"total": t, "fixed": f, "recall": _rate(f, t)} for k, (t, f) in sorted(rec.items())
        }
        | {ALL: {"total": total[0], "fixed": total[1], "recall": _rate(total[1], total[0])}},
        "identity": {
            "clean_paragraphs": clean,
            "altered": altered,
            "altered_rate": _rate(altered, clean),
            "preservation": _rate(clean - altered, clean),
        },
        "placeholders": {
            "expected": s["ph_expected"],
            "matched": s["ph_matched"],
            "extra": s["ph_extra"],
            "integrity": _rate(s["ph_matched"], s["ph_expected"], 1.0),
            "docs_exact": _rate(s["ph_exact"], len(rs)),
        },
        "structure": {
            "n": struct["n"],
            "accuracy": _rate(struct["correct"], struct["n"]),
            "split_position_accuracy": _rate(struct["split_pos_ok"], struct["split_pos_n"]),
        }
        | {
            c: {
                "gold": struct[f"gold:{c}"],
                "pred": struct[f"pred:{c}"],
                "precision": _rate(struct[f"tp:{c}"], struct[f"pred:{c}"]),
                "recall": _rate(struct[f"tp:{c}"], struct[f"gold:{c}"]),
            }
            for c in STRUCT
        },
    }


def summarize(results: list[DocResult]) -> dict[str, Any]:
    by_lang: dict[str, list[DocResult]] = defaultdict(list)
    for r in results:
        by_lang[r.lang].append(r)
    return {
        "overall": aggregate(results),
        "by_lang": {k: aggregate(v) for k, v in sorted(by_lang.items())},
    }


def compare(results: dict[str, list[DocResult]], exact_max_n: int = 25) -> list[dict[str, Any]]:
    """Paired Wilcoxon on per-document CER for every pair of systems (Holm-adjusted)."""
    rows = []
    for a, b in combinations(results, 2):
        x = [r.cer for r in results[a]]
        y = [r.cer for r in results[b]]
        diffs = [u - v for u, v in zip(x, y, strict=True)]
        w = wilcoxon(x, y, exact_max_n)
        rows.append({
            "a": a, "b": b, "docs": len(diffs),
            "mean_cer_a": round(sum(x) / len(x), 6) if x else None,
            "mean_cer_b": round(sum(y) / len(y), 6) if y else None,
            "mean_diff": round(sum(diffs) / len(diffs), 6) if diffs else None,
            "median_diff": round(median(diffs), 6) if diffs else None,
            "b_better": sum(d > 1e-12 for d in diffs),
            "a_better": sum(d < -1e-12 for d in diffs),
            "ties": sum(abs(d) <= 1e-12 for d in diffs),
            "wilcoxon": w.to_dict(),
        })  # fmt: skip
    for row, p in zip(rows, holm([r["wilcoxon"]["p"] for r in rows]), strict=True):
        row["p_holm"] = round(p, 8)
    return rows


# ---------------------------------------------------------------- run


def evaluate(
    gold_path: Path,
    systems: list[tuple[str, Path]],
    out_dir: Path,
    *,
    opts: dict[str, Any],
    errors_opts: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    from ocrfix.eval.report import write_report

    names = [n for n, _ in systems]
    if opts.get("baseline", True):
        names = [BASELINE, *names]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate system names: {names} (use name=path)")
    digest = gold_digest(gold_path)
    gold, skipped = load_gold(gold_path, limit)
    if not gold:
        raise ValueError(f"{gold_path}: no gold rows with source_raw/gold_raw")

    inputs = [SystemInput(BASELINE, None)] if opts.get("baseline", True) else []
    inputs += [load_system(n, p, gold) for n, p in systems]
    tol = int(opts.get("split_tolerance", 3))
    results: dict[str, list[DocResult]] = {}
    for si in inputs:
        results[si.name] = [
            evaluate_doc(
                d,
                [p.source for p in d.paras] if si.path is None else si.texts.get(d.doc_id),
                errors_opts=errors_opts,
                split_tolerance=tol,
            )
            for d in gold
        ]

    metrics = {
        "gold": digest
        | {
            "docs": len(gold),
            "paragraphs": sum(len(d.paras) for d in gold),
            "clean_paragraphs": sum(p.clean for d in gold for p in d.paras),
            "skipped_rows": skipped,
        },
        "systems": {
            si.name: {"path": str(si.path) if si.path else None, "input": dict(si.stats)}
            | summarize(results[si.name])
            for si in inputs
        },
        "comparison": compare(results, int(opts.get("exact_max_n", 25))),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    _write(
        out_dir / "per_doc.jsonl",
        "".join(
            json.dumps({"system": n} | r.summary(), ensure_ascii=False) + "\n"
            for n, rs in results.items()
            for r in rs
        ),
    )
    _write(
        out_dir / "paragraphs.jsonl",
        "".join(
            json.dumps(
                {
                    "system": n,
                    "doc_id": r.doc_id,
                    "para_id": p.para_id,
                    "lang": p.lang,
                    "cer": round(p.cer, 6),
                    "edits": p.edits,
                    "gold_len": p.gold_len,
                    "clean": p.clean,
                    "altered": p.altered,
                    "pred": p.pred,
                },
                ensure_ascii=False,
            )
            + "\n"
            for n, rs in results.items()
            for r in rs
            for p in r.paras
        ),
    )
    _write(out_dir / "report.md", write_report(metrics, results, int(opts.get("worst_n", 20))))
    return metrics


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_eval(cfg, gold: Path, systems: list[str], out_dir: Path, limit: int | None = None):
    from omegaconf import OmegaConf

    opts = OmegaConf.to_container(cfg.eval, resolve=True)
    ec = cfg.errors
    errors_opts = {
        "header_footer_min_chars": int(ec.header_footer_min_chars),
        "confusion_max_len": int(ec.confusion_max_len),
    }
    return evaluate(
        gold, [parse_system(s) for s in systems], out_dir,
        opts=opts, errors_opts=errors_opts, limit=limit,
    )  # fmt: skip
