"""`ocrfix refine eval <truth_dir> <system>... <out_dir>`: corrected documents vs hand truth.

Each system is ``name=dir`` (or a dir, named after it): `refine` output (or plain chunks) of
the gold eval documents (`gold chunks` eval/). The documents complete in every system and
with a truth file are compared whole (``reconstruct --source current``), so chunk cuts never
matter. The first system's original text is also scored, as system ``ocr``. Both sides are
normalized as the chunker reads them (no BOM, LF, NFC).

Micro-averaged over documents, overall and per document language (the language with the
most chunk characters):
  cer, wer, ser      char / word edits to the truth over its length; spacing disagreements
                     at aligned non-space chars over the truth's spaces
  false_fix          system edits (ocr -> output regions) touching no truth edit (ocr ->
                     truth), over all system edits; also per 1000 OCR chars
  recall[type]       truth edits (`errors.align.diff_ops`, the `errors table` types) that the
                     output gets right: no remaining output -> truth edit touches them
  uncertain          each 〈?〉 the truth adds (a lost word): the output has 〈?〉 there
                     (marked), still the OCR text (left), or anything else (invented); and
                     false_uncertain: 〈?〉 in the output where the truth has none
Paired Wilcoxon signed-rank on per-document CER for each pair of systems, Holm-adjusted.
Writes metrics.json, per_doc.jsonl, report.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, groupby
from pathlib import Path
from statistics import median
from typing import Any

from rapidfuzz.distance import Levenshtein

from ocrfix.chunks.rebuild import rebuild_doc
from ocrfix.chunks.split import normalize
from ocrfix.errors.align import diff_ops
from ocrfix.eval.align import touches
from ocrfix.eval.metrics import spacing_errors, word_edits
from ocrfix.eval.stats import holm, wilcoxon
from ocrfix.io import list_shards, open_text
from ocrfix.schema import UNCERTAIN

_Q = re.compile(re.escape(UNCERTAIN))
OUTCOMES = ("marked", "left", "invented")


@dataclass
class Doc:
    path: str
    lang: str
    ocr: str
    truth: str
    truth_ops: list  # diff_ops(ocr, truth)


@dataclass
class Counts:
    docs: int = 0
    truth_chars: int = 0
    ocr_chars: int = 0
    char_edits: int = 0
    truth_words: int = 0
    word_edits: int = 0
    truth_spaces: int = 0
    space_errors: int = 0
    sys_edits: int = 0
    false_fixes: int = 0
    ops: Counter = field(default_factory=Counter)
    fixed: Counter = field(default_factory=Counter)
    uncertain: Counter = field(default_factory=Counter)
    false_uncertain: int = 0

    def add(self, o: Counts) -> None:
        for k, v in o.__dict__.items():
            if isinstance(v, Counter):
                getattr(self, k).update(v)
            else:
                setattr(self, k, getattr(self, k) + v)

    def summary(self) -> dict[str, Any]:
        def r(a, b):
            return round(a / b, 6) if b else None

        n_q = sum(self.uncertain.values())
        return {
            "docs": self.docs,
            "truth_chars": self.truth_chars,
            "cer": r(self.char_edits, self.truth_chars),
            "wer": r(self.word_edits, self.truth_words),
            "ser": r(self.space_errors, self.truth_spaces),
            "false_fix_rate": r(self.false_fixes, self.sys_edits),
            "false_fix_per_1000": r(1000 * self.false_fixes, self.ocr_chars),
            "edits": self.sys_edits,
            "recall": {k: r(self.fixed[k], n) for k, n in sorted(self.ops.items())},
            "truth_ops": dict(sorted(self.ops.items())),
            "uncertain": {k: r(self.uncertain[k], n_q) for k in OUTCOMES} | {"n": n_q},
            "false_uncertain": self.false_uncertain,
        }


def _regions(a: str, b: str, side: str) -> list[tuple[int, int]]:
    ops = Levenshtein.opcodes(a, b)
    if side == "src":
        return [(o.src_start, o.src_end) for o in ops if o.tag != "equal"]
    return [(o.dest_start, o.dest_end) for o in ops if o.tag != "equal"]


def _map(ops, a: int, b: int) -> tuple[int, int]:
    """[a, b) in the opcodes' source -> the covering range of their destination."""
    lo = hi = None
    for o in ops:
        if o.src_end < a or o.src_start > b:
            continue
        if o.tag == "equal":
            s = o.dest_start + max(a - o.src_start, 0)
            e = o.dest_start + min(b, o.src_end) - o.src_start
        else:
            s, e = o.dest_start, o.dest_end
        lo = s if lo is None else min(lo, s)
        hi = e if hi is None else max(hi, e)
    return (lo or 0, hi or 0)


def score_doc(d: Doc, out: str, opts: dict[str, Any]) -> Counts:
    c = Counts(docs=1, truth_chars=len(d.truth), ocr_chars=len(d.ocr))
    c.char_edits = Levenshtein.distance(out, d.truth)
    c.truth_words = len(d.truth.split())
    c.word_edits = word_edits(out, d.truth)
    c.space_errors, c.truth_spaces = spacing_errors(out, d.truth)
    gold_src = [(op.src_start, op.src_end) for op in d.truth_ops]
    edits = _regions(d.ocr, out, "src")
    c.sys_edits = len(edits)
    c.false_fixes = sum(not any(touches(a, b, s, e) for s, e in gold_src) for a, b in edits)
    out_truth = Levenshtein.opcodes(out, d.truth)
    residual = [(o.dest_start, o.dest_end) for o in out_truth if o.tag != "equal"]
    truth_out = Levenshtein.opcodes(d.truth, out)
    for op in d.truth_ops:
        c.ops[op.key] += 1
        ok = not any(touches(a, b, op.gold_start, op.gold_end) for a, b in residual)
        c.fixed[op.key] += ok
        if UNCERTAIN in op.gold and UNCERTAIN not in op.src:
            lo, hi = _map(truth_out, op.gold_start, op.gold_end)
            seg = out[lo:hi]
            if UNCERTAIN in seg:
                k = "marked"
            elif seg.strip() and Levenshtein.normalized_distance(seg.strip(), op.src.strip()) < 0.5:
                k = "left"
            else:
                k = "invented"
            c.uncertain[k] += 1
    # 〈?〉 in the output that the alignment doesn't put on a truth 〈?〉
    equal = [(o.src_start, o.src_end) for o in out_truth if o.tag == "equal"]
    for m in _Q.finditer(out):
        c.false_uncertain += not any(a <= m.start() and m.end() <= b for a, b in equal)
    return c


def _read_dir(d: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for shard in list_shards(d):
        with open_text(shard) as f:
            rows = (json.loads(x) for x in f if x.strip())
            for path, cs in groupby(rows, key=lambda c: c["source_path"]):
                cs = sorted(cs, key=lambda c: c["chunk_index"])
                if len(cs) == cs[0]["n_chunks"]:
                    out[path] = cs
    return out


def parse_system(arg: str) -> tuple[str, Path]:
    if "=" in arg and not Path(arg).exists():
        name, path = arg.split("=", 1)
        return name, Path(path)
    return Path(arg).name, Path(arg)


def _lang(chunks: list[dict[str, Any]]) -> str:
    w: Counter = Counter()
    for c in chunks:
        w[c.get("lang") or "other"] += c["char_end"] - c["char_start"]
    return w.most_common(1)[0][0] if w else "other"


def run_refine_eval(
    truth_dir: Path, systems: list[str], out_dir: Path, *, opts: dict[str, Any],
    exact_max_n: int = 25,
) -> dict[str, Any]:  # fmt: skip
    """``opts``: `errors.align.diff_ops` options (errors.header_footer_min_chars, ...)."""
    named = [parse_system(s) for s in systems]
    names = [n for n, _ in named]
    if len(set(names)) != len(names) or "ocr" in names:
        raise ValueError(f"system names must be unique and not 'ocr': {names}")
    data = {n: _read_dir(p) for n, p in named}
    paths = sorted(set.intersection(*(set(v) for v in data.values())))
    missing = [p for p in paths if not (truth_dir / p).is_file()]
    paths = [p for p in paths if p not in missing]
    if not paths:
        raise ValueError("no document is complete in every system and has a truth file")
    first = data[names[0]]
    per_doc: list[dict[str, Any]] = []
    totals: dict[str, dict[str, Counts]] = defaultdict(lambda: defaultdict(Counts))
    cers: dict[str, list[float]] = defaultdict(list)
    for p in paths:
        ocr = normalize(rebuild_doc(first[p], "original").decode("utf-8"))
        truth = normalize((truth_dir / p).read_bytes().decode("utf-8"))
        d = Doc(p, _lang(first[p]), ocr, truth, diff_ops(ocr, truth, **opts))
        row: dict[str, Any] = {"path": p, "lang": d.lang, "truth_chars": len(truth)}
        for name, chunks in [("ocr", first[p]), *((n, data[n][p]) for n in names)]:
            src = "original" if name == "ocr" else "current"
            out = normalize(rebuild_doc(chunks, src).decode("utf-8"))
            c = score_doc(d, out, opts)
            for g in ("all", d.lang):
                totals[name][g].add(c)
            cer = c.char_edits / max(len(truth), 1)
            cers[name].append(cer)
            row[name] = {"cer": round(cer, 6), "edits": c.sys_edits,
                         "false_fixes": c.false_fixes,
                         "uncertain": dict(c.uncertain)}  # fmt: skip
        per_doc.append(row)

    systems_out = {
        name: {g: c.summary() for g, c in sorted(groups.items())} for name, groups in totals.items()
    }
    comps = []
    for a, b in combinations(["ocr", *names], 2):
        x, y = cers[a], cers[b]
        diffs = [u - v for u, v in zip(x, y, strict=True)]
        comps.append({
            "a": a, "b": b, "docs": len(diffs),
            "median_diff": round(median(diffs), 6),
            "b_better": sum(t > 1e-12 for t in diffs), "a_better": sum(t < -1e-12 for t in diffs),
            "wilcoxon": wilcoxon(x, y, exact_max_n).to_dict(),
        })  # fmt: skip
    for row, p in zip(comps, holm([r["wilcoxon"]["p"] for r in comps]), strict=True):
        row["p_holm"] = round(p, 8)
    res = {"truth_dir": str(truth_dir), "systems": dict(named) | {"ocr": "original"},
           "docs": len(paths), "missing_truth": missing, "metrics": systems_out,
           "comparisons": comps}  # fmt: skip
    res["systems"] = {k: str(v) for k, v in res["systems"].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    with (out_dir / "per_doc.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in per_doc)
    (out_dir / "report.md").write_text(format_report(res), "utf-8")
    return res


def _pct(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def format_report(res: dict[str, Any]) -> str:
    m = res["metrics"]
    names = list(m)
    groups = sorted({g for v in m.values() for g in v}, key=lambda g: (g != "all", g))
    lines = [f"# Eval: {res['docs']} documents vs {res['truth_dir']}", ""]
    for g in groups:
        lines += [f"## {g}", "", "| system | docs | CER | WER | SER | false fixes | "
                  "〈?〉 marked | left | invented | false 〈?〉 |", "|---" * 10 + "|"]  # fmt: skip
        for n in names:
            s = m[n].get(g)
            if not s:
                continue
            u = s["uncertain"]
            lines.append(
                f"| {n} | {s['docs']} | {_pct(s['cer'])} | {_pct(s['wer'])} | {_pct(s['ser'])} | "
                f"{_pct(s['false_fix_rate'], 1)} | {_pct(u['marked'], 1)} | {_pct(u['left'], 1)} "
                f"| {_pct(u['invented'], 1)} | {s['false_uncertain']} |"
            )
        lines.append("")
    types = sorted({k for v in m.values() for k in v["all"]["truth_ops"]})
    lines += ["## Recall by error type (all documents)", "",
              "| type | truth edits | " + " | ".join(names) + " |",
              "|---" * (len(names) + 2) + "|"]  # fmt: skip
    for t in types:
        n = m[names[0]]["all"]["truth_ops"].get(t, 0)
        lines.append(f"| {t} | {n} | " + " | ".join(
            _pct(m[s]["all"]["recall"].get(t), 1) for s in names) + " |")  # fmt: skip
    lines += ["", "## Paired comparison (per-document CER, Wilcoxon, Holm)", "",
              "| a | b | docs | b better | a better | median diff | p (Holm) |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in res["comparisons"]:
        lines.append(f"| {c['a']} | {c['b']} | {c['docs']} | {c['b_better']} | {c['a_better']} "
                     f"| {_pct(c['median_diff'], 3)} | {c['p_holm']:.4g} |")  # fmt: skip
    if res["missing_truth"]:
        lines += ["", f"No truth file for {len(res['missing_truth'])} documents (skipped)."]
    return "\n".join(lines) + "\n"
