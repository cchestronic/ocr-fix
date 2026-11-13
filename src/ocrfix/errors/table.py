"""`ocrfix errors table`: OCR error statistics from gold chunks.

Each gold row (`gold chunks` output, ``gold_status: ok``) is compared in placeholder form:
``source_text`` with its ``misclassified-text`` spans expanded back to raw text, vs
``gold_text``. Kept placeholders (``⟨M1⟩``) match on both sides, so math and tables never
produce ops. Rates are ops per 1000 gold chars, with placeholders excluded from the count.

Outputs in ``out_dir``: ``error_table.json`` (groups per lang x stratum, per lang and
overall: counts, per-1000 rates, histogram of per-paragraph op rate; top char confusions),
``confusion_map.json`` (hangul_garbled: source glyph run -> gold syllables + jamo, counts),
``summary.md``.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice, pairwise
from pathlib import Path
from typing import Any

from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.errors.align import TYPES, EditOp, diff_ops
from ocrfix.io import iter_jsonl
from ocrfix.schema import nfc

try:
    with warnings.catch_warnings():  # jamo 0.4.1 has invalid escapes (SyntaxWarning on 3.12)
        warnings.simplefilter("ignore", SyntaxWarning)
        from jamo import h2j, j2hcj
except ImportError:  # the `text` extra is optional; same result for precomposed syllables
    h2j = j2hcj = None

MISCLASSIFIED = "misclassified-text"
ALL = "*"
# per-paragraph total ops per 1000 chars; bin i is [EDGES[i], EDGES[i+1])
HIST_EDGES = (0, 1, 2, 5, 10, 20, 50, 100, 200)
_TEX = re.compile(r"\\[A-Za-z]+|[${}\s]|\\[()\[\]]")

_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]


def to_jamo(text: str) -> str:
    """Hangul syllables -> compatibility jamo (결과 -> ㄱㅕㄹㄱㅘ); other chars unchanged."""
    if h2j is not None:
        return j2hcj(h2j(text))
    out = []
    for ch in text:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out += [_L[o // 588], _V[o % 588 // 28], _T[o % 28]]
        else:
            out.append(ch)
    return "".join(out)


def glyphs(raw: str) -> str:
    """The visible glyphs of a garbled run: TeX commands, $, braces and spaces removed."""
    return _TEX.sub("", raw)


def _labels(row: dict[str, Any]) -> dict[str, str]:
    labels = row.get("span_labels") or {}
    return labels if isinstance(labels, dict) else {}


def compare_forms(row: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """(source, gold, kept placeholders): misclassified spans expanded in the source."""
    src, gold = nfc(row["source_text"]), nfc(row["gold_text"])
    mapping = row.get("placeholders") or {}
    labels = _labels(row)
    kept = {}
    for ph, raw in mapping.items():
        if labels.get(ph.strip("⟨⟩")) == MISCLASSIFIED:
            src = src.replace(ph, nfc(raw), 1)
        else:
            kept[ph] = nfc(raw)
    return src, gold, kept


def n_chars(gold: str) -> int:
    return len(PLACEHOLDER.sub("", gold))


@dataclass
class Group:
    n_paragraphs: int = 0
    n_chars: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    para_rates: list[float] = field(default_factory=list)

    def add(self, chars: int, ops: list[EditOp]) -> None:
        self.n_paragraphs += 1
        self.n_chars += chars
        self.counts.update(op.key for op in ops)
        self.para_rates.append(1000 * len(ops) / chars if chars else 0.0)

    def to_dict(self) -> dict[str, Any]:
        per = 1000 / self.n_chars if self.n_chars else 0.0
        by_type = Counter[str]()
        for k, n in self.counts.items():
            by_type[k.split(".")[0]] += n
        hist = Counter[str]()
        for r in self.para_rates:
            hist[_bin(r)] += 1
        return {
            "n_paragraphs": self.n_paragraphs,
            "n_chars": self.n_chars,
            "n_ops": sum(self.counts.values()),
            "total_per_1000": round(sum(self.counts.values()) * per, 3),
            "counts": {t: by_type[t] for t in TYPES if by_type[t]},
            "per_1000": {t: round(by_type[t] * per, 3) for t in TYPES if by_type[t]},
            "spacing_per_1000": {
                k: round(n * per, 3) for k, n in sorted(self.counts.items()) if "." in k
            },
            "histogram": {b: hist[b] for b in _bins()},
        }


def _bins() -> list[str]:
    pairs = [f"[{a},{b})" for a, b in pairwise(HIST_EDGES)]
    return [*pairs, f"[{HIST_EDGES[-1]},inf)"]


def _bin(rate: float) -> str:
    for a, b in pairwise(HIST_EDGES):
        if rate < b:
            return f"[{a},{b})"
    return f"[{HIST_EDGES[-1]},inf)"


@dataclass
class Table:
    groups: dict[tuple[str, str], Group] = field(default_factory=lambda: defaultdict(Group))
    confusions: dict[str, Counter[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    garbled: Counter[tuple[str, str]] = field(default_factory=Counter)
    garbled_raw: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add(self, lang: str, stratum: str, chars: int, ops: list[EditOp]) -> None:
        for key in ((lang, stratum), (lang, ALL), (ALL, ALL)):
            self.groups[key].add(chars, ops)
        for op in ops:
            if op.type == "char_confusion":
                self.confusions[lang][(op.src, op.gold)] += 1
                self.confusions[ALL][(op.src, op.gold)] += 1
            elif op.type == "hangul_garbled":
                k = (glyphs(op.src) or op.src.strip(), op.gold.strip())
                self.garbled[k] += 1
                self.garbled_raw[k][op.src.strip()] += 1

    def to_dict(self, top_k: int) -> dict[str, Any]:
        groups = [
            {"lang": lang, "stratum": stratum, **g.to_dict()}
            for (lang, stratum), g in sorted(
                self.groups.items(), key=lambda kv: _group_order(kv[0])
            )
        ]
        confusions = {
            lang: [{"src": s, "gold": g, "count": n} for (s, g), n in _top(c, top_k)]
            for lang, c in sorted(self.confusions.items())
        }
        return {"groups": groups, "char_confusions": confusions}

    def confusion_map(self) -> list[dict[str, Any]]:
        return [
            {
                "source_glyphs": src,
                "gold": gold,
                "gold_jamo": to_jamo(gold),
                "count": n,
                "source_raw": dict(self.garbled_raw[(src, gold)].most_common(5)),
            }
            for (src, gold), n in _top(self.garbled, None)
        ]


def _top(c: Counter, k: int | None) -> list:
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _group_order(key: tuple[str, str]) -> tuple:
    lang, stratum = key
    return (lang == ALL, lang, stratum != ALL, stratum)


@dataclass
class TableResult:
    table: Table
    n_rows: int


def build(gold_rows: Iterable[dict[str, Any]], **opts: Any) -> TableResult:
    table = Table()
    n_rows = 0
    for row in gold_rows:
        n_rows += 1
        src, gold, _ = compare_forms(row)
        lang, stratum = row.get("lang") or "other", row.get("stratum") or ""
        table.add(lang, stratum, n_chars(gold), diff_ops(src, gold, **opts))
    return TableResult(table, n_rows)


def _md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return [*out, ""]


def summary_md(res: TableResult, top_k: int) -> str:
    t = res.table.to_dict(top_k)
    overall = [g for g in t["groups"] if g["stratum"] == ALL]
    langs = [g["lang"] for g in overall]

    lines = [
        "# OCR error table",
        "",
        f"{res.n_rows} gold chunks. Rates: ops per 1000 gold chars (placeholders excluded).",
        "",
    ]
    header = ["type", *langs]
    rows = []
    for k in ["total", *TYPES]:
        cells: list[Any] = [f"**{k}**" if k == "total" else k]
        cells += [_rate(g, k) for g in overall]
        if any(c is not None for c in cells[1:]):
            rows.append(cells)
    lines += ["## Rates per language", "", *_md_table(header, rows)]

    spacing = sorted({k for g in overall for k in g["spacing_per_1000"]})
    if spacing:
        rows = [[k, *(g["spacing_per_1000"].get(k) for g in overall)] for k in spacing]
        lines += ["## Spacing ops by script", "", *_md_table(["op", *langs], rows)]

    rows = [[b, *(g["histogram"][b] for g in overall)] for b in _bins()]
    lines += ["## Chunks by total op rate", "", *_md_table(["ops/1000", *langs], rows)]

    conf = t["char_confusions"].get(ALL, [])[:20]
    if conf:
        rows = [[f"`{c['src']}`", f"`{c['gold']}`", c["count"]] for c in conf]
        lines += ["## Top char confusions", "", *_md_table(["src", "gold", "count"], rows)]
    cmap = res.table.confusion_map()[:20]
    if cmap:
        rows = [[f"`{c['source_glyphs']}`", c["gold"], c["gold_jamo"], c["count"]] for c in cmap]
        lines += [
            "## Garbled Hangul (source glyphs -> gold)",
            "",
            *_md_table(["glyphs", "gold", "jamo", "count"], rows),
        ]
    return "\n".join(lines)


def _rate(g: dict[str, Any], k: str) -> float | None:
    if k == "total":
        return g.get("total_per_1000")
    return g.get("per_1000", {}).get(k)


def write_outputs(res: TableResult, out_dir: Path, top_k: int = 50) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {"n_rows": res.n_rows, **res.table.to_dict(top_k)}
    cmap: dict[str, Any] = {"hangul_garbled": res.table.confusion_map()}
    files = {
        out_dir / "error_table.json": json.dumps(table, ensure_ascii=False, indent=1) + "\n",
        out_dir / "confusion_map.json": json.dumps(cmap, ensure_ascii=False, indent=1) + "\n",
        out_dir / "summary.md": summary_md(res, top_k) + "\n",
    }
    for path, text in files.items():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return list(files)


def run_table(cfg, gold: Path, out_dir: Path, limit: int | None = None) -> TableResult:
    ec = cfg.gold.errors
    # `gold chunks` rows: only chunks whose gold could be cut cleanly (gold_status ok)
    rows = (r for r in iter_jsonl(gold) if r.get("gold_status", "ok") == "ok")
    res = build(
        islice(rows, limit),
        header_footer_min_chars=int(ec.header_footer_min_chars),
        confusion_max_len=int(ec.confusion_max_len),
    )
    write_outputs(res, out_dir, int(ec.top_k))
    return res
