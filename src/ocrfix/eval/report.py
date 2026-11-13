"""report.md for `ocrfix eval`: tables per system, by language, by error type, structure,
paired comparison, and the worst paragraphs per system."""

from __future__ import annotations

from typing import Any

from ocrfix.eval.metrics import DocResult
from ocrfix.eval.run import ALL, BASELINE

MAX_SHOW = 400  # chars of each text in the worst-paragraph listing


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def _num(x: float | None, nd: int = 3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _p(x: float | None) -> str:
    if x is None:
        return "–"
    return f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"


def _prf(c: dict[str, Any]) -> str:
    return f"{c['gold']} / {_pct(c['precision'])} / {_pct(c['recall'])}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    def cell(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return [*out, ""]


def _short(text: str) -> str:
    return text if len(text) <= MAX_SHOW else text[:MAX_SHOW] + " …"


def write_report(
    metrics: dict[str, Any], results: dict[str, list[DocResult]], worst_n: int = 20
) -> str:
    g = metrics["gold"]
    systems = metrics["systems"]
    md = [
        "# OCR correction eval",
        "",
        f"Gold: `{g['path']}` ({g['docs']} documents, {g['paragraphs']} paragraphs, "
        f"{g['clean_paragraphs']} clean). sha256 `{g['sha256']}`"
        + (" (verified against the .sha256 file)." if g["verified"] else "."),
        "",
        "Texts are compared per document: paragraphs in order, joined by a space, blank-line "
        "breaks flattened, so merges and splits don't count as edits (see Structure). "
        "CER/WER/SER are micro-averaged (total edits / total gold length). A document missing "
        "from a system's output is scored as its unchanged source.",
        "",
        "## Overall",
        "",
    ]
    rows = []
    for name, s in systems.items():
        o = s["overall"]
        rows.append([
            name, str(o["docs"]), str(o["missing_docs"]), _pct(o["cer"]), _pct(o["cer_doc_mean"]),
            _pct(o["wer"]), _pct(o["ser"]), _pct(o["false_fix_rate"]),
            _num(o["false_fixes_per_1000"], 2), _pct(o["identity"]["preservation"]),
            _pct(o["placeholders"]["integrity"]), _pct(o["placeholders"]["docs_exact"]),
            _pct(o["structure"]["accuracy"]),
        ])  # fmt: skip
    md += _table(
        [
            "system",
            "docs",
            "missing",
            "CER",
            "CER (doc mean)",
            "WER",
            "SER",
            "false-fix rate",
            "false fixes /1k chars",
            "identity kept",
            "placeholders",
            "placeholder docs exact",
            "structure acc.",
        ],
        rows,
    )
    md += [
        "- **false-fix rate**: system edits that touch no gold edit, over all system edits.",
        "- **identity kept**: clean gold paragraphs (gold = source) the system left untouched.",
        "- **placeholders**: gold math/table spans found verbatim and in order in the output.",
        "",
        "## By language",
        "",
        "Korean is scored by CER and spacing error rate (SER), English by CER and WER.",
        "",
    ]
    langs = sorted({lang for s in systems.values() for lang in s["by_lang"]})
    for lang in langs:
        cols = {"ko": ["CER", "SER"], "en": ["CER", "WER"]}.get(lang, ["CER", "WER", "SER"])
        rows = []
        for name, s in systems.items():
            b = s["by_lang"].get(lang)
            if b is None:
                continue
            vals = {"CER": b["cer"], "WER": b["wer"], "SER": b["ser"]}
            rows.append([
                name, str(b["docs"]), *[_pct(vals[c]) for c in cols],
                _pct(b["false_fix_rate"]), _pct(b["identity"]["preservation"]),
            ])  # fmt: skip
        md += [f"### {lang}", ""]
        md += _table(["system", "docs", *cols, "false-fix rate", "identity kept"], rows)

    md += [
        "## Recall by error type",
        "",
        "Gold edits (typed as in `ocrfix errors table`) the system got exactly right: fixed / "
        "total.",
        "",
    ]
    totals: dict[str, int] = {}
    for s in systems.values():
        for k, c in s["overall"]["recall"].items():
            if k != ALL:
                totals[k] = max(totals.get(k, 0), c["total"])
    keys = sorted(totals, key=lambda k: (-totals[k], k))
    rows = []
    for k in [*keys, ALL]:
        row = [k if k != ALL else "**all**"]
        for s in systems.values():
            c = s["overall"]["recall"].get(k)
            row.append("–" if not c else f"{c['fixed']}/{c['total']} ({_pct(c['recall'])})")
        rows.append(row)
    md += _table(["type", *systems], rows)

    md += [
        "## Structure",
        "",
        "Per paragraph, the gold label (none / merge-with-next / split-at) vs the system's "
        "record boundaries. A merge-with-next whose next paragraph isn't in the eval set is "
        "not scored. Split position: a boundary within the tolerance of the gold offset.",
        "",
    ]
    rows = []
    for name, s in systems.items():
        st = s["overall"]["structure"]
        rows.append([
            name, str(st["n"]), _pct(st["accuracy"]),
            _prf(st["merge"]),
            _prf(st["split"]),
            _pct(st["split_position_accuracy"]),
        ])  # fmt: skip
    md += _table(
        [
            "system",
            "paragraphs",
            "accuracy",
            "merge: gold / P / R",
            "split: gold / P / R",
            "split position",
        ],
        rows,
    )

    md += [
        "## Paired comparison",
        "",
        "Per-document CER, two-sided Wilcoxon signed-rank test (zero differences dropped; exact "
        "for small n). diff = CER(A) − CER(B), so a positive median means B is better. "
        "p (Holm) is adjusted over all pairs.",
        "",
    ]
    rows = []
    for c in metrics["comparison"]:
        w = c["wilcoxon"]
        rows.append([
            c["a"], c["b"], str(c["docs"]), _pct(c["mean_cer_a"]), _pct(c["mean_cer_b"]),
            _pct(c["median_diff"]), f"{c['b_better']}/{c['a_better']}/{c['ties']}",
            _num(w["w_plus"], 1), _p(w["p"]), _p(c["p_holm"]), w["method"],
        ])  # fmt: skip
    md += (
        _table(
            [
                "A",
                "B",
                "docs",
                "mean CER A",
                "mean CER B",
                "median diff",
                "B better / A better / tie",
                "W+",
                "p",
                "p (Holm)",
                "method",
            ],
            rows,
        )
        if rows
        else ["Only one system: nothing to compare.", ""]
    )

    shown = [n for n in results if n != BASELINE] or list(results)
    md += [f"## Worst {worst_n} paragraphs per system", ""]
    for name in shown:
        paras = [(r.doc_id, p) for r in results[name] for p in r.paras if p.edits]
        paras.sort(key=lambda x: (-x[1].cer, -x[1].edits, x[0], x[1].para_id))
        md += [f"### {name}", ""]
        if not paras:
            md += ["No paragraph with errors.", ""]
        for i, (doc_id, p) in enumerate(paras[:worst_n], 1):
            md += [
                f"**{i}. `{doc_id}` / `{p.para_id}`** ({p.lang}) CER {_pct(p.cer)} "
                f"({p.edits} edits / {p.gold_len} chars)",
                "",
                "````",
                f"source: {_short(p.source)}",
                f"output: {_short(p.pred)}",
                f"gold:   {_short(p.gold)}",
                "````",
                "",
            ]
    return "\n".join(md).rstrip() + "\n"
