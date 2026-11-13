"""summary.md and matplotlib PNGs for `ocrfix calibrate`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ocrfix.calibrate.fit import roc_curve

if TYPE_CHECKING:
    from ocrfix.calibrate.rules import Calibration
    from ocrfix.calibrate.run import Report, Scores

ALL = "*"


def _md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return [*out, ""]


def summary_md(cal: Calibration, rep: Report, pngs: list[Path]) -> str:
    langs = list(rep.auroc)
    lines = [
        "# Calibration",
        "",
        f"tau = **{cal.tau:g}** ({cal.variant} features), chosen by pooled AUROC of spike_frac. "
        f"Target recall {cal.target_recall}, max false-flag rate {cal.max_false_flag}.",
        "",
        "## Rows",
        "",
    ]
    labels = sorted({k for c in rep.counts.values() for k in c})
    lines += _md_table(
        ["lang", *labels], [[lg, *(rep.counts[lg].get(k) for k in labels)] for lg in langs]
    )

    lines += ["## tau candidates (pooled AUROC of spike_frac)", ""]
    taus = list(next(iter(rep.tau_auroc.values())))
    lines += _md_table(
        ["variant", *taus], [[v, *(a.get(t) for t in taus)] for v, a in rep.tau_auroc.items()]
    )

    lines += ["## Flag rules", ""]
    rows = []
    for lg, r in cal.rules.items():
        met = "yes" if r.target_met else "**no**"
        rows.append(
            [lg, r.spike_frac, r.p90, r.bpc, r.recall, r.false_flag, met, r.n_clean, r.n_corrupted]
        )
    lines += _md_table(
        [
            "lang",
            "spike_frac >",
            "p90 >",
            "bpc >",
            "recall",
            "false flag",
            "target met",
            "clean",
            "corrupted",
        ],
        rows,
    )
    lines += ["Share flagged per corruption label (clean = false-flag rate):", ""]
    labels = sorted({k for v in rep.recall_by_label.values() for k in v})
    lines += _md_table(
        ["lang", *labels],
        [[lg, *(v.get(k) for k in labels)] for lg, v in rep.recall_by_label.items()],
    )

    if rep.gold:
        lines += ["## Real gold source paragraphs", ""]
        keys = ["n", "n_noisy", "n_flagged", "precision", "recall", "false_flag", "auroc_spike"]
        lines += _md_table(
            ["lang", *keys], [[lg, *(g[k] for k in keys)] for lg, g in rep.gold.items()]
        )

    if rep.noise_chunk:
        lines += ["## Whole-chunk noise (score.noise.whole_chunks)", "",
                  "A chunk with bpc above the threshold is dropped as noise: the highest bpc of "
                  "any clean or damaged chunk of that language, plus calibrate.noise_chunk.margin. "
                  "A first value from synthetic data: "
                  "`ocrfix calibrate-noise-chunk` sets it from the gold (run it after this).",
                  ""]  # fmt: skip
        keys = ["max_clean", "max_damaged", "threshold"]
        lines += _md_table(
            ["lang", *keys], [[lg, *(v[k] for k in keys)] for lg, v in rep.noise_chunk.items()]
        )

    feats = list(rep.auroc[ALL])
    lines += ["## AUROC clean vs corrupted", ""]
    lines += _md_table(
        ["feature", *langs], [[f, *(rep.auroc[lg].get(f) for lg in langs)] for f in feats]
    )
    lines += ["## Spearman with corruption rate", ""]
    lines += _md_table(
        ["feature", *langs], [[f, *(rep.spearman[lg].get(f) for lg in langs)] for f in feats]
    )
    if pngs:
        lines += ["## Plots", ""] + [f"![{p.stem}]({p.name})" for p in pngs]
    return "\n".join(lines)


def write_plots(
    cal: Calibration, rep: Report, data: Scores, out_dir: Path, *, base_tau: float
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ocrfix.calibrate.run import CLEAN, spike_column

    out: list[Path] = []
    langs = [lg for lg in rep.auroc if lg != ALL] or [ALL]
    s_col = spike_column(cal.tau, base_tau, cal.prefix)
    p_col = cal.prefix + "p90"
    is_clean = data.label == CLEAN

    # 1. AUROC per feature and language
    feats = list(rep.auroc[ALL])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(feats) * len(langs))))
    h = 0.8 / len(langs)
    for i, lg in enumerate(langs):
        vals = [rep.auroc[lg].get(f) or np.nan for f in feats]
        ax.barh(np.arange(len(feats)) + i * h, vals, height=h, label=lg)
    ax.set_yticks(np.arange(len(feats)) + 0.4 - h / 2, feats)
    ax.axvline(0.5, color="grey", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC clean vs corrupted")
    ax.legend()
    out.append(_save(fig, out_dir / "auroc.png"))

    # 2. ROC of the chosen spike feature, with each rule's operating point
    fig, ax = plt.subplots(figsize=(5, 5))
    for lg in langs:
        m = np.ones(len(data), bool) if lg == ALL else data.lang == lg
        bad = m & np.isin(data.label, rep.recall_labels)
        fpr, tpr = roc_curve(data.col(s_col)[m & is_clean], data.col(s_col)[bad])
        (line,) = ax.plot(fpr, tpr, label=lg)
        r = cal.rule(lg)
        ax.plot([r.false_flag], [r.recall], "o", color=line.get_color())
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.axvline(cal.max_false_flag or 0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("false-flag rate (clean)")
    ax.set_ylabel(f"recall ({', '.join(rep.recall_labels)})")
    ax.set_title(f"{s_col}")
    ax.legend()
    out.append(_save(fig, out_dir / "roc.png"))

    # 3. Feature means by corruption label (ordered by mean error rate)
    labels = [
        CLEAN,
        *sorted(
            set(data.label[~is_clean]),
            key=lambda lb: (
                np.nanmean(data.rate[data.label == lb]) if (data.label == lb).any() else 0
            ),
        ),
    ]
    show = [c for c in (s_col, p_col, cal.prefix + "bpc") if c in data.cols]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.5), squeeze=False)
    for ax, col in zip(axes[0], show, strict=True):
        for lg in langs:
            m = np.ones(len(data), bool) if lg == ALL else data.lang == lg
            ys = [
                np.nanmean(data.col(col)[m & (data.label == lb)])
                if (m & (data.label == lb)).any()
                else np.nan
                for lb in labels
            ]
            ax.plot(range(len(labels)), ys, marker="o", label=lg)
        ax.set_xticks(range(len(labels)), labels, rotation=30)
        ax.set_title(col)
    axes[0][0].legend()
    out.append(_save(fig, out_dir / "by_corruption.png"))

    # 4. spike_frac vs p90 with the rule, per language
    fig, axes = plt.subplots(1, len(langs), figsize=(4.5 * len(langs), 4), squeeze=False)
    for ax, lg in zip(axes[0], langs, strict=True):
        m = np.ones(len(data), bool) if lg == ALL else data.lang == lg
        for sel, name, color in (
            (m & ~is_clean, "corrupted", "tab:red"),
            (m & is_clean, "clean", "tab:blue"),
        ):
            ax.scatter(
                data.col(p_col)[sel], data.col(s_col)[sel], s=4, alpha=0.4, label=name, color=color
            )
        r = cal.rule(lg)
        if r.spike_frac is not None:
            ax.axhline(r.spike_frac, color="k", lw=1)
        if r.p90 is not None:
            ax.axvline(r.p90, color="k", lw=1, ls="--")
        ax.set_xlabel(p_col)
        ax.set_ylabel(s_col)
        ax.set_title(lg)
    axes[0][0].legend()
    out.append(_save(fig, out_dir / "rule.png"))
    return out


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
