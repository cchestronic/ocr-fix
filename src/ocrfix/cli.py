"""`ocrfix` command line: chunk -> score -> correct -> reconstruct, plus gold, calibration,
training data, training and eval. docs/real_data.md runs them in order."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf

from ocrfix.config import load_config
from ocrfix.offline import enforce_offline

enforce_offline()

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

LimitOpt = Annotated[int | None, typer.Option(help="Process at most N records.")]
ProfileOpt = Annotated[
    str | None, typer.Option(help="Hardware profile (configs/profiles/<name>.yaml).")
]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Config override key=value (repeatable).")
]


def _load(profile: str | None, overrides: list[str] | None):
    try:
        return load_config(profile, overrides)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from None


@app.command("chunk")
def chunk_cmd(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Markdown tree.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output folder of shards.")],
    limit: Annotated[int | None, typer.Option(help="Chunk at most N documents.")] = None,
    sample_frac: Annotated[
        float | None, typer.Option(help="Keep this share of documents, by path hash.")
    ] = None,
    workers: Annotated[
        int | None, typer.Option(help="Processes [chunk.workers, else all cores]; 0 = in-process.")
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Markdown -> masked chunks of chunk.min_tokens..max_tokens tokens, in parallel."""
    from ocrfix.chunks.run import run_chunk

    cfg = _load(profile, overrides)
    try:
        st = run_chunk(
            cfg, input_dir, out_dir, limit=limit, sample_frac=sample_frac, workers=workers
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    mb = st.bytes / 1e6 / st.seconds if st.seconds else 0
    typer.echo(
        f"{out_dir}: {st.docs} docs -> {st.chunks} chunks, {st.tokens} tokens "
        f"(max {st.max_tokens} per chunk); "
        f"{st.short} short ({st.mid_short} not at a document end), {st.with_raw} with raw; "
        f"{st.parts - st.parts_skipped}/{st.parts} parts run in {st.seconds:.0f}s ({mb:.1f} MB/s)"
    )
    for e in st.errors[:20]:
        typer.echo(f"  skipped {e['path']}: {e['error']}", err=True)


@app.command("reconstruct")
def reconstruct_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="Chunk shards folder or file.")],
    out_dir: Annotated[
        Path | None, typer.Argument(file_okay=False, help="Write .md files here; none = check.")
    ] = None,
    source: Annotated[
        str, typer.Option(help="original (byte-exact, checked) or current (corrected text).")
    ] = "original",
    limit: Annotated[int | None, typer.Option(help="At most N documents.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Chunks -> markdown files, placeholders restored. With --source current, documents noise
    removal left nearly empty (reconstruct.min_letters / min_kept_share) are not written but
    listed in <out_dir>/dropped_docs.jsonl."""
    from ocrfix.chunks.rebuild import run_rebuild

    cfg = _load(profile, overrides)
    r = cfg.reconstruct
    try:
        st = run_rebuild(chunks, out_dir, source=source, limit=limit,
                         min_letters=int(r.min_letters),
                         min_kept_share=float(r.min_kept_share))  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"{st.docs} docs, {st.bytes} bytes; {len(st.mismatched)} differ from the source")
    if source == "current":
        where = f" (listed in {out_dir / 'dropped_docs.jsonl'})" if out_dir else ""
        typer.echo(f"{st.dropped} nearly empty docs dropped{where}")
    for x in [*st.mismatched, *st.errors][:20]:
        typer.echo(f"  {x}", err=True)
    if st.mismatched or st.errors:
        raise typer.Exit(1)


data_app = typer.Typer(no_args_is_help=True, help="Training data: SFT examples, DPO pairs.")
app.add_typer(data_app, name="data")
ShardLimit = Annotated[int | None, typer.Option(help="Process only the first N shards.")]


@app.command("score")
def score_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`ocrfix chunk` output.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Scored chunk shards.")],
    limit: ShardLimit = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Score every chunk (score.backend: vllm or transformers); write the scores into it."""
    from ocrfix.stages.run import run_score

    cfg = _load(profile, overrides)
    try:
        st = run_score(cfg, chunks, out_dir, limit=limit)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out_dir}: {st.chunks} chunks in {st.shards - st.shards_skipped} shards "
        f"({st.shards_skipped} already done), {st.seconds:.0f}s, "
        f"{st.to_dict()['tokens_per_s']} tokens/s; {dict(st.status)}"
    )


@app.command("correct")
def correct_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`score` output.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Refined chunk shards.")],
    select_all: Annotated[
        bool, typer.Option("--all", help="Send every scored chunk, not only flagged ones.")
    ] = False,
    limit: ShardLimit = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Per flagged chunk: correct, re-score at once, verify; accepted text replaces it."""
    from ocrfix.stages.run import format_summary, run_correct, summarize

    cfg = _load(profile, overrides)
    try:
        st = run_correct(cfg, chunks, out_dir, select_all=select_all, limit=limit)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"{out_dir}: {st.shards - st.shards_skipped} shards in {st.seconds:.0f}s")
    if st.status.get("skipped_lang"):
        typer.echo(f"  not sent (language not in correct.langs): {st.status['skipped_lang']}")
    typer.echo(format_summary(summarize(out_dir)))


@app.command("corrupt")
def corrupt_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="Clean chunks (`ocrfix chunk`).")],
    error_table: Annotated[Path, typer.Argument(exists=True, help="error_table.json.")],
    confusion_map: Annotated[Path, typer.Argument(exists=True, help="confusion_map.json.")],
    out: Annotated[Path, typer.Argument(help="Corrupted chunks .jsonl.")],
    rate: Annotated[
        float | None, typer.Option(help="Fixed ops per prose char (0.01 = 1%) [table rates].")
    ] = None,
    n_variants: Annotated[
        int | None, typer.Option(help="Damaged copies per chunk [corrupt.variants].")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Corrupt only the first N chunks.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Synthetic OCR damage on clean chunks at the gold set's error rates (for calibration)."""
    from ocrfix.stages.corrupt import run_corrupt

    cfg = _load(profile, overrides)
    n = run_corrupt(cfg, chunks, error_table, confusion_map, out, rate=rate,
                           n_variants=n_variants, limit=limit)  # fmt: skip
    typer.echo(f"{out}: {n} corrupted chunks")


@data_app.command("sft")
def data_sft_cmd(
    out_dir: Annotated[
        Path, typer.Argument(file_okay=False, help="train.jsonl, val.jsonl, stats.json.")
    ],
    synthetic: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`corrupt` output (repeatable): damaged -> clean."),
    ] = None,
    identity: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="Clean chunks (repeatable): returned unchanged."),
    ] = None,
    gold: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`gold chunks` sft/ (repeatable): OCR -> hand-corrected."),
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Chunk SFT examples in the exact form `correct` sends (for `train sft`)."""
    from omegaconf import OmegaConf

    from ocrfix.stages.data import build_sft_data

    cfg = _load(profile, overrides)
    if not synthetic and not identity and not gold:
        raise typer.BadParameter("give --synthetic, --gold and/or --identity")
    opts = OmegaConf.to_container(cfg.sft, resolve=True)
    try:
        st = build_sft_data(
            {"synthetic": synthetic or [], "gold": gold or [], "identity": identity or []},
            out_dir, opts=opts, seed=int(cfg.seed),
        )  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out_dir}: train {st.train}, val {st.val} "
        f"({st.with_junk} with junk, {st.with_lost} with lost words)"
    )
    for s, langs in st.taken.items():
        typer.echo(f"  {s:<9} taken {langs} of {st.pool[s]}; dropped {dict(st.dropped[s])}")


@data_app.command("dpo")
def data_dpo_cmd(
    out: Annotated[Path, typer.Argument(help="Pairs .jsonl (+ .responses.jsonl, .stats.json).")],
    synthetic: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`corrupt` output (repeatable): damaged -> clean."),
    ] = None,
    identity: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="Clean chunks (repeatable): returned unchanged."),
    ] = None,
    gold: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`gold chunks` dpo/ (repeatable): OCR -> hand-corrected."),
    ] = None,
    exclude: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="SFT data (repeatable): skip prompts it already holds."),
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Chunk DPO pairs: chosen = the clean chunk, rejected = the served model's own mistakes."""
    from omegaconf import OmegaConf

    from ocrfix.stages.dpo import build_dpo_pairs
    from ocrfix.stages.score import Encoder

    cfg = _load(profile, overrides)
    if not synthetic and not identity and not gold:
        raise typer.BadParameter("give --synthetic, --gold and/or --identity")
    from ocrfix.config import served_name

    c = lambda node: OmegaConf.to_container(node, resolve=True)  # noqa: E731
    opts = c(cfg.sft) | {
        "empty_think": False, "dpo": c(cfg.dpo), "llm": c(cfg.correct) | {"seed": int(cfg.seed)},
        "model": served_name(cfg, "correct"), "urls": c(cfg.hw.infer.correct_urls),
        "concurrency": int(cfg.hw.infer.concurrency),
    }  # fmt: skip
    transport = None
    if str(cfg.correct.backend) == "transformers":
        from ocrfix.stages.run import generator_transport

        transport = generator_transport(cfg)
    try:
        st = build_dpo_pairs(
            {"synthetic": synthetic or [], "gold": gold or [], "identity": identity or []},
            out, opts=opts, seed=int(cfg.seed), exclude=exclude or [],
            encode=Encoder(str(cfg.model)), transport=transport,
        )  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out}: {st.pairs} pairs from {sum(st.prompts.values())} prompts {st.prompts} "
        f"({st.excluded} excluded); greedy right {st.greedy_right}, all answers right "
        f"{st.all_right}; requests {st.requests}"
    )
    for k, v in st.types.most_common():
        typer.echo(f"  {k:<28} {v}")


@app.command("eval")
def eval_cmd(
    truth_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="The gold truth/ folder.")
    ],
    paths: Annotated[
        list[str],
        typer.Argument(
            help="System outputs (name=dir or dir: `correct` output of the gold eval chunks), "
            "then the output dir."
        ),
    ],
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Whole corrected documents vs the hand truth: CER/WER/SER, false fixes, recall by error
    type, how each lost word (〈?〉) was answered, paired Wilcoxon; report.md."""
    from ocrfix.stages.eval import parse_system, run_eval

    if len(paths) < 2:
        raise typer.BadParameter("need at least one system output and the output dir")
    *systems, out_dir = paths
    for s in systems:
        if not parse_system(s)[1].exists():
            raise typer.BadParameter(f"system output not found: {s}")
    cfg = _load(profile, overrides)
    ec = cfg.gold.errors
    opts = {"header_footer_min_chars": int(ec.header_footer_min_chars),
            "confusion_max_len": int(ec.confusion_max_len)}  # fmt: skip
    try:
        res = run_eval(
            truth_dir, systems, Path(out_dir), opts=opts, exact_max_n=int(cfg.eval.exact_max_n)
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    for name, groups in res["metrics"].items():
        o = groups["all"]
        u = o["uncertain"]
        q = f"; 〈?〉 {u['n']}: marked {u['marked']}, invented {u['invented']}" if u["n"] else ""
        typer.echo(f"{name:<12} CER {o['cer']:.4%}  false-fix {o['false_fix_rate']}{q}")
    typer.echo(f"{out_dir}/report.md ({res['docs']} documents)")


@app.command("compare")
def compare_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="Chunks (`score` output).")],
    out: Annotated[Path, typer.Argument(dir_okay=False, help="Report .json.")],
    limit: Annotated[int, typer.Option(help="Compare the first N chunks with prose.")] = 200,
    score: Annotated[
        bool, typer.Option(help="Compare scores (hw.infer.score_urls vs the base in this process).")
    ] = True,
    generate: Annotated[
        bool, typer.Option(help="Compare greedy corrections (correct_urls vs this process).")
    ] = True,
    examples: Annotated[int, typer.Option(help="Differing answers to show.")] = 5,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """vLLM vs transformers on the same chunks: per-token bits, features, flags, noise spans,
    and greedy answers. Run before using one backend's results for the other."""
    from ocrfix.stages.compare import format_compare, run_compare

    cfg = _load(profile, overrides)
    try:
        rep = run_compare(
            cfg, chunks, out, limit=limit, score=score, generate=generate, examples=examples
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(format_compare(rep))
    typer.echo(str(out))


@app.command("summary")
def summary_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`score` / `correct` output.")],
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Status counts, accept rate, failed checks, per language."""
    from ocrfix.stages.run import format_summary, summarize

    _load(profile, overrides)
    typer.echo(format_summary(summarize(chunks)))


@app.command("report")
def report_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`correct` (or score) output.")],
    clean: Annotated[
        Path | None,
        typer.Option(exists=True, help="`score` output of clean text, as the baseline."),
    ] = None,
    calibration: Annotated[
        Path | None, typer.Option(exists=True, help="calibration.yaml [score.calibration].")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the report here.")
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Corpus quality per language before / after noise removal and correction: bpc, flag rate."""
    from ocrfix.calibrate.rules import load_calibration
    from ocrfix.stages.report import corpus_report, format_report

    cfg = _load(profile, overrides)
    path = calibration or cfg.score.calibration
    cal = load_calibration(path) if path and Path(path).is_file() else None
    rep = corpus_report(chunks, cal, clean)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    typer.echo(format_report(rep))


def _maybe_torchrun(cfg) -> None:
    """Re-exec this command under torchrun with hw.n_gpus processes (once). Also sets
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fewer out-of-memory errors from
    fragmentation; harmless elsewhere) unless it is set already."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    n_gpus = int(cfg.hw.n_gpus)
    if n_gpus > 1 and "WORLD_SIZE" not in os.environ:
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={n_gpus}",
            "-m",
            "ocrfix.cli",
            *sys.argv[1:],
        ]
        os.execv(sys.executable, argv)


gold_app = typer.Typer(no_args_is_help=True, help="Gold: hand-corrected .md pairs -> chunk gold.")
app.add_typer(gold_app, name="gold")


@gold_app.command("chunks")
def gold_chunks_cmd(
    gold_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Folder with ocr/ and truth/.")
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output folder.")],
    workers: Annotated[int | None, typer.Option(help="Chunking processes.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Hand-corrected .md pairs -> chunk gold, split by document into sft / dpo / eval."""
    from ocrfix.gold.chunks import build_gold_chunks

    cfg = _load(profile, overrides)
    try:
        st = build_gold_chunks(cfg, gold_dir, out_dir, workers=workers)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"{out_dir}: documents {st['docs']}")
    for side, counts in st["chunks"].items():
        typer.echo(
            f"  {side:<4} chunks {counts}; edited {st['edited_chunks'].get(side, 0)}, "
            f"〈?〉 {st['uncertain'].get(side, 0)}"
        )
    if st["no_truth"]:
        typer.echo(f"  no truth file for {len(st['no_truth'])} documents, e.g. "
                   f"{st['no_truth'][:3]}", err=True)  # fmt: skip
    for e in st["chunk_errors"][:20]:
        typer.echo(f"  skipped {e['path']}: {e['error']}", err=True)


errors_app = typer.Typer(no_args_is_help=True, help="OCR error analysis on gold pairs.")
app.add_typer(errors_app, name="errors")


@errors_app.command("table")
def errors_table_cmd(
    gold: Annotated[Path, typer.Argument(exists=True, help="`gold chunks` sft/.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output directory.")],
    limit: Annotated[int | None, typer.Option(help="Read only the first N gold rows.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Error-type rates per language and stratum, char confusions, garbled-Hangul map."""
    from ocrfix.errors import run_table

    cfg = _load(profile, overrides)
    res = run_table(cfg, gold, out_dir, limit=limit)
    overall = res.table.groups.get(("*", "*"))
    n_ops = sum(overall.counts.values()) if overall else 0
    typer.echo(f"{out_dir}: {res.n_rows} chunks, {n_ops} ops")


@app.command("calibrate")
def calibrate_cmd(
    clean_scores: Annotated[
        Path, typer.Argument(exists=True, help="`score` output of clean chunks.")
    ],
    corrupted_scores: Annotated[
        Path, typer.Argument(exists=True, help="`score` output of damaged chunks.")
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output directory.")],
    gold_scores: Annotated[
        Path | None,
        typer.Option(exists=True, help="Scores of gold source paragraphs (`gold records`)."),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Read only the first N rows of each.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
    plots: Annotated[bool, typer.Option(help="Write PNGs (needs matplotlib).")] = True,
) -> None:
    """Choose tau and per-language flag thresholds; write calibration.yaml."""
    from ocrfix.calibrate.run import run_calibrate

    cfg = _load(profile, overrides)
    cal, rep = run_calibrate(
        cfg, clean_scores, corrupted_scores, out_dir, gold_scores, limit=limit, plots=plots
    )
    typer.echo(f"{out_dir / 'calibration.yaml'}: tau={cal.tau:g} ({cal.variant})")
    for lang, r in cal.rules.items():
        parts = [
            f"{k} > {v:g}"
            for k, v in (("spike_frac", r.spike_frac), ("p90", r.p90), ("bpc", r.bpc))
            if v is not None
        ]
        met = "" if r.target_met else "  (target recall NOT met)"
        typer.echo(
            f"  {lang:<6} {' or '.join(parts) or 'never'}: recall {r.recall}, "
            f"false flag {r.false_flag}{met}"
        )  # fmt: skip
    if rep.gold:
        g = rep.gold["*"]
        typer.echo(f"  gold: precision {g['precision']}, recall {g['recall']} (n={g['n']})")


train_app = typer.Typer(no_args_is_help=True, help="Fine-tuning (TRL): SFT, DPO, adapter merge.")
app.add_typer(train_app, name="train")


@train_app.command("sft")
def train_sft_cmd(
    data: Annotated[Path, typer.Argument(exists=True, help="`data sft` train.jsonl.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Adapter / checkpoint dir.")],
    eval_data: Annotated[
        Path | None, typer.Option("--eval", exists=True, help="Eval file [held-out slice].")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Read only the first N examples.")] = None,
    max_steps: Annotated[
        int, typer.Option(help="Stop after N optimizer steps (-1 = epochs).")
    ] = -1,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """SFT on the assistant turns: QLoRA / LoRA / full per hw.train.mode (torchrun if >1 GPU)."""
    cfg = _load(profile, overrides)
    _maybe_torchrun(cfg)
    from ocrfix.train.sft import train_sft

    m = train_sft(cfg, data, out_dir, eval_data=eval_data, limit=limit, max_steps=max_steps)
    typer.echo(
        f"{out_dir}: train_loss={m.get('train_loss')} peak_vram_gib={m.get('peak_vram_gib')}"
    )


@train_app.command("dpo")
def train_dpo_cmd(
    data: Annotated[Path, typer.Argument(exists=True, help="`data dpo` pairs.")],
    sft: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="SFT result: a LoRA adapter or a full model dir."
        ),
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="DPO adapter dir.")],
    limit: Annotated[int | None, typer.Option(help="Read only the first N pairs.")] = None,
    max_steps: Annotated[
        int, typer.Option(help="Stop after N optimizer steps (-1 = epochs).")
    ] = -1,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """DPO on the SFT result in hw.train.mode (qlora | lora | full); reference per
    train.dpo.reference."""
    cfg = _load(profile, overrides)
    _maybe_torchrun(cfg)
    from ocrfix.train.dpo import train_dpo

    m = train_dpo(cfg, data, sft, out_dir, limit=limit, max_steps=max_steps)
    typer.echo(
        f"{out_dir}: train_loss={m.get('train_loss')} peak_vram_gib={m.get('peak_vram_gib')}"
    )


@train_app.command("merge")
def train_merge_cmd(
    adapter: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="LoRA adapter dir.")
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Merged bf16 checkpoint dir.")],
    dequantize: Annotated[
        bool | None,
        typer.Option(
            "--dequantize/--no-dequantize",
            help="Merge into the dequantized nf4 base "
            "[default: yes if the adapter was trained with qlora].",
        ),
    ] = None,
    base: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Base model [the one the adapter was trained on (ocrfix_train.json), "
            "else the base `model`].",
        ),
    ] = None,
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Merge an adapter into a bf16 checkpoint loadable by vLLM."""
    from ocrfix.train.common import read_meta
    from ocrfix.train.merge import merge

    cfg = _load(profile, overrides)
    if base is None:
        recorded = read_meta(adapter).get("base_model")
        base = Path(recorded) if recorded and Path(recorded).is_dir() else Path(cfg.model)
    typer.echo(f"base: {base}")
    try:
        info = merge(str(base), adapter, out_dir, dequantize=dequantize)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out_dir}: merged ({'dequantized nf4 base' if info['dequantized'] else 'bf16 base'})"
    )


@app.command("calibrate-noise")
def calibrate_noise_cmd(
    gold_chunks: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="`gold chunks` output folder.")
    ],
    out_dir: Annotated[
        Path, typer.Argument(file_okay=False, help="noise_report.json goes here.")
    ] = Path("data/calibration"),
    write: Annotated[
        bool, typer.Option(help="Write the result into configs/stages/score.yaml.")
    ] = True,
    limit: Annotated[int | None, typer.Option(help="Use only the first N gold chunks.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Fit noise removal (score.noise core / extend / min_chars) on the gold OCR text, with
    junk inserted; write the result into configs/stages/score.yaml."""
    from ocrfix.calibrate.noise import run_calibrate_noise
    from ocrfix.config import config_dir

    cfg = _load(profile, [*(overrides or []), "score.noise.remove=false"])
    target = config_dir() / "stages" / "score.yaml" if write else None
    try:
        fit = run_calibrate_noise(cfg, gold_chunks, out_dir, limit=limit, score_yaml=target)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    p = fit["previous"]

    def line(r: dict) -> str:
        return (f"noise found {r['recall']:.1%}, real text removed {r['text_removed_chars']} chars "
                f"({r['text_removed']:.4%})")  # fmt: skip

    typer.echo(f"{fit['chunks']} gold chunks (+ a copy each with junk): {fit['text_chars']} text "
               f"chars, {fit['noise_chars']} inserted noise chars")  # fmt: skip
    typer.echo(f"  previous {p['core']} {p['extend']} min_chars {p['min_chars']}: {line(p)}")
    if fit["use"]:
        typer.echo(f"  fitted   {fit['core']} {fit['extend']} min_chars {fit['min_chars']}: "
                   f"{line(fit)}")  # fmt: skip
        for ex in fit["text_removed_examples"][:5]:
            typer.echo(f"    removed text: {ex}")
        typer.echo(f"  written to {fit['written']}" if fit.get("written") else "  not written")
    else:
        typer.echo(f"  no result: {fit['reason']}; score.yaml unchanged")
    typer.echo(f"{out_dir / 'noise_report.json'}")


@app.command("calibrate-noise-chunk")
def calibrate_noise_chunk_cmd(
    gold_chunks: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="`gold chunks` output folder.")
    ],
    scored: Annotated[
        Path,
        typer.Argument(exists=True, help="Corpus scores of a `score` run with noise removal off."),
    ],
    out_dir: Annotated[
        Path, typer.Argument(file_okay=False, help="Report and samples go here.")
    ] = Path("data/calibration"),
    write: Annotated[
        bool, typer.Option(help="Write the limits into the calibration file (score.calibration).")
    ] = True,
    limit: Annotated[int | None, typer.Option(help="Use only the first N gold chunks.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Whole-chunk noise limit per language: the highest bpc of real gold OCR text + a margin;
    reports what it would drop in the corpus, with samples to read."""
    from ocrfix.calibrate.noise_chunk import run_calibrate_noise_chunk

    cfg = _load(profile, [*(overrides or []), "score.noise.remove=false"])
    target = None
    if write:
        if not cfg.score.calibration or not Path(cfg.score.calibration).is_file():
            raise typer.BadParameter(
                f"no calibration file {cfg.score.calibration}: run `ocrfix calibrate` first, "
                "or --no-write"
            )
        target = Path(cfg.score.calibration)
    try:
        rep = run_calibrate_noise_chunk(cfg, gold_chunks, scored, out_dir, calibration=target,
                                        limit=limit)  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"gold: {rep['gold_scored']} chunks scored ({rep['gold_rows']} rows)")
    for lang, g in rep["gold"].items():
        c = rep["corpus"][lang]
        d = next(x for x in c["drop"] if x["limit"] == g["limit"])
        share = f"{d['chunk_share']:.3%}" if d["chunk_share"] is not None else "-"
        typer.echo(f"  {lang:6} gold real text: {g['real_chunks']} chunks, bpc p50 {g['real_p50']}"
                   f" p99 {g['real_p99']} max {g['real_max']} -> limit {g['limit']}; gold noise "
                   f"chunks caught {g['noise_caught']}/{g['noise_chunks']}; corpus: "
                   f"{d['chunks']} of {c['chunks']} chunks dropped ({share})")  # fmt: skip
    typer.echo(f"  written to {rep['written']}" if rep.get("written") else "  not written")
    for f in ("noise_chunk_report.json", "noise_chunk_samples.md", "noise_chunk_hist.png"):
        typer.echo(str(out_dir / f))


@app.command("bpc-hist")
def bpc_hist_cmd(
    scored: Annotated[
        list[Path], typer.Argument(exists=True, help="`score` output folder(s) or file(s).")
    ],
    out: Annotated[Path, typer.Option(help="The PNG to write.")] = Path("bpc_hist.png"),
    calibration: Annotated[
        Path | None,
        typer.Option(
            help="Draw its noise_chunk limits (default: score.calibration, if it exists)."
        ),
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Histogram of chunk bpc per language from score output (no model): a PNG, plus a table."""
    from ocrfix.calibrate.noise_chunk import BIN, bpc_histograms, plot_bpc
    from ocrfix.calibrate.rules import load_calibration

    cfg = _load(profile, overrides or [])
    cal_path = calibration or (Path(cfg.score.calibration) if cfg.score.calibration else None)
    limits = load_calibration(cal_path).noise_chunk if cal_path and cal_path.is_file() else {}
    hists = bpc_histograms(scored, limits)
    if not hists:
        raise typer.BadParameter("no scored chunks with a bpc")
    title = "Chunk bpc: " + ", ".join(str(p) for p in scored)
    plot_bpc(hists, out, title)
    for lang, h in hists.items():
        lim = h["limit"]
        typer.echo(f"{lang:6} {h['chunks']:>10,} chunks  p50 {h['p50']:<6} p99 {h['p99']:<6} "
                   f"p99.9 {h['p99.9']:<6} max {h['max']:<6} limit {lim} "
                   f"({h['above']} chunks above)")  # fmt: skip
        hist = h["histogram"]
        peak = max(hist.values())
        for k, v in hist.items():
            bar = "#" * max(1, round(40 * v / peak))
            mark = "  <- above limit" if lim is not None and float(k) + BIN > lim else ""
            typer.echo(f"   {float(k):5.2f}-{float(k) + BIN:<5.2f} {v:>10,} {bar}{mark}")
    typer.echo(str(out))


@app.command("serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def serve_cmd(
    ctx: typer.Context,
    correct: Annotated[
        bool, typer.Option(help="Serve correct.model (as `tuned`) instead of the base.")
    ] = False,
    port: Annotated[int | None, typer.Option(help="First port [hw.infer.base_port].")] = None,
    engines: Annotated[int | None, typer.Option(help="Engines [hw.infer.engines].")] = None,
    gpus: Annotated[str | None, typer.Option(help="GPU ids, e.g. 0,1 [0..engines-1].")] = None,
    memory: Annotated[
        float | None, typer.Option(help="GPU memory share [hw.infer.gpu_memory_utilization].")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the commands only.")] = False,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Start the vLLM engines (base model, or --correct); vLLM arguments go after `--`."""
    from ocrfix import serve

    cfg = _load(profile, overrides)
    try:
        engs = serve.engines(
            cfg, correct=correct, port=port, n=engines, memory=memory, extra=list(ctx.args),
            gpus=[int(g) for g in gpus.split(",")] if gpus else None,
        )  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    if dry_run:
        for e in engs:
            typer.echo(f"CUDA_VISIBLE_DEVICES={e.gpus} {' '.join(e.cmd)}")
        return
    raise typer.Exit(serve.run(cfg, engs, echo=typer.echo))


@app.command("config")
def show_config(
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
    key: Annotated[str | None, typer.Option(help="Print only this dotted key.")] = None,
) -> None:
    """Print the fully resolved config for a profile (or one key of it)."""
    cfg = _load(profile, overrides)
    if key is None:
        typer.echo(OmegaConf.to_yaml(cfg, resolve=True))
        return
    value = OmegaConf.select(cfg, key, throw_on_missing=True)
    typer.echo(OmegaConf.to_yaml(value, resolve=True) if OmegaConf.is_config(value) else value)


if __name__ == "__main__":
    app()
