"""`ocrfix` command line. One command per pipeline stage."""

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


@app.command("ingest")
def ingest_cmd(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="MinerU dir.")],
    out: Annotated[Path, typer.Argument(help="Output .jsonl or .jsonl.gz.")],
    limit: Annotated[int | None, typer.Option(help="Ingest at most N new documents.")] = None,
    sample_frac: Annotated[
        float | None,
        typer.Option(help="Keep this share of documents, by path hash [corpus.sample_frac]."),
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Split MinerU markdown into paragraph Records with prose/math/table spans."""
    from ocrfix.corpus import ingest

    cfg = _load(profile, overrides)
    frac = sample_frac if sample_frac is not None else cfg.corpus.get("sample_frac")
    stats = ingest(
        input_dir,
        out,
        pattern=cfg.corpus.input_glob,
        limit=limit,
        lang_opts=dict(cfg.corpus.lang),
        sample_frac=frac,
        sample_seed=int(cfg.seed),
    )
    sampled = f", {stats.docs_sampled_out} outside the sample" if frac is not None else ""
    typer.echo(
        f"{out}: {stats.docs_written} docs ({stats.paragraphs} paragraphs) written, "
        f"{stats.docs_skipped} already present{sampled}"
    )


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
    """Chunks -> markdown files, placeholders restored."""
    from ocrfix.chunks.rebuild import run_rebuild

    _load(profile, overrides)
    try:
        st = run_rebuild(chunks, out_dir, source=source, limit=limit)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"{st.docs} docs, {st.bytes} bytes; {len(st.mismatched)} differ from the source")
    for x in [*st.mismatched, *st.errors][:20]:
        typer.echo(f"  {x}", err=True)
    if st.mismatched or st.errors:
        raise typer.Exit(1)


refine_app = typer.Typer(
    no_args_is_help=True, help="Refinement on chunks: score, then correct -> re-score -> verify."
)
app.add_typer(refine_app, name="refine")
ShardLimit = Annotated[int | None, typer.Option(help="Process only the first N shards.")]


@refine_app.command("score")
def refine_score_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`ocrfix chunk` output.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Scored chunk shards.")],
    limit: ShardLimit = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Score every chunk (refine.scorer: vllm or transformers); write the scores into it."""
    from ocrfix.refine.run import run_score

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


@refine_app.command("correct")
def refine_correct_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`refine score` output.")],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Refined chunk shards.")],
    select_all: Annotated[
        bool, typer.Option("--all", help="Send every scored chunk, not only flagged ones.")
    ] = False,
    limit: ShardLimit = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Per flagged chunk: correct, re-score at once, verify; accepted text replaces it."""
    from ocrfix.refine.run import format_summary, run_correct, summarize

    cfg = _load(profile, overrides)
    try:
        st = run_correct(cfg, chunks, out_dir, select_all=select_all, limit=limit)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"{out_dir}: {st.shards - st.shards_skipped} shards in {st.seconds:.0f}s")
    typer.echo(format_summary(summarize(out_dir)))


@refine_app.command("corrupt")
def refine_corrupt_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="Clean chunks (`ocrfix chunk`).")],
    error_table: Annotated[Path, typer.Argument(exists=True, help="error_table.json.")],
    confusion_map: Annotated[Path, typer.Argument(exists=True, help="confusion_map.json.")],
    out: Annotated[Path, typer.Argument(help="Corrupted chunks .jsonl.")],
    rate: Annotated[
        float | None, typer.Option(help="Fixed ops per prose char (0.01 = 1%) [table rates].")
    ] = None,
    n_variants: Annotated[
        int | None, typer.Option(help="Corruptions per chunk [corrupt.variants_per_para].")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Corrupt only the first N chunks.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Synthetic OCR damage on clean chunks at the gold set's error rates (for calibration)."""
    from ocrfix.refine.corrupt import run_refine_corrupt

    cfg = _load(profile, overrides)
    n = run_refine_corrupt(cfg, chunks, error_table, confusion_map, out, rate=rate,
                           n_variants=n_variants, limit=limit)  # fmt: skip
    typer.echo(f"{out}: {n} corrupted chunks")


@refine_app.command("data")
def refine_data_cmd(
    out_dir: Annotated[
        Path, typer.Argument(file_okay=False, help="train.jsonl, val.jsonl, stats.json.")
    ],
    synthetic: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`refine corrupt` output (repeatable): damaged -> clean."),
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
    """Chunk SFT examples in the exact form `refine correct` sends (for `train sft`)."""
    from omegaconf import OmegaConf

    from ocrfix.refine.data import build_refine_sft

    cfg = _load(profile, overrides)
    if not synthetic and not identity and not gold:
        raise typer.BadParameter("give --synthetic, --gold and/or --identity")
    opts = OmegaConf.to_container(cfg.refine.data, resolve=True)
    d = cfg.data
    opts |= {"val_frac": d.val_frac, "empty_think": bool(d.empty_think),
             "exclude_split": d.exclude_split}  # fmt: skip
    try:
        st = build_refine_sft(
            {"synthetic": synthetic or [], "gold": gold or [], "identity": identity or []},
            out_dir,
            opts=opts, seed=int(d.seed),
        )  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out_dir}: train {st.train}, val {st.val} "
        f"({st.with_junk} with junk, {st.with_lost} with lost words)"
    )
    for s, langs in st.taken.items():
        typer.echo(f"  {s:<9} taken {langs} of {st.pool[s]}; dropped {dict(st.dropped[s])}")


@refine_app.command("dpo")
def refine_dpo_cmd(
    out: Annotated[Path, typer.Argument(help="Pairs .jsonl (+ .responses.jsonl, .stats.json).")],
    synthetic: Annotated[
        list[Path] | None,
        typer.Option(exists=True, help="`refine corrupt` output (repeatable): damaged -> clean."),
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

    from ocrfix.refine.dpo import build_refine_dpo
    from ocrfix.refine.score import Encoder

    cfg = _load(profile, overrides)
    if not synthetic and not identity and not gold:
        raise typer.BadParameter("give --synthetic, --gold and/or --identity")
    c = lambda node: OmegaConf.to_container(node, resolve=True)  # noqa: E731
    d = cfg.data
    opts = c(cfg.refine.data) | {
        "val_frac": d.val_frac, "empty_think": False, "exclude_split": d.exclude_split,
        "dpo": c(cfg.refine.dpo), "llm": c(cfg.llm), "urls": c(cfg.refine.correct_urls),
        "concurrency": int(cfg.refine.concurrency),
    }  # fmt: skip
    transport = None
    if str(cfg.refine.generator) == "transformers":
        from ocrfix.refine.run import generator_transport

        transport = generator_transport(cfg)
    try:
        st = build_refine_dpo(
            {"synthetic": synthetic or [], "gold": gold or [], "identity": identity or []},
            out, opts=opts, seed=int(d.seed), exclude=exclude or [],
            encode=Encoder(str(cfg.paths.tokenizer)), transport=transport,
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


@refine_app.command("eval")
def refine_eval_cmd(
    truth_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="The gold truth/ folder.")
    ],
    paths: Annotated[
        list[str],
        typer.Argument(
            help="System outputs (name=dir or dir: `refine` output of the gold eval chunks), "
            "then the output dir."
        ),
    ],
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Whole corrected documents vs the hand truth: CER/WER/SER, false fixes, recall by error
    type, how each lost word (〈?〉) was answered, paired Wilcoxon; report.md."""
    from ocrfix.refine.eval import parse_system, run_refine_eval

    if len(paths) < 2:
        raise typer.BadParameter("need at least one system output and the output dir")
    *systems, out_dir = paths
    for s in systems:
        if not parse_system(s)[1].exists():
            raise typer.BadParameter(f"system output not found: {s}")
    cfg = _load(profile, overrides)
    ec = cfg.errors
    opts = {"header_footer_min_chars": int(ec.header_footer_min_chars),
            "confusion_max_len": int(ec.confusion_max_len)}  # fmt: skip
    try:
        res = run_refine_eval(
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


@refine_app.command("compare")
def refine_compare_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="Chunks (refine score output).")],
    out: Annotated[Path, typer.Argument(dir_okay=False, help="Report .json.")],
    limit: Annotated[int, typer.Option(help="Compare the first N chunks with prose.")] = 200,
    score: Annotated[
        bool, typer.Option(help="Compare scores (score_urls vs paths.scorer_model).")
    ] = True,
    generate: Annotated[
        bool, typer.Option(help="Compare greedy corrections (correct_urls vs llm.model).")
    ] = True,
    examples: Annotated[int, typer.Option(help="Differing answers to show.")] = 5,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """vLLM vs transformers on the same chunks: per-token bits, features, flags, noise spans,
    and greedy answers. Run before using one backend's results for the other."""
    from ocrfix.refine.compare import format_compare, run_compare

    cfg = _load(profile, overrides)
    try:
        rep = run_compare(
            cfg, chunks, out, limit=limit, score=score, generate=generate, examples=examples
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(format_compare(rep))
    typer.echo(str(out))


@refine_app.command("summary")
def refine_summary_cmd(
    chunks: Annotated[Path, typer.Argument(exists=True, help="`refine` output.")],
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Status counts, accept rate, failed checks, per language."""
    from ocrfix.refine.run import format_summary, summarize

    _load(profile, overrides)
    typer.echo(format_summary(summarize(chunks)))


@refine_app.command("report")
def refine_report_cmd(
    chunks: Annotated[
        Path, typer.Argument(exists=True, help="`refine correct` (or score) output.")
    ],
    clean: Annotated[
        Path | None,
        typer.Option(exists=True, help="`refine score` output of clean text, as the baseline."),
    ] = None,
    calibration: Annotated[
        Path | None, typer.Option(exists=True, help="calibration.yaml [scorer.calibration].")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the report here.")
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Corpus quality per language before / after noise removal and correction: bpc, flag rate."""
    from ocrfix.calibrate.rules import load_calibration
    from ocrfix.refine.report import corpus_report, format_report

    cfg = _load(profile, overrides)
    path = calibration or cfg.scorer.calibration
    cal = load_calibration(path) if path and Path(path).is_file() else None
    rep = corpus_report(chunks, cal, clean)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    typer.echo(format_report(rep))


def _maybe_torchrun(cfg) -> None:
    """Re-exec this command under torchrun with hw.n_gpus processes (once)."""
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


@app.command("score")
def score_cmd(
    paragraphs: Annotated[Path, typer.Argument(exists=True, help="Paragraph Records JSONL.")],
    out: Annotated[Path, typer.Argument(help="Output scores .jsonl.")],
    limit: Annotated[int | None, typer.Option(help="Score only the first N paragraphs.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
    dump_tokens: Annotated[
        Path | None, typer.Option(help="Also write per-paragraph token arrays (.npy) here.")
    ] = None,
) -> None:
    """Per-paragraph LM surprisal features with the local Qwen3.5 (torchrun, hw.n_gpus)."""
    cfg = _load(profile, overrides)
    n_gpus = int(cfg.hw.n_gpus)
    _maybe_torchrun(cfg)  # one process per GPU, full model on each
    from ocrfix.scorer.run import score

    n = score(cfg, paragraphs, out, limit=limit, dump_tokens=dump_tokens)
    if int(os.environ.get("RANK", 0)) == 0:
        typer.echo(f"{out}: scored {n} paragraphs on rank 0 ({cfg.hw.name}, {n_gpus} GPU)")


gold_app = typer.Typer(
    no_args_is_help=True, help="Human gold annotation: sample paragraphs, import labels."
)
app.add_typer(gold_app, name="gold")


@gold_app.command("sample")
def gold_sample_cmd(
    scores: Annotated[Path, typer.Argument(exists=True, help="scores.jsonl from `score`.")],
    paragraphs: Annotated[Path, typer.Argument(exists=True, help="Paragraph Records JSONL.")],
    out: Annotated[Path, typer.Argument(help="Annotation JSONL to write.")],
    limit: Annotated[int | None, typer.Option(help="Read only the first N paragraphs.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
    exclude: Annotated[
        list[Path] | None,
        typer.Option(help="Skip paragraphs already in this JSONL (repeatable)."),
    ] = None,
) -> None:
    """Stratified sample for annotation (+ Label Studio import JSON and labeling config)."""
    from ocrfix.gold import run_sample

    cfg = _load(profile, overrides)
    res = run_sample(cfg, scores, paragraphs, out, exclude=exclude or [], limit=limit)
    typer.echo(f"{out}: {len(res.rows)} paragraphs sampled from {res.n_paragraphs}")
    for stratum, n in res.wanted.items():
        typer.echo(f"  {stratum:<24} {res.got[stratum]:>5} / {n}")
    if res.shortfalls:
        typer.echo(f"  short: {', '.join(res.shortfalls)}", err=True)


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


@gold_app.command("import")
def gold_import_cmd(
    annotated: Annotated[
        Path, typer.Argument(exists=True, help="Annotated JSONL, or a Label Studio .json export.")
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output directory.")],
    limit: Annotated[int | None, typer.Option(help="Read only the first N annotations.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
    strict: Annotated[bool, typer.Option(help="Exit 1 if any violation was found.")] = False,
) -> None:
    """Validate annotations, inter-annotator distance, 50/50 doc split into train/eval."""
    from ocrfix.gold import run_import

    cfg = _load(profile, overrides)
    rep = run_import(cfg, annotated, out_dir, limit=limit)
    typer.echo(
        f"{out_dir}: {rep.n_accepted}/{rep.n_paragraphs} paragraphs accepted "
        f"({rep.n_rows} annotations); train {rep.n_train} ({rep.train_docs} docs), "
        f"eval {rep.n_eval} ({rep.eval_docs} docs)"
    )
    for rule, n in sorted(rep.violations.items()):
        typer.echo(f"  violation {rule:<20} {n:>5}", err=True)
    if rep.agreement.get("n_pairs"):
        a = rep.agreement
        typer.echo(
            f"  inter-annotator CED: mean {a['mean_ced']}, median {a['median_ced']} "
            f"over {a['n_pairs']} pairs"
        )
    if strict and rep.violations:
        raise typer.Exit(1)


errors_app = typer.Typer(no_args_is_help=True, help="OCR error analysis on gold pairs.")
app.add_typer(errors_app, name="errors")


@errors_app.command("table")
def errors_table_cmd(
    gold: Annotated[
        Path, typer.Argument(exists=True, help="`gold chunks` sft/ (or gold_train.jsonl).")
    ],
    out_dir: Annotated[Path, typer.Argument(file_okay=False, help="Output directory.")],
    fixed: Annotated[
        Path | None,
        typer.Option(exists=True, help="Fixer output (Records keyed like gold): residual table."),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Read only the first N gold rows.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Error-type rates per language and stratum, char confusions, garbled-Hangul map."""
    from ocrfix.errors import run_table

    cfg = _load(profile, overrides)
    res = run_table(cfg, gold, out_dir, fixed=fixed, limit=limit)
    overall = res.table.groups.get(("*", "*"))
    n_ops = sum(overall.counts.values()) if overall else 0
    typer.echo(f"{out_dir}: {res.n_rows} paragraphs, {n_ops} ops")
    if res.residual is not None:
        r = res.residual.groups.get(("*", "*"))
        typer.echo(
            f"  residual: {sum(r.counts.values()) if r else 0} ops "
            f"({res.n_fixed_missing} paragraphs without fixer output)"
        )


@app.command("corrupt")
def corrupt_cmd(
    clean: Annotated[Path, typer.Argument(exists=True, help="Clean paragraphs or gold rows.")],
    error_table: Annotated[Path, typer.Argument(exists=True, help="error_table.json.")],
    confusion_map: Annotated[Path, typer.Argument(exists=True, help="confusion_map.json.")],
    out: Annotated[Path, typer.Argument(help="Output pairs .jsonl.")],
    n_variants: Annotated[
        int | None, typer.Option(help="Corruptions per paragraph [corrupt.variants_per_para].")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Seed [seed].")] = None,
    rate: Annotated[
        float | None,
        typer.Option(
            help="Fixed error rate in ops per prose char (0.01 = 1%) instead of the table's."
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Corrupt only the first N paragraphs.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Synthetic OCR noise on clean text at the measured per-type rates."""
    from ocrfix.corrupt import run_corrupt

    cfg = _load(profile, overrides)
    n = run_corrupt(
        cfg, clean, error_table, confusion_map, out,
        n_variants=n_variants, seed=seed, rate=rate, limit=limit,
    )  # fmt: skip
    typer.echo(f"{out}: {n} corrupted paragraphs")


@gold_app.command("records")
def gold_records_cmd(
    gold: Annotated[Path, typer.Argument(exists=True, help="gold_train.jsonl / gold_eval.jsonl.")],
    out: Annotated[Path, typer.Argument(help="Records JSONL (the OCR source side) to score.")],
    side: Annotated[
        str, typer.Option(help="source (OCR text, with cer) or gold (the clean text).")
    ] = "source",
    limit: Annotated[int | None, typer.Option(help="Read only the first N gold rows.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Gold rows -> source-side Records with `cer`, for `score` and `calibrate --gold-scores`
    (`--side gold`: clean Records for `corrupt` and the clean scores of `calibrate`)."""
    from ocrfix.gold.records import gold_records

    _load(profile, overrides)
    if side not in ("source", "gold"):
        raise typer.BadParameter("--side must be source or gold")
    n, skipped = gold_records(gold, out, limit, side=side)
    typer.echo(f"{out}: {n} records ({skipped} rows without restored text skipped)")


@app.command("calibrate")
def calibrate_cmd(
    clean_scores: Annotated[Path, typer.Argument(exists=True, help="Scores of clean paragraphs.")],
    corrupted_scores: Annotated[
        Path, typer.Argument(exists=True, help="Scores of the same paragraphs corrupted.")
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


@app.command("fix")
def fix_cmd(
    paragraphs: Annotated[Path, typer.Argument(exists=True, help="Paragraph Records JSONL.")],
    out: Annotated[Path, typer.Argument(help="Fixed Records JSONL.")],
    ops: Annotated[Path | None, typer.Option(help="Ops log JSONL [<out stem>.ops.jsonl].")] = None,
    limit: Annotated[int | None, typer.Option(help="Fix only the first N documents.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Run the enabled fixers (fix.order) per document; log every change to --ops."""
    from ocrfix.fixers.run import run_fix

    cfg = _load(profile, overrides)
    if ops is None:
        ops = out.with_name(out.name.split(".")[0] + ".ops.jsonl")
    stats = run_fix(cfg, paragraphs, out, ops, limit=limit)
    typer.echo(
        f"{out}: {stats.docs} docs, {stats.paragraphs_in} -> {stats.paragraphs_out} paragraphs; "
        f"{sum(stats.ops.values())} ops in {ops}"
    )
    for k, n in sorted(stats.ops.items()):
        typer.echo(f"  {k:<32} {n:>6}")


@app.command("correct")
def correct_cmd(
    paragraphs: Annotated[
        Path, typer.Argument(exists=True, help="Paragraph Records (fix output).")
    ],
    scores: Annotated[Path, typer.Argument(exists=True, help="Scores of those paragraphs.")],
    out: Annotated[Path, typer.Argument(help="Corrected Records JSONL.")],
    responses: Annotated[
        Path | None, typer.Option(help="Resumable raw responses [<out stem>.responses.jsonl].")
    ] = None,
    select_all: Annotated[
        bool, typer.Option("--all", help="Correct every paragraph, not only flagged ones.")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="Send at most N chunks.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """LLM correction of flagged paragraphs via the vLLM server(s) (scripts/serve_vllm.sh)."""
    from ocrfix.llm.correct import run_correct

    cfg = _load(profile, overrides)
    st = run_correct(cfg, paragraphs, scores, out, responses, select_all=select_all, limit=limit)
    typer.echo(
        f"{out}: {st.selected} selected, {st.chunks} chunks ({st.sent} sent), "
        f"{st.corrected} paragraphs changed"
    )
    for err, n in sorted(st.errors.items()):
        typer.echo(f"  error {err:<24} {n:>6}", err=True)


@app.command("verify")
def verify_cmd(
    corrections: Annotated[Path, typer.Argument(exists=True, help="`ocrfix correct` output.")],
    scores_before: Annotated[
        Path, typer.Argument(exists=True, help="Scores of the paragraphs sent to the LLM.")
    ],
    out: Annotated[Path, typer.Argument(help="Verified Records JSONL.")],
    limit: Annotated[int | None, typer.Option(help="Verify only the first N rows.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Guardrails on each LLM correction (markers, length, edits, language, gaps, scorer)."""
    from ocrfix.verify.run import format_summary, run_verify, summarize

    cfg = _load(profile, overrides)
    run_verify(cfg, corrections, scores_before, out, limit=limit)
    typer.echo(f"{out}:")
    typer.echo(format_summary(summarize(out)))


@app.command("verify-summary")
def verify_summary_cmd(
    verified: Annotated[Path, typer.Argument(exists=True, help="`ocrfix verify` output.")],
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Accept rate and a histogram of failed checks."""
    from ocrfix.verify.run import format_summary, summarize

    _load(profile, overrides)
    typer.echo(format_summary(summarize(verified)))


data_app = typer.Typer(no_args_is_help=True, help="Training data: SFT examples, DPO pairs.")
app.add_typer(data_app, name="data")


@data_app.command("sft")
def data_sft_cmd(
    out_dir: Annotated[
        Path, typer.Argument(file_okay=False, help="Writes train/val.jsonl, stats.json.")
    ],
    synthetic: Annotated[Path | None, typer.Option(exists=True, help="`corrupt` output.")] = None,
    verified: Annotated[Path | None, typer.Option(exists=True, help="`verify` output.")] = None,
    gold: Annotated[Path | None, typer.Option(exists=True, help="gold_train.jsonl.")] = None,
    identity: Annotated[
        Path | None, typer.Option(exists=True, help="Clean paragraph Records.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Read at most N rows per source.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Mixed, language-balanced SFT chat examples in the `correct` prompt format."""
    from ocrfix.data.run import run_sft

    cfg = _load(profile, overrides)
    st = run_sft(
        cfg,
        out_dir,
        synthetic=synthetic,
        verified=verified,
        gold=gold,
        identity=identity,
        limit=limit,
    )
    typer.echo(f"{out_dir}: {st.total} examples (train {st.train}, val {st.val})")
    if st.unmaskable:
        typer.echo(
            f"  {st.unmaskable} pairs dropped: the target changes a math/table/protected span "
            "(e.g. garbled Hangul), which stays a placeholder at inference"
        )
    for s, langs in st.taken.items():
        typer.echo(
            f"  {s:<10} " + ", ".join(f"{k} {v}/{st.pool[s].get(k, 0)}" for k, v in langs.items())
        )


@data_app.command("dpo")
def data_dpo_cmd(
    out: Annotated[Path, typer.Argument(help="Preference pairs JSONL.")],
    gold: Annotated[
        Path | None, typer.Option(exists=True, help="gold_train.jsonl (chosen = gold).")
    ] = None,
    verified: Annotated[
        Path | None,
        typer.Option(exists=True, help="`verify` output (chosen = accepted correction)."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Served model/adapter to sample from.")] = None,
    n_samples: Annotated[
        int | None, typer.Option(help="Completions per input [data.dpo.n_samples].")
    ] = None,
    responses: Annotated[Path | None, typer.Option(help="Resumable responses JSONL.")] = None,
    limit: Annotated[int | None, typer.Option(help="Read at most N rows per input file.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """prompt/chosen/rejected pairs: failing samples plus constructed negatives."""
    from ocrfix.data.run import run_dpo

    cfg = _load(profile, overrides)
    st = run_dpo(cfg, out, gold=gold, verified=verified, model=model, n_samples=n_samples,
                 responses=responses, limit=limit)  # fmt: skip
    typer.echo(f"{out}: {st.pairs} pairs from {st.inputs} inputs ({st.sampled_requests} sampled)")
    if st.unmaskable:
        typer.echo(f"  {st.unmaskable} inputs dropped: chosen changes a placeholder span")
    for k, n in st.by_type.most_common():
        typer.echo(f"  {k:<24} {n:>6}")


@data_app.command("stats")
def data_stats_cmd(
    files: Annotated[list[Path], typer.Argument(exists=True, help="SFT or DPO JSONL file(s).")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    limit: LimitOpt = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Counts by source, language and rejected_type."""
    from ocrfix.data.run import format_stats, stats, stats_json

    _load(profile, overrides)
    s = stats(files)
    typer.echo(stats_json(s) if as_json else format_stats(s))


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
            "else paths.base_model].",
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
        base = (
            Path(recorded) if recorded and Path(recorded).is_dir() else Path(cfg.paths.base_model)
        )
    typer.echo(f"base: {base}")
    try:
        info = merge(str(base), adapter, out_dir, dequantize=dequantize)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"{out_dir}: merged ({'dequantized nf4 base' if info['dequantized'] else 'bf16 base'})"
    )


@app.command("eval")
def eval_cmd(
    gold: Annotated[Path, typer.Argument(exists=True, help="gold_eval.jsonl from `gold import`.")],
    paths: Annotated[
        list[str],
        typer.Argument(
            help="System outputs (name=path or path; records from the gold eval sources), "
            "then the output dir."
        ),
    ],
    limit: Annotated[int | None, typer.Option(help="Evaluate at most N gold documents.")] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """Score system outputs against gold (CER/WER/SER, false fixes, recall by error type,
    structure, identity, placeholders), paired Wilcoxon across systems, report.md."""
    from ocrfix.eval.run import parse_system, run_eval

    if len(paths) < 2:
        raise typer.BadParameter("need at least one system output and the output dir")
    *systems, out_dir = paths
    for s in systems:
        if not parse_system(s)[1].exists():
            raise typer.BadParameter(f"system output not found: {s}")
    cfg = _load(profile, overrides)
    m = run_eval(cfg, gold, systems, Path(out_dir), limit=limit)
    for name, s in m["systems"].items():
        o = s["overall"]
        typer.echo(
            f"{name}: CER {o['cer']:.4f}  WER {o['wer']:.4f}  false-fix {o['false_fix_rate']:.3f}  "
            f"placeholders {o['placeholders']['integrity']:.3f}"
        )
    typer.echo(f"{out_dir}/report.md")


@app.command("pipeline")
def pipeline_cmd(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="pipeline.yaml.")],
    limit: Annotated[
        int | None, typer.Option(help="Ingest at most N documents (smoke runs).")
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option(help="Run only this stage (repeatable).")
    ] = None,
    force: Annotated[
        list[str] | None, typer.Option(help="Rerun this stage even if up to date ('all').")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the stage commands only.")] = False,
    work_dir: Annotated[
        Path | None, typer.Option(help="Override the config's work_dir (one per system).")
    ] = None,
    profile: ProfileOpt = None,
    overrides: SetOpt = None,
) -> None:
    """ingest -> score -> fix -> rescore -> correct -> verify -> eval over a directory,
    resumable per stage (see configs/pipeline/*.yaml and docs/runbook.md)."""
    from ocrfix.pipeline import Pipeline, PipelineError, load_pipeline

    try:
        pc = load_pipeline(config)
        if work_dir is not None:
            pc.work_dir = work_dir
        _load(profile, [*pc.sets, *(overrides or [])])  # fail early on a bad profile/override
        pl = Pipeline(pc, profile=profile, sets=overrides, limit=limit, only=only, force=force,
                      echo=typer.echo)  # fmt: skip
        if dry_run:
            for line in pl.dry_run():
                typer.echo(line)
            return
        results = pl.run()
    except PipelineError as e:
        typer.echo(f"pipeline: {e}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"\n{pc.work_dir}:")
    for r in results:
        secs = f"{r.seconds:8.1f}s" if r.status == "ran" else " " * 9
        typer.echo(f"  {r.stage:<8} {r.status:<8}{secs}  {r.note}".rstrip())


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
