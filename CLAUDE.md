# ocrfix

Corrects OCR errors in MinerU markdown output (English and Korean, with LaTeX math and
HTML tables) using a locally hosted Qwen3.5 model. Runs on an **air-gapped** network: today
one RTX 5090 (32 GB, Blackwell sm_120) with the 4B, later 4x H200 with the 9B.

Models: `model` is set per profile (rtx5090: `Qwen3.5-4B` in the project folder; h200x4:
`/models/Qwen3.5-9B`). A relative path resolves against the current folder, then the project root.

The unit of work is the 512-768 token chunk. Commands: `chunk`, `reconstruct`,
`score`, `correct`, `corrupt`, `data sft|dpo`, `eval`, `compare`, `summary`, `report`, `gold chunks`,
`errors table`, `calibrate`, `calibrate-noise`, `calibrate-noise-chunk`, `bpc-hist`, `serve`, `train sft|dpo|merge`, `config`, `markup`, `pipeline`. `docs/real_data.md` runs them in
order on real data; `docs/runbook.md` covers install, the H200 track and adding a profile.
The older paragraph pipeline (ingest, score, fix, correct, verify, data, eval, pipeline, gold
sample/import) was removed on 2026-11-13; a copy is in `backups/before_paragraph_removal_*.tar.gz`.

## Layout

```
ocrfix, ocrfix.cmd             launchers (Linux / Windows): `./ocrfix <command>` = .venv's Python + run.py
run.py                         UTF-8 mode, profile = --profile / $OCRFIX_PROFILE / cpu (printed), then cli.app
configs/default.yaml           shared: seed, work_dir, lang (and `model: ???`, `hw: ???` for the profile)
configs/stages/*.yaml          one file per step, one section of the same name (see Config)
configs/profiles/*.yaml        per machine: `model` + `hw` only (rtx5090, h200x4, h200x8); cluster.yaml.example
src/ocrfix/cli.py              typer app; one command per step
src/ocrfix/schema.py           Span, placeholder constants, nfc()
src/ocrfix/io.py               streaming JSONL I/O (files or folders of shards)
src/ocrfix/config.py           load_config(profile, overrides): default + stages + profile + --set
src/ocrfix/offline.py          enforce_offline() env vars
src/ocrfix/chunks/             split.py (one document -> chunks), run.py (parallel, shards), rebuild.py (reconstruct)
src/ocrfix/stages/             score.py (masks, features, vLLM/transformers backends), noise.py, prompt.py, run.py
                               (score, correct), corrupt.py, data.py (SFT examples), dpo.py (DPO pairs),
                               eval.py (docs vs truth), compare.py (vLLM vs transformers), report.py
src/ocrfix/corpus/             tagger.py (spans), lang.py, placeholders.py, protect.py (the one masking rule),
                               files.py (markdown files, doc ids, sampling)
src/ocrfix/scorer/             tokens.py (token masks), model.py (batching, NLL, load_scorer), features.py
src/ocrfix/gold/               chunks.py (md pairs -> chunk gold, split)
src/ocrfix/errors/             align.py (alignment + op types), table.py (rates, maps, summary)
src/ocrfix/corrupt/            ops.py (one generator per error type), run.py (rates, corrupt_text)
src/ocrfix/calibrate/          fit.py (AUROC, rule search), rules.py (calibration.yaml), run.py, report.py,
                               noise.py (calibrate-noise), noise_chunk.py (calibrate-noise-chunk)
src/ocrfix/verify/             checks.py (the checks `correct` runs)
src/ocrfix/data/               common.py (validation split, eval docs, quotas) for data sft / dpo
src/ocrfix/train/              common.py (settings, loading, chat render), sft.py, dpo.py, merge.py
src/ocrfix/eval/               align.py (opcodes, touches), metrics.py (spacing, word edits), stats.py (Wilcoxon, Holm)
src/ocrfix/llm/                client.py (vLLM HTTP client), hf.py (transformers behind the same API)
src/ocrfix/editdist.py         Levenshtein (rapidfuzz if installed)
scripts/                       make_wheelhouse.py, install_offline.py, build_causal_conv1d.py, make_test_corpus.py
src/ocrfix/serve.py            `ocrfix serve`: the vLLM engines
src/ocrfix/pipeline.py         `ocrfix pipeline prepare|calibrate|train|corpus|all`: the README's commands as
                               stages (subprocess per step, skip-if-done markers, starts/stops `serve`)
tests/fixtures/                sample_mineru/ (tiny tokenizer), records.jsonl
docs/annotation_guideline.md   gold .md pairs: what to fix, checks, examples
docs/real_data.md              real data step by step: gold chunks -> calibration -> SFT/DPO -> eval -> corpus
docs/runbook.md                install, model copy, the H200 track, adding a profile
```

## Steps

All outputs are folders of shards `part-NNNNN.jsonl[.gz]` (or one JSONL file). Every command
takes `--profile NAME` and `--set key=value` (repeatable); most take `--limit N`.
`ocrfix config --profile P [--key k]` prints the resolved config.

| # | command | input(s) | output |
|---|---|---|---|
| 1 | `gold chunks` | `gold/ocr/**.md` + `gold/truth/**.md` | `{sft,dpo,eval}/` chunk gold, `split.json`, `stats.json` |
| 2 | `errors table` | gold `sft/` | `error_table.json`, `confusion_map.json`, `summary.md` |
| 3 | `chunk` | markdown tree | chunk shards (`source_text` masked, `placeholders`, offsets) |
| 4 | `corrupt` | clean chunks + error tables | damaged chunks (`source_text`, `gold_text`, `applied_ops`) |
| 5 | `score` | chunks | + `score`, `flag`, noise removed (`history`) |
| 6 | `calibrate` | clean + damaged scores | `calibration.yaml` (per-language flag rule), report |
| 7 | `data sft` | corrupt, gold sft, clean chunks | `train.jsonl`, `val.jsonl` (chat messages) |
| 8 | `train sft` / `merge` | train.jsonl / adapter | LoRA adapter (+ `ocrfix_train.json`) / merged bf16 model |
| 9 | `data dpo` | corrupt, gold dpo, clean + the SFT model | DPO pairs (`prompt, chosen, rejected, rejected_type`) |
| 10 | `train dpo` | pairs + SFT adapter | DPO adapter |
| 11 | `correct` | scored chunks | `text` corrected where accepted; `history` of attempts |
| 12 | `eval` | gold `truth/` + corrected eval chunks | `metrics.json`, `per_doc.jsonl`, `report.md` |
| 13 | `reconstruct` / `report` | corrected chunks | `.md` files / corpus quality before vs after |

## Config (`src/ocrfix/config.py`, `configs/`)

- **Files:** `default.yaml` (seed, work_dir, lang), then every `stages/*.yaml` (sorted), then
  `profiles/<profile>.yaml`, then `--set` dotlist. Stage files: `chunk` (chunk), `score` (score),
  `gold` (gold.split, gold.errors), `corrupt`, `calibrate`, `sft` (sft + the shared `train`),
  `dpo` (dpo + train.dpo), `correct` (correct incl. correct.hf and correct.checks), `eval`, `reconstruct`.
- **One place per setting:** a leaf key in two of default/stage files raises (`load_base`); a
  profile may only set `model` and `hw` (`PROFILE_KEYS`), else it raises naming the keys.
- **Servers:** `hw.servers` = list of groups `{model: base|trained, host, gpus: all|[..],
  memory?}` (base = `model`, trained = `correct.model`); `hw.infer` = `gpus_per_engine`,
  `ports: {base, trained}`, `max_model_len`, `max_num_seqs`, `concurrency`. `server_plan(cfg)`
  -> `EngineSpec`s in list order: one engine per `gpus_per_engine` GPUs of a group; port =
  `ports[model]` + k, counted per (host, model); memory = the group's `memory`, else
  `MEMORY_TOTAL` (0.88) / the number of groups on that host sharing its busiest GPU (two ->
  0.44). Old keys (`OLD_INFER`: engines, tensor_parallel_size, gpu_memory_utilization, host,
  base_port, correct_port) raise; `pipeline.both_memory` is gone.
- **Derived, never set twice:** `_derive_servers`: `hw.infer.score_urls` null -> every base
  engine's url (0.0.0.0 -> 127.0.0.1); `correct_urls` null -> every trained engine when
  `correct.model` differs from the base and no adapter, else score_urls; `concurrency` null ->
  max(len urls) x max_num_seqs. An explicit url list is an advanced override (load balancer,
  test fakes), never in profiles. Relative `model`,
  `correct.model`, `correct.adapter` resolve against cwd then the project root. The tokenizer and
  scorer are `model`. Served names (`served_name`): the base is `base`, `correct.adapter` is
  `adapter`, a `correct.model` other than the base is `tuned`; `ocrfix serve` uses the same, so
  no name settings exist. `correct_model(cfg)` = correct.model or the base. One `seed` (splits,
  damage, sampling, request seed).
- **`ocrfix serve [--only base|trained] [--host ADDR] [--dry-run] [-- vllm args]`**
  (`serve.py`): every `server_plan` engine whose host is this machine (`local_specs`: loopback
  or `my_addresses()`, plus `--host` when detection fails), both kinds in one command; one
  `vllm serve` per engine (`CUDA_VISIBLE_DEVICES` = its GPUs, `--served-model-name`, host,
  port, dtype, tp = len(gpus), max_model_len, the engine's memory, max_num_seqs, prefix
  caching, seed, `--attention-backend` from `hw.vllm_attention_backend` or
  `OCRFIX_ATTENTION_BACKEND`); base engines get `correct.adapter` as `--enable-lora
  --lora-modules adapter=<dir> --max-lora-rank <adapter r>`; trained engines serve
  `correct.model` as `tuned`. No `correct.model`: trained groups skipped with a note
  (`--only trained`: error); no local group: error naming the hosts. Several machines = the
  same profile on each, `./ocrfix serve` on each (`cluster.yaml.example`). Offline env vars
  passed on; logs `<work_dir>/vllm/<name>-<port>.log`; waits for every `/health`, then until
  one exits or Ctrl-C/SIGTERM, then stops all.
- **Scripts are Python:** `scripts/make_wheelhouse.py`, `install_offline.py`,
  `build_causal_conv1d.py` (stdlib only), `make_test_corpus.py`. `ocrfix train` re-execs under
  torchrun when `hw.n_gpus > 1` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Pipeline (`src/ocrfix/pipeline.py`, `configs/stages/pipeline.yaml`)

**`ocrfix pipeline <stage> [--dry-run] [--redo] [--from N]`**: `plan(stage, cfg)` lists the
README's commands for `prepare | calibrate | train | corpus` (`all` = the four) as `Step`s
(args, `needs` = cpu / base / both / free, `done` marker path, step `sets`, `tuned` model).
Each step runs as `python -X utf8 -m ocrfix ...` with the pipeline's `--profile` / `--set`
then the step's own sets (so they win); output shown and logged to
`<pipeline.data_dir>/pipeline_logs/NN-name.log`. A step with an existing `done` file is
skipped (`--redo` ignores that); resumable commands (markup, chunk, score, correct) always
run. `Servers.need(step)`: with vllm backends starts this machine's groups as subprocesses,
`serve --only base` and `serve --only trained --set correct.model=<step.tuned>`, and waits
for every url of that kind's `/health`; never starts other machines' groups (one of them
down -> stops, saying to run `./ocrfix serve --only <kind>` there). No memory restarts
(memory is per group from `server_plan`); a different tuned model restarts trained; `free`
(training) stops all (warns about healthy local engines it didn't start); engines already
healthy before are used, never stopped. No
servers with transformers backends. Tested on CPU end to end (tiny model, transformers);
the vLLM start/stop sequence only by unit test (`tests/test_pipeline.py`).

## Markup (`src/ocrfix/markup/`)

**`ocrfix markup <input_dir> <out_dir> [--limit N] [--workers N]`**: rule-based, CPU-only
repair of MinerU's LaTeX and tables, run on the corpus before `chunk` (`configs/stages/markup.yaml`
toggles each part). Writes the same tree into `out_dir` (never in place; unchanged files byte
for byte, BOM and CRLF kept, non-UTF-8 copied and listed), `markup_report.jsonl` (changed or
flagged docs: fix counts, issues with a snippet) and `markup_stats.json`.
- **Never worse than the input:** each repaired formula / table is checked before use
  (`latex.check_formula`, `tables._check_table`): the tagger must find exactly one whole math /
  table span, braces must balance, and no text may be lost (only spaces, env names of a renamed
  `\end`, and the stray `}` the repair dropped). Otherwise the original is kept byte for byte,
  counted as `latex.kept_original` / `table.kept_original` and reported with a reason.
- **Big corpora:** one process per core (`markup.workers`), files handed out in batches;
  resumable (an existing output file is complete, so a rerun skips it and appends to the
  report; delete `out_dir` to redo). Test corpus, 12 cores: 10 MB in 2.5 s with 1 process,
  1.1 s with 12 (mostly start-up at that size).
- **Order** (`run.fix_document`): HTML tables, pipe tables, math delimiters outside tables,
  then each math span the tagger finds; finally an unclosed `<table>` or a `$` left in prose is
  reported.
- **Tables** (`tables.py`): HTML rebuilt from its tags (missing `</td> </tr> </table>`, cells
  outside rows), short rows padded (colspan/rowspan counted), empty rows dropped, wider rows
  reported, nested tables left alone; pipe tables get a missing/malformed `|---|` line and
  rows padded or trimmed (only empty extra cells); `|` in `$..$`, code or `\|` never splits.
- **Delimiters** (`latex.repair_delimiters`): top-level `\begin{env}` closed at its paragraph
  end, a mismatched `\end` renamed; blank lines inside `$$..$$` / `\[..\]` (the tagger never
  lets these cross a blank line, so the formula would stay unmasked prose; also a LaTeX error)
  collapsed to single newlines by `join_display_blank_lines` when every blank-line-separated
  piece in between is `_math_only` (non-empty, no Hangul, no `$`, no run of 3+ words of 2+
  letters outside `\text{..}`-like groups; `$$` paired in order, a failed pair shifts by one),
  counted `latex.display_blank_lines_joined`; then `$$` / `\[` left open closed at the paragraph end, `\(` at
  the line end, only when the rest looks like math (no Hangul, some LaTeX: `looks_math`);
  `$ x $` tightened when the content looks like math; orphan closers only reported.
- **Formulas** (`latex.fix_formula`): inner envs, `\left`/`\right` (`\left.`/`\right.`) and
  braces balanced; spacing (`x _ { i }` -> `x_{i}`; spaces before `_ ^ { }` and after
  `_ ^ {` dropped, never in `\text{..}`-like groups, never after a lone `\`); `\(..\)` ->
  `$..$`, `\[..\]` -> `$$..$$`.
- **Measured** (scratch test, 20 test-corpus docs with injected MinerU-style damage: 791
  spaced formulas, 18 dropped braces, 28 lost `\end`, 40 lost `</td>`, 23 lost `</table>`,
  28 lost `|---|`): math spans tagged 2,893 / damaged 2,846 / fixed 2,893, tables 166 / 79 /
  166; on the clean corpus 0 of 167 docs changed.

## Chunk (`src/ocrfix/chunks/`)

Phase 1 of the chunk-based redesign: the 512-768 token chunk becomes the unit that is scored,
corrected and re-scored.

- **`ocrfix chunk <input_dir> <out_dir> [--limit DOCS] [--sample-frac F] [--workers N]`**: every
  `chunk.glob` file under the tree -> `part-NNNNN.jsonl[.gz]` (`chunk.gzip`), one worker
  process per shard of `chunk.shard_bytes` input (processes: tagging is pure Python).
- **Masking:** as elsewhere (`corpus/protect.py`), numbered per chunk; a literal placeholder in
  the source becomes a `⟨P#⟩`. Masked items are never cut.
- **Size:** `chunk.min_tokens`..`max_tokens` on the masked text, counted exactly per chunk with
  `model`. Cuts prefer blank line > sentence end > line end > whitespace > token
  boundary. Only a document's end can be short (`short: true`); the last cut is moved to avoid
  that when the remainder allows it.
- **Row fields:** see the `chunks/run.py` docstring. `text` is null until a chunk is corrected.
- **Resume:** `manifest.json` records settings and parts; finished shards are skipped; other
  files or settings in the same folder are refused. Files that aren't UTF-8 are listed in
  `stats.json` (`errors`) and skipped.
- **`ocrfix reconstruct <chunks> [out_dir] [--source original|current]`:** `original` is
  byte-exact (BOM, CRLF and non-NFC bytes via `raw`) and checked against `doc_sha256`; without
  `out_dir` it only checks. `current` applies corrections and drops a nearly empty document
  (`chunks.rebuild.drop_reason`: letters/digits `[^\W_]` of the corrected text <
  `reconstruct.min_letters` (300) or < `min_kept_share` (0.1) of the original's; 0 = off;
  never a document with a table `⟨T#⟩`): not written (an old copy is deleted), listed in
  `<out_dir>/dropped_docs.jsonl`, `st.dropped`.

## Stages: score, correct, corrupt, data, eval (`src/ocrfix/stages/`)

Score, correct (checks + re-score), damage, training data, DPO pairs, eval, compare and
report, all on chunks. Input and output are chunk shards; each input shard gives one output
shard of the same name (`.tmp` + rename; a rerun skips finished
shards; `--limit` counts shards). Details in the `stages/run.py` docstring.

- **`ocrfix score <chunks> <out_dir>`:** tokenizes the current masked text (`text`, else
  `source_text`) with `model`; a token is prose iff it overlaps no placeholder, and
  the first `score.boundary` (2) prose tokens after a placeholder and at the chunk start are
  not scored. Writes `score` (features as in Score, plus `n_tokens`), `flag` (calibration, else
  null) and `status: scored` (or `score_error` + `error`).
- **Backend choice:** `score.backend` / `correct.backend` default `auto`, resolved by
  `config._resolve_backends` after `_derive_servers`: `vllm` if the derived `score_urls` /
  `correct_urls` are non-empty (a group in `hw.servers`), else `transformers` (the cpu
  profile, `servers: []`). One engine per machine, so scores stay comparable; `--set` forces either.
- **Backends** (`score.backend`): `vllm` = POST `/v1/completions` with prompt = `[eos] + ids`,
  `prompt_logprobs: 0`, `max_tokens: 1` (`score.parse_prompt_logprobs`; vLLM skips the prefix
  cache for such requests), round-robin over `hw.infer.score_urls`, model `base`
  (the base, never the adapter); `transformers` = `scorer.model.score_sequences` in-process.
  The start token is the model config's EOS.
- **`ocrfix correct <scored> <out_dir> [--all]`:** sends a chunk when `status: scored`,
  not `low_confidence`, and flagged (stored `flag`, else the calibration now; neither → error
  unless `--all`). Per chunk, concurrently: chat request with `stages.prompt.SYSTEM` + the
  whole masked chunk (no markers; `max_tokens` = 1.3 × n_tokens + 16; greedy, thinking off) to
  `hw.infer.correct_urls` → `parse_answer` (drops `<think>`, strips) → `text_checks`
  (`placeholders` via `restore`, then on restored text `markers` and `language`, on masked text
  `length_ratio`, `edit_ratio`) → re-score → `check_scorer` → `accepted` (text and
  score replaced) / `rejected` / `unchanged` / `error`. Each attempt is appended to `history`
  (`model, candidate, checks, score_before, score_after, accepted` or `error`).
- **`ocrfix summary <dir>`:** status counts, accept rate, failed checks, per language,
  mean bpc gain of accepted chunks. `<out_dir>/score_stats.json` / `correct_stats.json` hold
  run counts and tokens/s.
- **`ocrfix report <dir> [--clean scored_clean] [--calibration yaml] [--json f]`**
  (`stages/report.py`): corpus quality per language at three stages, all from one folder: raw
  (the denoise entry's `score_before`), denoised (the first correct entry's `score_before`),
  final (`score`); a missing stage falls back to the next. Per stage: bpc weighted by chars
  (sum bpc x n_chars / sum n_chars) and the share of chunks the calibration flags (recomputed
  per stage). With `--clean` (clean text scored with `score.noise.remove=false`), its bpc and
  flag rate, and the share of the raw bpc excess over it closed by denoise and by correct.
  Plus noise chunks/chars and sent/accepted. On the test data the scale matched the real gain
  (doc CER 0.58% -> 0.23% after noise removal; bpc excess closed 42%, flags 46% -> 13% vs
  5% clean), but the real measure is gold.
- **Whole-chunk noise** (`score.noise.whole_chunks`, needs a calibration with `noise_chunk`):
  after word-level removal and re-scoring, a chunk whose bpc is above
  `cal.noise_chunk_threshold(lang)` (`*` for other languages; never `low_confidence`) gets
  `text` = its placeholders only (joined by spaces), `status: noise`, `flag: null`, history
  `{stage: noise_chunk, bpc, threshold}` (`stages.run._drop_noise_chunk`); `report` counts
  them. `calibrate` sets a first limit per language with >= `min_rows` rows (and `*`): max bpc
  of clean and damaged rows + `calibrate.noise_chunk.margin` (1.0); `calibrate-noise-chunk`
  replaces it from gold (below). Test data (4B): en 3.92, ko
  5.10, mixed 4.17; 0 of 13,614 OCR chunks above (max 3.46); 60/60 all-junk chunks dropped
  (bpc after word removal 5.4-10.4), 60/60 half-junk and 60/60 real kept.
- **Noise removal** (`stages/noise.py`, in `score` when `score.noise.remove`; a word with 2+ Han/kana
  chars making up at least half of it is never noise, so Chinese/Japanese text is kept; measured
  on 25k Chinese / 23k Japanese chars of real local text (locale messages, GNOME help), 4B: per
  word bits/char median 3.4 / 2.1, p90 5.6 / 3.8; without the protection the current values
  would remove 16.8% of Chinese and 2.2% of Japanese chars, the fitted 9.0/5.0 values 11.2% /
  1.4%): per-token
  bits are spread over each token's chars (placeholders and unscored tokens: NaN); each
  whitespace word gets its mean bits per char and a level by its script (`hangul` if half its
  chars are Hangul, else `other`): above `core` (8.0 / 5.0) starts a run, above `extend`
  (6.0 / 3.5) joins a neighbouring run; a word with a placeholder ends a run, an unscored one
  joins only if it has no Latin/Hangul letter or digit. Runs of `>= min_chars` (20) non-space
  chars are removed (`remove_spans`: the surrounding whitespace becomes one separator) and the
  cleaned text is re-scored. History entry `{stage: denoise, removed: [{start, end,
  bits_per_char, text}], score_before}`; no prose left -> `status: noise` (`correct`
  skips it). Per-token surprise alone does not separate noise (odd symbols are 2-3 byte tokens,
  the later ones predictable); bits per char does. Measured with vLLM on the 4B (24-char
  windows, p99.9 of text far from noise: Latin 3.7, Hangul/digits/punct 5.8; noise: Latin junk
  ~6, symbols ~12, garbled Hangul ~14). The earlier rule (24-char windows, one threshold per
  chunk language) cut Korean text in `mixed` chunks and text right after noise (polluted
  context); on 3,000 test chunks the word rule takes precision 0.79 -> 0.96, recall 0.86 ->
  0.74 (misses: mostly the generator's repeated junk, which the model learns in context).
  `min_chars` 12 -> 20: wrongly removed chars 277 -> 154 (by hand, one real false removal
  left), noise found 0.74 -> 0.58. The gap to the model's top guess instead of the surprise
  was measured too: no better for noise, worse for flagging (ko recall at r0.01 0.68 -> 0.45).
  Score calibration inputs with `score.noise.remove=false`. A random model (the tiny test
  model) finds everything noisy, so tests using it turn removal off.
- **`ocrfix gold chunks <gold_dir> <out_dir>`** (`gold/chunks.py`): real gold is `.md` pairs,
  `gold/ocr/**.md` (MinerU, unedited) + `gold/truth/**.md` (corrected;
  guideline `docs/annotation_guideline.md`). Only ocr/ is chunked (`run_chunk` into
  `<out>/chunks`, rebuilt each run); `TruthMap` (Levenshtein opcodes, adjacent non-equal ops
  merged) maps each cut to the truth: insertions at a chunk's start/end go to that chunk, a
  deletion across a cut vanishes, a replacement across a cut -> `gold_status: cut_in_edit`
  (also when non-space truth text falls between chunks). The truth piece (whitespace-trimmed) is
  masked from a whole-document tagging (`render`); placeholders must equal the OCR chunk's
  (`placeholders_changed`), no text -> `empty`. `gold_suffix` = the truth's whitespace after the
  chunk; `suffix_changed` counts cuts at a line break the truth joined (unfixable by a chunk
  corrector). Split by doc hash (`gold.split` eval 0.2 /
  dpo 0.2 / rest sft), pinned by `split.json` (`{eval,dpo,sft}_docs`; `the gold eval split`
  reads `eval_docs`). Writes `{sft,dpo,eval}/part-00000.jsonl` (complete documents: chunk rows
  + `gold_text, gold_placeholders, gold_status, gold_suffix, split`) and `stats.json`. `errors table` reads the sft dir
  (only `gold_status: ok`).
- **Gold in `data sft` / `data dpo`:** `--gold` = `gold chunks` rows, input `source_text`,
  target `gold_text`, only `ok` rows, never `split: eval`; no junk added;
  `sft.gold_repeats` (2) copies with ids `<id>#k` (SFT only; DPO uses 1). Default mix
  synthetic 0.6 / gold 0.2 / identity 0.2, renormalized over the sources given.
- **`ocrfix eval <truth_dir> <name=dir>... <out_dir>`** (`stages/eval.py`): documents
  complete in every system dir and with a truth file, rebuilt (`current`; the first system's
  `original` is system `ocr`), normalized like the chunker, compared whole: CER, WER, SER,
  false-fix rate (ocr->output regions touching no ocr->truth region), recall per
  `diff_ops` type (no output->truth residual region touches it); overall and per doc language (most chunk chars); paired Wilcoxon on
  per-doc CER, Holm. `metrics.json`, `per_doc.jsonl`, `report.md`. The step-by-step real-data
  workflow is `docs/real_data.md`.
- **Test data noise:** `make_test_corpus.py --noise-docs 0.3` inserts noise runs after sentence
  ends and whole noise paragraphs into `ocr/` only, logging each string in `ocr_errors.jsonl`
  (`noise`), so detection can be measured against it.
- **`ocrfix corrupt <chunks> <error_table.json> <confusion_map.json> <out.jsonl> [--rate R]
  [--n-variants N]`** (`stages/corrupt.py`): per chunk and variant, restore the placeholders,
  damage with `corrupt.corrupt_text` (prose only: every masked item, plus the whitespace on each
  side, is a block, so words never get glued onto a URL/citation), mask again with the chunk
  rule. Rows: `chunk_id = <id>~<r0.01|table>~v<n>`, `source_text` (damaged, masked),
  `placeholders`, `gold_text`, `gold_placeholders`, `corruption`, `variant`, `error_rate`,
  `applied_ops`, `status: new`; `score` takes them. Deterministic per seed. On the test
  data 5,999/6,000 keep their placeholders (the rest: a DOI whose `https://` lost its space).
- **`ocrfix data sft <out_dir> [--synthetic corrupt.jsonl]... [--identity chunks]...`**
  (`stages/data.py`): chunk SFT examples in the exact `correct` form: `messages` =
  `stages.prompt.messages(input)` + assistant (target; the template adds the empty think block), for
  `train sft`. `synthetic` = `corrupt` rows, `source_text` -> `gold_text`; dropped
  (`placeholders_changed`) unless `placeholders == gold_placeholders`. A share
  (`sft.junk.prob` 0.3) also gets 1..3 junk runs (3-30 non-space chars: symbol blocks,
  `lI1`-like Latin, random Hangul; never `$ | \` < > [ ]` or placeholder brackets) after a
  sentence end or as their own paragraph; the target leaves them out (`junk` lists them).
  The alphabets overlap `make_test_corpus.py`'s noise, so junk removal on the test set is
  optimistic. `identity` = clean chunks,
  target = input. Quotas `sft.mix` (0.75 / 0.25) x `lang_balance` via
  `data.common.quotas`; eval-split docs skipped; split by doc (`sft.val_frac`). Two passes
  (count ids, then write), so memory holds ids only. Rows `{id, source, lang, doc_id,
  chunk_id, placeholders, messages, corruption?, error_rate?, junk?}`. Test data: 3,998 examples, rendered
  max 2,025 tokens with the 4B template (rtx5090 `max_seq_len` 2048).
- **`ocrfix data dpo <out.jsonl> [--synthetic f]... [--identity f]... [--exclude sft.jsonl]...`**
  (`stages/dpo.py`): prompts made as `data sft` (same junk; `empty_think` forced off since
  chosen is compared with raw answers); prompts whose input or target is in an `--exclude`
  file are skipped (the SFT model knows those). The served model (`the correction model`, else
  the adapter if `correct.adapter`, else the base) answers each prompt greedily (`stages.prompt`
  body) and `n_samples` times (`temperature`, `top_p`, seed); answers cached in
  `<stem>.responses.jsonl` (`llm.client.run_requests`). Chosen is exact, so every distinct
  answer != chosen is rejected, typed by `classify`: placeholders (restore fails or
  placeholder list differs), truncated, false_fix (an edit touching no input->chosen region,
  1-char slack; identity: any edit), missed (a damaged region no edit touches), mixed (both),
  wrong_fix (neither). Up to `max_pairs` (2): greedy first, then sampled by distance to
  chosen. Prompts the model always got right: a `constructed_share` (0.25) of them get one
  off-policy pair, kinds `constructed` in turn (unchanged, half_fixed, placeholder_dropped).
  Rows TRL conversational + `rejected_type` (`greedy:|sampled:|constructed:<kind>`), `source,
  lang, doc_id, chunk_id`; `<stem>.stats.json` (greedy_right, all_right, types).
- **First SFT/DPO on the 5090 (test data):** QLoRA SFT on `data sft` output (3,935 rows,
  micro-batch 8, 1 epoch, 64 min, 19.7 GiB peak). Doc CER on 71 OCR test docs (after noise
  removal 0.58%): base 0.23%, SFT flagged 0.13%, SFT `--all` 0.01% (no rejections; clean chunks
  come back unchanged). Remaining errors: destroyed Korean words replaced by plausible
  guesses. `data dpo` with the merged SFT model: greedy right
  on 1,299/1,300 unseen prompts, 10 on-policy pairs, so no DPO training on test data; a
  5-step `train dpo` check ran (328 pairs, 73 s/step, 11.8 GiB). vLLM 0.33 crashed (illegal
  memory access in `punica_gpu.update_metadata`) serving the adapter while base scoring
  requests shared batches: serve a merged model on a second engine instead (README).
- **Calibration on chunks:** `calibrate` reads `score` output (`calibrate.run.flat_scores`
  lifts the `score` dict to the top level). Workflow in README.md, "Calibration (chunks)".
- **Corrupt speed:** generators pick positions by rejection sampling (`ops._sample`: 64 random
  tries, then the exact candidate list; uniform either way) and `Paragraph.matches` caches each
  pattern's matches, so cost no longer grows with ops × text length (rate 0.1 on 1,500 chunks:
  >300 s -> 12 s).
- **transformers generator** (`llm/hf.py`, `correct.backend: transformers`): `HFTransport`, an
  httpx transport answering vLLM's `/v1/chat/completions` (and `/health`) in this process, so
  `correct` / `data dpo` send the same bodies. Renders with the tokenizer's chat
  template (`chat_template_kwargs`), left-pads concurrent requests (collected for `correct.hf.wait`)
  into `generate` batches of <= `correct.hf.batch_size` (32) sequences (halved and retried on
  out-of-memory) grouped by greedy / (temperature,
  top_p, n), stops at the generation config's EOS ids + `<|im_end|>` and at each request's own
  `max_tokens` (`finish_reason: length`), decodes skipping special tokens. Model `correct.model`
  (+ `correct.adapter` via peft, unmerged; serve a QLoRA adapter's `train merge` output to match
  vLLM); reuses the in-process scorer's model when it is the same checkpoint and no adapter
  (`run.generator_transport`). Tests (tiny model, fp32): greedy = a step-by-step argmax recompute
  (hybrid cache), batched = single (left padding), batching through `Client`, `correct`
  through it. Sampled answers never match vLLM token for token.
- **`ocrfix compare <chunks> <out.json> [--limit 200] [--no-score] [--no-generate]`**
  (`stages/compare.py`, needs `score.backend=vllm`): the first N chunks with prose, current text.
  Scoring: VLLMScorer (score_urls) vs TransformersScorer (model): |bits| per scored
  token (mean/p99/max), |bpc|/|p90|/|spike_frac| per chunk, flag agreement under the calibration
  (+ the disagreements), identical noise spans; tokens/s each. Generation: the greedy `correct` body to correct_urls vs the transformers transport: identical-answer share, char
  distance, completion tokens/s, truncations, first differing examples. `/v1/models` of both
  servers recorded. `pass`: flags >= 0.99, greedy identical >= 0.95. Test: a fake vLLM that is the
  same tiny model gives max |bits| < 1e-3, flags 1.0, identical answers 1.0.
- **Measured on the 5090 (4B, test data, vLLM 0.33 at 0.45 GPU memory, transformers bf16 SDPA
  without the optional kernels):** scoring 1,500 chunks: |bits| per token mean 0.033, p99 0.22
  (rare tokens up to 12.6), bpc per chunk mean 0.0011, max 0.0099; flags agree 99.6% (6
  borderline chunks), noise spans identical on all; vLLM 20.7k vs transformers 11.4k tokens/s.
  Greedy correction with the merged SFT model, 100 chunks the model edits: 100% identical
  answers; vLLM 3,272 vs transformers 380 completion tokens/s at batch 16 next to vLLM.
  transformers alone: batch 16 423, 32 738, 48 794 tokens/s (peak 13 / 18 / 23 GiB), so
  `correct.hf.batch_size` 32; ~4-8x slower than vLLM. Batch 48 next to a 15 GiB vLLM ran out of
  memory, hence the split-on-OOM retry in `HFGenerator.generate`.
- **Client:** `llm.client.Client.post` (generic retrying POST, also behind `complete`);
  `hw.infer.concurrency` requests in flight per kind.
- **Tests** (`tests/test_score_correct.py`) run against an httpx `MockTransport` fake vLLM and the
  tiny model for the transformers backend. Not yet run against a real vLLM.

## Text analysis (`src/ocrfix/corpus/`)

- **Files** (`files.py`): `doc_id` = first 16 hex of sha256 of the NFC relative POSIX path.
  Files are walked in sorted order (`chunk.glob`). `--sample-frac F` on `chunk`
  (`chunk.sample_frac`): keep a document iff sha256(seed, NFC relative path) as a fraction < F
  (deterministic, spread over the corpus, stable across reruns).
- **Tagger** (`tagger.py`): tables first (HTML `<table>` with nesting, case-insensitive; GFM
  pipe tables), then math only between tables, so `$` in a table is table. Math:
  `$..$`, `$$..$$`, `\(..\)`, `\[..\]` (never across a blank line), and
  `\begin{equation|align|gather|multline}[*]..\end{..}` (star must match; may cross blank
  lines). Odd backslash count before a delimiter = escaped. Inline `$` uses pandoc's rule
  plus: a `$` that can't close ends the attempt (so `$5 and $10` is prose). Spans tile the
  text; everything else is prose.
- **Language** (`lang.py`): letters in prose only (fallback: table text without tags; math never
  counts). Hangul syllables and Han chars weigh `lang.hangul_weight` (2). `cjk` if Han + kana
  >= `cjk_min` (0.5) of the letters; else Han/kana are dropped from the count (Hanja in Korean
  text) and `ko`/`en` if that share >= `dominant` (0.8), `mixed` if both >= `mixed_min` (0.1),
  else `other` (math-only, number-only, Cyrillic, ...). `correct` sends only `correct.langs`
  (`[ko, en, mixed]`); others pass through (counted as `skipped_lang`).
- **Masking** (`protect.py`, used by chunk): every math span `⟨M#⟩`, table `⟨T#⟩`, and the
  `chunk.protect` kinds in prose `⟨P#⟩`, numbered per chunk.

## Scoring model (`src/ocrfix/scorer/`)

- **Token masks** (`tokens.py`): a token is prose iff it overlaps no placeholder. The first
  `score.boundary` (2) prose tokens after a placeholder and at the chunk start are not scored.
  **Scored tokens = prose and not boundary.**
- **Model** (`model.py`, used by the transformers backend `score.backend=transformers` and
  `compare`): every sequence is `[EOS] + tokens`, so each token gets a surprisal.
  Plain torch, `hw.dtype`. `choose_mode` picks:
  - **packed** only if `hw.attn_impl == flash_attention_2`, `flash_attn` imports, and (Qwen3.5
    is hybrid) `fla` + `causal_conv1d` are installed. Rows of <= `hw.scorer.max_batch_tokens`,
    position_ids reset, `cu_seq_lens_*` (flash varlen and fla), `seq_idx` (causal-conv1d). All
    three are needed: without `seq_idx` the conv leaks 3 tokens across sequences, without
    `cu_seqlens` the recurrent state leaks.
  - **padded** otherwise: SDPA, sorted by length, right-padded, rows x width <=
    `max_batch_tokens`. Right padding is exact for causal attention and linear-attention layers.
  - Both give the same bits (tests: within 1e-3). NLL goes hidden states -> `lm_head` ->
    cross-entropy in chunks of `hw.scorer.logits_chunk` tokens (fp32 only per chunk).
- **Features** (`features.py`), per chunk over its scored tokens, in bits:
  - `n_prose_tokens`, and `n_chars` (chars covered by scored tokens);
  - `bpc, mean_bits, p50, p90, p99, max_bits`, and `max_pos` (a char offset into the
    **masked** text);
  - `spike_frac` (share > `score.tau`), plus `spike_frac@<t>` for each other
    `score.tau_candidates` value;
  - `low_confidence`: fewer than `score.min_scored_tokens` (10) scored tokens. Never
    flagged, and left out of calibration fitting.
- **Calibration:** if `score.calibration` (default `data/calibration/calibration.yaml`)
  exists, its tau replaces `score.tau` and each chunk gets `flag` (the per-language rule).
  Use `--set score.calibration=null` when scoring inputs for calibration itself.
- **Measured on the RTX 5090** (random weights, real Qwen3.5-9B shape, bf16, padded/SDPA, 6144
  tokens per forward, 1024-token logits chunks): weights 16.7 GiB, peak 19.7 GiB, ~6.4k
  tokens/s with plain transformers; with the optional kernels ~2.9k tokens/s (probably Triton
  re-tuning per batch shape), so scoring does not need them.

## Errors (`src/ocrfix/errors/`)

- **`ocrfix errors table <gold_chunks/sft> <out_dir>`** (only `gold_status: ok` rows). Never
  the eval side: the tables feed calibration and training data.
- **Compare form:** `source_text` vs `gold_text`, both masked. Kept `⟨M#⟩ ⟨T#⟩` match on both sides, so math and tables never
  produce ops.
- **Alignment** (`align.py`): rapidfuzz Levenshtein opcodes (difflib fallback). Adjacent
  non-equal opcodes are merged into one region, so `rn`→`m` is one op. Two pure deletions
  (or insertions) split by an equal run of at most 3 chars are joined when an equal-cost
  alignment allows it, so a long inserted header is one op, not two fragments. Each op keeps
  its src/gold positions and text.
- **Op types**, checked in this order: `header_footer_insert` (a dropped run of at least
  `gold.errors.header_footer_min_chars` (8) non-space chars, or a bare page number, at word
  boundaries), `hyphen_wrap`, `newline_in_sentence`, `missing_space`,
  `extra_space`, `ligature`, `quote_dash`, `hangul_garbled` (non-Hangul run → Hangul),
  `dropped_char`, `duplicated_char` (the deleted text repeats its neighbor),
  `char_confusion` (at most `gold.errors.confusion_max_len` (4) chars each side, no whitespace),
  and `other`.
- **Script:** spacing ops carry `script` (`ko` if Hangul is next to the edit, else `en` for
  Latin letters or digits, else `other`), and are reported as `missing_space.ko` etc.
- **error_table.json:** groups for each lang × stratum, each lang, and `*`. Each group has
  counts, `per_1000` (ops per 1000 gold chars, placeholders excluded), `spacing_per_1000`
  and a histogram of the per-chunk op rate. Also the top `gold.errors.top_k` (50)
  `char_confusions`, per lang and overall.
- **confusion_map.json:** `hangul_garbled` entries `{source_glyphs (TeX commands, $, braces
  and spaces stripped), gold, gold_jamo (jamo library), count, source_raw}`.
- **summary.md:** the same tables as markdown.

## Damage generator (`src/ocrfix/corrupt/`)

Used by `ocrfix corrupt` (stages/corrupt.py applies it to the restored chunk text,
then masks again; see the Refine section) for calibration inputs and synthetic training data.

- **RNG:** one stream per (chunk, variant): `sha256(seed, doc_id + tag, chunk_id, variant)`.
  Output is deterministic and independent of input order.
- **Rate:** each chunk draws a rate (ops per 1000 prose chars) from its language's
  histogram (`<lang>/*` group): a bin by count, uniform inside it, the open last bin spanning
  `[a, 2a)`. `corrupt.match_total` scales the draw so its mean equals `total_per_1000`;
  `rate_scale` multiplies on top. Op count = rate × prose chars / 1000, rounded
  stochastically. `--rate R`: a fixed R ops per prose char instead (tag `rR`, else `table`).
- **Op kinds:** drawn with weight = measured rate, with spacing types split by script
  (`missing_space.ko`). A kind with no place in the chunk is dropped for that chunk.
- **Generators** (`ops.py`), one per error type, built so `errors.diff_ops` reads each edit
  back as the same type:
  - Korean/English space insertion and deletion, and a space → `\n`.
  - `hyphen_wrap` at simulated line ends (`corrupt.line_width`): `pro-\nposed` or
    `pro- posed`, or a break after the hyphen of a compound.
  - `char_confusion` reversed from the measured pairs (by count).
  - `hangul_garbled` reversed from the confusion map. It is wrapped in `$...$` with the
    share of measured raw forms that were math (else `garbled_wrap_prob`), and adds a math
    span.
  - `header_footer_insert` (page number and/or a `corrupt.header_phrases` entry).
  - The rest: ligature glyph or dropped ligature, typographic↔ASCII quotes and dashes,
    dropped and duplicated chars and words, and junk symbols for `other`.
- **Safety:** edits touch prose only and keep a 2-char gap from each other, so math and
  tables are copied verbatim.
- **Ops:** `applied_ops` entries are `{type, script?, gold_start, gold_end, src_start, src_end,
  src, gold, wrapped?, pair?}`; the returned spans are the shifted clean spans plus the new
  garbled math spans (tests check they equal `tag_spans(source)`).

## Calibrate (`src/ocrfix/calibrate/`)

**Workflow** (docs/real_data.md step 4): `corrupt` clean chunks at the gold rates and
at fixed rates (`--rate 0.01`, `0.03`), `score` the clean and damaged chunks with
`--set score.calibration=null --set score.noise.remove=false`, then `ocrfix calibrate
clean_scored damaged_scored data/calibration`. `calibrate.run.flat_scores` lifts each chunk's
`score` dict to the top level.

**How it works:**
- **Rows:** `low_confidence` rows are dropped first.
- **Metrics:** AUROC clean vs corrupted for every feature (`bpc p90 p99 spike_frac
  spike_frac@t`), per language and per corruption label. Spearman correlation
  with `error_rate` (clean = 0). Ranks are averaged on ties, and NaN (null) is dropped.
- **tau:** the spike_frac column with the best pooled AUROC sets `tau`. `variant` is
  always `raw` (kept in the file format).
- **Rules** (`fit.fit_rule`) are fit per language with at least `calibrate.min_rows` clean
  and recall-set rows; other languages use `*`. The recall set is the `calibrate.recall_on`
  labels (default `table`). Thresholds are strict (`>`) and taken from clean values, so the
  false-flag rate is exact.
  - The rule is `spike_frac > A or p90 > B or bpc > C`; any part may be null (off).
  - First single features at the whole budget (`max_false_flag`): spike_frac, and bpc. If
    the better one reaches `target_recall`, done.
  - Otherwise a grid over spike_frac and bpc thresholds (each off, or a clean value at or
    above its single-feature threshold, up to 60 per axis); the budget left goes to the
    lowest `p90` threshold. Best recall wins, ties go to fewer false flags.
    `target_met: false` is recorded if even that misses.
  - Why bpc: per language it separates best (4B test chunks, AUROC en 0.97, ko 0.93 vs
    spike_frac 0.95 / 0.87). Pooled over languages it is poor (0.77, different baselines),
    which per-language rules avoid. On the test data, recall at 1 error / 100 chars went
    en 0.95 -> 1.0, ko 0.51 -> 0.71; document CER after correcting 4 shards 0.27% -> 0.23%.
- **Gold:** `--gold-scores` = `score` output of `gold chunks` sft/ (+ dpo/), noise removal
  off. `run.gold_rows` keeps `gold_status: ok`, never `split: eval` (counted in
  `gold_skipped`; none left -> error); `run.gold_cer` = levenshtein(source_text, gold_text) /
  len(gold_text) on the masked text (a `cer` field wins). Reports precision, recall and
  false-flag rate of the rules (noisy = `cer > calibrate.gold_noisy_cer`, 0) and
  `gold_bands`: share flagged per cer band (`GOLD_BANDS`: 0, 0-0.5%, 0.5-2%, 2-5%, >5%).
  Report only: the fitted rules don't change.
- **Outputs:**
  - `calibration.yaml`: `{version, tau, variant, rules: {lang|*: {spike_frac, p90, bpc,
    recall, false_flag, target_met, n_clean, n_corrupted}}}` (no `bpc` key = off: older files
    load unchanged). `rules.load_calibration(path).flag(feats,
    lang)` applies it in `score` and `correct`.
    Plus `noise_chunk: {lang|*: bpc}` (absent = never drop); `is_noise_chunk(feats, lang)`.
  - `report.json`, `summary.md`, and PNGs: `auroc.png`, `roc.png`, `by_corruption.png`,
    `rule.png`.

## Noise calibration (`src/ocrfix/calibrate/noise.py`)

**`ocrfix calibrate-noise <gold_chunks> [out_dir=data/calibration] [--no-write] [--limit N]`**:
fits `score.noise` core (per script) / extend / min_chars on the OCR side of `gold chunks` sft/ +
dpo/ (`gold_status: ok`; never eval/).
- **Labels** (`noise_labels`): Levenshtein opcodes OCR -> gold, adjacent edits merged into
  regions; a region whose gold side has no letter/digit was deleted by the
  annotator (REAL: page numbers, footers, junk) and is **ignored** (neither noise nor text);
  everything the truth keeps (as is or corrected) is **text**.
- **Noise = inserted junk only** (`insert_junk`): every chunk is also scored as a copy with
  `calibrate.noise.insert_runs` (1-3) runs of `insert_chars` (5-60) non-space chars (symbols,
  `lI1` Latin, random Hangul, loose jamo, digit/symbol table leftovers; no markdown/math/
  placeholder syntax) at whitespace, 20% as their own paragraph; `insert_chars` [20, 60] since
  only runs of 20+ are removed.
- **Scoring** with `score.noise.remove=false` through `stages.run.Context` (vLLM or
  transformers). **Search**: core hangul 6-14 x other 3.5-8 x margin 0.5-3 (extend = core -
  margin) x `calibrate.noise.min_chars` ([20, 25, 30, 40, 60]: long runs only);
  `removed_words` is the `noise_spans` rule vectorized over all words (`noise.word_values` /
  `level`; a test checks they agree exactly). Best recall (inserted junk removed) with
  text_removed (removed text chars / all text chars) <= `max_text_removed` (0.0001);
  `min_noise_chars` (2000) inserted chars needed.
- **Output:** `noise_report.json` (counts, the previous values' result, top 20, up to 30
  `text_removed_examples` as `context[[removed word]]context`) and, unless `--no-write`,
  `write_score_yaml` rewrites the three lines under `score.noise` in configs/stages/score.yaml
  plus a `# fitted by ...` comment, keeping every other line.
- **Test gold (4B, 697 chunks, 46,157 inserted noise chars, 2.7M text chars, 87 s):** previous
  8.0/5.0, 6.0/3.5, 20: 72.2% found, 0.0239% text removed; fitted 9.0/5.0, 8.0/4.0, 20: 59.9%
  found, 0.0096% text removed. The text still removed is mostly the first word after a junk
  run (the model's context is polluted there). Not written to score.yaml (test data).

## Whole-chunk noise limit (`src/ocrfix/calibrate/noise_chunk.py`)

**`ocrfix calibrate-noise-chunk <gold_chunks> <scored> [out_dir=data/calibration] [--no-write]
[--limit N]`**: workflow = corpus `score` pass 1 with `score.noise.remove=false
score.noise.whole_chunks=false`, this command, then `score` pass 2 with both on (new folder).
- **Gold:** sft/ + dpo/ `gold_status: ok` OCR text scored (noise off, `Context.feats`); a chunk
  with `deleted_share` (annotator-deleted share of non-space chars, `noise_labels` REAL) >=
  `calibrate.noise_chunk.deleted_share` (0.5) is noise, the rest real. limit = max real bpc +
  `margin` per lang in (en, ko, mixed) with >= `calibrate.min_rows` real rows, plus `*`.
- **Corpus:** `Corpus` streams `<scored>` once (status scored, not low_confidence; other langs
  -> `*`): 0.25-bpc histogram, chunks/chars dropped per candidate (gold max + `candidates`),
  reservoir `samples` (5) per 0.5 band from limit - 1 to limit + 4 (last band open).
- **Writes** `noise_chunk` into `score.calibration` (rest kept), `noise_chunk_report.json`,
  `noise_chunk_samples.md` (non-empty bands, kept/dropped), `noise_chunk_hist.png`
  (`plot_histograms`: per lang, corpus share bars + gold real-text step line per 0.25 bin, log y,
  gold noise ×, gold max dashed, limit solid). Test gold (4B, 697 chunks): en 2.04,
  ko 4.65, mixed 2.57; 0 of 13,614 test OCR chunks above; 58 s.

**`ocrfix bpc-hist <scored>... [--out bpc_hist.png] [--calibration yaml]`** (no model):
`bpc_histograms` (per lang: 0.25-bin counts, p50/p99/p99.9/max, `limit` = noise_chunk[lang]
else `*`, `above` exact) + `plot_bpc` (PNG, log y) + a text histogram on stdout.

## Checks (`src/ocrfix/verify/checks.py`)

What `correct` (`stages.run.text_checks` + `check_scorer`) requires of a correction, each
`{check, passed, reason}`:
- `placeholders`: every placeholder of the input comes back once, in order (`restore`); then on
  the restored text `markers`: no math/table span lost or added.
- `length_ratio` within `correct.checks.length_ratio`; `edit_ratio`: char edit distance / len(source)
  <= `max_edit_ratio` (both on the masked text).
- `language`: same `detect_lang`, and the Hangul share of prose letters moves by at most
  `max_script_shift`.
- `scorer` (the re-score): bpc(before) − bpc(after) >= `min_bpc_gain` and spike_frac(after) <=
  `spike_ratio` × before. With `rescore_failed: false`, a correction that already failed a text
  check is not re-scored.

## Train (`src/ocrfix/train/`)

**Commands:**
- `ocrfix train sft <train.jsonl> <out_dir> [--eval file] [--max-steps N]`
- `ocrfix train dpo <pairs.jsonl> <sft> <out_dir>` (`sft` = a LoRA adapter or a full model dir)
- `ocrfix train merge <adapter> <out_dir> [--base DIR] [--dequantize/--no-dequantize]`
- Launch through `ocrfix train <cmd> ... --profile P`; the CLI also re-execs under
  torchrun when `hw.n_gpus > 1`.

**Settings** (`common.resolve`), all from `hw.train`:
- `mode` is `qlora | lora | full` (`--set hw.train.mode=...`), for SFT and DPO alike. `full`
  = no LoRA: every weight trains. A 9B full fine-tune needs ~150 GB with Adam, so it is for
  the H200s (FSDP), not the 5090.
- `distributed: auto` gives DDP for LoRA on more than one GPU and FSDP for `full`
  (`hw.train.fsdp`, layer class `Qwen3_5DecoderLayer`).
- Micro-batch is `per_device_batch_size`; grad accumulation = ceil(`train.effective_batch` /
  (micro × world)).
- `max_seq_len`, `optim`, `gradient_checkpointing`, and `packing` (refused without the
  kernels).
- Profiles:
  - **rtx5090**: QLoRA (nf4 double-quant base, bf16 compute, LoRA r/alpha/targets from
    `train.lora`), `paged_adamw_8bit`, seq 2048, micro-batch 2, gradient checkpointing.
  - **h200x4**: bf16 LoRA, DDP ×4, packing, seq 8192; `--set hw.train.mode=full` gives FSDP
    full fine-tuning.
- LoRA targets missing from the model are skipped with a log line (Qwen3.5: attention, MLP,
  and the Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`).
- transformers 5 names: `warmup_steps` takes `train.warmup_ratio` (a float < 1 is a ratio),
  and `train_sampling_strategy=group_by_length`.

**Rendering** (`common.render`): the prompt is the tokenizer chat template with
`add_generation_prompt=True` and `enable_thinking = correct.enable_thinking`, the same text
vLLM serves. The target is the rest of the full rendering (assistant content +
end-of-turn), so TRL `completion_only_loss` puts the loss on assistant tokens only. A
mismatch, e.g. the data's empty think block versus the template, raises. Rows over
`max_seq_len` are dropped and counted, never truncated. TRL appends `eos_token` to a
completion that doesn't end with it, so `common.turn_end` finds the template's end-of-turn
token every completion ends with (`<|im_end|>`) and passes it as the EOS: SFTConfig
`eos_token`, and a copy of the tokenizer for DPO (the saved tokenizer is unchanged). No
`<|endoftext|>` is trained after `<|im_end|>`. Tests decode the labelled tokens of real SFT
batches: exactly the answer + `<|im_end|>`, one contiguous block, none of the prompt.

**SFT:** TRL `SFTTrainer` on prompt/completion strings. Eval is `--eval` or a held-out
slice (`train.eval_frac` by id hash, ≤ `max_eval`). Peak VRAM (allocated, per rank) is
logged at every log step and saved in `train_summary.json`. `ocrfix_train.json` records
mode, `quantized_base`, settings, row counts and metrics.

**DPO** (`dpo.build_policy`): the policy starts as the SFT result, in `hw.train.mode`.
- **qlora / lora on an SFT adapter:** the base it was trained on (`base_model` in its
  `ocrfix_train.json`; nf4 for qlora) + that adapter, trainable.
- **qlora / lora on a full SFT model dir:** that model + a new LoRA.
- **full on a full model dir:** all weights train.
- **full on an adapter:** refused with a hint to `train merge` first.
- **LoRA references** (`ref_model=None`, no second weight copy; TRL 1.17 copies the adapter
  into a frozen `ref` adapter):
  - `train.dpo.reference: base` deletes that copy, so the reference is the adapter-disabled
    base (the full SFT model when that is the base);
  - `sft` keeps it (the usual DPO reference).
- **Full references,** with `train.dpo.precompute_ref` (default), log-probs are computed once
  before training:
  - `sft` is the starting policy, with no copy kept (TRL loads one under FSDP for that pass);
  - `base` is `model`, loaded as a frozen `ref_model`.
- **Metadata:** `ocrfix_train.json` records the base actually used (`base_model`) and
  `sft_kind`.
- Uses `beta`, `loss_type`, `lr` and `epochs` (1) from `train.dpo`.
- **Not tested:** full DPO under FSDP; only single-process runs are.

**Merge:**
- For a qlora adapter (per its `ocrfix_train.json`), the base is loaded nf4 as in
  training, every `Linear4bit` is dequantized to bf16, then the adapter is merged. This
  reproduces the training-time model (tested); `--no-dequantize` merges into the original
  bf16 weights instead. Either way a warning says the merged weights differ slightly from
  the training-time model.
- The output is plain bf16 safetensors with no `quantization_config`, plus the tokenizer
  and generation config. vLLM 0.33 registers `Qwen3_5ForCausalLM`.
- The base is `--base`, else the `base_model` in the adapter's `ocrfix_train.json` (a LoRA
  DPO on a full SFT model merges into that model), else `model`.
- A full model dir (full SFT/DPO output) is refused: it needs no merge. Serve it with
  `correct.model`.

**Tests** (`tests/test_train.py`): on the tiny model, with a Qwen-style chat template added:
- 2-step SFT, LoRA and full;
- DPO on an adapter (both references), full DPO (both references), and a LoRA DPO on a full
  SFT model then merged;
- the loss mask, the end-of-turn handling, and a merge. The QLoRA SFT + dequantized merge runs
only with CUDA + bitsandbytes.

## Conventions

- **Stable IDs.** `doc_id` is a hash of the relative source path (stable across reruns);
  `chunk_id = <doc_id>-c<index>`; corrupt variants `<chunk_id>~<tag>~v<n>`.
- **Resumable.** Chunk, score and correct write each output shard via `.tmp` + rename and skip
  finished shards on a rerun; `data dpo` caches answers in `<stem>.responses.jsonl`.
- **`--profile` on every command** (default `$OCRFIX_PROFILE`, else `rtx5090`); `--limit`
  where it makes sense (documents for `chunk`, shards for score / correct).
- **No network access anywhere.** `import ocrfix` sets `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  VLLM_NO_USAGE_STATS=1` (and friends); `ocrfix serve` passes them to vLLM. All model and
  tokenizer paths come from config (`model`, `correct.model`) and point to local directories. Never
  pass a hub repo id. Never add code that downloads.
- **NFC on every input.** The chunker normalizes each document (no BOM, LF, NFC; `raw` keeps
  the original bytes for a byte-exact `reconstruct`). Korean OCR output often contains
  decomposed Hangul jamo.
- **Hardware differences live only in profiles.** Code reads `cfg.hw.*` (n_gpus, dtype,
  attn_impl, train.mode `qlora|lora|full`, batch sizes, seq lens, vLLM parallelism); it never
  branches on GPU model. `default.yaml` declares `hw: ???` so a profile is mandatory (config: default.yaml + stages/*.yaml + profile + --set). To add a
  machine, add a profile, don't add `if`s.
- **JSONL everywhere between stages**, optionally gzipped (`chunk.gzip: true`), sharded.
  Read with `io.iter_jsonl` (a file or a folder of shards); never load a whole stage into
  memory.

## Placeholders

Before text reaches a model (the scorer, the LLM, and the LLM's training data), non-prose
content is masked with one rule (`corpus/protect.py`), so the LLM cannot "fix" it and every
stage sees the same text:

| token   | meaning |
|---------|---------|
| `⟨M#⟩`  | math span (inline `$...$` or display `$$...$$`), `#` = 1-based index per kind within the chunk |
| `⟨T#⟩`  | HTML table (`<table>...</table>`) |
| `⟨P#⟩`  | other protected content: URLs, code, citations/refs, image links |

`⟨ ⟩` are U+27E8/U+27E9.
`corpus/placeholders.py`: `substitute(text, spans) -> (masked, mapping)` numbers math and
table spans in order of appearance (⟨M1⟩ ⟨M2⟩ ⟨T1⟩ ...; mapping is position-ordered);
`restore(masked, mapping)` raises `PlaceholderError` unless every placeholder appears exactly
once and in the original order. The `placeholders` check builds on this. Constants are in `schema.py`
(`MATH_PH`, `TABLE_PH`, `PROTECTED_PH`).

## Install (offline)

On an online machine with the same Python (3.12) and arch (x86_64):

```bash
python3 scripts/make_wheelhouse.py wheelhouse
```

Copy `wheelhouse/` plus the model directory named by your profile's `model` to the air-gapped
machine, then:

```bash
python3 scripts/install_offline.py wheelhouse .venv
```

`requirements.lock` is pinned with hashes, generated from `requirements.in` (command at the
top of that file). Update it only on an online machine.

### Optional Qwen3.5 kernels

Qwen3.5-9B is a hybrid model (Gated DeltaNet linear-attention + full-attention layers). By
default we install **plain transformers only**: it runs those layers with its pure-PyTorch
fallback (correct, slower; logs "falling back to its reference PyTorch implementation").
vLLM ships its own kernels and is unaffected.

The optional kernels, `flash-linear-attention` (`requirements-kernels.lock`, locked against the
main lock) and `causal-conv1d`, speed up training and cut memory, and are required for
`hw.train.packing: true` (the h200x4 profile sets it); `ocrfix train` refuses packing without
them at start-up. The rtx5090 profile trains unpacked with `train.group_by_length: true`. To
use the kernels:

```bash
python3 scripts/make_wheelhouse.py --with-kernels wheelhouse   # online
python3 scripts/install_offline.py --with-kernels wheelhouse   # air-gapped
```

`causal-conv1d` has no PyPI wheel for our torch, so `scripts/build_causal_conv1d.py` compiles it
(~10 min) with a pip nvcc pinned to torch's CUDA minor (13.0; nvcc 13.4 with 13.0 headers
fails). The wheel covers sm_90 and sm_120 and is checked by `SHA256SUMS`, not a lock. Rebuild it
whenever torch changes in the lock. A build for torch 2.14.0+cu130 is kept in `wheelhouse/`.

With `flash-linear-attention` installed, transformers uses its Triton kernels on every device,
so Qwen3.5 then **only runs on CUDA**, and the first forward pass JIT-compiles (~1 min, cached).

## Running

The full sequence on real data is `docs/real_data.md`. In short:

```bash
ocrfix serve                                                             # vLLM engines (hw.infer)
ocrfix chunk data/real/corpus data/real/chunks --profile rtx5090
ocrfix score data/real/chunks data/real/scored --profile rtx5090
ocrfix correct data/real/scored data/real/refined --profile rtx5090
ocrfix reconstruct data/real/refined data/real/corrected --source current
ocrfix train sft data/real/sft/train.jsonl runs/sft --profile h200x4   # torchrun, hw.n_gpus procs
ocrfix train merge runs/sft runs/sft_merged --profile rtx5090
```

## Tests and lint

```bash
pytest
ruff check . && ruff format --check .
```

Tests must never touch the network: `tests/conftest.py` forces offline env vars and makes any
non-loopback socket connection fail. Tests that need a model use the `tiny_model_dir` fixture and put it on `model_device`
(`cuda` if available, else `cpu`; skipped on CPU when the optional kernels are installed). `tiny_model_dir` is the fixture
which builds a tiny randomly initialized Qwen3.5 (`qwen3_5_text`, one linear-attention and
one full-attention layer) and a BPE tokenizer trained on `tests/fixtures/sample_mineru/` (skipped if
`transformers` is not installed); never reference a real checkpoint in tests.
