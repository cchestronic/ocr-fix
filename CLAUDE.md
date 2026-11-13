# ocrfix

Corrects OCR errors in MinerU markdown output (English and Korean, with LaTeX math and
HTML tables) using a locally hosted Qwen3.5-9B. Runs on an **air-gapped** network: today
one RTX 5090 (32 GB, Blackwell sm_120), later 4x H200.

Models: `paths.base_model` is set per profile (rtx5090: `Qwen3.5-4B` in the project folder; h200x4:
`/models/Qwen3.5-9B`). A relative path resolves against the current folder, then the project root.

Status: every stage command works: `chunk`, `reconstruct`, `refine score|correct|corrupt|data|dpo|eval|compare|summary|report`, `ingest`, `score`, `gold chunks|sample|import|records`, `errors table`,
`corrupt`, `calibrate`, `fix` (rule-based fixers), `correct` (LLM), `verify`, `verify-summary`,
`data sft|dpo|stats`, `train sft|dpo|merge`, `eval`, `pipeline` and `config`.
`docs/runbook.md` is the operator guide (install, sample run, gold, calibration, training,
eval, the H200 schedule, adding a profile).

## Layout

```
configs/default.yaml           base config (no hardware settings)
configs/profiles/*.yaml        per machine: `paths.base_model` (4B on rtx5090, 9B on h200x4) + the `hw` block
configs/pipeline/*.yaml        `ocrfix pipeline` runs: smoke (fixtures, echo LLM), sample (2%), full
src/ocrfix/cli.py              typer app; one command per stage
src/ocrfix/schema.py           Record / Span, placeholder constants, nfc()
src/ocrfix/io.py               streaming sharded JSONL I/O, read_ids()
src/ocrfix/config.py           load_config(profile, overrides)
src/ocrfix/offline.py          enforce_offline() env vars
src/ocrfix/chunks/             chunk: split.py (one document -> chunks), run.py (parallel, shards), rebuild.py (reconstruct)
src/ocrfix/refine/             refine: score.py (masks, features, vLLM/transformers backends), prompt.py, run.py, report.py,
                               data.py (chunk SFT examples), dpo.py (chunk DPO pairs), eval.py (docs vs truth),
                               compare.py (vLLM vs transformers)
src/ocrfix/corpus/             ingest: tagger.py (spans), lang.py, placeholders.py, ingest.py,
                               protect.py (the one masking rule: score, correct, data)
src/ocrfix/scorer/             score: tokens.py (masking, windows), model.py (batching, NLL), features.py, run.py
src/ocrfix/gold/               gold: chunks.py (md pairs -> chunk gold), features.py, sample.py, labelstudio.py (+ labeling_config.xml),
                               validate.py, importer.py
src/ocrfix/errors/             errors: align.py (alignment + op types), table.py (rates, maps, summary)
src/ocrfix/corrupt/            corrupt: ops.py (one generator per error type), run.py (rates, spans, IO)
src/ocrfix/calibrate/          calibrate: fit.py (AUROC, rule search), rules.py (calibration.yaml), run.py, report.py
src/ocrfix/fixers/             fix: base.py (Fixer, registry, Op, span-safe edits), unicode.py, line_join.py,
                               layout_junk.py, run.py
src/ocrfix/verify/             verify: checks.py (guardrails), run.py (verify, summary)
src/ocrfix/data/               data: common.py (pairs -> chat), sources.py, sft.py, dpo.py, run.py
src/ocrfix/train/              train: common.py (settings, loading, chat render), sft.py, dpo.py, merge.py
src/ocrfix/editdist.py         Levenshtein (rapidfuzz if installed)
src/ocrfix/pipeline.py         `ocrfix pipeline`: stage subprocesses, resume markers, LLM server lifecycle
src/ocrfix/eval/               eval: align.py (doc texts, position map), metrics.py, stats.py (Wilcoxon, Holm),
                               run.py, report.py
src/ocrfix/llm/                correct: protect.py (⟨P#⟩ masking), prompt.py, client.py (vLLM HTTP), hf.py (transformers behind the same API), parse.py, correct.py,
                               echo_server.py (stdlib stand-in for vLLM, smoke runs)
scripts/                       env.sh, launchers (serve_vllm.sh, train.sh), offline install
tests/fixtures/                small JSONL / markdown samples
docs/annotation_guideline.md   gold .md pairs: what to fix, 〈?〉, checks, examples
docs/annotation_guideline_paragraphs.md   the older paragraph tasks (gold sample/import)
docs/real_data.md              real data step by step: gold chunks -> calibration -> SFT/DPO -> eval -> corpus
docs/runbook.md                operator guide for the 5090 and 4x H200 tracks
```

## Pipeline

All paths are relative to `paths.data_dir` (default `data/`). Each output is a directory of
shards `part-NNNNN.jsonl[.gz]`. "Record" means the shared schema below.

| # | command     | module    | input(s)                         | output                | output contents |
|---|-------------|-----------|----------------------------------|-----------------------|-----------------|
| 1 | `ingest`    | corpus    | `<input_dir>` (MinerU `*.md`)    | `<out.jsonl>`         | Record per paragraph, spans prose/math/table |
| 2 | `score`     | scorer    | `<paragraphs.jsonl>`             | `<scores.jsonl>`      | `{doc_id, para_id}` + surprisal features (no text) |
| 3 | `gold sample` / `gold import` | gold | `scores.jsonl`, `corpus.jsonl` / annotated JSONL | `<out.jsonl>` / `gold/` | rows to annotate / `gold_train.jsonl`, `gold_eval.jsonl` |
| 4 | `errors table` | errors | `gold/gold_train.jsonl` [`--fixed`] | `<out_dir>`        | `error_table.json`, `confusion_map.json`, `summary.md` |
| 5 | `calibrate` | calibrate | clean scores, corrupted scores [`--gold-scores`] | `<out_dir>`  | `calibration.yaml` (tau, flag rules per lang), PNGs, `summary.md` |
| 6 | `corrupt`   | corrupt   | clean JSONL, `error_table.json`, `confusion_map.json` | `<out.jsonl>`  | `source_text` (noisy), `gold_text` (clean), `spans`, `applied_ops` |
| 7 | `data sft` / `data dpo` | data | corrupt, verify, gold, clean outputs | `<out_dir>/{train,val}.jsonl` / `<out.jsonl>` | chat `messages` + `source`, `lang` / `prompt, chosen, rejected, rejected_type` |
| 8 | `train sft` / `dpo` / `merge` | train | `train.jsonl` / dpo pairs + SFT result / adapter | `<out_dir>` | LoRA adapter or full model (+ `ocrfix_train.json`) / merged bf16 checkpoint |
| 9 | `fix`       | fixers    | `<paragraphs.jsonl>`             | `<out.jsonl>` + `--ops` | Record with fixed `text` + `original_text`, `fixer`; ops log |
| 9b| `correct`   | llm       | `<paragraphs.jsonl>`, `<scores.jsonl>` | `<out.jsonl>` + responses | flagged paragraphs corrected by the LLM; `llm_error` on failures |
|10 | `verify`    | verify    | `<corrections.jsonl>`, `<scores_before.jsonl>` | `<out.jsonl>` | Record + `accepted`, `failed_checks`, `before`/`after` |
|11 | `eval`      | eval      | `gold_eval.jsonl`, system outputs | `<out_dir>`          | `metrics.json`, `per_doc.jsonl`, `paragraphs.jsonl`, `report.md` |

Every command takes `--in` (repeatable), `--out`, `--limit N`, `--profile NAME`, and
`--set key=value` (repeatable OmegaConf override). `ingest` instead takes positional
`<input_dir> <out.jsonl[.gz]>`, and its `--limit` counts documents; `score` takes positional
`<paragraphs.jsonl> <scores.jsonl>` plus `--dump-tokens DIR`; `gold sample`, `gold import`,
`errors table`, `corrupt`, `calibrate`, `fix`, `correct`, `verify`, `data *`, `train *`, `eval` and `pipeline` take positional paths too (below). `ocrfix config --profile h200x4` prints
the resolved config; `--key hw.n_gpus` prints one value.

## Chunk (`src/ocrfix/chunks/`)

Phase 1 of the chunk-based redesign: the 512-768 token chunk becomes the unit that is scored,
corrected and re-scored. The later stages below still read paragraph Records until they move over.

- **`ocrfix chunk <input_dir> <out_dir> [--limit DOCS] [--sample-frac F] [--workers N]`**: every
  `corpus.input_glob` file under the tree -> `part-NNNNN.jsonl[.gz]` (`io.gzip`), one worker
  process per shard of `chunk.shard_bytes` input (processes: tagging is pure Python).
- **Masking:** as elsewhere (`corpus/protect.py`), numbered per chunk; a literal placeholder in
  the source becomes a `⟨P#⟩`. Masked items are never cut.
- **Size:** `chunk.min_tokens`..`max_tokens` on the masked text, counted exactly per chunk with
  `paths.tokenizer`. Cuts prefer blank line > sentence end > line end > whitespace > token
  boundary. Only a document's end can be short (`short: true`); the last cut is moved to avoid
  that when the remainder allows it.
- **Row fields:** see the `chunks/run.py` docstring. `text` is null until a chunk is corrected.
- **Resume:** `manifest.json` records settings and parts; finished shards are skipped; other
  files or settings in the same folder are refused. Files that aren't UTF-8 are listed in
  `stats.json` (`errors`) and skipped.
- **`ocrfix reconstruct <chunks> [out_dir] [--source original|current]`:** `original` is
  byte-exact (BOM, CRLF and non-NFC bytes via `raw`) and checked against `doc_sha256`; without
  `out_dir` it only checks. `current` applies corrections.

## Refine (`src/ocrfix/refine/`)

Chunk-based score / correct / verify (replaces the paragraph `score`, `correct`, `verify` once
calibration and training data move to chunks). Input and output are chunk shards; each input
shard gives one output shard of the same name (`.tmp` + rename; a rerun skips finished
shards; `--limit` counts shards). Details in the `refine/run.py` docstring.

- **`ocrfix refine score <chunks> <out_dir>`:** tokenizes the current masked text (`text`, else
  `source_text`) with `paths.tokenizer`; a token is prose iff it overlaps no placeholder, and
  the first `scorer.boundary` (2) prose tokens after a placeholder and at the chunk start are
  not scored. Writes `score` (features as in Score, plus `n_tokens`), `flag` (calibration, else
  null) and `status: scored` (or `score_error` + `error`).
- **Backends** (`refine.scorer`): `vllm` = POST `/v1/completions` with prompt = `[eos] + ids`,
  `prompt_logprobs: 0`, `max_tokens: 1` (`score.parse_prompt_logprobs`; vLLM skips the prefix
  cache for such requests), round-robin over `refine.score_urls`, model `refine.score_model`
  (the base, never the adapter); `transformers` = `scorer.model.score_sequences` in-process.
  The start token is the model config's EOS, as in the paragraph scorer.
- **`ocrfix refine correct <scored> <out_dir> [--all]`:** sends a chunk when `status: scored`,
  not `low_confidence`, and flagged (stored `flag`, else the calibration now; neither → error
  unless `--all`). Per chunk, concurrently: chat request with `refine.prompt.SYSTEM` + the
  whole masked chunk (no markers; `max_tokens` = 1.3 × n_tokens + 16; greedy, thinking off) to
  `refine.correct_urls` → `parse_answer` (drops `<think>`, strips) → `text_checks`
  (`placeholders` via `restore`, then on restored text `markers` and `language`, on masked text
  `length_ratio`, `edit_ratio`, `gap_fill`) → re-score → `check_scorer` → `accepted` (text and
  score replaced) / `rejected` / `unchanged` / `error`. Each attempt is appended to `history`
  (`model, candidate, checks, score_before, score_after, accepted` or `error`).
- **`ocrfix refine summary <dir>`:** status counts, accept rate, failed checks, per language,
  mean bpc gain of accepted chunks. `<out_dir>/score_stats.json` / `correct_stats.json` hold
  run counts and tokens/s.
- **`ocrfix refine report <dir> [--clean scored_clean] [--calibration yaml] [--json f]`**
  (`refine/report.py`): corpus quality per language at three stages, all from one folder: raw
  (the denoise entry's `score_before`), denoised (the first correct entry's `score_before`),
  final (`score`); a missing stage falls back to the next. Per stage: bpc weighted by chars
  (sum bpc x n_chars / sum n_chars) and the share of chunks the calibration flags (recomputed
  per stage). With `--clean` (clean text scored with `refine.noise.remove=false`), its bpc and
  flag rate, and the share of the raw bpc excess over it closed by denoise and by correct.
  Plus noise chunks/chars and sent/accepted. On the test data the scale matched the real gain
  (doc CER 0.58% -> 0.23% after noise removal; bpc excess closed 42%, flags 46% -> 13% vs
  5% clean), but the real measure is gold.
- **Noise removal** (`refine/noise.py`, in `refine score` when `refine.noise.remove`): per-token
  bits are spread over each token's chars (placeholders and unscored tokens: NaN); each
  whitespace word gets its mean bits per char and a level by its script (`hangul` if half its
  chars are Hangul, else `other`): above `core` (8.0 / 5.0) starts a run, above `extend`
  (6.0 / 3.5) joins a neighbouring run; a word with a placeholder ends a run, an unscored one
  joins only if it has no Latin/Hangul letter or digit. Runs of `>= min_chars` (20) non-space
  chars are removed (`remove_spans`: the surrounding whitespace becomes one separator) and the
  cleaned text is re-scored. History entry `{stage: denoise, removed: [{start, end,
  bits_per_char, text}], score_before}`; no prose left -> `status: noise` (`refine correct`
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
  Score calibration inputs with `refine.noise.remove=false`. A random model (the tiny test
  model) finds everything noisy, so tests using it turn removal off.
- **Test data lost words:** `make_test_corpus.py --lost-words P` replaces a share P of prose
  words in `ocr/` by garbage from other alphabets than `garble()` (loose jamo, `▢▣◘`, mixed
  capitals/digits) and by `〈?〉` in `truth/` (as a hand truth file has it); the log keeps
  `lost: [{start, end, word, garbage}]` (offsets before the replacement). `data/test_lost`
  (48 docs, 776 lost words; `gold/` links its ocr/ and truth/) is such a set.
- **`ocrfix gold chunks <gold_dir> <out_dir>`** (`gold/chunks.py`): real gold is `.md` pairs,
  `gold/ocr/**.md` (MinerU, unedited) + `gold/truth/**.md` (corrected, `〈?〉` for lost words;
  guideline `docs/annotation_guideline.md`). Only ocr/ is chunked (`run_chunk` into
  `<out>/chunks`, rebuilt each run); `TruthMap` (Levenshtein opcodes, adjacent non-equal ops
  merged) maps each cut to the truth: insertions at a chunk's start/end go to that chunk, a
  deletion across a cut vanishes, a replacement across a cut -> `gold_status: cut_in_edit`
  (also when non-space truth text falls between chunks). The truth piece (whitespace-trimmed) is
  masked from a whole-document tagging (`render`); placeholders must equal the OCR chunk's
  (`placeholders_changed`), no text -> `empty`. `gold_suffix` = the truth's whitespace after the
  chunk; `suffix_changed` counts cuts at a line break the truth joined (unfixable by a chunk
  corrector; 5 of 845 cuts on test_lost). Split by doc hash (`gold.chunks.split` eval 0.2 /
  dpo 0.2 / rest sft), pinned by `split.json` (`{eval,dpo,sft}_docs`; `data.exclude_split`
  reads `eval_docs`). Writes `{sft,dpo,eval}/part-00000.jsonl` (complete documents: chunk rows
  + `gold_text, gold_placeholders, gold_status, gold_suffix, split`) and `stats.json`. On
  test_lost: 844/845 ok (one URL glued by a missing space). `errors table` reads the sft dir
  (only `gold_status: ok`).
- **Gold in `refine data` / `refine dpo`:** `--gold` = `gold chunks` rows, input `source_text`,
  target `gold_text`, only `ok` rows, never `split: eval`; no junk or lost words added;
  `refine.data.gold_repeats` (2) copies with ids `<id>#k` (SFT only; DPO uses 1). Default mix
  synthetic 0.6 / gold 0.2 / identity 0.2, renormalized over the sources given.
- **`ocrfix refine eval <truth_dir> <name=dir>... <out_dir>`** (`refine/eval.py`): documents
  complete in every system dir and with a truth file, rebuilt (`current`; the first system's
  `original` is system `ocr`), normalized like the chunker, compared whole: CER, WER, SER,
  false-fix rate (ocr->output regions touching no ocr->truth region), recall per
  `diff_ops` type (no output->truth residual region touches it), per truth-added `〈?〉`:
  marked / left (output ~ the OCR text there) / invented, `false_uncertain` (output `〈?〉`
  outside equal alignment); overall and per doc language (most chunk chars); paired Wilcoxon on
  per-doc CER, Holm. `metrics.json`, `per_doc.jsonl`, `report.md`. The step-by-step real-data
  workflow is `docs/real_data.md`.
- **Test data noise:** `make_test_corpus.py --noise-docs 0.3` inserts noise runs after sentence
  ends and whole noise paragraphs into `ocr/` only, logging each string in `ocr_errors.jsonl`
  (`noise`), so detection can be measured against it.
- **`ocrfix refine corrupt <chunks> <error_table.json> <confusion_map.json> <out.jsonl> [--rate R]
  [--n-variants N]`** (`refine/corrupt.py`): per chunk and variant, restore the placeholders,
  damage with `corrupt.corrupt_text` (prose only: every masked item, plus the whitespace on each
  side, is a block, so words never get glued onto a URL/citation), mask again with the chunk
  rule. Rows: `chunk_id = <id>~<r0.01|table>~v<n>`, `source_text` (damaged, masked),
  `placeholders`, `gold_text`, `gold_placeholders`, `corruption`, `variant`, `error_rate`,
  `applied_ops`, `status: new`; `refine score` takes them. Deterministic per seed. On the test
  data 5,999/6,000 keep their placeholders (the rest: a DOI whose `https://` lost its space).
- **`ocrfix refine data <out_dir> [--synthetic corrupt.jsonl]... [--identity chunks]...`**
  (`refine/data.py`): chunk SFT examples in the exact `refine correct` form: `messages` =
  `refine.prompt.messages(input)` + assistant (target; the template adds the empty think block), for
  `train sft`. `synthetic` = `refine corrupt` rows, `source_text` -> `gold_text`; dropped
  (`placeholders_changed`) unless `placeholders == gold_placeholders`. A share
  (`refine.data.junk.prob` 0.3) also gets 1..3 junk runs (3-30 non-space chars: symbol blocks,
  `lI1`-like Latin, random Hangul; never `$ | \` < > [ ]` or placeholder brackets) after a
  sentence end or as their own paragraph; the target leaves them out (`junk` lists them).
  The alphabets overlap `make_test_corpus.py`'s noise, so junk removal on the test set is
  optimistic. A share (`refine.data.lost.prob` 0.1) gets 1..2 lost words
  (`add_lost`): a word the input holds intact and the chunk holds once (2-8 syllables / 4+
  letters) becomes garbage drawn without its letters (random syllables or letters, boxes,
  symbols) in the input and one `〈?〉` in the target (`lost` lists `{word, garbage}`). `identity` = clean chunks,
  target = input. Quotas `refine.data.mix` (0.75 / 0.25) x `lang_balance` via
  `data.sft.quotas`; eval-split docs skipped; split by doc (`data.val_frac`). Two passes
  (count ids, then write), so memory holds ids only. Rows `{id, source, lang, doc_id,
  chunk_id, placeholders, messages, corruption?, error_rate?, junk?}`. Test data: 3,998 examples, rendered
  max 2,025 tokens with the 4B template (rtx5090 `max_seq_len` 2048).
- **`ocrfix refine dpo <out.jsonl> [--synthetic f]... [--identity f]... [--exclude sft.jsonl]...`**
  (`refine/dpo.py`): prompts made as `refine data` (same junk; `empty_think` forced off since
  chosen is compared with raw answers); prompts whose input or target is in an `--exclude`
  file are skipped (the SFT model knows those). The served model (`refine.dpo.model`, else
  the adapter if `llm.adapter`, else the base) answers each prompt greedily (`refine.prompt`
  body) and `n_samples` times (`temperature`, `top_p`, seed); answers cached in
  `<stem>.responses.jsonl` (`llm.client.run_requests`). Chosen is exact, so every distinct
  answer != chosen is rejected, typed by `classify`: placeholders (restore fails or
  placeholder list differs), truncated, false_fix (an edit touching no input->chosen region,
  1-char slack; identity: any edit), missed (a damaged region no edit touches), mixed (both),
  wrong_fix (neither). Up to `max_pairs` (2): greedy first, then sampled by distance to
  chosen. Prompts the model always got right: a `constructed_share` (0.25) of them get one
  off-policy pair, kinds `constructed` in turn (unchanged, half_fixed, placeholder_dropped,
  guessed = a lost word's `〈?〉` replaced by another word of the chunk (tried first when the
  prompt has lost words), overcautious = a fixed word replaced by `〈?〉`). `classify` also
  types `guessed` (fewer `〈?〉` than chosen, every lost region edited) and `overcautious`
  (more `〈?〉` than chosen).
  Rows TRL conversational + `rejected_type` (`greedy:|sampled:|constructed:<kind>`), `source,
  lang, doc_id, chunk_id`; `<stem>.stats.json` (greedy_right, all_right, types).
- **First SFT/DPO on the 5090 (test data):** QLoRA SFT on `refine data` output (3,935 rows,
  micro-batch 8, 1 epoch, 64 min, 19.7 GiB peak). Doc CER on 71 OCR test docs (after noise
  removal 0.58%): base 0.23%, SFT flagged 0.13%, SFT `--all` 0.01% (no rejections; clean chunks
  come back unchanged). Remaining errors: unrecoverable Korean words replaced by plausible
  guesses (the data never teaches 〈?〉). `refine dpo` with the merged SFT model: greedy right
  on 1,299/1,300 unseen prompts, 10 on-policy pairs, so no DPO training on test data; a
  5-step `train dpo` check ran (328 pairs, 73 s/step, 11.8 GiB). vLLM 0.33 crashed (illegal
  memory access in `punica_gpu.update_metadata`) serving the adapter while base scoring
  requests shared batches: serve a merged model on a second engine instead (README).
- **Calibration on chunks:** `calibrate` reads `refine score` output (`calibrate.run.flat_scores`
  lifts the `score` dict to the top level). Workflow in README.md, "Calibration (chunks)".
- **Corrupt speed:** generators pick positions by rejection sampling (`ops._sample`: 64 random
  tries, then the exact candidate list; uniform either way) and `Paragraph.matches` caches each
  pattern's matches, so cost no longer grows with ops × text length (rate 0.1 on 1,500 chunks:
  >300 s -> 12 s).
- **transformers generator** (`llm/hf.py`, `refine.generator: transformers`): `HFTransport`, an
  httpx transport answering vLLM's `/v1/chat/completions` (and `/health`) in this process, so
  `refine correct` / `refine dpo` send the same bodies. Renders with the tokenizer's chat
  template (`chat_template_kwargs`), left-pads concurrent requests (collected for `refine.hf.wait`)
  into `generate` batches of <= `refine.hf.batch_size` sequences grouped by greedy / (temperature,
  top_p, n), stops at the generation config's EOS ids + `<|im_end|>` and at each request's own
  `max_tokens` (`finish_reason: length`), decodes skipping special tokens. Model `llm.model`
  (+ `llm.adapter` via peft, unmerged; serve a QLoRA adapter's `train merge` output to match
  vLLM); reuses the in-process scorer's model when it is the same checkpoint and no adapter
  (`run.generator_transport`). Tests (tiny model, fp32): greedy = a step-by-step argmax recompute
  (hybrid cache), batched = single (left padding), batching through `Client`, `refine correct`
  through it. Sampled answers never match vLLM token for token.
- **`ocrfix refine compare <chunks> <out.json> [--limit 200] [--no-score] [--no-generate]`**
  (`refine/compare.py`, needs `refine.scorer=vllm`): the first N chunks with prose, current text.
  Scoring: VLLMScorer (score_urls) vs TransformersScorer (paths.scorer_model): |bits| per scored
  token (mean/p99/max), |bpc|/|p90|/|spike_frac| per chunk, flag agreement under the calibration
  (+ the disagreements), identical noise spans; tokens/s each. Generation: the greedy `refine
  correct` body to correct_urls vs the transformers transport: identical-answer share, char
  distance, completion tokens/s, truncations, first differing examples. `/v1/models` of both
  servers recorded. `pass`: flags >= 0.99, greedy identical >= 0.95. Test: a fake vLLM that is the
  same tiny model gives max |bits| < 1e-3, flags 1.0, identical answers 1.0. Not yet run on the 4B.
- **Client:** `llm.client.Client.post` (generic retrying POST, also behind `complete`);
  `refine.concurrency` requests in flight per kind.
- **Tests** (`tests/test_refine.py`) run against an httpx `MockTransport` fake vLLM and the
  tiny model for the transformers backend. Not yet run against a real vLLM.

## Ingest (`src/ocrfix/corpus/`)

- `doc_id` = first 16 hex of sha256 of the NFC relative POSIX path; `para_id` = `p00000`...;
  `order` = paragraph index. Files are walked in sorted order (`corpus.input_glob`).
- Paragraphs split on blank lines, except blank lines inside a math or table span (an
  `align` environment or HTML table with blank lines stays one paragraph). Edges stripped.
- Tagger (`tagger.py`): tables first (HTML `<table>` with nesting, case-insensitive; GFM
  pipe tables), then math only between tables, so `$` in a table is table. Math:
  `$..$`, `$$..$$`, `\(..\)`, `\[..\]` (never across a blank line), and
  `\begin{equation|align|gather|multline}[*]..\end{..}` (star must match; may cross blank
  lines). Odd backslash count before a delimiter = escaped. Inline `$` uses pandoc's rule
  plus: a `$` that can't close ends the attempt (so `$5 and $10` is prose). Spans tile the
  text; everything else is prose.
- `--sample-frac F` (`corpus.sample_frac`): keep a document iff sha256(seed, NFC relative
  path) as a fraction < F. Deterministic, spread over the corpus, stable across reruns.
- Language (`lang.py`): letters in prose only (fallback: table text without tags; math never
  counts). Hangul syllables weigh `corpus.lang.hangul_weight` (2). `ko`/`en` if that share
  >= `dominant` (0.8); `mixed` if both >= `mixed_min` (0.1); else `other` (includes
  math-only and number-only paragraphs).
- Resume: records are written per document; on restart the last document in the output
  (possibly incomplete) and any torn line are truncated and redone. `.jsonl.gz` output is
  rebuilt via `*.tmp` + rename instead.

## Score (`src/ocrfix/scorer/`)

- **Run:** `ocrfix score corpus.jsonl scores.jsonl --profile rtx5090`. With `hw.n_gpus > 1` the
  CLI re-execs itself under `torchrun` (one process per GPU, full model on each, no
  `device_map`). Each rank owns the **documents** with `crc32(doc_id) % world == rank`
  (windows never cross documents) and appends to `scores.jsonl.parts/rank-N.jsonl`; rank 0
  merges into `scores.jsonl` (skipping keys already there) and removes the parts.
- **Resume:** skip keys in the output or any part. Shards (`scorer.shard_size` paragraphs)
  hold whole documents, and a document with some keys left is re-windowed whole, but only
  the missing rows are written. A test checks that limit + resume gives the same features as
  one run.
- **Masking** (`tokens.py`): the text is scored masked exactly like the LLM input, with the
  one rule in `corpus/protect.py` (`mask`; `llm.protect` re-exports it).
  - Every math span (inline and display) becomes `⟨M#⟩`, every table `⟨T#⟩`, and the
    `llm.protect` kinds in prose `⟨P#⟩`. No exceptions (no garbled-Hangul detector).
  - A token is prose iff it overlaps no placeholder. The first `scorer.boundary` (2) prose
    tokens after a placeholder are not scored. **Scored tokens = prose and not boundary.**
  - A text that already holds a literal placeholder is scored unmasked (`mask_error: true`).
- **Windows** (`tokens.py`): consecutive paragraphs of one group are packed into one sequence,
  joined by the tokens of `"\n\n"`.
  - **Group:** `doc_id`, plus `corruption` and `variant` for corrupt rows. Paragraphs join
    only when their `order` is consecutive.
  - **Size:** a window closes once it holds `scorer.window.target_tokens` (512) and never
    exceeds `max_tokens` (768; must be < `hw.scorer.max_seq_len`).
  - **Long paragraphs:** a paragraph longer than `max_tokens` is cut at the last sentence end
    in [target, max] (else at max). Each piece starts a window, its first `boundary` tokens
    are not scored, and the paragraph after it starts a new window.
  - **Effect:** short paragraphs get the earlier ones as context, and batch rows are nearly
    uniform. There is no newline-collapsed (`flat`) pass and no truncation any more.
  - **Tokenization:** each paragraph is tokenized once (HF fast tokenizer with offsets, in a
    spawn pool, `hw.scorer.tokenize_workers`; 0 = in-process). Window ids are the
    concatenation, and `para_bits` maps window bits back.
  - **Arrays:** token arrays stay in memory. `--dump-tokens DIR` writes
    `DIR/<doc_id>/<para_id>.npy` (ids, offsets, bits, prose, boundary). Never in JSONL.
- **Model** (`model.py`): every sequence is `[EOS] + tokens`, so each token gets a surprisal.
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
- **Features** (`features.py`), per paragraph over its own scored tokens, in bits:
  - `n_prose_tokens`, and `n_chars` (chars covered by scored tokens);
  - `bpc, mean_bits, p50, p90, p99, max_bits`, and `max_pos` (a char offset into the
    **masked** text);
  - `spike_frac` (share > `scorer.tau`), plus `spike_frac@<t>` for each other
    `scorer.tau_candidates` value;
  - `low_confidence`: fewer than `scorer.min_scored_tokens` (10) scored tokens. Never
    flagged, and left out of calibration fitting.
  - With no prose, the stats are null. Every row also has `lang`, plus any `scorer.passthrough`
    field the input record carries (`corruption`, `error_rate`, `cer`, ...).
- **Calibration:** if `scorer.calibration` (default `data/calibration/calibration.yaml`)
  exists, its tau replaces `scorer.tau` and each row gets `flag` (the per-language rule).
  Delete or override it (`--set scorer.calibration=null`) when scoring inputs for calibration
  itself.
- **Measured on the RTX 5090** before windows and masking (random weights, real Qwen3.5-9B shape, 8.95B params, bf16,
  padded/SDPA, 6144 tokens per forward, 1024-token logits chunks): weights 16.7 GiB, peak
  19.7 GiB, ~6.4k tokens/s with plain transformers. With the optional kernels the same run was
  ~2.9k tokens/s (probably Triton re-tuning per batch shape; not investigated), so scoring does
  not need them.

## Gold (`src/ocrfix/gold/`)

Human annotation (paragraph tasks; the chunk pipeline uses `gold chunks`, above). Annotators follow `docs/annotation_guideline_paragraphs.md` (gold = PDF text with OCR
damage removed, nothing else).

- **`ocrfix gold sample <scores.jsonl> <paragraphs.jsonl> <out.jsonl> [--exclude done.jsonl]`**
  - Strata are filled in this order, and each paragraph lands in at most one:
    1. `feature:<f>` fixed quotas (`gold.sample.features`, any language). The features are
       `inline_math`, `display_math`, `table`, `garbled_hangul` (a math span with at most 8
       chars inside, no operators, no TeX commands except formatting ones, and Hangul next to
       it allowing one space), `mixed_koen` (ko and en each at least `corpus.lang.mixed_min`
       of prose letters) and `hyphenated` (`exam-\nple` or `exam- ple`).
    2. `<lang>:low|mid|high`: `gold.sample.per_lang[lang]` split 40/30/30 (largest
       remainder) over that language's `spike_frac` tertiles. Tertiles are taken by rank over
       all scored paragraphs, so ties at 0 still give equal bins. Unscored paragraphs only
       enter feature strata.
  - Candidates are ordered by `sha256(seed, doc_id, para_id)`, so the sample doesn't depend on
    file order.
  - At most `max_per_doc` (8) paragraphs per document. Paragraphs that already contain a
    literal placeholder are skipped.
  - Each output row has the fields `doc_id para_id order stratum lang features spike_frac
    source_path source_text placeholders context_prev context_next page_image_path`, plus
    empty annotation fields `gold_text span_labels structure_label unrecoverable_count
    annotator_id notes`. `source_text` has math and tables as `⟨M#⟩ ⟨T#⟩`, and `placeholders`
    maps each one to its raw text. Context is the raw neighbor text.
  - `page_image_path` comes from the `gold.page_image` template (`{source_stem}` etc.) or is
    null.
  - Also writes `<out>.labelstudio.json` (tasks prefilled as predictions: gold = source,
    every span `ok`) and `<out>.labelstudio.xml` (labeling config: image, context, source,
    gold textarea, span-label lines `M1=ok`, structure choice + offset, notes).
- **`ocrfix gold import <annotated.jsonl | labelstudio-export.json> <out_dir> [--strict]`**
  - Rules are in `validate.py`: `empty_gold`, `length_ratio` (0.8–1.25, sources under 20
    chars are exempt), `span_label`, `placeholders` (same as the source, exactly once, in
    order, minus `misclassified-text` spans, which must be gone), `latin_in_ko` (a new Latin
    word in `ko` text unless it is within a small edit distance of a source word, i.e. an
    OCR fix), `structure_label` (`merge-with-next` or `split-at:<n>`, where n is an offset
    into the gold text), `unrecoverable_count` (must match the 〈?〉 count; derived when
    empty), and `duplicate` (same annotator twice).
  - Any violation drops that annotation. Per paragraph, the valid annotation with the lowest
    `annotator_id` is kept.
  - Inter-annotator agreement is the normalized char edit distance between the gold texts of
    each pair of annotators on the same paragraph (mean, median, max, exact-match rate), in
    `import_report.json`.
  - The split is by document, `eval_frac` 0.5, `gold.split.seed`. An existing `split.json` in
    `out_dir` pins earlier docs to their side, so a re-import never moves eval docs into
    train. New docs are shuffled in sorted order and fill eval up to the target.
  - Outputs: `gold_train.jsonl`, `gold_eval.jsonl` (sorted, canonical JSON, adding
    `source_raw`/`gold_raw` with placeholders restored and `n_annotations`),
    `gold_eval.manifest.json` (file sha256 plus per-row sha256), `gold_eval.jsonl.sha256`,
    `split.json`, `violations.jsonl` and `import_report.json`.
- **`ocrfix gold records <gold.jsonl> <out.jsonl> [--side source|gold]`**: source-side Records
  (`text` = `source_raw`, re-tagged) with `cer` vs `gold_raw`, for `score` and
  `calibrate --gold-scores`. `--side gold` writes the gold text (`cer` 0): the clean side for
  `corrupt` and calibration.

## Errors (`src/ocrfix/errors/`)

- **`ocrfix errors table <gold.jsonl> <out_dir> [--fixed fixed.jsonl]`**. Pass
  `gold_train.jsonl` when the output feeds `calibrate`, so the eval half doesn't leak into
  training.
- **Compare form:** `source_text` with its `misclassified-text` placeholders expanded back to
  raw text, vs `gold_text`. Kept `⟨M#⟩ ⟨T#⟩` match on both sides, so math and tables never
  produce ops.
- **Alignment** (`align.py`): rapidfuzz Levenshtein opcodes (difflib fallback). Adjacent
  non-equal opcodes are merged into one region, so `rn`→`m` is one op. Two pure deletions
  (or insertions) split by an equal run of at most 3 chars are joined when an equal-cost
  alignment allows it, so a long inserted header is one op, not two fragments. Each op keeps
  its src/gold positions and text.
- **Op types**, checked in this order: `header_footer_insert` (a dropped run of at least
  `errors.header_footer_min_chars` (8) non-space chars, or a bare page number, at word
  boundaries), `unrecoverable` (gold has 〈?〉; not in the original spec, added so these
  don't land in `other`), `hyphen_wrap`, `newline_in_sentence`, `missing_space`,
  `extra_space`, `ligature`, `quote_dash`, `hangul_garbled` (non-Hangul run → Hangul),
  `dropped_char`, `duplicated_char` (the deleted text repeats its neighbor),
  `char_confusion` (at most `errors.confusion_max_len` (4) chars each side, no whitespace),
  and `other`.
- **Script:** spacing ops carry `script` (`ko` if Hangul is next to the edit, else `en` for
  Latin letters or digits, else `other`), and are reported as `missing_space.ko` etc.
- **error_table.json:** groups for each lang × stratum, each lang, and `*`. Each group has
  counts, `per_1000` (ops per 1000 gold chars, placeholders excluded), `spacing_per_1000`
  and a histogram of the per-paragraph op rate. Also the top `errors.top_k` (50)
  `char_confusions`, per lang and overall.
- **confusion_map.json:** `hangul_garbled` entries `{source_glyphs (TeX commands, $, braces
  and spaces stripped), gold, gold_jamo (jamo library), count, source_raw}`.
- **summary.md:** the same tables as markdown.
- **`--fixed`:** fixer output (`text`, keyed by doc_id/para_id) is masked back to
  placeholder form, so a fixer that altered math shows up as residual errors. It is compared
  to the same gold to give the `residual` table and map.

## Corrupt (`src/ocrfix/corrupt/`)

- **`ocrfix corrupt <clean.jsonl> <error_table.json> <confusion_map.json> <out.jsonl>
  [--n-variants N] [--seed S]`**
- **Input:** paragraph Records (`text` + `spans`), or gold rows (`gold_raw`, re-tagged).
- **RNG:** one stream per (paragraph, variant): `sha256(seed, doc_id, para_id, variant)`.
  Output is deterministic and independent of input order.
- **Rate:** each paragraph draws a rate (ops per 1000 prose chars) from its language's
  histogram (`<lang>/*` group): a bin by count, uniform inside it, the open last bin spanning
  `[a, 2a)`. `corrupt.match_total` scales the draw so its mean equals `total_per_1000`;
  `rate_scale` multiplies on top. Op count = rate × prose chars / 1000, rounded
  stochastically.
- **Op kinds:** drawn with weight = measured rate, with spacing types split by script
  (`missing_space.ko`). A kind with no place in the paragraph is dropped for that paragraph.
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
    dropped and duplicated chars and words, junk symbols for `other`, and junk glyphs for an
    existing `〈?〉`.
- **Safety:** edits touch prose only and keep a 2-char gap from each other, so math and
  tables are copied verbatim.
- **`--rate R`:** a fixed R ops per prose char replaces the histogram draw (tag `rR`, else
  `table`). Rows also carry `text` (= `source_text`), `src_para_id`, `corruption`, `order`
  and `source_path`, so they can be fed straight to `score`.
- **Output:** `spans` are the shifted clean spans plus the new garbled math spans (the tests
  check they equal `tag_spans(source_text)`); `gold_spans` are the clean spans. Each output
  row is `{doc_id, para_id, variant, lang, stratum, source_text, gold_text, spans, gold_spans,
  error_rate, applied_ops}` (with `para_id = <orig>~<tag>~v<n>`), where `applied_ops` entries are `{type, script?, gold_start,
  gold_end, src_start, src_end, src, gold, wrapped?, pair?}`.
- **Writes:** the whole file via `.tmp` + rename. A rerun regenerates it identically.

## Calibrate (`src/ocrfix/calibrate/`)

**Workflow.** Steps 1–4 build the corrupted and gold inputs, step 5 scores them, step 6
calibrates:

1. Pick clean paragraphs `clean.jsonl`. Paragraph Records work; `corrupt` also accepts
   `gold_train.jsonl` rows.
2. Corrupt them at each rate:
   `ocrfix corrupt clean.jsonl error_table.json confusion_map.json c01.jsonl --rate 0.01`,
   and the same for 0.03, 0.1, and without `--rate` (table-sampled).
3. Concatenate the outputs. Every corrupt row is a scorable Record: `text` = corrupted
   source, and `para_id = <orig>~<r0.01|table>~v<n>`, so rows from different settings never
   collide.
4. Optionally run `ocrfix gold records gold_train.jsonl gold_src.jsonl`.
5. Score the clean, corrupted and gold-source files with `scorer.calibration=null`.
6. `ocrfix calibrate clean_scores.jsonl corrupted_scores.jsonl data/calibration
   [--gold-scores gold_src_scores.jsonl]`.

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
- **Gold:** with `--gold-scores`, precision, recall and false-flag rate of the rules on real
  source paragraphs, where noisy means `cer > calibrate.gold_noisy_cer`.
- **Outputs:**
  - `calibration.yaml`: `{version, tau, variant, rules: {lang|*: {spike_frac, p90, bpc,
    recall, false_flag, target_met, n_clean, n_corrupted}}}` (no `bpc` key = off: older files
    load unchanged). `rules.load_calibration(path).flag(feats,
    lang)` applies it in score and verify.
  - `report.json`, `summary.md`, and PNGs: `auroc.png`, `roc.png`, `by_corruption.png`,
    `rule.png`.

## Fix (`src/ocrfix/fixers/`)

- **`ocrfix fix <paragraphs.jsonl> <out.jsonl> [--ops ops.jsonl]`**. `--ops` defaults to
  `<out stem>.ops.jsonl`; `--limit` counts documents.
- **Input:** paragraph Records grouped by document, as ingest writes them. A doc_id that
  reappears later is an error.
- **Fixer:** a class `@register("name")` in a module of this package, with
  `apply(doc_paragraphs) -> (paragraphs, ops)` and `defaults()` for its options.
  `discover()` imports every module here, so a new fixer needs only its module plus a
  `fix.<name>` block (`enabled: true`, options) and an entry in `fix.order`. The CLI never
  changes. Unknown fixers or options fail at startup.
- **Ops:** `{doc_id, op: merge|split|delete|edit, src_para_ids, dst_para_ids, rule
  (<fixer>.<rule>), evidence}`, logged for every change, text edits included.
  - After each fixer, `check_ops` enforces that every para_id which disappeared, appeared
    or changed text is in an op; otherwise `FixerError`.
  - A merge may reuse the first source id; a split adds new ids.
  - Output Records carry `original_text` (merges join their sources with a blank line) and
    `fixer` (names of the fixers that touched them).
- **Editing:** fixers edit through `base.map_spans` / `base.apply_edits`, which edit only
  prose (`apply_edits`) or map each span by kind, then rebuild the spans exactly as ingest
  would. Tests check they equal `tag_spans(text)`.
- **`unicode`** (enabled by default):
  - All spans: NFC, ligatures, and invisible chars stripped (control chars except `\n` and
    `\t`, plus Cf: ZWSP, ZWJ, BOM, soft hyphen, bidi marks).
  - Prose only: fullwidth→ASCII (keeping `＄ ＼ ｜` so no new math/table syntax),
    ideographic space, halfwidth forms, NBSP and tabs → space, runs of spaces → one, and
    trailing spaces dropped. A paragraph left empty is deleted.
- **`layout_junk`** (disabled):
  - Whole prose lines only: running headers/footers, i.e. a first or last line of a
    paragraph, not ending in sentence punctuation, repeated verbatim on at least
    `min_repeats` pages. The page is the record's `page`/`page_idx` field, else each
    paragraph counts as a page.
  - Bare page numbers (`12`, `- 12 -`, `Page 12`, `12 / 30`).
  - Line-number gutters: increasing numbers starting every line (≥ 3 lines) are stripped;
    a paragraph of only such numbers is deleted.
- **`line_join`** (disabled): `\n` → space unless the line ends with `.!?:;。！？…` or a
  closing bracket, the next line starts a block (`- `, `1. `, `#`, `>`, `|`, blank), or math
  or a table touches the break.
  - `letters-\nlowercase`: the hyphen is dropped if the joined word is in `wordlist`
    (word [count] per line, at least `min_count`), else kept (`well-known`). Without a
    list it is always kept.
  - Other hyphen breaks (`COVID-\n19`) are left alone. Korean lines are joined with a
    space.
- **Default order** `[unicode, layout_junk, line_join]`: headers go before joining, so they
  aren't glued into sentences. Enable the last two only if `errors table` shows
  `header_footer_insert` / `newline_in_sentence` / `hyphen_wrap` are common.

## Correct (`src/ocrfix/llm/`)

**Serving.** `scripts/serve_vllm.sh --profile <p> [--set k=v]... [--dry-run] [-- extra vllm args]`
reads everything from config (`llm.model`, `hw.dtype`, `hw.infer.*`).
- **Engines:** one `vllm serve` engine per `data_parallel_size`, GPU i on port
  `base_port + i`, with `--max-model-len`, `--gpu-memory-utilization`,
  `--enable-prefix-caching` and `--attention-backend` from `hw.vllm_attention_backend`.
  Override the backend with `OCRFIX_ATTENTION_BACKEND=TRITON_ATTN`; vLLM 0.33 has no
  `VLLM_ATTENTION_BACKEND` env var. A LoRA adapter (`llm.adapter`) is served as
  `llm.adapter_name`.
- **Profiles:** rtx5090 runs 1 engine with max-model-len 4096 and 0.90 GPU memory; h200x4
  runs 4 engines on ports 8000–8003 with max-model-len 8192. No tensor parallelism.
  `hw.infer.urls` must list the same ports (a test checks this).
- The script header documents the vLLM/CUDA requirements for Blackwell.

**`ocrfix correct <paragraphs.jsonl> <scores.jsonl> <out.jsonl> [--responses r.jsonl] [--all]`**
- **Select:** score rows with `flag: true`, or else flagged by `scorer.calibration`, or every
  paragraph with `--all`. Score the paragraphs you correct (the `fix` output), so the
  para_ids match.
- **Mask** (`corpus/protect.py`, the rule score and the training data use too): math → `⟨M#⟩`, tables → `⟨T#⟩`, and `⟨P#⟩` for `llm.protect`
  (images, inline code, URLs, emails, DOIs, `[12]` citations). Prose only, numbered 1-based
  per kind in position order, so `placeholders.restore` checks them.
- **Chunk** per document, in order:
  - At most `llm.chunk.max_paragraphs`, and user-message tokens ≤ `max_input_tokens`
    (default: the largest n with n + ceil(1.3 n) + 16 + system + `template_overhead` ≤
    `hw.infer.max_model_len`).
  - A paragraph too long alone gets `llm_error: too_long`.
  - `chunk_id = <doc_id>:<sha1(para_ids + masked texts)[:12]>`, so it changes when the
    input does.
- **Prompt** (`prompt.py`): a fixed `SYSTEM` instruction, then the user message, with each
  paragraph after a `<<<n>>>` marker line; the model answers with the same markers. The
  request body has `temperature 0`, `top_p 1`, `seed`, `max_tokens = ceil(1.3 ×
  input_tokens) + 16`, and `chat_template_kwargs.enable_thinking = llm.enable_thinking`
  (false).
- **Client** (`client.py`, httpx async): round-robin over `hw.infer.urls` with
  `hw.infer.client_concurrency` requests in flight.
  - Retries go to the next URL with exponential backoff on transport errors, timeouts, 429
    and 5xx; other 4xx are not retried.
  - Responses are appended to `--responses` as they arrive. A rerun skips chunks that
    succeeded or were truncated (`finish_reason=length`, deterministic under greedy).
- **Parse** (`parse.py`): a `<think>` block is dropped, then the answer is split on markers
  and placeholders restored. Per-paragraph errors: `marker_missing`, `marker_missing_next`
  (the next marker is lost, so this body likely swallowed it), `marker_duplicate`, `empty`,
  `placeholders: ...`, and request errors. A single-paragraph chunk without a marker is
  accepted.
- **Output:** every input paragraph. Corrected ones get the new `text` (re-tagged spans),
  `fixer` + `llm` and `llm_chunk_id`, and keep `original_text` from `fix`. Failed ones keep
  their text and get `llm_error`.

## Verify (`src/ocrfix/verify/`)

- **`ocrfix verify <corrections.jsonl> <scores_before.jsonl> <out.jsonl>`**;
  `ocrfix verify-summary <out.jsonl>` prints the accept rate per language and a histogram
  of failed checks.
- **Candidates:** `correct` output rows the LLM changed (`llm` in `fixer`, no `llm_error`,
  text ≠ `llm_source_text`). `llm_source_text`, written by `correct`, is the text the LLM
  saw; `scores_before` are its scores.
- **Checks** (`checks.py`), each `{check, passed, reason}`:
  - `markers`: the source's math/table spans (the `⟨M#⟩ ⟨T#⟩`) are in the correction
    verbatim, once, in order, with no new ones.
  - `length_ratio` within `verify.length_ratio`.
  - `edit_ratio`: char edit distance / len(source) ≤ `max_edit_ratio`.
  - `language`: same `detect_lang`, and the Hangul share of prose letters moves by at most
    `max_script_shift`.
  - `gap_fill`: each source `〈?〉` (adjacent ones count as one) is kept or filled with 1–
    `gap_fill_max_syllables` Hangul syllables, located by 8 chars of source context on each
    side. If that context changed, the check fails.
  - `scorer`: `before` comes from `scores_before`. `after` is
    `InProcessScorer.rescore(source rows, {key: corrected})`, on one device:
    - it rebuilds the windows `score` used, from every row's source text in file order;
    - it swaps in the corrected paragraphs and scores each window only up to its last
      corrected paragraph (causal, so later tokens don't matter);
    - a split long paragraph is re-split on its own.

    The check needs bpc(before) − bpc(after) ≥ `min_bpc_gain` and spike_frac(after) ≤
    `spike_ratio` × before. Paragraphs without prose pass. With `rescore_failed: false`, a
    correction that already failed a text check is not re-scored (`scorer` is skipped).
- **Output:** every input row. Candidates get `original`, `corrected`, `accepted`, `checks`,
  `failed_checks`, `before`, `after`, and `text` = corrected if accepted, else the source,
  with `llm` dropped from `fixer`. Other rows pass through unchanged.

## Data (`src/ocrfix/data/`)

**Format** (`common.py`): exactly what `correct` sends and parses: the `SYSTEM` prompt,
`<<<n>>>` markers, and `llm.protect` placeholders.
- The assistant turn repeats the markers with the targets. `data.empty_think` (default false)
  would put `<think>\n\n</think>\n\n` before them; Qwen3.5's template adds that block itself
  with `enable_thinking=false` (and strips one from the content first), so both render the
  same tokens.
- Masking is the same rule as score and correct (`corpus.protect.mask`): every math, table
  and protected span of the input becomes a placeholder. The target must keep each one
  verbatim, in order, where it gets the same placeholder. Otherwise the pair is dropped and
  counted as `unmaskable`. Examples: a gold `misclassified-text` span (`$\mathrm{Ol}$` → 이),
  or a synthetic `hangul_garbled` op. At inference such a span is a placeholder the model
  must return unchanged, so garbled Hangul inside math is not fixed by the LLM, and eval
  shows it as missed `hangul_garbled` recall.
- Chunks hold 1–3 consecutive paragraphs (`data.chunk.sizes/weights`, at most
  `max_chars`) from one document and, for synthetic data, one corruption variant.
- Docs in `data.exclude_split` (the gold `split.json` eval half) are dropped from every
  source, for SFT and DPO.

**`ocrfix data sft <out_dir> [--synthetic corrupt.jsonl] [--verified verified.jsonl]
[--gold gold_train.jsonl] [--identity clean.jsonl]`**
- **Sources** (`sources.py`):
  - `synthetic`: corrupt output re-run through the configured fixers per (doc, corruption,
    variant); input = fixed, target = clean.
  - `verified`: accepted rows with bpc gain ≥ `data.sft.verified_min_bpc_gain`.
  - `gold`: `source_raw` → `gold_raw`, repeated `gold_repeats` times, each repeat chunked
    differently.
  - `identity`: target = input.
- **Quotas:** `mix` weight × `lang_balance` share per (source, language); both are
  renormalized over what is present. The total is the largest that needs no reuse, capped
  by `total`. Cells are filled by a seeded shuffle.
- **Split** by document (`val_frac`). Writes `train.jsonl`, `val.jsonl` and `stats.json`
  (pool vs taken per source and language). Rows are `{id, source, lang, doc_id, para_ids,
  messages, repeat?}`.

**`ocrfix data dpo <out.jsonl> [--gold ...] [--verified ...] [--model name] [--n-samples N]`**
- **Inputs:** one paragraph per prompt, chosen = gold or accepted correction.
- **Sampling:** `n_samples` completions per input (one request with `n`, at
  `data.dpo.temperature/top_p`, round-robin client, resumable `<out stem>.responses.jsonl`),
  each labeled with the verify text guardrails. Completions passing every guardrail are
  never negatives.
- **Rejected order,** capped at `max_pairs_per_input`, deduplicated, never equal to chosen:
  1. Fluent samples that fail only `edit_ratio`/`length_ratio`.
  2. Constructed `unchanged`, `half_fixed` (half of the input→chosen edits undone),
     `marker_dropped` and `gap_filled` (one 〈?〉 guessed).
  3. Other sampled failures (`sampled:<check>`, `sampled:marker_missing`, ...).
- **Rows:** TRL conversational `{prompt, chosen, rejected}` + `rejected_type, source, lang,
  doc_id, para_id`.

**`ocrfix data stats <file>... [--json]`**: counts by source, language and rejected_type.

## Train (`src/ocrfix/train/`)

**Commands:**
- `ocrfix train sft <train.jsonl> <out_dir> [--eval file] [--max-steps N]`
- `ocrfix train dpo <pairs.jsonl> <sft> <out_dir>` (`sft` = a LoRA adapter or a full model dir)
- `ocrfix train merge <adapter> <out_dir> [--base DIR] [--dequantize/--no-dequantize]`
- Launch through `scripts/train.sh <cmd> ... --profile P`; the CLI also re-execs under
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
`add_generation_prompt=True` and `enable_thinking = llm.enable_thinking`, the same text
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
  - `base` is `paths.base_model`, loaded as a frozen `ref_model`.
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
  DPO on a full SFT model merges into that model), else `paths.base_model`.
- A full model dir (full SFT/DPO output) is refused: it needs no merge. Serve it with
  `llm.model`.

**Tests** (`tests/test_train.py`): on the tiny model, with a Qwen-style chat template added:
- 2-step SFT, LoRA and full;
- DPO on an adapter (both references), full DPO (both references), and a LoRA DPO on a full
  SFT model then merged;
- the loss mask, the end-of-turn handling, and a merge. The QLoRA SFT + dequantized merge runs
only with CUDA + bitsandbytes.

## Eval (`src/ocrfix/eval/`)

**`ocrfix eval <gold_eval.jsonl> <system>... <out_dir> [--limit DOCS]`**, where each system is
`name=path` or a path (named after the file).
- **Inputs:** system outputs are paragraph Records (`correct`, `verify` or `fix` output) produced
  from the gold eval sources (`ocrfix gold records gold_eval.jsonl`), so the ids match.
  - Only gold documents are read. A record with an ingest-style id (`p00012`) that isn't a gold
    paragraph is dropped (`non_gold_paragraphs`); other ids are kept, since splits make them.
  - A missing document is scored as its unchanged source (`missing_docs`), so pairs stay
    complete.
  - If `<gold>.sha256` exists, the gold file must match it.
  - With `eval.baseline`, the unchanged source is scored as system `source`.
- **Document texts** (`align.py`): paragraphs in order (gold by `order`, system in file order),
  blank-line breaks flattened to a space, joined by a space. So merges and splits cost no edits.
  `Aligner` maps positions between versions through the Levenshtein opcodes.
- **Metrics** (`metrics.py`), micro-averaged per system, overall and per document language
  (gold-char-weighted majority):
  - CER, WER (word Levenshtein) and SER: spacing disagreements at aligned non-space chars over
    gold spaces. The report shows ko as CER + SER and en as CER + WER.
  - False-fix rate: system edits (source → output regions) touching no gold edit (source → gold),
    over all system edits; also per 1000 source chars.
  - Recall per error type: gold edits from `errors.align.diff_ops` (same types and options as
    `errors table`) with no residual edit (output → gold) touching their gold range.
  - Identity: clean gold paragraphs (gold = source) that a system edit touches.
  - Placeholder integrity: gold math/table spans, raw and in order, found verbatim in the output
    (LCS), plus extra spans and the share of documents with an exact match.
  - Structure, per paragraph none / merge / split:
    - Predicted merge: the source joint after the paragraph maps to no record joint (±1 char).
    - Predicted split: a record joint falls inside the paragraph.
    - Split position is right within `eval.split_tolerance` chars.
    - A merge-with-next whose next paragraph (`order + 1`) isn't in the eval set is not scored.
- **Comparison** (`stats.py`, no scipy): two-sided paired Wilcoxon signed-rank on per-document
  CER for every pair of systems. Zero differences are dropped and ties get average ranks. The p
  is exact (enumerated over doubled ranks) up to `eval.exact_max_n` pairs, else normal with tie
  and continuity correction. Holm-adjusted across pairs.
- **Outputs:** `metrics.json`, `per_doc.jsonl`, `paragraphs.jsonl` and `report.md` (overall, by
  language, recall by type, structure, paired comparison, the `eval.worst_n` (20) worst
  paragraphs per system by CER, with source, output and gold).

## Pipeline (`src/ocrfix/pipeline.py`)

**`ocrfix pipeline <pipeline.yaml> [--limit DOCS] [--only STAGE]... [--force STAGE|all]
[--work-dir DIR] [--dry-run]`** runs ingest → score → fix → rescore → correct → verify → eval.
- **The YAML:** `input_dir` (markdown) or `input_records` (Records, ingest skipped),
  `work_dir`, `stages`, `sample_frac`, `gold_eval` (null skips eval), `correct_all`,
  `serve`, `serve_timeout`, and `set` (overrides for every stage, before the CLI's `--set`).
  Unknown keys are an error.
- **Stages are subprocesses** (`python -m ocrfix.cli <stage> ... --profile P --set ...`).
  GPU memory is freed between stages, multi-GPU stages re-exec under torchrun as usual, and
  each stage's output is teed to `logs/<stage>.log`.
- **Outputs in `work_dir`:** `corpus.jsonl`, `scores.jsonl`, `fixed.jsonl` (+ `.ops.jsonl`),
  `scores_fixed.jsonl`, `corrected.jsonl`, `corrected.responses.<llm hash>.jsonl`,
  `verified.jsonl`, `eval/`, and `pipeline.json` (seconds per stage).
- **Score stages** run with `scorer.calibration=null`, so no `flag` is stored and correct
  always applies the current calibration. So `STAGE_KEYS` leaves the calibration out of the
  score fingerprint, and a new calibration needs no rescore.
- **Correct** is skipped with a note when there is no calibration and no `correct_all`;
  verify and eval then lack inputs and are skipped too. Responses go to a file per hash of
  the `llm` config, because `correct` caches answers by input text only.
- **Eval** compares `fixed` and `verified` (if present) against `gold_eval`.
- **Resume** (`.stages/<stage>.json`): a settings fingerprint (the stage's own arguments,
  without `--profile`/`--set`, plus the `STAGE_KEYS` config sections) and an inputs
  fingerprint (size plus a hash of the first and last MiB of each file; for ingest, the
  markdown files' count, size and newest mtime, plus `--limit`), and `complete`.
  - **cached:** completed, both fingerprints equal, outputs present.
  - **resume** (outputs kept): interrupted with equal fingerprints, or only the inputs
    changed for an `APPEND_SAFE` stage (ingest; score on the append-only corpus).
  - **rebuilt** (outputs, `.parts` and `.tmp` deleted first): settings changed, inputs of
    any other stage changed, or `--force`. Rescore is not append-safe: fix rewrites
    `fixed.jsonl`, and score skips ids it already has.
- **LLM server** for correct (`serve`):
  - `auto`: use `hw.infer.urls` if every one answers `/health`; else start
    `scripts/serve_vllm.sh --profile P --set ...` in its own process group and SIGTERM it
    afterwards. On one GPU, verify's scorer needs the memory back.
  - `external`: the servers must already answer.
  - `echo`: `ocrfix.llm.echo_server` on the same ports. It answers with the user message
    after hyphen joins, space collapsing and a few whole-word OCR fixes (`Tab1e` → `Table`).
  - A port that accepts connections but has no `/health` is reported as taken by another
    process.
- **Smoke run** (`configs/pipeline/smoke.yaml`): the fixtures, the echo server, ports
  18300+, and `tests/fixtures/sample_mineru_gold_eval.jsonl`. That file is real
  `gold import` output for the 3 fixture docs: 13 paragraphs, one fixing `Tab1e`. The
  end-to-end test runs it with the tiny model (verified CER 0), then checks every stage
  reports `cached` on a rerun.
- **Profiles:** the tests check every file in `configs/profiles/` for the required `hw` keys,
  `urls` = `base_port + i`, engines ≤ GPUs, and `max_model_len ≥ train.max_seq_len`.

## Record schema (`src/ocrfix/schema.py`)

```json
{"doc_id": "...", "para_id": "...", "order": 0, "lang": "ko", "text": "...",
 "spans": [{"start": 0, "end": 12, "kind": "prose"}], "source_path": "..."}
```

- `spans[].kind` is `prose | math | table`. Offsets are Python `str` indices (code points)
  into the **NFC** text, half-open `[start, end)`.
- Stage-specific fields (`ocr_score`, `target`, ...) are extra top-level keys; they land in
  `Record.extra` on read and are written back flat. They may not shadow core fields.
- `Record.from_dict` NFC-normalizes `text` and refuses non-NFC text that has spans
  (the offsets would be wrong).

## Conventions

- **Stable IDs.** `doc_id` is derived from the source document (stable across reruns, e.g.
  a hash of the relative source path); `para_id` is stable within a doc. `(doc_id, para_id)`
  is the key everywhere. `order` is the paragraph position within the doc.
- **Idempotent and resumable.** A stage first calls `io.read_ids(output)` and skips keys
  already present; rerunning a finished stage is a no-op. `ShardedWriter` appends new shards
  after the existing ones and writes via `*.tmp` + rename, so only complete shards are seen.
- **`--limit` and `--profile` on every command.** `--limit` caps records processed (for smoke
  runs); the profile defaults to `$OCRFIX_PROFILE`, else `rtx5090`.
- **No network access anywhere.** `import ocrfix` sets `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  VLLM_NO_USAGE_STATS=1` (and friends); every launcher sources `scripts/env.sh`. All model and
  tokenizer paths come from config (`paths.*`, `llm.*`) and point to local directories. Never
  pass a hub repo id. Never add code that downloads.
- **NFC on every input.** Every stage NFC-normalizes text as it reads it (`schema.nfc`,
  `Record.from_dict`). Korean OCR output often contains decomposed Hangul jamo.
- **Hardware differences live only in profiles.** Code reads `cfg.hw.*` (n_gpus, dtype,
  attn_impl, train.mode `qlora|lora|full`, batch sizes, seq lens, vLLM parallelism); it never
  branches on GPU model. `default.yaml` declares `hw: ???` so a profile is mandatory. To add a
  machine, add a profile, don't add `if`s.
- **JSONL everywhere between stages**, optionally gzipped (`io.gzip: true`), sharded
  (`io.shard_size`). Read with `io.iter_records` / `io.iter_jsonl`; never load a whole stage
  into memory.

## Placeholders

Before text reaches a model (the scorer, the LLM, and the LLM's training data), non-prose
content is masked with one rule (`corpus/protect.py`), so the LLM cannot "fix" it and every
stage sees the same text:

| token   | meaning |
|---------|---------|
| `⟨M#⟩`  | math span (inline `$...$` or display `$$...$$`), `#` = 1-based index per kind within the paragraph |
| `⟨T#⟩`  | HTML table (`<table>...</table>`) |
| `⟨P#⟩`  | other protected content: URLs, code, citations/refs, image links |
| `〈?〉`  | emitted by the model where the source is illegible and it cannot decide; never invented away |

`⟨ ⟩` are U+27E8/U+27E9; `〈 〉` in `〈?〉` are U+3008/U+3009, chosen so they never collide.
`corpus/placeholders.py`: `substitute(text, spans) -> (masked, mapping)` numbers math and
table spans in order of appearance (⟨M1⟩ ⟨M2⟩ ⟨T1⟩ ...; mapping is position-ordered);
`restore(masked, mapping)` raises `PlaceholderError` unless every placeholder appears exactly
once and in the original order. `verify` builds on this. Constants are in `schema.py`
(`MATH_PH`, `TABLE_PH`, `PROTECTED_PH`, `UNCERTAIN`).

## Install (offline)

On an online machine with the same Python (3.12) and arch (x86_64):

```bash
scripts/make_wheelhouse.sh wheelhouse
```

Copy `wheelhouse/` plus the model directory named by your profile's `paths.base_model` to the air-gapped
machine, then:

```bash
scripts/install_offline.sh wheelhouse .venv
```

`requirements.lock` is pinned with hashes, generated from `requirements.in` (command at the
top of that file). Update it only on an online machine.

### Optional Qwen3.5 kernels

Qwen3.5-9B is a hybrid model (Gated DeltaNet linear-attention + full-attention layers). By
default we install **plain transformers only**: it runs those layers with its pure-PyTorch
fallback (correct, slower; logs "falling back to its reference PyTorch implementation").
vLLM (fix/eval) ships its own kernels and is unaffected.

The optional kernels, `flash-linear-attention` (`requirements-kernels.lock`, locked against the
main lock) and `causal-conv1d`, speed up training and cut memory, and are required for
`hw.train.packing: true` (the h200x4 profile sets it); `ocrfix train` refuses packing without
them at start-up. The rtx5090 profile trains unpacked with `train.group_by_length: true`. To
use the kernels:

```bash
scripts/make_wheelhouse.sh --with-kernels wheelhouse   # online
scripts/install_offline.sh --with-kernels wheelhouse   # air-gapped
```

`causal-conv1d` has no PyPI wheel for our torch, so `scripts/build_causal_conv1d.sh` compiles it
(~10 min) with a pip nvcc pinned to torch's CUDA minor (13.0; nvcc 13.4 with 13.0 headers
fails). The wheel covers sm_90 and sm_120 and is checked by `SHA256SUMS`, not a lock. Rebuild it
whenever torch changes in the lock. A build for torch 2.14.0+cu130 is kept in `wheelhouse/`.

With `flash-linear-attention` installed, transformers uses its Triton kernels on every device,
so Qwen3.5 then **only runs on CUDA**, and the first forward pass JIT-compiles (~1 min, cached).

## Running

```bash
scripts/ocrfix.sh ingest /data/mineru data/corpus.jsonl --limit 100
ocrfix pipeline configs/pipeline/sample.yaml --profile rtx5090          # all stages, resumable
scripts/serve_vllm.sh --profile rtx5090                                # LLM for correct / data dpo
scripts/train.sh sft data/sft/train.jsonl runs/sft --profile h200x4      # torchrun, hw.n_gpus procs
scripts/train.sh merge runs/sft runs/merged --profile rtx5090
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
