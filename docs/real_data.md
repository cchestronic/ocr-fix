# Real data, step by step

From your gold and clean documents to a trained correction model, an honest measurement and
the corrected corpus. Run the steps in order. Each one is resumable: run the same command again
after an interruption.

The commands use the default profile, `rtx5090` (the 4B; vLLM on ports 8100+, since 8000 is
taken on this machine). For the H200s run `export OCRFIX_PROFILE=h200x4` first (the 9B); the
commands stay the same.

## 0. What you prepare

```
data/real/
  corpus/        the MinerU .md tree to correct (50 GB), as it is
  gold/ocr/      MinerU .md files picked for gold, copied unchanged
  gold/truth/    the same files, hand-corrected (same relative paths)
  clean/         born-digital markdown in the same style, no OCR
```

- **Gold:** pairs of `.md` files; how to correct them is in
  [annotation_guideline.md](annotation_guideline.md).
  - **Size:** 5,000 chunks is about 10M characters. Many short documents of different types
    is best.
- **Clean:** as much as you have, ideally 50–100 MB. It must not be copies of gold documents,
  and it must have the same markdown for math (`$...$`) and tables (`<table>`).

**Settings:** each step reads its own file in `configs/stages/` (`sft.yaml`: the SFT mix and
learning rate; `dpo.yaml`: sampling and beta; `calibrate.yaml`; `correct.yaml`: the correction
model, the checks and the re-score rule; `eval.yaml`). The machine (model path, GPUs, ports,
batch sizes) is `configs/profiles/rtx5090.yaml`. Edit a file, or change one run with
`--set key=value`.

The model, scores and calibration belong together. After switching 4B → 9B, redo steps 3–9.

**vLLM engines** run in their own terminal (or with `&`): `ocrfix serve` starts them and waits
until they are ready; Ctrl-C (or `pkill -f "[o]crfix serve"`) stops them. Steps 7, 9 and 10 use
two engines on one GPU: the base model on 8100 (scoring) and the tuned model on 8101
(correction). Set this once per shell for them:

```bash
E="data/real/errors/error_table.json data/real/errors/confusion_map.json"
TUNED=(--set correct.model=runs/sft_merged)    # the tuned engine on correct_port (8101)
```

## 1. Gold → chunks (CPU)

```bash
ocrfix gold chunks data/real/gold data/real/gold_chunks
```

- **What it does:**
  - chunks the `ocr/` files exactly like the corpus;
  - cuts each `truth/` file at the matching places;
  - splits by document into `sft/` (60%), `dpo/` (20%) and `eval/` (20%) (`gold.split`).
- **Check the output:**
  - The `ok` count per split is what training can use.
  - `placeholders_changed` means math, a table or a link differs between the two files. A few
    are normal (e.g. OCR glued a word to a URL); many mean the truth files changed math.
  - `cut_in_edit` and `empty` should be rare.
  - Documents without a truth file are listed.
- **The split is fixed:** `split.json` pins every document to its side. Adding gold later and
  running the same command again never moves a document, and the eval documents are never
  trained on.

## 2. Error rates (CPU)

```bash
ocrfix errors table data/real/gold_chunks/sft data/real/errors
```

`summary.md` shows the real error types and rates per language. The training-data generator
copies these.

## 3. Clean chunks (CPU)

```bash
ocrfix chunk data/real/clean data/real/chunks_clean
```

## 4. Calibration (GPU: base model)

Two settings come from your data here: which text counts as noise (and is deleted), and which
chunk scores count as "damaged" (and get corrected).

```bash
ocrfix serve &                                    # the base model on 8100
ocrfix calibrate-noise data/real/gold_chunks      # noise thresholds -> configs/stages/score.yaml
ocrfix corrupt data/real/chunks_clean $E data/real/calib/corrupt/table.jsonl
ocrfix corrupt data/real/chunks_clean $E data/real/calib/corrupt/r0.01.jsonl --rate 0.01
ocrfix corrupt data/real/chunks_clean $E data/real/calib/corrupt/r0.03.jsonl --rate 0.03
S=(--set score.calibration=null --set score.noise.remove=false)
ocrfix score data/real/chunks_clean data/real/calib/clean_scored "${S[@]}"
ocrfix score data/real/calib/corrupt data/real/calib/corrupt_scored "${S[@]}"
ocrfix calibrate data/real/calib/clean_scored data/real/calib/corrupt_scored data/calibration
```

- **`calibrate-noise`** scores the OCR side of the gold (sft/ and dpo/ only) plus a copy of
  each chunk with junk inserted (runs of 20–60 characters). The inserted junk is the noise to find; everything the truth
  keeps (corrected words included) is real text; what the annotators deleted
  (page numbers, footers) counts neither way. It tries `min_chars` 20 and up only (long
  runs), and picks the `core` / `extend` that find the most junk while removing at most 1 in
  10,000 real characters (`calibrate.noise.max_text_removed`). It writes them into
  `configs/stages/score.yaml` with a comment (date, model, results); the report,
  `data/calibration/noise_report.json`, shows the previous values' result and up to 30
  examples of real text the new values would still remove. On the test gold: 60% of the junk
  found, 0.0096% of real text removed (the old values: 72%, 0.024%). `--no-write` only
  reports.
- **A few thousand clean chunks are enough here:** `--limit N` on `corrupt` counts
  input shards.
- **Check `data/calibration/summary.md`:** recall per language at ≤ 5% clean chunks flagged.
  `data/calibration/calibration.yaml` is the file `score.calibration` points to.
- **Whole-chunk noise:** `calibrate` also writes a first `noise_chunk` limit per language (the
  highest bpc of its clean and damaged chunks + 1.0). It is replaced in step 10 by
  `calibrate-noise-chunk`, from the gold and the corpus. `score` drops a chunk whose bpc
  (after word-level removal) is above it: its prose is removed, math/tables/links stay,
  `status: noise`, a `noise_chunk` history entry keeps the bpc; `source_text` is untouched.
  Off: `--set score.noise.whole_chunks=false`.

## 5. SFT data (CPU)

```bash
ocrfix corrupt data/real/chunks_clean $E data/real/train/corrupt_table.jsonl
ocrfix corrupt data/real/chunks_clean $E data/real/train/corrupt_r0.03.jsonl --rate 0.03
ocrfix data sft data/real/sft \
  --synthetic data/real/train/corrupt_table.jsonl --synthetic data/real/train/corrupt_r0.03.jsonl \
  --gold data/real/gold_chunks/sft --identity data/real/chunks_clean
```

- **Mix** (`sft.mix`):
  - synthetic 60%: clean chunks damaged at the gold rates. 30% of them also get junk.
  - gold 20%: each gold chunk twice (`sft.gold_repeats`).
  - identity 20%: clean chunks returned unchanged.
- **Size:** the total is the most the mix allows without reusing anything. Cap it with
  `--set sft.total=30000`.
- **Check:** the output line and `stats.json` show what was taken and dropped per source and
  language.

## 6. SFT (GPU: stop the engines first)

```bash
ocrfix train sft data/real/sft/train.jsonl runs/sft --eval data/real/sft/val.jsonl
ocrfix train merge runs/sft runs/sft_merged
```

On the 5090, about 1 hour per 4,000 examples (QLoRA, 4B). With several GPUs (`hw.n_gpus`),
`train` runs itself under torchrun.

## 7. DPO pairs (GPU: base + SFT model)

```bash
ocrfix serve --memory 0.44 &                                              # base, 8100
ocrfix serve --correct --memory 0.42 "${TUNED[@]}" &                      # SFT model, 8101
ocrfix corrupt data/real/chunks_clean $E data/real/train/corrupt_dpo.jsonl --set seed=7
ocrfix data dpo data/real/dpo/pairs.jsonl \
  --synthetic data/real/train/corrupt_dpo.jsonl --gold data/real/gold_chunks/dpo \
  --exclude data/real/sft/train.jsonl "${TUNED[@]}"
```

- **What happens:** the SFT model answers every prompt once greedily and 4 times sampled.
  Every answer that isn't the right one becomes a pair, typed:
  `missed`, `false_fix`, `wrong_fix`, `mixed`, `placeholders`, `truncated`.
- **Check `pairs.stats.json`:**
  - You want several thousand pairs, most of them `greedy:` or `sampled:` (the model's own
    mistakes).
  - If the model is right on almost every prompt, DPO has little to learn. Use harder
    synthetic prompts (`--rate 0.03` on the DPO corrupt file) or skip DPO.

## 8. DPO (GPU: stop the engines first)

```bash
ocrfix train dpo data/real/dpo/pairs.jsonl runs/sft runs/dpo
ocrfix train merge runs/dpo runs/dpo_merged
```

The reference model is the SFT adapter (`train.dpo.reference: sft`).

## 9. Eval on the gold eval documents (GPU)

Score the eval chunks once with the base model, then correct them with each model:

```bash
ocrfix serve --memory 0.44 &                                              # base, 8100
ocrfix score data/real/gold_chunks/eval data/real/eval/scored
ocrfix correct data/real/eval/scored data/real/eval/base           # the base corrects
ocrfix serve --correct --memory 0.42 "${TUNED[@]}" &                      # SFT model, 8101
ocrfix correct data/real/eval/scored data/real/eval/sft "${TUNED[@]}"
ocrfix correct data/real/eval/scored data/real/eval/sft_all --all "${TUNED[@]}"
# the DPO model: stop the 8101 engine, start it with correct.model=runs/dpo_merged, then
ocrfix correct data/real/eval/scored data/real/eval/dpo --set correct.model=runs/dpo_merged

ocrfix eval data/real/gold/truth \
  base=data/real/eval/base sft=data/real/eval/sft sft_all=data/real/eval/sft_all dpo=data/real/eval/dpo \
  data/real/eval/report
```

- **`report.md`** compares whole documents with the truth files. System `ocr` is the text
  before correction.
  - **CER, WER and SER**, overall and per language.
  - **False fixes:** edits where the truth has none.
  - **Recall per error type:** the same types as `errors table`.
  - **Paired Wilcoxon tests** between systems.
- **Pick the corrector and the mode:**
  - correct only flagged chunks, or every chunk with `--all`;
  - SFT or DPO.

## Optional: transformers instead of vLLM for the small jobs

DPO sampling (step 7) and eval corrections (step 9) can run in-process with transformers
(`--set correct.backend=transformers`, model `correct.model`), with no second engine and no
merge needed for an adapter (`--set correct.adapter=runs/sft`). The corpus (step 10) stays on
vLLM. Before trusting it, check that both backends give the same results, one engine at a time
(leave room for transformers with `--memory 0.45` on the engine):

```bash
# scores: the base engine on 8100 vs transformers loading the base
ocrfix compare data/real/eval/scored data/real/compare_score.json --no-generate
# greedy answers: the tuned engine on 8101 vs transformers loading the same dir
ocrfix compare data/real/eval/scored data/real/compare_sft.json --no-score "${TUNED[@]}"
```

- **`pass`** needs flags to agree on ≥ 99% of chunks and ≥ 95% of greedy answers to be
  identical.
- **Small differences are normal:** bf16 with different kernels, and a greedy answer can split at
  a near-tie. The report shows the first differing answers.
- **Calibration** should come from the same scorer as the corpus (vLLM), unless the score check
  passes.
- **Speed:** transformers generation is much slower than vLLM without the optional kernels
  (`flash-linear-attention`, `causal-conv1d`). The report gives tokens/s for both.
- **Measured on the 5090 (4B, test data):**
  - **scores:** flags agreed on 99.6% of 1,500 chunks; the 6 that differed were on a
    threshold. Noise removal was identical on every chunk, and bpc differed by 0.001 on
    average.
  - **greedy answers:** the SFT model's corrections were identical on 100 of 100 chunks.
  - **speed:** transformers alone did about 750 tokens/s at batch 32, vLLM about 3,300 at
    half the GPU memory, so transformers is 4–8× slower.
  - **what that means:** DPO sampling of 5,000 prompts (1 greedy + 4 samples, about 16M
    tokens) takes about 6 hours with transformers, against roughly 1–1.5 hours with vLLM.
    Eval on 1,000 chunks takes about 15 minutes.

Then, for example, DPO pairs without the second engine:
```bash
ocrfix data dpo data/real/dpo/pairs.jsonl --synthetic data/real/train/corrupt_dpo.jsonl \
  --gold data/real/gold_chunks/dpo --exclude data/real/sft/train.jsonl \
  --set correct.backend=transformers --set correct.model=runs/sft_merged
```

## 10. The corpus (GPU)

With the base engine on 8100 and the chosen model on 8101 (as in step 9):

```bash
ocrfix chunk data/real/corpus data/real/chunks

# pass 1: scores only, no noise removal
ocrfix score data/real/chunks data/real/scored_raw \
  --set score.noise.remove=false --set score.noise.whole_chunks=false

# the whole-chunk noise limit: from the gold, checked on the corpus (-> calibration.yaml)
ocrfix calibrate-noise-chunk data/real/gold_chunks data/real/scored_raw
# look at data/calibration/noise_chunk_hist.png (bpc histograms) and noise_chunk_samples.md;
# edit noise_chunk in calibration.yaml if needed

# pass 2: scores with both noise removals (words, then whole chunks) and flags
ocrfix score data/real/chunks data/real/scored
ocrfix correct data/real/scored data/real/refined "${TUNED[@]}"     # add --all if step 9 says so
ocrfix report data/real/refined --clean data/real/calib/clean_scored
ocrfix reconstruct data/real/refined data/real/corrected --source current
```

- **Pass 1** is only for choosing the limit; pass 2 scores everything again (a new folder:
  `score` skips the shards already in its output folder).
- **`calibrate-noise-chunk`** (the limit for whole chunks of noise):
  - **Gold:** it scores the OCR side of the gold chunks (sft/ and dpo/, never eval/) with
    the base model, noise removal off. A gold chunk whose annotator deleted at least half of
    its characters (`calibrate.noise_chunk.deleted_share`) is a noise chunk; every other one is
    real text. **limit = the highest bpc of real-text gold chunks + 1.0**
    (`calibrate.noise_chunk.margin`), per language (en, ko, mixed); a language with fewer than
    `calibrate.min_rows` (50) real gold chunks, and every other language, uses `*`.
  - **Corpus:** it reads the pass-1 scores (no model) and reports, per language, how many
    chunks and characters the limit would drop, and the same for the gold maximum + 0.5, 1.0,
    1.5, 2.0, 3.0 (`calibrate.noise_chunk.candidates`).
  - **Writes** `noise_chunk` in `data/calibration/calibration.yaml` (the rest of the file is
    kept; `--no-write` only reports); `noise_chunk_report.json` (gold bpc p50 / p99 / max,
    the gold noise chunks it catches, corpus histogram and drops), `noise_chunk_samples.md`
    and **`noise_chunk_hist.png`**: per language, the share of corpus chunks (bars) and of
    real-text gold chunks (line) in each 0.25-bpc bin, on a log scale so the few high chunks
    show; gold noise chunks as ×; the gold maximum (dashed) and the limit (solid, with the
    number of corpus chunks it drops). Garbage chunks show up as a separate group of bars far
    right of the gold line; the limit belongs in the empty gap before them.
- **`ocrfix bpc-hist <scored>... [--out bpc_hist.png] [--calibration yaml]`:** the bpc
  histogram of any `score` output, on its own (no model, no GPU; reads only the scores): per
  language a PNG panel (share of chunks per 0.25-bpc bin, log scale) and a text table in the
  terminal (count per bin, p50 / p99 / p99.9 / max), with the `noise_chunk` limit of the
  calibration file (default `score.calibration`) and how many chunks are above it. Edit
  `calibration.yaml` and run it again to see what a new value drops. Examples:

  ```bash
  ocrfix bpc-hist data/real/scored_raw --out data/real/bpc_raw.png       # pass 1
  ocrfix bpc-hist data/real/scored --out data/real/bpc_final.png         # pass 2
  ```
- **How to decide:** open `noise_chunk_samples.md`. Per language it shows up to 5 random corpus
  chunks in each 0.5-bpc band, from 1.0 below the limit to 4.0 above it (and above that),
  each marked kept or dropped. Every band marked dropped should be garbage. Real text in a
  dropped band: raise that language's value in `calibration.yaml` above it. Garbage in a kept
  band right below the limit: lower it, but not below the gold maximum (real text lives
  there). Real garbage usually sits far above real text, so the bands just above the limit
  are often empty.
- **The test data (4B):** gold limits en 2.04, ko 4.65, mixed 2.57 (gold real-text max 1.04 /
  3.65 / 1.57); 0 of 13,614 test corpus chunks above them.
- **Pass-1 bpc is before word-level removal**, so it is never lower than the bpc pass 2
  compares: a chunk you saw under the limit stays under it.
- **`report`:** bpc and flag rates before and after, per language; with pass 2, how many
  chunks were dropped whole.
- **`ocrfix summary data/real/refined`:** accept rate and the reasons for rejections.
- **Output:** `data/real/corrected/` holds the corrected `.md` files. `data/real/corpus/` is
  never changed.
- **Nearly empty documents are dropped:** when noise removal took (almost) all of a document,
  `reconstruct --source current` does not write it. A document is dropped when the corrected
  text has fewer than `reconstruct.min_letters` (300) letters/digits (any script; math and
  tables count), or keeps less than `reconstruct.min_kept_share` (0.1) of the original's.
  Each one is listed in `data/real/corrected/dropped_docs.jsonl` (`source_path, letters,
  letters_original, kept_share, chunks, noise_chunks`), so you can check them; the command
  prints how many. Turn it off with `--set reconstruct.min_letters=0 --set
  reconstruct.min_kept_share=0`. **A document with a table is never dropped** (noise removal
  never removes tables, so a table-only document stays). Math stays in a dropped chunk, so a garbage
  document full of formulas keeps some letters: raise `min_letters` if such documents remain.
  On the test data no document was dropped (the smallest kept 1,218 letters).
- **Several machines:** see `configs/profiles/cluster.yaml.example`.
