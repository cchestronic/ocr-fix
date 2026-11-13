# ocrfix

Finds and fixes OCR errors in Markdown documents (MinerU output, Korean and English, with
LaTeX math and tables) using a locally hosted Qwen3.5 model. Runs fully **offline**.

- **What it fixes:** character confusions (`Tab1e` → `Table`), wrong spacing, broken words and
  lines, stray page numbers.
- **What it never touches:** math, tables, links, code and citations. They are replaced by
  placeholders (`⟨M1⟩`, `⟨T1⟩`, `⟨P1⟩`) before any model sees the text, and restored afterwards.
- **Machines:** today one RTX 5090 with **Qwen3.5-4B**; later 4× H200 with **Qwen3.5-9B**.

The detailed reference for every command and setting is [CLAUDE.md](CLAUDE.md); the operator
guide is [docs/runbook.md](docs/runbook.md). **Running on real data, step by step** (gold `.md`
pairs → calibration → SFT/DPO → eval → the corpus): [docs/real_data.md](docs/real_data.md);
how to correct gold files: [docs/annotation_guideline.md](docs/annotation_guideline.md).

## How it works

Two parts in one repository:

**1. Refinement** (vLLM)
```
.md files ─► chunk ─► score ─► correct (flagged chunks) ─► re-score ─► verify ─► reconstruct ─► .md files
```
- **chunk:** every `.md` file → pieces of 512–768 tokens, masked, one JSONL row each.
- **score:** the language model (through vLLM) measures how surprising each token is; very surprising text is
  likely OCR damage. Placeholders and the 2 tokens after each are not scored.
- **correct:** the fine-tuned model rewrites flagged chunks; every placeholder must come back.
- **re-score + verify:** right after a correction, score it again and run safety checks (math
  unchanged, not rewritten too much, same language, score improved). A failed correction is
  rejected and the original text is kept.
- **reconstruct:** chunks → `.md` files: the original (byte-exact) or the corrected version.

**2. Training** the correction model: a human gold set gives the real error rates; clean `.md`
files are damaged on purpose at those rates to make SFT pairs; DPO pairs come from the SFT
model's own answers, judged by vLLM scores and the verify checks; then SFT → DPO → merge.

## Status

| stage | works on | status |
|---|---|---|
| `chunk`, `reconstruct` | 512–768-token chunks | **done**, tested byte-exact on 50 MB |
| `refine score`, `refine correct` (correct → re-score → verify per chunk) | chunks | **done**, tested against a fake vLLM and the transformers backend, and run on vLLM with the 4B on the 5090 |
| `refine corrupt`, `calibrate` | chunks (and paragraphs) | **done** on chunks, tested with the fake vLLM; real thresholds need the GPU |
| `refine data` (chunk SFT examples: damaged → clean, clean → unchanged, short junk) | chunks | **done**; 3,998 examples from the test data, all ≤ 2,048 tokens. QLoRA SFT on the 4B (1 epoch, 64 min): test doc CER 0.58% → 0.13% (flagged chunks) / 0.01% (`--all`), vs 0.23% for the base model |
| `refine dpo` (chosen = clean chunk, rejected = the model's own mistakes) | chunks | **done**; `train dpo` runs on its pairs. On the test data the SFT model is right on 1,299/1,300 new prompts, so DPO there has nothing to learn |
| `gold`, `errors table`, `corrupt`, `data sft/dpo` | paragraphs | working; gold and DPO moving to chunks |
| `train sft/dpo/merge` (QLoRA, LoRA or full) | — | **done** |
| `ingest`, `score`, `correct`, `verify`, `eval`, `pipeline` | paragraphs | the older paragraph pipeline; still used for calibration and training data until those move to chunks |

## Install

On an online machine (same Python 3.12, x86_64), or directly if this machine is online:

```bash
scripts/make_wheelhouse.sh wheelhouse          # online: download all pinned packages
scripts/install_offline.sh wheelhouse .venv    # offline: install from the wheelhouse
source .venv/bin/activate && source scripts/env.sh
```

Copy the model folder by hand. Its path is set **per machine** in the profile, so switching
4B ↔ 9B is one line:

| profile | file | `paths.base_model` |
|---|---|---|
| `rtx5090` (default) | `configs/profiles/rtx5090.yaml` | `Qwen3.5-4B` (in the project folder) |
| `h200x4` | `configs/profiles/h200x4.yaml` | `/models/Qwen3.5-9B` |

Check: `ocrfix config --profile rtx5090 --key paths.base_model`.

Every command takes `--profile NAME` (default `rtx5090`, or `$OCRFIX_PROFILE`),
`--set key=value` to override any setting, and `--limit N` for a quick test on a few items.

## Commands

### Chunk
```bash
ocrfix chunk /data/mineru data/chunks                    # every **/*.md under the folder
ocrfix chunk /data/mineru data/chunks --limit 100        # first 100 documents only
ocrfix chunk /data/mineru data/chunks --sample-frac 0.003   # 0.3% of documents, by path hash
ocrfix chunk /data/mineru data/chunks --workers 12 --set io.gzip=true   # 12 processes, gzipped
```
- Output: `data/chunks/part-NNNNN.jsonl`, plus `manifest.json` and `stats.json` (counts, token
  histogram, files that could not be read).
- Chunk size: `chunk.min_tokens` / `chunk.max_tokens` (512 / 768) in `configs/default.yaml`.
- Interrupted? Run the same command again: finished shards are skipped. A different input or
  setting into the same folder is refused; use a new folder.
- One row per chunk; the main fields:

  | field | meaning |
  |---|---|
  | `chunk_id`, `doc_id`, `source_path`, `chunk_index`, `n_chunks` | where it comes from |
  | `source_text` | the masked original text |
  | `text` | the masked corrected text; `null` = unchanged |
  | `placeholders` | `{"⟨M1⟩": "$x^2$", "⟨T1⟩": "<table>…"}` |
  | `prefix`, `suffix`, `raw` | whitespace and original bytes, for exact reconstruction |
  | `n_tokens`, `short`, `lang` | size (`short` = under 512, only at a document's end), language |
  | `status`, `score`, `history` | filled by later stages |

### Reconstruct
```bash
ocrfix reconstruct data/chunks                              # check only: rebuild in memory, compare
ocrfix reconstruct data/chunks out/original                 # write the original files (byte-exact)
ocrfix reconstruct data/chunks out/fixed --source current   # write the corrected files
```
The check compares every rebuilt file with the sha256 of the source file and exits with code 1
if any differs.

### Score, correct, verify (chunks)

```bash
# 1. start vLLM (one engine per GPU, settings from the profile); wait for "ready"
scripts/serve_vllm.sh --profile rtx5090 --set llm.adapter=runs/sft   # base model + correction adapter

# 2. score every chunk; the scores are written into each chunk row
ocrfix refine score data/chunks data/scored

# 3. correct flagged chunks; each correction is re-scored and verified at once
ocrfix refine correct data/scored data/refined
ocrfix refine correct data/scored data/refined --all     # every chunk, without a calibration

# 4. status counts, accept rate, failed checks
ocrfix refine summary data/refined

# 5. corpus quality per language: before noise removal, before and after correction
ocrfix refine report data/refined --clean data/clean_scored --json data/refined/report.json

# 6. the corrected markdown files
ocrfix reconstruct data/refined out/fixed --source current
```
- **`refine report`** shows, per language, bpc (all bits / all characters, so long chunks weigh
  more) and the share of chunks the calibration flags, at three stages: before noise removal,
  before correction, after. `--clean` adds clean text in the same language as a baseline
  (score it with `--set refine.noise.remove=false`) and how much of the gap to it noise
  removal and correction each closed. Compare languages only with their own clean baseline:
  clean Korean scores about 1.13 bpc, clean English about 0.46. These are the model's view;
  check the real error rate on gold chunks.
- **Flagging** uses the calibration (`scorer.calibration`, default
  `data/calibration/calibration.yaml`); without one, `correct` stops and asks for `--all`.
- **What `correct` does per chunk:** send the whole masked chunk to the correction model → check
  placeholders, new math/tables, length, edit size, language, `〈?〉` → re-score the corrected
  text → accept only if the score improved. Accepted: `text` is replaced and `status` is
  `accepted`. Otherwise the text stays (`rejected`, `unchanged` or `error`). Every attempt is
  kept in `history`, including the model's answer and the checks.
- **Scoring** uses the base model, never the adapter (`refine.score_model`), so scores before
  and after correction are comparable.
- **Separate servers:** `--set 'refine.score_urls=[http://127.0.0.1:8000]' --set
  'refine.correct_urls=[http://127.0.0.1:8001,http://127.0.0.1:8002]'`. Both default to
  `hw.infer.urls`, the engines `serve_vllm.sh` starts.
- **Without vLLM:** `--set refine.scorer=transformers` scores with the local model on this
  process's GPU (slower; correction still needs vLLM).
- **Port 8000 in use?** Add `--set hw.infer.base_port=8100 --set
  'hw.infer.urls=[http://127.0.0.1:8100]'` to both `serve_vllm.sh` and `refine`.
- **Interrupted?** Run the same command again: finished shards are skipped. `--limit N`
  processes only the first N shards.

The older paragraph commands (`ingest` → `fix` → `score` → `correct` → `verify`, or
`ocrfix pipeline`) still work; see [docs/runbook.md](docs/runbook.md).

### Random noise removal

`refine score` also finds random noise (`ⅡⅢ▯ ¤ʘ…`, `lIl1 :;. rnvv`, garbled Hangul) and
**cuts it out automatically**, then scores the cleaned chunk again.
- **How:** from the scores, as **bits per character** of each word. Noise packs many
  surprising tokens into few characters; real text, even with OCR errors, doesn't. A word
  above the `core` threshold starts a noise run; neighbouring words above `extend` join it.
- **Thresholds** by the word's script, not the chunk's language (a Hangul syllable carries
  more bits than a Latin letter): `refine.noise.core` (Hangul 8.0, other 5.0) and
  `refine.noise.extend` (6.0 / 3.5). Set on the 4B with the test data (removed text that is
  noise: 97%, noise found: 58% at 20 characters). Check them on real chunks, and again for
  the 9B.
- **Only long stretches** (`refine.noise.min_chars`, 20 non-space characters) are removed;
  shorter ones are left for the corrector. Placeholders are never removed.
- **Reversible:** each removal is recorded in `history` (`stage: denoise`, position and removed
  text); `source_text` keeps the original. A chunk with no text left gets `status: noise` and
  is never sent to the corrector.
- **Off:** `--set refine.noise.remove=false` (always off when scoring calibration inputs).
- **Not caught:** repeated junk such as `. . . .`, `1 1 1 1`, or the same junk string again
  (the model predicts repetition), and noise runs under 20 characters.

### Calibration (chunks)

Decides which scores count as "damaged". Inputs: error rates from the gold set, and clean
chunks (from your clean `.md` files).

```bash
ocrfix errors table data/gold/gold_train.jsonl data/errors          # error rates from gold
ocrfix chunk /data/clean_md data/clean_chunks                       # clean chunks
E="data/errors/error_table.json data/errors/confusion_map.json"
ocrfix refine corrupt data/clean_chunks $E data/calib/corrupt/table.jsonl          # gold rates
ocrfix refine corrupt data/clean_chunks $E data/calib/corrupt/r0.01.jsonl --rate 0.01
ocrfix refine corrupt data/clean_chunks $E data/calib/corrupt/r0.03.jsonl --rate 0.03
ocrfix refine corrupt data/clean_chunks $E data/calib/corrupt/r0.1.jsonl  --rate 0.1
S="--set scorer.calibration=null --set refine.noise.remove=false"
ocrfix refine score data/clean_chunks data/calib/clean_scored $S
ocrfix refine score data/calib/corrupt data/calib/corrupt_scored $S
ocrfix calibrate data/calib/clean_scored data/calib/corrupt_scored data/calibration
```
- `--rate R` = R errors per prose character (0.01 = 1 in 100 characters, 0.1 = 1 in 10, very
  heavy). Without `--rate`, each chunk gets a rate drawn from the gold set's measurements.
- `refine corrupt` damages prose only: math, tables, links, citations and the spaces next to
  them are never touched, so the placeholders stay the same.
- `calibrate` writes `data/calibration/calibration.yaml` (the default `scorer.calibration`).
  From then on `refine score` writes `flag` into each chunk, and `refine correct` sends only
  flagged chunks. A chunk is flagged when `spike_frac > A or p90 > B or bpc > C` (per
  language; `calibrate` picks the parts and thresholds with the best recall at <= 5% of clean
  chunks flagged). Use a few thousand clean chunks (`--limit` on `refine corrupt`).
- Scores and calibration belong to one model: after switching 4B → 9B, score and calibrate
  again.

### Training data (chunks)

`refine correct` sends one whole chunk with its own prompt, so the model is trained on exactly
that form: `refine data` builds the examples, `train sft` trains on them.
```bash
# damaged copies of clean chunks, at the gold set's error rates (and a fixed rate for variety)
ocrfix refine corrupt data/chunks_clean data/errors/error_table.json data/errors/confusion_map.json data/train/corrupt_table.jsonl
ocrfix refine corrupt data/chunks_clean data/errors/error_table.json data/errors/confusion_map.json data/train/corrupt_r0.03.jsonl --rate 0.03

ocrfix refine data data/sft_chunks \
  --synthetic data/train/corrupt_table.jsonl --synthetic data/train/corrupt_r0.03.jsonl \
  --identity data/chunks_clean
scripts/train.sh sft data/sft_chunks/train.jsonl runs/sft --eval data/sft_chunks/val.jsonl --profile rtx5090
```
- **synthetic:** damaged chunk → clean chunk. 30% of them also get 1–3 short junk runs
  (symbols, `lI1`-like strings, random Hangul; 3–30 characters, after a sentence or as their
  own paragraph) that the answer leaves out: the junk `refine score` doesn't remove.
- **identity:** clean chunks with themselves as the answer (25% of the mix), so the model
  learns to leave correct text alone.
- Settings in `refine.data` (mix, `lang_balance`, `total`, `junk`); the split is by document
  (`data.val_frac`), and documents in the gold eval split are never used.
- `stats.json`: chunks per source and language, what was taken, what was dropped and why.
- **lost words:** 10% of synthetic chunks get 1–2 words replaced by garbage that holds nothing
  of them (random syllables or letters, boxes, symbols), answered with `〈?〉`, so the model
  learns to mark what it can't recover instead of inventing a word.
- **gold:** `--gold data/real/gold_chunks/sft` (`ocrfix gold chunks` output from hand-corrected
  `.md` pairs), each chunk twice; see [docs/real_data.md](docs/real_data.md).

**DPO pairs** come from the SFT model's own mistakes on prompts it hasn't seen (chosen is
always the known clean chunk):
```bash
scripts/serve_vllm.sh --profile rtx5090 --set llm.adapter=runs/sft     # base + SFT adapter
ocrfix refine corrupt data/chunks_clean data/errors/error_table.json data/errors/confusion_map.json data/train/corrupt_dpo.jsonl --set seed=7
ocrfix refine dpo data/dpo_chunks.jsonl --synthetic data/train/corrupt_dpo.jsonl \
  --identity data/chunks_clean --exclude data/sft_chunks/train.jsonl --set llm.adapter=runs/sft
scripts/train.sh dpo data/dpo_chunks.jsonl runs/sft runs/dpo --profile rtx5090
```
- Each prompt is answered once greedily (as `refine correct` does) and 4 times sampled.
  Every answer that isn't exactly the clean chunk is a rejected answer: the greedy one first,
  then the sampled ones closest to the right answer, at most 2 per prompt. Each pair is typed:
  `missed` (damage left), `false_fix` (correct text changed), `wrong_fix`, `mixed`,
  `placeholders`, `truncated`, `guessed` (a word where `〈?〉` belongs), `overcautious` (`〈?〉`
  where a fix was possible).
- Prompts the model already gets right every time get no pair, except a quarter of them that
  get one constructed wrong answer (the damaged input as is, half the fixes undone, or a
  placeholder dropped), so DPO doesn't drift on what already works.
- `--exclude`: prompts already in the SFT data are skipped; the SFT model learned their answers.
- **Serving a LoRA adapter crashed vLLM 0.33** (illegal memory access in its LoRA code when
  base-model scoring and adapter requests share a batch). Merge instead and run two engines:
  ```bash
  scripts/train.sh merge runs/sft runs/sft_merged --profile rtx5090
  scripts/serve_vllm.sh --profile rtx5090 --set hw.infer.gpu_memory_utilization=0.44 --set paths.work_dir=runs/vllm_base
  scripts/serve_vllm.sh --profile rtx5090 --set hw.infer.gpu_memory_utilization=0.44 --set paths.work_dir=runs/vllm_sft \
    --set hw.infer.base_port=8001 --set 'hw.infer.urls=[http://127.0.0.1:8001]' \
    --set llm.model=runs/sft_merged --set llm.served_model_name=ocrfix-sft
  ocrfix refine correct data/scored data/refined --set 'refine.score_urls=[http://127.0.0.1:8000]' \
    --set 'refine.correct_urls=[http://127.0.0.1:8001]' --set llm.served_model_name=ocrfix-sft --set refine.score_model=qwen3.5
  ```

**transformers instead of vLLM for small jobs** (DPO sampling, eval): `--set
refine.generator=transformers` answers the same requests in-process (`llm.model`, or
`llm.adapter` without merging). Check first that both backends agree: `ocrfix refine compare`
(scores, flags, noise spans, greedy answers); see [docs/real_data.md](docs/real_data.md).

### Training (summary, paragraph format)
```bash
ocrfix errors table data/gold/gold_train.jsonl data/errors     # error rates from the gold set
ocrfix corrupt clean.jsonl data/errors/error_table.json data/errors/confusion_map.json data/train/synthetic.jsonl
ocrfix data sft data/sft --synthetic data/train/synthetic.jsonl --gold data/gold/gold_train.jsonl
scripts/train.sh sft data/sft/train.jsonl runs/sft --profile rtx5090
ocrfix data dpo data/dpo.jsonl --gold data/gold/gold_train.jsonl --model ocrfix-lora --set llm.adapter=runs/sft
scripts/train.sh dpo data/dpo.jsonl runs/sft runs/dpo --profile rtx5090
scripts/train.sh merge runs/dpo runs/merged --profile rtx5090
```
Mode: `--set hw.train.mode=qlora|lora|full` (5090: `qlora`; H200: `lora` or `full`). The full
sequence with gold and calibration is in [docs/runbook.md](docs/runbook.md).

## Test data

Generate a synthetic test corpus (no download, no model, ~2 s):
```bash
python scripts/make_test_corpus.py data/test --ocr-mb 50 --clean-mb 20
```
- `data/test/ocr/`: documents with injected OCR errors (to refine)
- `data/test/truth/`: the same documents without errors (answer key, for testing only)
- `data/test/clean/`: other error-free documents (for SFT data)
- `data/test/ocr_errors.jsonl`: what was injected per document (for testing only; on real data
  the error rates come from the gold set)

A fake gold set of chunk pairs from the test data (test only; real gold comes from annotators):
```bash
python scripts/make_test_gold.py data/test/chunks/truth data/test/gold --n 5000
ocrfix errors table data/test/gold/gold_train.jsonl data/test/errors
```

The sentences come from templates: fine for testing the pipeline and speed, not for judging
model quality.

## Tests

```bash
pytest                                  # ~3 min, tiny random model, never uses the network
ruff check . && ruff format --check .
```

## Layout

```
configs/default.yaml        all settings except hardware
configs/profiles/           one file per machine: model path + GPU settings
src/ocrfix/chunks/          chunk, reconstruct
src/ocrfix/refine/          refine score / correct / summary (chunks)
src/ocrfix/scorer/          paragraph scoring; its model code is the transformers backend
src/ocrfix/llm/             correct (vLLM client, prompt, parsing)
src/ocrfix/verify/          verify
src/ocrfix/calibrate/       flag thresholds
src/ocrfix/gold/ errors/ corrupt/ data/ train/ eval/   training side
scripts/                    install, serve_vllm.sh, train.sh, make_test_corpus.py
docs/                       runbook, annotation guideline
```
