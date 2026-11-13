# ocrfix

Finds and fixes OCR errors in Markdown documents (MinerU output, Korean and English, with LaTeX
math and tables) using a locally hosted Qwen3.5 model. Runs fully **offline**.

- **What it fixes:** character confusions (`Tab1e` → `Table`, `방볍` → `방법`), wrong spacing,
  broken lines and hyphenation, stray page numbers, and garbage text.
- **What it never touches:** math, tables, links, code and citations. They are replaced by
  placeholders (`⟨M1⟩`, `⟨T1⟩`, `⟨P1⟩`) before any model sees the text, and put back afterwards.
- **Other languages:** chunks mostly in Chinese or Japanese (Han, kana) are tagged `cjk` and
  passed through unchanged (`correct.langs: [ko, en, mixed]`), and noise removal never deletes
  Chinese or Japanese text: measured on real Chinese text, it would otherwise remove about 17%
  of it (Chinese has no spaces, so a whole sentence is judged as one word, and one in six such
  words scores above the threshold). Hanja inside Korean text keeps the chunk `ko`.
- **Machines:** an RTX 5090 with **Qwen3.5-4B**, or H200s with **Qwen3.5-9B**.

Guides: [docs/real_data.md](docs/real_data.md) (every command on real data, in order),
[docs/annotation_guideline.md](docs/annotation_guideline.md) (how to write gold truth files),
[docs/runbook.md](docs/runbook.md) (install details, the H200s, adding a machine), and
[CLAUDE.md](CLAUDE.md) (the full reference for every command and setting).

## How it works

```
gold .md pairs ─► gold chunks ─► errors table ─► corrupt ─┬─► score ─► calibrate            (when is a chunk damaged)
                      │                                    └─► data sft ─► train sft ─► data dpo ─► train dpo
                      └─► calibrate-noise                                         (what is noise)

corpus .md ─► chunk ─► score (+ noise removal, flags) ─► correct (+ checks, re-score) ─► reconstruct ─► corrected .md
```

| command | what it does |
|---|---|
| `chunk` | cuts every `.md` file into 512–768-token chunks and masks math, tables and links |
| `score` | the base model measures how surprising each token is; deletes long runs of garbage (**noise removal**) and whole chunks that are garbage (bpc above the calibration's `noise_chunk` limit); flags damaged chunks (with the calibration) |
| `correct` | the trained model rewrites each flagged chunk (or all with `--all`); the result must pass checks and score better than before, else the original is kept |
| `reconstruct` | chunks back to `.md` files, the original (byte-exact) or the corrected text; a document noise removal left nearly empty is not written (listed in `dropped_docs.jsonl`); documents with tables are always kept |
| `gold chunks` | your hand-corrected `.md` pairs → chunk pairs, split by document into sft / dpo / eval |
| `errors table` | the real error types and rates in your gold |
| `corrupt` | adds fake OCR errors to clean text at those rates (for calibration and training data) |
| `calibrate` | decides which scores mean "damaged", and above which bpc a whole chunk is garbage (`data/calibration/calibration.yaml`) |
| `calibrate-noise` | decides which text is noise to delete (writes `configs/stages/score.yaml`) |
| `bpc-hist` | histogram of chunk bpc per language from any `score` output, with the noise-chunk limit (a PNG and a text table; no model needed) |
| `calibrate-noise-chunk` | decides above which bpc a whole chunk is garbage, from the gold, and shows what that drops in the corpus (writes `noise_chunk` in `calibration.yaml`) |
| `data sft` / `data dpo` | the training files: damaged → correct examples; the model's own mistakes as DPO pairs |
| `train sft` / `dpo` / `merge` | QLoRA, LoRA or full fine-tuning; merge an adapter into a model |
| `eval` | corrected documents vs your truth files: CER, false fixes, recall per error type |
| `serve` | starts the vLLM engines the other commands send requests to |
| `summary`, `report`, `compare`, `config` | run statistics, corpus quality before/after, vLLM vs transformers, the settings in effect |

## Install

On an online machine with the same Python (3.12, x86_64):

```bash
python3 scripts/make_wheelhouse.py wheelhouse          # downloads every pinned package
```

Copy `wheelhouse/` and the model folder to each offline machine, then:

```bash
python3 scripts/install_offline.py wheelhouse .venv    # never touches the network
source .venv/bin/activate
```

## What you prepare

```
data/real/
  corpus/        the MinerU .md tree to correct (e.g. 50 GB), as it is
  gold/ocr/      MinerU .md files picked for gold, copied unchanged
  gold/truth/    the same files, hand-corrected, same relative paths
  clean/         born-digital markdown in the same style (no OCR errors)
```

How to correct the truth files: [docs/annotation_guideline.md](docs/annotation_guideline.md).

## How to run

### A. One machine with a GPU (the RTX 5090)

The default profile is `rtx5090`: the 4B in the project folder, vLLM on ports 8100 (base
model) and 8101 (trained model; 8000 is taken on this machine).

```bash
E="data/real/errors/error_table.json data/real/errors/confusion_map.json"

# 1. gold, error rates, clean chunks (CPU)
ocrfix gold chunks data/real/gold data/real/gold_chunks
ocrfix errors table data/real/gold_chunks/sft data/real/errors
ocrfix chunk data/real/clean data/real/chunks_clean

# 2. noise and damage thresholds (base model)
ocrfix serve &                                         # wait for "ready"
ocrfix calibrate-noise data/real/gold_chunks           # -> configs/stages/score.yaml
ocrfix corrupt data/real/chunks_clean $E data/real/calib/corrupt/table.jsonl
ocrfix corrupt data/real/chunks_clean $E data/real/calib/corrupt/r0.03.jsonl --rate 0.03
S=(--set score.calibration=null --set score.noise.remove=false)
ocrfix score data/real/chunks_clean data/real/calib/clean_scored "${S[@]}"
ocrfix score data/real/calib/corrupt data/real/calib/corrupt_scored "${S[@]}"
ocrfix calibrate data/real/calib/clean_scored data/real/calib/corrupt_scored data/calibration

# 3. SFT (stop the engine first: Ctrl-C, or pkill -f "[o]crfix serve")
ocrfix corrupt data/real/chunks_clean $E data/real/train/corrupt_table.jsonl
ocrfix data sft data/real/sft --synthetic data/real/train/corrupt_table.jsonl \
  --gold data/real/gold_chunks/sft --identity data/real/chunks_clean
ocrfix train sft data/real/sft/train.jsonl runs/sft --eval data/real/sft/val.jsonl
ocrfix train merge runs/sft runs/sft_merged

# 4. DPO (optional; base + SFT model, each with ~half the GPU)
T=(--set correct.model=runs/sft_merged)
ocrfix serve --memory 0.44 &
ocrfix serve --correct --memory 0.42 "${T[@]}" &
ocrfix corrupt data/real/chunks_clean $E data/real/train/corrupt_dpo.jsonl --set seed=7
ocrfix data dpo data/real/dpo/pairs.jsonl --synthetic data/real/train/corrupt_dpo.jsonl \
  --gold data/real/gold_chunks/dpo --exclude data/real/sft/train.jsonl "${T[@]}"
# stop the engines, then
ocrfix train dpo data/real/dpo/pairs.jsonl runs/sft runs/dpo
ocrfix train merge runs/dpo runs/dpo_merged

# 5. measure on the gold eval documents (engines as in 4)
ocrfix score data/real/gold_chunks/eval data/real/eval/scored
ocrfix correct data/real/eval/scored data/real/eval/sft "${T[@]}"
ocrfix eval data/real/gold/truth sft=data/real/eval/sft data/real/eval/report

# 6. the corpus (engines as in 4)
ocrfix chunk data/real/corpus data/real/chunks
ocrfix score data/real/chunks data/real/scored_raw \
  --set score.noise.remove=false --set score.noise.whole_chunks=false   # pass 1: scores only
ocrfix calibrate-noise-chunk data/real/gold_chunks data/real/scored_raw    # whole-chunk limit
#   look at data/calibration/noise_chunk_hist.png and noise_chunk_samples.md
#   (any time, no GPU: ocrfix bpc-hist data/real/scored_raw --out bpc_hist.png);
#   edit noise_chunk in calibration.yaml if needed
ocrfix score data/real/chunks data/real/scored                             # pass 2: noise removed
ocrfix correct data/real/scored data/real/refined "${T[@]}"
ocrfix report data/real/refined --clean data/real/calib/clean_scored
ocrfix reconstruct data/real/refined data/real/corrected --source current
```

What to check after each step, and the options, are in [docs/real_data.md](docs/real_data.md).
Every command resumes where it stopped: run it again after an interruption.

### B. A CPU server and a GPU server (H200x8)

Keep the project and all data on the **CPU server**. The **GPU server** only serves the models
and trains. Commands that send requests (`score`, `correct`, `data dpo`, `calibrate-noise`) run
on the CPU server and talk to the GPU server over the network. So does
`calibrate-noise-chunk` (it scores the gold).

| step | runs on |
|---|---|
| `gold chunks`, `errors table`, `chunk`, `corrupt`, `data sft`, `calibrate`, `bpc-hist`, `eval`, `report`, `reconstruct` | CPU server |
| `score`, `correct`, `data dpo`, `calibrate-noise`, `calibrate-noise-chunk` (send requests to the GPU server) | CPU server |
| `serve` (the vLLM engines), `train sft` / `dpo` / `merge` | **GPU server** |

**Set up once:**
1. Install ocrfix on both machines (same wheelhouse).
2. The GPU server needs the full model, e.g. `/models/Qwen3.5-9B`. The CPU server needs only
   its small files at the **same path**: `config.json`, `generation_config.json`,
   `tokenizer.json`, `tokenizer_config.json` (to count tokens; it never loads the weights).
3. In `configs/profiles/h200x8.yaml`, set `hw.infer.host` to the GPU server's LAN address
   (e.g. `10.0.0.20`) on both machines. The engines listen on it; the CPU server sends to it.
   Base engines are on ports 8000–8007, tuned engines on 8100–8107, both on every GPU.
4. On both machines: `export OCRFIX_PROFILE=h200x8`.
5. Check from the CPU server, once the engines run: `curl http://10.0.0.20:8000/health`.
6. If both machines can mount the same storage, put the project there: then nothing needs
   copying.

**Run** (`[cpu]` / `[gpu]` = where; the same commands as in A):

```
[cpu] gold chunks, errors table, chunk clean                      (step 1)
[gpu] ocrfix serve                                                (base engines; leave running)
[cpu] calibrate-noise, corrupt, score, calibrate                  (step 2)
[cpu] corrupt, data sft                                           (step 3, data part)
      copy data/real/sft/ to the GPU server
[gpu] stop the engines; ocrfix train sft ...; ocrfix train merge runs/sft /models/ocrfix-sft
[gpu] ocrfix serve &   ocrfix serve --correct --set correct.model=/models/ocrfix-sft &
[cpu] data dpo ... --set correct.model=/models/ocrfix-sft          (step 4)
      copy pairs.jsonl to the GPU server; [gpu] stop the engines; train dpo; merge
[gpu] serve base + tuned again
[cpu] score, correct, eval (step 5); chunk, score (pass 1), calibrate-noise-chunk,
      score (pass 2), correct, report, reconstruct (step 6)
      with --set correct.model=<the merged model's path on the GPU server>
```

- **Network:** scoring sends token ids and gets one number back per token: roughly 1–2 TB of
  responses for a 15B-token corpus. Fine on 10 Gb Ethernet.
- **Security:** the engines have no password; the GPU server must be reachable only from your
  internal network.
- **Interrupted?** Every `score` / `correct` run skips finished shards; run it again.

## Settings

Each step has one file, `configs/stages/<step>.yaml`; the machine has one file,
`configs/profiles/<machine>.yaml`. Edit a file, or change one run with `--set key=value`.

| file | what you set there |
|---|---|
| `stages/chunk.yaml` | chunk size, which files, what is masked |
| `stages/score.yaml` | vLLM or transformers, spike threshold, noise removal (set by `calibrate-noise`), whole-chunk removal on/off (`score.noise.whole_chunks`), the calibration file |
| `stages/gold.yaml` | gold split (eval / dpo shares), error-table options |
| `stages/corrupt.yaml` | the damage generator |
| `stages/calibrate.yaml` | allowed false flags; the whole-chunk limit's margin over real gold text (`noise_chunk.margin`, 1.0 bits/char); how much real text noise removal may delete (1 in 10,000) |
| `stages/sft.yaml` | SFT data mix, junk, lost words (`sft.lost.prob`, off), gold repeats; lr, epochs, LoRA, batch |
| `stages/dpo.yaml` | DPO sampling and pair types; beta, lr, epochs, reference |
| `stages/correct.yaml` | the correction model (`correct.model`), which languages to correct (`langs`), vLLM or transformers, decoding, the checks and the re-score rule |
| `stages/eval.yaml` | eval options |
| `stages/reconstruct.yaml` | when a nearly empty corrected document is dropped (`min_letters` 300, `min_kept_share` 0.1) |
| `profiles/<machine>.yaml` | the base model path, GPUs, training batch and mode, vLLM engines (count, memory, host, ports) |
| `default.yaml` | seed, work folder, language rule |

- **Each setting lives in exactly one place.** A setting in two stage files is an error, and a
  profile may set only `model` and `hw`.
- **Derived, never set by hand:** the tokenizer is the model's; the server addresses come from
  `hw.infer` (`host`, `base_port`, `correct_port`); the names vLLM serves the models under
  come from `model` / `correct.model`; one `seed`.
- **Precedence:** `default.yaml`, every `stages/*.yaml`, the profile, then `--set`.
- **Check a value:** `ocrfix config --key score.noise.min_chars`.
- **Profiles:** `rtx5090` (default), `h200x4`, `h200x8`; several machines serving different
  models: `configs/profiles/cluster.yaml.example`.
- **After switching 4B → 9B:** run `calibrate-noise` and `calibrate` again; scores belong to
  one model.

## Results so far (generated test data)

| what | result |
|---|---|
| `chunk` / `reconstruct` | byte-exact on 50 MB |
| SFT on the 4B (QLoRA, 1 epoch, 64 min) | document CER 0.58% → 0.13% (flagged chunks), 0.01% (`--all`); base model 0.23% |
| DPO | runs; on test data the SFT model was right on 1,299 of 1,300 new prompts, so DPO had nothing to learn |
| `〈?〉` for destroyed words | the model marked 63% of them instead of guessing (old model: guessed 97%). **Off by default** (`sft.lost.prob: 0`): a marked sentence would have to be dropped from training text |
| noise removal (`calibrate-noise`, 697 test gold chunks) | at most 1 in 10,000 real characters removed: 60% of inserted junk found, 0.0096% of real text removed (the hand-set values: 72%, 0.024%) |
| whole-chunk noise (4B; limits from synthetic data en 3.9 / ko 5.1 / mixed 4.2, from the test gold en 2.0 / ko 4.6 / mixed 2.6 bits per char) | 0 of 13,614 test OCR chunks dropped (highest bpc 3.5); 60 of 60 all-garbage chunks dropped; 60 of 60 half-garbage chunks kept (word-level removal cleaned them) |
| vLLM vs transformers (`compare`) | flags agree on 99.6% of chunks, greedy answers identical on 100% |

The test text comes from templates, so these numbers are optimistic. The real measure is your
gold (`ocrfix eval`).

## Test data and tests

```bash
python scripts/make_test_corpus.py data/test --ocr-mb 50 --clean-mb 20   # generated, ~2 s
pytest                                     # ~1 min, a tiny random model, never uses the network
ruff check . && ruff format --check .
```

`make_test_corpus.py` writes `ocr/` (with injected errors), `truth/` (the answer key), `clean/`
and `ocr_errors.jsonl`. With `--lost-words 0.003`, `ocr/` + `truth/` work as a gold set
(`mkdir -p data/test/gold && ln -s ../ocr data/test/gold/ocr && ln -s ../truth data/test/gold/truth`).

## Layout

```
configs/default.yaml        shared settings (seed, work folder, language rule)
configs/stages/*.yaml       one file per step
configs/profiles/           one file per machine: model path + GPU settings
src/ocrfix/cli.py           the `ocrfix` commands
src/ocrfix/chunks/          chunk, reconstruct
src/ocrfix/stages/          score, correct, corrupt, data sft|dpo, eval, compare, summary, report
src/ocrfix/serve.py         ocrfix serve (vLLM engines)
src/ocrfix/calibrate/       calibrate (flag rule), calibrate-noise (noise thresholds)
src/ocrfix/gold/ errors/ corrupt/ train/   gold chunks, error tables, damage generator, training
src/ocrfix/scorer/ llm/ verify/ eval/ data/ corpus/   scoring model, vLLM client, checks, metrics, text analysis
scripts/                    make_wheelhouse.py, install_offline.py, make_test_corpus.py
docs/                       real_data.md, annotation_guideline.md, runbook.md
```
