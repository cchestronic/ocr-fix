# ocrfix

Finds and fixes OCR errors in Markdown made by MinerU (Korean and English, with LaTeX math
and tables) with a Qwen3.5 model on your own machines. Runs fully **offline**.

- **Fixes:** misread characters (`Tab1e` → `Table`, `방볍` → `방법`), wrong spacing, broken
  lines and hyphenation, stray page numbers and headers, garbage text; and, before anything
  else, broken LaTeX and tables (`markup`, rule-based, CPU only).
- **Never lets the model touch:** math, tables, links, code, citations. They are replaced by
  placeholders (`⟨M1⟩` math, `⟨T1⟩` table, `⟨P1⟩` other) before a model sees the text, and put
  back afterwards.
- **Skips:** chunks that are mostly Chinese or Japanese pass through unchanged.

This file is the guide: what to run, in which order, and where every setting lives.
More detail: [docs/real_data.md](docs/real_data.md) (what to check after each step),
[docs/annotation_guideline.md](docs/annotation_guideline.md) (how to write gold truth files),
[docs/runbook.md](docs/runbook.md) (install, H200s, adding a machine),
[CLAUDE.md](CLAUDE.md) (the full technical reference).

---

## 1. The pipeline at a glance

There are three tracks. You run them in this order: **A** once, **B** once per model, **C**
on the corpus.

```
A. PREPARE (CPU)
   markup  ──► gold chunks ──► errors table          chunk (clean)
   (corpus, gold, clean)        (error rates)          │
                                                       ▼
B. LEARN (GPU)                                       corrupt (fake damage at the gold rates)
   calibrate-noise ─► score (clean + damaged) ─► calibrate        what is noise / damage
   data sft ─► train sft ─► train merge ─► [data dpo ─► train dpo ─► train merge]
   score + correct the gold eval docs ─► eval          how good is the model

C. CORRECT THE CORPUS (GPU)
   chunk ─► score (pass 1) ─► calibrate-noise-chunk ─► score (pass 2) ─► correct
         ─► report ─► reconstruct ─► corrected .md files
```

| step | command | needs | input → output |
|---|---|---|---|
| A1 | `markup` | CPU | `.md` tree → fixed `.md` tree (LaTeX, tables) |
| A2 | `gold chunks` | CPU | `gold/ocr` + `gold/truth` → chunk pairs, split sft / dpo / eval |
| A3 | `errors table` | CPU | gold sft chunks → error rates and types |
| A4 | `chunk` | CPU | `.md` tree → chunks of 512–768 tokens (masked) |
| A5 | `corrupt` | CPU | clean chunks + error tables → fake-damaged chunks |
| B1 | `serve` | GPU | starts the model servers (vLLM) that score / correct use |
| B2 | `calibrate-noise` | GPU | gold chunks → noise-removal thresholds (into `score.yaml`) |
| B3 | `score` | GPU | chunks → + surprise scores, noise removed, `flag` |
| B4 | `calibrate` | CPU | clean + damaged scores → `calibration.yaml` (which chunks are damaged) |
| B4b | `calibrate --gold-scores` | GPU (score), CPU | the same thresholds checked on real gold OCR chunks |
| B5 | `data sft` | CPU | damaged / gold / clean chunks → `train.jsonl`, `val.jsonl` |
| B6 | `train sft`, `train merge` | GPU | → LoRA adapter → merged model |
| B7 | `data dpo`, `train dpo` | GPU | optional second training round |
| B8 | `correct`, `eval` | GPU, CPU | gold eval docs → CER, false fixes, recall per error type |
| C1 | `calibrate-noise-chunk` | GPU | gold + corpus scores → limit for whole-garbage chunks |
| C2 | `correct` | GPU | scored chunks → corrected chunks |
| C3 | `report`, `summary` | CPU | quality before / after, accept rates |
| C4 | `reconstruct` | CPU | chunks → `.md` files |

Every step is resumable: if it stops, run the same command again and it continues.

**Two ways to run it:** one command per stage that runs every step with the default settings
(`./ocrfix pipeline prepare|calibrate|train|corpus`, section 5), or each step by hand
(sections 6–8). Section 9 lists every step's input, output and main settings.

---

## 2. How to run a command

The same way on every machine, from the project folder:

```bash
./ocrfix <command> <arguments> [options]
```

```bash
./ocrfix --help                 # every command
./ocrfix chunk --help           # one command's arguments and options
```

`./ocrfix` is a small launcher in the project folder (`ocrfix.cmd` on Windows, `ocrfix` on
Linux). It starts the project's own Python (`.venv`) with [run.py](run.py), so there is
nothing to activate. Works in PowerShell, cmd (`ocrfix ...`) and bash.

### Tell each machine which profile it is (once)

The **profile** (`configs/profiles/<name>.yaml`) is the only thing that differs between
machines: where the model is, which GPUs, how the model servers run. Every command prints the
one in use first, e.g. `[ocrfix] profile: cpu`.

| machine | profile | set it once |
|---|---|---|
| this Windows PC (no GPU) | `cpu` | nothing to do: it is the default |
| RTX 5090 | `rtx5090` | `echo 'export OCRFIX_PROFILE=rtx5090' >> ~/.bashrc` |
| H200 server | `h200x8` (or `h200x4`) | `echo 'export OCRFIX_PROFILE=h200x8' >> ~/.bashrc` |

For a single run, `--profile NAME` overrides it.

### Options every command takes

| option | what it does | example |
|---|---|---|
| `--profile NAME` | which machine file in `configs/profiles/` | `--profile h200x8` |
| `--set key=value` | change one setting for this run (repeatable) | `--set sft.total=30000` |
| `--limit N` | only the first N documents / shards (quick tests) | `--limit 2` |

### Which engine runs the model: vLLM, transformers, or none

The model is used in three ways, and each has its own engine:

- **vLLM** is a separate *server* program (`./ocrfix serve`) that holds the model on the GPU
  and answers requests over the network. Fast, and several GPUs or servers can share the work.
  Only for **using** the model (scoring, answering), never for training. Linux + GPU only.
- **transformers (HF)** loads the model *inside* the command itself: no server. Used for all
  **training**, and as a slower stand-in for vLLM on any machine (one GPU, or the CPU).
- **tokenizer only:** some CPU steps only need the model's tokenizer files (to count
  tokens); they never load the weights.

Two settings choose between vLLM and transformers for using the model. Both default to
**`auto`: the machine decides.** A profile with servers (a non-empty `hw.servers` list: the
5090 and H200 profiles) gets `vllm`; a profile without (`servers: []`: `cpu`, this PC) gets
`transformers`. So every command on one machine uses
the same engine, the scores stay comparable, and you never type it. Check with
`./ocrfix config --key score.backend`.

| setting | decides how ... | `vllm` | `transformers` |
|---|---|---|---|
| `score.backend` | the **base** model scores text (how surprising each token is) | requests to the base servers (the `model: base` groups of `hw.servers`) | the base model loaded in the command, on GPU 0 (or CPU) |
| `correct.backend` | the **correction** model writes answers | requests to the correction servers (the `model: trained` groups; the base ones while `correct.model` is not set) | `correct.model` loaded in the command, on GPU 0 (or CPU) |

Every command:

| command | model engine | GPU? |
|---|---|---|
| `markup`, `errors table`, `calibrate`, `data sft`, `eval`, `summary`, `report`, `bpc-hist`, `config` | **none** | no |
| `chunk`, `gold chunks`, `corrupt`, `reconstruct` | **tokenizer only** (`model`'s tokenizer files) | no |
| `serve` | **vLLM** (it *is* the vLLM server) | yes |
| `score` | `score.backend` | vLLM servers, or transformers on one GPU / CPU |
| `calibrate-noise`, `calibrate-noise-chunk` | `score.backend` (they score the gold) | as `score` |
| `correct` | **both:** answers with `correct.backend`, re-scores each answer with `score.backend` | as `score` |
| `data dpo` | `correct.backend` (the trained model answers the prompts) | as `score` |
| `compare` | **both on purpose:** vLLM servers vs transformers in the command, same chunks | yes (`score.backend` must be vllm) |
| `train sft`, `train dpo` | **transformers** + PEFT (LoRA) + TRL; QLoRA also bitsandbytes; several GPUs via torchrun | yes (all of the machine's `hw.n_gpus`) |
| `train merge` | **transformers** + PEFT | a QLoRA adapter needs a GPU (to unpack the 4-bit base); `--no-dequantize` or a LoRA adapter works on CPU |

What that means in practice:

- **GPU machine (auto = vllm):** start `./ocrfix serve` (base and, with `correct.model` set,
  trained model in one command) before the scoring / correcting steps, and stop them before training
  (training needs the GPU memory). `./ocrfix pipeline` does that for you.
- **This PC (auto = transformers):** nothing to start; every command loads the model itself.
  Nearly the same results, slower (measured on the RTX 5090: scoring ~2x, correction 4-8x),
  one GPU or the CPU.
- **Forcing one:** `--set score.backend=transformers --set correct.backend=transformers` (e.g.
  a GPU machine without vLLM installed), or `=vllm`. Mixing is allowed, but scoring is best
  kept on one engine from calibration to corpus.
- **Same model, same scores:** `compare` checks that vLLM and transformers agree (on the
  5090: flags 99.6% equal, greedy answers 100%). Calibrate with the backend you will use.
- Which machine: the "no model" and "tokenizer only" commands run anywhere (this PC
  included); the rest need a GPU machine, or its servers over the network (section 10,
  setups 2 and 3), or patience on CPU with transformers.

## 3. Install

**This PC (CPU, for development and the CPU steps):** already done: `.venv` with Python 3.12,
CPU torch and the locked packages, without vLLM / bitsandbytes (GPU only). To redo it:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install typer omegaconf rapidfuzz jamo matplotlib httpx regex numpy transformers accelerate peft trl datasets pytest ruff
.\.venv\Scripts\pip install --no-deps -e .
```

(use the versions in `requirements.lock`.) After installing, `./ocrfix --help` must list the
commands.

**GPU machines (air-gapped):** on an online machine with the same Python (3.12, x86_64):

```bash
python3 scripts/make_wheelhouse.py wheelhouse          # downloads every pinned package
```

Copy `wheelhouse/` and the model folder to the GPU machine, then:

```bash
python3 scripts/install_offline.py wheelhouse .venv    # never touches the network
```

---

## 4. What you prepare

```
data/real/
  corpus/        the MinerU .md tree to correct (e.g. 50 GB), as it is
  gold/ocr/      some MinerU .md files, copied unchanged
  gold/truth/    the same files, corrected by hand, same relative paths
  clean/         born-digital Markdown in the same style (no OCR errors)
```

How to write the truth files: [docs/annotation_guideline.md](docs/annotation_guideline.md).
More gold gives better error rates and a more reliable eval (20% of it is kept for eval
only). No real data yet? Generate test data (section 12).

---

## 5. The easy way: run a whole stage automatically

One command runs every step of a stage, in order, with the default settings, as section 9
lists them. It starts and stops the model servers itself, skips steps whose output already
exists, and stops after each stage so you can check the results before spending GPU time.

```bash
./ocrfix pipeline prepare      # Track A, CPU (this PC is fine)
./ocrfix pipeline calibrate    # B2-B4b, GPU
./ocrfix pipeline train        # B5-B8, GPU (add --set pipeline.dpo=true for DPO)
./ocrfix pipeline corpus       # Track C, GPU
```

| stage | steps it runs | then check |
|---|---|---|
| `prepare` | markup ×4 (corpus, gold ocr, gold truth, clean), gold chunks, errors table, chunk clean, corrupt ×3 | `corpus_fixed/markup_report.jsonl` (`kept_original` entries), `gold_chunks/stats.json` (most chunks `ok`), `errors/summary.md` |
| `calibrate` | calibrate-noise, score clean / damaged / gold, calibrate with the gold check | `data/calibration/summary.md`: the rules, and "Real gold OCR chunks" (heavily damaged chunks flagged?) |
| `train` | data sft, train sft, merge, [corrupt + data dpo + train dpo + merge], score + correct the eval docs (base, sft, [dpo]), eval | `eval/report/report.md`: the trained model's CER below `base` and `ocr`, few false fixes |
| `corpus` | chunk, score pass 1, calibrate-noise-chunk, score pass 2, correct, summary, report, bpc by stage, reconstruct | `corrected/` (the result), `bpc_stages.png` / `.json` (quality raw → denoised → corrected), `corpus_report.json`, `data/calibration/noise_chunk_samples.md` |

- `./ocrfix pipeline all` runs the four in a row, without the checks in between (for a
  rerun when you trust the data).
- **Before running:** `./ocrfix pipeline train --dry-run` prints every step's exact command
  and which ones are already done; it runs nothing.
- **Interrupted or failed:** fix the cause, run the same command again: finished steps are
  skipped. `--from N` starts at step N (numbers as in `--dry-run`), `--redo` runs finished
  steps again (or delete a step's output folder to redo just that one).
- **Settings:** every `--set` on the pipeline command reaches every step, e.g.
  `./ocrfix pipeline train --set sft.total=30000`. The pipeline's own settings are in
  `configs/stages/pipeline.yaml`:

| setting | default | what it does |
|---|---|---|
| `pipeline.data_dir` | `data/real` | where `corpus/`, `gold/`, `clean/` are; every output goes under it |
| `pipeline.calibration_dir` | `data/calibration` | `calibration.yaml` and the noise reports |
| `pipeline.runs_dir` | `runs` | `sft/`, `sft_merged/`, `dpo/`, `dpo_merged/` |
| `pipeline.dpo` | false | the train stage also does DPO, and evaluates it |
| `pipeline.model` | null | the corpus stage's correction model; null = `runs/dpo_merged` with DPO, else `runs/sft_merged` |
| `pipeline.max_steps` | null | stop training after N steps (quick tests) |
| `pipeline.write_noise` | true | calibrate-noise writes its thresholds into `configs/stages/score.yaml` |
| `pipeline.ready_timeout` | 1800 | seconds to wait for the servers to start |

- **Servers:** a step that scores starts this machine's base groups (`serve --only base`); a
  step that corrects with the trained model also starts the trained groups (`serve --only
  trained --set correct.model=...`); training stops them first. GPU memory is set per group
  (`hw.servers`, section 10), so nothing restarts when the second model joins. Servers you
  started yourself are used and left running.
- **Several GPU servers:** the pipeline starts servers only on the machine it runs on. Groups
  of `hw.servers` on other machines (section 10, setup 3) you start
  yourself, once, on each of them (`./ocrfix serve`); the pipeline then just uses them. If one
  doesn't answer, the stage stops and says to run `./ocrfix serve --only <kind>` there. Training
  (`train` stage) runs on the machine the pipeline runs on, using its GPUs: run that stage on
  a GPU server. On a machine without servers (this PC: the backends are `auto` =
  transformers) no server is started: slow, but it works.
- **Logs:** each step's full output in `<data_dir>/pipeline_logs/`; the screen shows it too.
- The steps are exactly the commands of sections 6–8, so you can always run one by hand.

---

## 6. Track A: prepare (CPU, this PC is fine)

### A1. Repair LaTeX and tables: `markup`

```bash
./ocrfix markup data/real/corpus      data/real/corpus_fixed
./ocrfix markup data/real/gold/ocr    data/real/gold_fixed/ocr
./ocrfix markup data/real/gold/truth  data/real/gold_fixed/truth
./ocrfix markup data/real/clean       data/real/clean_fixed
```

- Fixes unclosed `$$` / `\[` / `\begin{equation}`, blank lines inside `$$…$$` / `\[…\]`
  (they cut the formula in two), unbalanced braces and `\left`/`\right`, MinerU spacing
  (`x _ { i }` → `x_{i}`), broken HTML tables (missing `</td>` `</tr>` `</table>`, short
  rows), Markdown tables without a `|---|` line or with short rows.
- **What the model never sees** (masked as placeholders when chunking): every table as one
  `⟨T1⟩`, including any LaTeX inside its cells (HTML or Markdown tables); every formula as one
  `⟨M1⟩`, also over several lines (`$$…$$`, `\[…\]`, `\begin{align}…`). A blank line inside
  `$$…$$` or `\[…\]` would end the formula, so `markup` removes it first, but only when
  everything in between looks like math (no Korean, no run of ordinary words): a stray `$$`
  never swallows real paragraphs.
- **If a formula or table can't be fixed safely, the original is kept** and reported.
- Writes a copy; the input is never changed. Uses every CPU core; resumable.
- **Run it on gold too, both sides:** `gold chunks` keeps a chunk only if its math and tables
  are byte-identical in ocr and truth, so both must get the same repairs.
- From here on use the `_fixed` folders (or rename them to the original names).
- Check: the summary it prints, and `<out>/markup_report.jsonl` (each changed or flagged file;
  `kept_original` entries are the ones it didn't dare to fix).

### A2. Gold into chunk pairs: `gold chunks`

```bash
./ocrfix gold chunks data/real/gold_fixed data/real/gold_chunks
```

- Chunks `ocr/`, cuts the truth at the same places, splits **by document**: eval 20% (never
  trained on), dpo 20%, sft the rest (`gold.split` in `gold.yaml`). `split.json` pins the
  split, so reruns keep every document on its side.
- Check: most chunks `ok`. `placeholders_changed` = math/table differs between ocr and truth.

### A3. Error statistics: `errors table`

```bash
./ocrfix errors table data/real/gold_chunks/sft data/real/errors
```

- Output: `error_table.json`, `confusion_map.json`, `summary.md` (read this one: which
  errors your OCR makes, how often).

### A4. Chunk the clean text: `chunk`

```bash
./ocrfix chunk data/real/clean_fixed data/real/chunks_clean
```

### A5. Fake damage for calibration and training: `corrupt`

```bash
./ocrfix corrupt data/real/chunks_clean data/real/errors/error_table.json data/real/errors/confusion_map.json data/real/calib/corrupt/table.jsonl
./ocrfix corrupt data/real/chunks_clean data/real/errors/error_table.json data/real/errors/confusion_map.json data/real/calib/corrupt/r0.03.jsonl --rate 0.03
./ocrfix corrupt data/real/chunks_clean data/real/errors/error_table.json data/real/errors/confusion_map.json data/real/train/corrupt_table.jsonl --set seed=2
```

- Damages clean chunks at the rates measured in A3 (`--rate 0.03` = a fixed 3 errors per 100
  chars instead). The first two are for calibration (B), the third (another seed) for training.

---

## 7. Track B: learn (GPU)

Run these on the GPU machine.

### B1. Start the model servers: `serve`

In a terminal of its own, leave it running:

```bash
./ocrfix serve                       # the base model; wait until it prints "ready"
```

This starts this machine's groups of `hw.servers` (section 10). Without
`correct.model` only the base model starts (the trained groups are skipped with a note).
`score`, `correct`, `calibrate-noise`, `calibrate-noise-chunk` and `data dpo` send requests to
these servers. Stop them (Ctrl-C in that terminal) before training: training needs the whole
GPU.

### B2. Noise thresholds: `calibrate-noise`

```bash
./ocrfix calibrate-noise data/real/gold_chunks
```

Writes the noise-removal settings into `configs/stages/score.yaml` (`score.noise`).

### B3 + B4. When is a chunk damaged: `score` + `calibrate`

```bash
./ocrfix score data/real/chunks_clean data/real/calib/clean_scored --set score.calibration=null --set score.noise.remove=false
./ocrfix score data/real/calib/corrupt data/real/calib/corrupt_scored --set score.calibration=null --set score.noise.remove=false
./ocrfix calibrate data/real/calib/clean_scored data/real/calib/corrupt_scored data/calibration
```

- The two `--set`s score the text as it is: no calibration yet, no noise removal.
- `calibrate` writes `data/calibration/calibration.yaml`: per language, the scores above
  which a chunk is flagged as damaged (at most 5% of clean chunks flagged). From now on
  `score` uses it and every chunk gets `flag`.
- Check: `data/calibration/summary.md`.

### B4b. Check the thresholds on real OCR: `calibrate --gold-scores`

The thresholds above were fitted on *fake* damage (clean text + `corrupt`), because that gives
hundreds of examples per language, truly clean text for the 5% false-flag limit, and a known
error rate. Fake damage is never exactly like your OCR, so check the rules on real gold
chunks (sft and dpo only; the eval split is refused, it stays unseen):

```bash
./ocrfix score data/real/gold_chunks/sft data/real/calib/gold_scored --set score.calibration=null --set score.noise.remove=false
./ocrfix calibrate data/real/calib/clean_scored data/real/calib/corrupt_scored data/calibration --gold-scores data/real/calib/gold_scored
```

- Each gold chunk's real error rate (cer) is computed from its OCR text vs its truth.
- `calibrate` prints, and writes into `summary.md` ("Real gold OCR chunks"), how many truly
  damaged gold chunks the rules flag (recall), how many undamaged ones they flag (false
  flag), and the share flagged per damage band (cer 0, 0–0.5%, 0.5–2%, 2–5%, over 5%).
- **How to read it:** heavily damaged chunks should nearly all be flagged. If real recall is
  much lower than the recall on fake damage (printed above it), `corrupt` doesn't imitate
  your OCR well: look at `data/real/errors/summary.md`, or flag more by raising
  `calibrate.max_false_flag`. `calibrate.gold_noisy_cer` (default 0: any edit) sets what
  counts as damaged.
- It only reports: the thresholds in `calibration.yaml` are the same as in B4.

### B5 + B6. Train the corrector: `data sft`, `train sft`, `train merge`

Stop the servers first.

```bash
./ocrfix data sft data/real/sft --synthetic data/real/train/corrupt_table.jsonl --gold data/real/gold_chunks/sft --identity data/real/chunks_clean
./ocrfix train sft data/real/sft/train.jsonl runs/sft --eval data/real/sft/val.jsonl
./ocrfix train merge runs/sft runs/sft_merged
```

- `data sft` runs on the CPU too (this PC is fine); copy `data/real/sft/` to the GPU machine.
- Training mix (`sft.mix`): 60% fake-damaged → clean, 20% gold OCR → truth, 20% clean →
  itself (teaches the model to leave good text alone).
- RTX 5090: QLoRA, about an hour for 4k examples.
- `runs/sft_merged` is the trained model. Every later step names it with
  `--set correct.model=runs/sft_merged`.

### B7. Optional: DPO (a second round on the model's own mistakes)

Start the base and the SFT model in one terminal (one command starts both; on shared GPUs
each gets about half the memory, automatically):

```bash
./ocrfix serve --set correct.model=runs/sft_merged
```

Then, in another terminal:

```bash
./ocrfix corrupt data/real/chunks_clean data/real/errors/error_table.json data/real/errors/confusion_map.json data/real/train/corrupt_dpo.jsonl --set seed=7
./ocrfix data dpo data/real/dpo/pairs.jsonl --synthetic data/real/train/corrupt_dpo.jsonl --gold data/real/gold_chunks/dpo --exclude data/real/sft/train.jsonl --set correct.model=runs/sft_merged
```

Stop both servers, then:

```bash
./ocrfix train dpo data/real/dpo/pairs.jsonl runs/sft runs/dpo
./ocrfix train merge runs/dpo runs/dpo_merged
```

Skip DPO if `data/real/dpo/pairs.stats.json` shows the SFT model is already right on nearly
every prompt. If you do it, use `runs/dpo_merged` instead of `runs/sft_merged` below.

### B8. Measure: `score` + `correct` + `eval` on the gold eval documents

Servers as in B7 (`./ocrfix serve --set correct.model=runs/sft_merged`: base + trained),
then:

```bash
./ocrfix score data/real/gold_chunks/eval data/real/eval/scored
./ocrfix correct data/real/eval/scored data/real/eval/sft --set correct.model=runs/sft_merged
./ocrfix eval data/real/gold_fixed/truth sft=data/real/eval/sft data/real/eval/report
```

- `eval` compares whole documents with the truth: CER (character error rate), WER, false
  fixes (changes the truth didn't make), recall per error type, and a significance test.
  Give it several systems (`base=... sft=... dpo=...`) to compare them.
- Check: `data/real/eval/report/report.md`.

---

## 8. Track C: correct the corpus (GPU)

Servers as in B7 (`./ocrfix serve --set correct.model=runs/sft_merged`: base + trained),
then:

```bash
./ocrfix chunk data/real/corpus_fixed data/real/chunks
./ocrfix score data/real/chunks data/real/scored_raw --set score.noise.remove=false --set score.noise.whole_chunks=false
./ocrfix calibrate-noise-chunk data/real/gold_chunks data/real/scored_raw
./ocrfix score data/real/chunks data/real/scored
./ocrfix correct data/real/scored data/real/refined --set correct.model=runs/sft_merged
./ocrfix summary data/real/refined
./ocrfix report data/real/refined --clean data/real/calib/clean_scored
./ocrfix reconstruct data/real/refined data/real/corrected --source current
```

- **Pass 1 / pass 2:** the first `score` only measures; `calibrate-noise-chunk` then sets
  above which score a whole chunk is garbage (look at `data/calibration/noise_chunk_hist.png`
  and `noise_chunk_samples.md`, and edit `noise_chunk` in `calibration.yaml` if needed); the
  second `score` removes noise words and garbage chunks.
- **`correct`** sends only flagged chunks to the model (add `--all` for every chunk). A
  correction is accepted only if it passes the checks (placeholders intact, not too long or
  too different, same language) and the base model finds it less surprising than before;
  otherwise the original chunk is kept.
- **`reconstruct --source current`** writes the corrected `.md` files. Documents that noise
  removal left nearly empty are not written; they are listed in `dropped_docs.jsonl`.
  `--source original` instead rebuilds the input byte for byte (a check that nothing was lost).
- `./ocrfix bpc-hist data/real/scored_raw` (no GPU) draws the score histogram any time.

### Corpus quality: mean and histogram before / after

How surprising the base model finds the text (bpc, bits per char; lower = cleaner) at the
three stages, per language and over all, with no model needed:

```bash
./ocrfix bpc-hist raw=data/real/scored_raw denoised=data/real/scored corrected=data/real/refined --out data/real/bpc_stages.png --json data/real/bpc_stages.json
```

- **raw** = the corpus as it came (score pass 1), **denoised** = after noise removal (pass 2),
  **corrected** = after `correct` (a rejected chunk keeps its old score: this is the corpus
  as `reconstruct` writes it).
- The table: per stage `chunks`, `noise` (chunks dropped as whole garbage), `mean` (mean
  chunk bpc), `wmean` (weighted by characters: the whole corpus as one text), `p50 p90 p99
  max`. The PNG overlays the three histograms (log scale, so the few bad chunks show), each
  stage's mean dashed.
- **Read the means with the counts:** noise removal lowers the mean partly by cleaning chunks
  and partly by dropping garbage chunks (`noise` column). For "how much cleaner is the
  text", compare `wmean`.
- **While `correct` is still running** (or after stopping it), `refined/` holds only the
  finished shards, so "corrected" covers fewer documents than "raw" and "denoised": the
  numbers aren't comparable yet. Compare raw vs denoised until `correct` has finished.
- `./ocrfix pipeline corpus` runs it as its second-last step (`bpc_stages.png` / `.json`).
- `./ocrfix report data/real/refined` shows the same three stages per language from one
  folder, plus the flag rates and how much of the gap to clean text each step closed.

### What each step does, and where the time goes

| # | step | what it does | output | time |
|---|---|---|---|---|
| 1 | `chunk` | cuts every `.md` into 512–768-token chunks; masks math, tables, links | `chunks/` | small (CPU) |
| 2 | `score` pass 1 | the base model scores every chunk; nothing removed | `scored_raw/` | large: reads every token |
| 3 | `calibrate-noise-chunk` | gold + pass-1 scores → the score above which a whole chunk is garbage | limit in `calibration.yaml` | small |
| 4 | `score` pass 2 | scores again, removes noise words and garbage chunks, sets `flag` | `scored/` | large: reads every token again |
| 5 | `correct` | the trained model rewrites each **flagged** chunk, checks it, re-scores it; keeps the answer or the original | `refined/` | **the largest**: it *writes* text |
| 6–7 | `summary`, `report` | counts; quality before → after | screen, `corpus_report.json` | seconds |
| 8 | `reconstruct` | chunks → `.md` files | `corrected/` | small |

Scoring only reads the text; correcting writes every flagged chunk again, token by token, and
then scores it. If half the chunks are flagged, `correct` generates about half the corpus:
on 50 GB that can take days.

### Long runs: stopping, continuing, using a partial result

**How the work is saved.** `chunk` stores the chunks in *shards* (files of about
`chunk.shard_bytes` = 32 MB of Markdown each: about 1,600 for 50 GB); a document is never
split across two shards. `score` and `correct` work shard by shard: they process every chunk
of one shard (all at once, spread over the servers), write that shard's result
(`refined/part-NNNNN.jsonl`), then go to the next.

**Stopping** (Ctrl-C, a crash, a reboot) is safe at any time:

- finished shards are kept: a result file is written under a temporary name and renamed only
  when complete, so a file in `refined/` is never half-written;
- only the shard in progress is lost: at most ~32 MB of input, ~0.06% of a 50 GB corpus.

**Continuing:** run the same command again (or `./ocrfix pipeline corpus`). Finished shards
are skipped; it resumes at the first missing one:

```bash
./ocrfix correct data/real/scored data/real/refined --set correct.model=runs/sft_merged
```

Use the **same `correct.model`** when you continue; otherwise part of the corpus is corrected
by one model and part by another. The same holds for `score` (and its settings).

**Using the result before `correct` finishes:** each document is either in a finished shard
(corrected, in `refined/`) or not reached yet (noise removed but not corrected, in `scored/`).
Two `reconstruct` runs into the same folder give a complete corpus, the second overwriting
the documents it has with their corrected version:

```bash
./ocrfix reconstruct data/real/scored  data/real/corrected --source current
./ocrfix reconstruct data/real/refined data/real/corrected --source current
```

After `correct` has done more shards, run the second command again to update the corrected
documents. `./ocrfix summary data/real/refined` shows how far it got.

**Making `correct` shorter:**

- **The right GPU split:** see section 10, "How many GPUs for scoring vs correcting".
- **More servers:** add every GPU server's groups to `hw.servers` (section 10,
  setup 3); requests are spread over all of them. The main lever.
- **Fewer flags:** a lower `calibrate.max_false_flag` (default 0.05) flags fewer borderline
  chunks: `calibrate` again, then score pass 2 again. Check on the gold (B4b) what it misses.
- **Measure first:** `--sample-frac 0.01` on `chunk` (into another folder) runs the whole
  stage on 1% of the documents: real speed and quality, before committing days of GPU time.

---

## 9. Every step: input, output, main settings

`<chunks>` = a folder of `part-NNNNN.jsonl` shards (one JSON row per chunk). Settings are
`file.key` in `configs/stages/<file>.yaml` (or `hw.*` in the profile); options are on the
command line. Every command also takes `--profile`, `--set`, and most `--limit`.

### Track A (CPU)

| command | input | output | main settings |
|---|---|---|---|
| `markup <in> <out>` | a `.md` tree (MinerU output) | the same tree with LaTeX / tables repaired; `markup_report.jsonl` (each changed or flagged file), `markup_stats.json` | `markup.latex.delimiters` / `formulas` / `spacing` / `normalize_delimiters`, `markup.tables.html` / `markdown` (each on/off); `markup.workers`; `chunk.glob` (which files) |
| `gold chunks <gold> <out>` | `<gold>/ocr/**.md` + `<gold>/truth/**.md` (same paths) | `sft/`, `dpo/`, `eval/` chunk pairs (OCR `source_text` + truth `gold_text`, `gold_status`), `split.json` (which document is where), `stats.json` | `gold.split.eval` (0.2), `gold.split.dpo` (0.2); chunk size as `chunk` |
| `errors table <gold_chunks/sft> <out>` | gold sft chunks (`ok` only) | `error_table.json` (error rates per type and language), `confusion_map.json` (garbled Hangul), `summary.md` | `gold.errors.header_footer_min_chars` (8), `gold.errors.confusion_max_len` (4), `gold.errors.top_k` (50) |
| `chunk <in> <out>` | a `.md` tree | `<chunks>`: masked text (`source_text`, `⟨M1⟩` `⟨T1⟩` `⟨P1⟩`), `placeholders`, `lang`, offsets; `manifest.json`, `stats.json` | `chunk.min_tokens` (512), `chunk.max_tokens` (768), `chunk.protect` (what is masked), `chunk.glob`, `chunk.sample_frac` / `--sample-frac`, `chunk.workers`, `chunk.shard_bytes`; `model` (its tokenizer counts tokens) |
| `corrupt <chunks> <error_table> <confusion_map> <out.jsonl>` | clean `<chunks>` + the two error files | damaged chunks: `source_text` (damaged), `gold_text` (clean), `applied_ops`, `error_rate` | `--rate R` (fixed rate; default: the gold's rates), `--n-variants` / `corrupt.variants` (2), `corrupt.rate_scale`, `seed` |

### Track B (GPU)

| command | input | output | main settings |
|---|---|---|---|
| `serve [--only base\|trained]` | the profile: this machine's groups of `hw.servers` (`model`, and `correct.model` for trained groups) | running vLLM servers (no files); logs in `runs/vllm/` | `hw.servers` (model, host, gpus, memory), `hw.infer.gpus_per_engine`, `ports`, `max_num_seqs`; `--host`, `--dry-run` (section 10) |
| `calibrate-noise <gold_chunks> [out]` | gold chunks (sft + dpo) | `noise_report.json`; writes `score.noise.core` / `extend` / `min_chars` into `configs/stages/score.yaml` (`--no-write`: report only) | `calibrate.noise.max_text_removed` (1 in 10,000 real chars), `calibrate.noise.min_chars`, `insert_runs`, `insert_chars` |
| `score <chunks> <out>` | `<chunks>` (from `chunk`, `corrupt` or `gold chunks`) | the same rows + `score` (bpc, p90, spike_frac, ...), `flag` (damaged?), noise removed (`history`), `status`; `score_stats.json` | `score.backend` (vllm / transformers), `score.calibration` (null = no flags), `score.noise.remove`, `score.noise.whole_chunks`, `score.noise.core` / `extend` / `min_chars` |
| `calibrate <clean> <damaged> <out> [--gold-scores g]` | scored clean chunks, scored damaged chunks [, scored gold sft chunks] | `calibration.yaml` (flag rule per language), `summary.md`, `report.json`, PNGs | `calibrate.max_false_flag` (0.05), `target_recall` (0.9), `recall_on` (`table`), `min_rows` (50), `gold_noisy_cer` (0) |
| `data sft <out> --synthetic --gold --identity` | damaged chunks (`corrupt`), gold sft chunks, clean chunks | `train.jsonl`, `val.jsonl` (chat messages: prompt + right answer), `stats.json` | `sft.mix` (0.6 / 0.2 / 0.2), `sft.total`, `sft.lang_balance`, `sft.gold_repeats` (2), `sft.val_frac`, `sft.junk.*` |
| `train sft <train.jsonl> <out>` | `train.jsonl` [`--eval val.jsonl`] | a LoRA adapter (or a full model with `mode: full`), `ocrfix_train.json`, `train_summary.json` | `hw.train.mode`, `hw.train.per_device_batch_size`, `train.effective_batch` (32), `train.lr`, `train.sft.epochs` (2), `train.lora.r` / `alpha`, `--max-steps` |
| `train merge <adapter> <out>` | an adapter | a plain model folder vLLM can serve (`correct.model`) | `--base`, `--dequantize` |
| `data dpo <out.jsonl> --synthetic --gold --exclude` | damaged chunks, gold dpo chunks, the SFT data (to skip) + the served trained model | `pairs.jsonl` (prompt, chosen, rejected), `.responses.jsonl` (cache), `.stats.json` | `correct.model`, `dpo.n_samples` (4), `dpo.temperature`, `dpo.max_pairs` (2), `dpo.constructed`, `dpo.max_prompts` |
| `train dpo <pairs> <sft> <out>` | pairs + the SFT adapter | a DPO adapter | `train.dpo.beta` (0.1), `train.dpo.lr`, `train.dpo.epochs` (1), `train.dpo.reference`; `hw.train.*` as SFT |
| `correct <scored> <out> [--all]` | scored `<chunks>` | the same rows, `text` = the accepted correction, `status` (accepted / rejected / unchanged / error), `history`; `correct_stats.json` | `correct.model`, `correct.backend`, `correct.langs`, `correct.checks.length_ratio`, `max_edit_ratio` (0.15), `max_script_shift`, `min_bpc_gain`, `spike_ratio`; `--all` (every chunk, not only flagged) |
| `eval <truth> name=<dir>... <out>` | the gold `truth/` files + one or more `correct` outputs of the eval chunks | `report.md`, `metrics.json` (CER, WER, false fixes, recall per error type), `per_doc.jsonl` | `gold.errors.*` (error types), `eval.exact_max_n` |

### Track C (GPU) and tools

| command | input | output | main settings |
|---|---|---|---|
| `calibrate-noise-chunk <gold_chunks> <scored_raw> [out]` | gold chunks + corpus scores with noise removal off | `noise_chunk` (whole-garbage limit per language) written into `score.calibration`; `noise_chunk_report.json`, `noise_chunk_samples.md`, `noise_chunk_hist.png` | `calibrate.noise_chunk.margin` (1.0), `deleted_share` (0.5), `--no-write` |
| `summary <dir>` | `score` or `correct` output | on screen: status counts, accept rate, failed checks per language | |
| `report <dir> [--clean c]` | `correct` output [+ scored clean chunks] | on screen [+ `--json f`]: quality per language raw → noise removed → corrected | `score.calibration` |
| `reconstruct <chunks> [out] [--source]` | `<chunks>` (any stage) | `.md` files (`--source current`: corrected; `original`: the input byte for byte); `dropped_docs.jsonl`. No `out`: only checks | `reconstruct.min_letters` (300), `reconstruct.min_kept_share` (0.1) |
| `bpc-hist <scored>...` or `bpc-hist name=<dir>...` | `score` output(s); or several stages (`raw=… denoised=… corrected=…`) | a PNG + a table of chunk scores per language; with stages: mean, weighted mean, percentiles and noise chunks per stage, histograms overlaid (no model needed) | `--out`, `--json`, `--calibration` |
| `compare <chunks> <out.json>` | scored chunks; needs vLLM servers | vLLM vs transformers agreement (scores, flags, answers) | `--limit` (200) |
| `config [--key k]` | the settings | the resolved settings on screen | |
| `pipeline <stage>` | the folders of section 4 | everything above, stage by stage (section 5) | `pipeline.*` |

---

## 10. GPU servers (vLLM)

### How it works

On a GPU machine the model runs as **vLLM servers**: programs that keep the model on a GPU and
answer requests over the network. `score` sends text to be scored, `correct` sends chunks to be
rewritten. There are two kinds of server:

| kind | model | used by |
|---|---|---|
| `base` | the base model (`model` in the profile) | scoring, re-scoring each correction, and correcting while `correct.model` is null |
| `trained` | the correction model (`correct.model`, e.g. `runs/sft_merged`) | `correct`, `data dpo` |

**One list in the profile, `hw.servers`, describes every server.** Each line is a *group*: one
kind of model on some GPUs of one machine. Everything else follows from that list:

- `./ocrfix serve` on a machine starts the groups listed for **that** machine, both kinds in one
  command.
- `score` / `correct` send their requests to **every** group of the right kind, on every machine
  in the list, spread evenly.
- Ports and GPU memory are set automatically. Each GPU runs one copy of the model per group
  ("engine").

```yaml
# configs/profiles/h200x8.yaml
hw:
  servers:
    - {model: base,    host: 10.0.0.20, gpus: all}   # base model on all 8 GPUs
    - {model: trained, host: 10.0.0.20, gpus: all}   # trained model on the same 8 GPUs
```

That gives 8 base engines (ports 8000–8007) and 8 trained engines (8100–8107). Both kinds share
every GPU, so each engine gets 0.44 of the GPU memory.

### Setup 1: one GPU machine (the profiles as shipped)

Nothing to change. On the machine:

```bash
./ocrfix serve --set correct.model=runs/sft_merged      # starts base + trained
./ocrfix serve                                           # before training: base only
```

Without `correct.model` (nothing trained yet), `serve` starts only the base servers and says
so. `./ocrfix pipeline` starts and stops these servers itself; you only start them by hand for
single commands. Stop them with Ctrl-C (training needs the GPU memory).

### Setup 2: the commands on another machine (a CPU server)

The servers must listen on the GPU machine's **network address**: `127.0.0.1` refuses other
machines. Use the GPU machine's profile on **both** machines (e.g. `h200x8`: on the CPU machine
too, `export OCRFIX_PROFILE=h200x8`), with `host` set to the GPU machine's address:

```yaml
  servers:
    - {model: base,    host: 10.0.0.11, gpus: all}
    - {model: trained, host: 10.0.0.11, gpus: all}
```

- GPU machine: `./ocrfix serve --set correct.model=/models/ocrfix-sft-merged`
- CPU machine: run the commands as usual (`./ocrfix score ...`, `./ocrfix correct ...`); they
  send to `10.0.0.11`. It needs the project, the data, and the model's small files
  (`config.json`, tokenizer files) at the same path, not the weights.
- Check from the CPU machine: `curl http://10.0.0.11:8000/health`

### Setup 3: several GPU servers

Make **one profile file that lists every server's groups**, and copy it to every machine
(`configs/profiles/cluster.yaml.example` is a worked example; copy it to `cluster.yaml`).

```yaml
  servers:
    - {model: base,    host: 10.0.0.11, gpus: [0, 1]}               # gpu1: 2 GPUs base
    - {model: trained, host: 10.0.0.11, gpus: [2, 3, 4, 5, 6, 7]}   # gpu1: 6 GPUs trained
    - {model: trained, host: 10.0.0.12, gpus: all}                  # gpu2: 8 GPUs trained
    - {model: trained, host: 10.0.0.13, gpus: all}                  # gpu3: 8 GPUs trained
```

1. On **each** GPU server, the same command (it starts that server's lines only):
   ```bash
   ./ocrfix serve --set correct.model=/models/ocrfix-sft-merged
   ```
2. Anywhere (a CPU server or gpu1): run the commands as usual. Here scoring is spread over 2
   base engines and correcting over 6 + 8 + 8 = 22 trained engines.

Rules:

- **Same file everywhere**, with every machine's lines. Each line's `host` is the address of
  the machine that runs it.
- `correct.model` must be the **same path on every machine** (copy the merged model to each).
  Best: set it in `configs/stages/correct.yaml`; then just `./ocrfix serve`.
- `./ocrfix pipeline` **uses** servers on other machines but never starts or stops them: start
  them by hand. If one doesn't answer, the stage stops and says which one.
- **Training** runs on one machine at a time (all its GPUs), the one the pipeline or
  `train` command runs on. Stop that machine's servers first.
- The servers have no password: internal network addresses only; the firewall must let the
  command machine reach ports 8000+ and 8100+ on the GPU servers.

### Settings

A group in `hw.servers`:

| key | what it is | example |
|---|---|---|
| `model` | `base` or `trained` | `base` |
| `host` | the address of the machine that runs it. The servers listen on it, the commands send to it. `127.0.0.1` = this machine only | `10.0.0.11` |
| `gpus` | `all` (GPUs 0 .. `hw.n_gpus` − 1) or a list | `[0, 1]` |
| `memory` | optional. Automatic: groups on the same GPUs share 0.88 (two groups: 0.44 each); a group alone gets 0.88 | `0.6` |

How every server runs, `hw.infer` in the profile:

| setting | what it does | 5090 | H200 ×8 |
|---|---|---|---|
| `gpus_per_engine` | GPUs per copy of the model. 1 = fastest when the model fits on one GPU (4B, 9B do) | 1 | 1 |
| `ports` | first port of each kind on each machine; the k-th engine of a kind on a machine uses `ports[kind] + k` | base 8100, trained 8101 | base 8000, trained 8100 |
| `max_num_seqs` | requests one engine works on at once | 64 | 256 |
| `max_model_len` | longest request in tokens (a chunk and its answer fit easily) | 4096 | 8192 |
| `concurrency` | requests `score` / `correct` keep in flight. null = automatic (engines × `max_num_seqs`) | null → 64 | null → 2048 |

Shipped profiles: `rtx5090` (both kinds on its one GPU, `127.0.0.1`), `h200x4` (both on all 4
GPUs, `127.0.0.1`), `h200x8` (both on all 8, `host: 10.0.0.20`: change it to the server's
address), `cpu` (`servers: []`: no servers, the model runs inside each command).

### How many GPUs for scoring vs correcting

*What each costs.* Scoring *reads* a chunk: all ~700 tokens in one pass. Correcting *writes*
it again, one token per model step; vLLM works on many chunks at once (`max_num_seqs`), so each
step writes a token for all of them. Measured on the RTX 5090 (4B): scoring ~20,700 tokens/s,
generating ~3,300 tokens/s (only 16 at a time, sharing the GPU with the scoring server). So
one flagged chunk costs `correct` about 8 times what scoring it costs:

| work per flagged chunk | kind | cost (1 = scoring that chunk once) |
|---|---|---|
| read the prompt | trained | ~1 |
| write the answer (~700 tokens) | trained | ~6 |
| re-score the answer | base | ~1 |

With half the chunks flagged, `correct` takes about as long as the two scoring passes
together, or up to twice that: the longest step, but not hundreds of times longer.

*Who needs GPUs when.* Scoring needs only base servers; `correct` needs trained (~85–90% of the
work) and base (~10–15%, the re-score).

- **Share every GPU (the default and the recommendation).** Both kinds on `gpus: all`, 0.44
  memory each. A GPU's compute goes to whichever server has work, so the balance adjusts by
  itself. Half the memory is plenty for a 9B (~18 GB) on a 141 GB H200, tighter on the 32 GB
  5090 (tested there).
- **Or give each kind its own GPUs**, about **1 base : 7 trained**; each then gets 0.88 of its
  GPUs. Only worth it if sharing runs out of memory:
  ```yaml
  servers:
    - {model: base,    host: 127.0.0.1, gpus: [0]}
    - {model: trained, host: 127.0.0.1, gpus: [1, 2, 3, 4, 5, 6, 7]}
  ```
- **Check on the real machine** (the 1 : 7 is an estimate from the 5090):
  - `nvidia-smi` during `correct`: GPUs that only hold base servers and sit near 0% → too
    many GPUs for base.
  - each server's log (`runs/vllm/<name>-<port>.log`) shows requests *Running* and *Waiting*:
    a long Waiting queue on the trained servers and none on base → move GPUs to trained.
  - `refined/correct_stats.json` records tokens/s: compare layouts on a 1% sample
    (`--sample-frac 0.01` on `chunk`).

### Checking and troubleshooting

| what | how |
|---|---|
| what `serve` would start here | `./ocrfix serve --dry-run --set correct.model=runs/sft_merged`: every server's GPU, port, memory; starts nothing |
| where the commands will send requests | `./ocrfix config --key hw.infer.score_urls` / `hw.infer.correct_urls` |
| is a server up | `curl http://<host>:<port>/health` |
| "no server group of hw.servers is on this machine" | this machine's address isn't recognised: `./ocrfix serve --host <its address>` |
| one kind only | `./ocrfix serve --only base` / `--only trained` (`--only trained` needs `correct.model`) |
| extra vLLM options for every server | after `--`: `./ocrfix serve -- --enforce-eager` |
| a server crashed or won't start | its log: `runs/vllm/<name>-<port>.log` |
| no vLLM on a GPU machine | `--set score.backend=transformers --set correct.backend=transformers`: the model runs inside each command on one GPU (slower: scoring ~2×, correction 4–8× on the 5090; batch sizes `hw.scorer.max_batch_tokens`, `correct.hf.batch_size`) |

---

## 11. Settings

### Where they live

| file | what you set there |
|---|---|
| `configs/default.yaml` | `seed` (every random choice), `work_dir`, language rules |
| `configs/profiles/<machine>.yaml` | **the machine:** model path, GPUs, training mode and batch, vLLM servers (`hw.servers`: model, host, GPUs per group; `hw.infer`). Only `model` and `hw` |
| `configs/stages/markup.yaml` | which LaTeX / table repairs `markup` makes; `workers` |
| `configs/stages/chunk.yaml` | chunk size (`min_tokens` 512, `max_tokens` 768), files (`glob`), what is masked (`protect`) |
| `configs/stages/gold.yaml` | gold split (`split.eval` 0.2, `split.dpo` 0.2), error-table options |
| `configs/stages/corrupt.yaml` | the fake-damage generator (`rate_scale`, `variants`, header phrases) |
| `configs/stages/score.yaml` | `backend` (auto / vllm / transformers), noise removal (`noise.remove`, `noise.whole_chunks`, `noise.core` / `extend` / `min_chars`), `calibration` file |
| `configs/stages/calibrate.yaml` | `max_false_flag` (0.05), `target_recall` (0.9), noise limits |
| `configs/stages/sft.yaml` | SFT data (`sft.mix`, `total`, `gold_repeats`, `junk`) and training (`train.lr`, `train.sft.epochs`, `train.lora`, `effective_batch`) |
| `configs/stages/dpo.yaml` | DPO pairs (`n_samples`, `max_pairs`) and training (`train.dpo.beta`, `lr`) |
| `configs/stages/correct.yaml` | **the correction model** (`model`, `adapter`), `langs`, decoding, the checks (`length_ratio`, `max_edit_ratio`, `min_bpc_gain`) |
| `configs/stages/eval.yaml` | eval options |
| `configs/stages/reconstruct.yaml` | when a nearly empty document is dropped (`min_letters` 300, `min_kept_share` 0.1) |

### How to change one

- **For one run:** `--set key=value`, e.g. `--set correct.model=runs/sft_merged`,
  `--set sft.total=30000`, `--set markup.latex.spacing=false`.
- **For good:** edit the YAML file. Each setting lives in exactly one file; putting it in a
  second one is an error (so there is never doubt which value wins).
- **Order:** `default.yaml` → `stages/*.yaml` → profile → `--set`.
- **See what is in effect:** `./ocrfix config --key correct.model` (one key) or
  `./ocrfix config` (everything) — with the same `--profile` / `--set` as your run.

### Settings people change most

| setting | default | when to change |
|---|---|---|
| `correct.model` | null (= base model) | after training: `runs/sft_merged` |
| `score.calibration` | `data/calibration/calibration.yaml` | `null` when scoring calibration inputs (B3) |
| `score.noise.remove` | true | `false` for calibration inputs and pass 1 |
| `score.backend` / `correct.backend` | auto (vllm with servers, else transformers) | force `transformers` or `vllm` |
| `sft.total` | null (as much as the mix allows) | cap the training set size |
| `gold.split.eval` / `dpo` | 0.2 / 0.2 | bigger eval set for a more reliable measurement |
| `correct.langs` | `[ko, en, mixed]` | which chunk languages are corrected |
| `chunk.sample_frac` / `--sample-frac` | null (all) | try the corpus steps on a sample first |
| `seed` | 1234 | a second damage set (`--set seed=2`) |

### Machines (profiles)

| profile | machine | model |
|---|---|---|
| `cpu` | this PC, no GPU (the default) | `../Qwen3.5-4B` (`D:\cch\Qwen3.5-4B`) |
| `rtx5090` | one RTX 5090 | `Qwen3.5-4B` in the project folder |
| `h200x4`, `h200x8` | H200 servers | `/models/Qwen3.5-9B` |

After switching model (4B → 9B), run B2 and B3+B4 again: scores belong to one model. To add a
machine, copy a profile and change it ([docs/runbook.md](docs/runbook.md)).

### Parallelism: CPU processes and training

(GPU servers: section 10.)

All of these are in the **profile** (`hw.*`, per machine) or a stage file. Change them for
good in the YAML file, or for one run with `--set` (or the command's own option, shown).

**CPU steps: processes**

| step | setting | one run | default |
|---|---|---|---|
| `markup` | `markup.workers` | `--workers N` | every CPU core; `0` = one process (debugging) |
| `chunk`, `gold chunks` | `chunk.workers` | `--workers N` | every CPU core |

- `markup` hands each process a batch of files. `chunk` hands each process one *shard* of
  input (`chunk.shard_bytes`, 32 MB of Markdown): fewer shards than cores means idle cores, so
  for a small corpus lower `chunk.shard_bytes` (e.g. `--set chunk.shard_bytes=4194304`).
- More processes than cores doesn't help; on a 50 GB corpus the disk may be the limit.

**Training (`train sft`, `train dpo`): GPUs, batch, memory**

`train` uses `hw.n_gpus` GPUs: with more than 1 it restarts itself under `torchrun`, one
process per GPU. Everything below is under `hw.train.` in the profile, except
`train.effective_batch` (`sft.yaml`).

| setting | what it does | 5090 | H200 ×8 |
|---|---|---|---|
| `hw.n_gpus` | GPUs to train on (processes) | 1 | 8 |
| `hw.train.mode` | `qlora` (4-bit base + LoRA, least memory), `lora` (bf16 base + LoRA), `full` (every weight) | qlora | lora |
| `hw.train.distributed` | `auto`: LoRA → DDP (a full copy per GPU), full → FSDP (weights split over the GPUs); or `none` / `ddp` / `fsdp` | none | auto |
| `hw.train.per_device_batch_size` | sequences per GPU per step (the micro-batch); lower it when memory runs out | 2 | 8 |
| `train.effective_batch` | sequences per optimizer step over all GPUs; gradient accumulation = effective_batch ÷ (micro-batch × GPUs), rounded up | 32 | 32 |
| `hw.train.max_seq_len` | longer examples are dropped (counted), never cut | 2048 | 8192 |
| `hw.train.gradient_checkpointing` | less memory, somewhat slower | true | true |
| `hw.train.packing` | several short examples per sequence: faster; needs the optional kernels | false | true |
| `hw.train.optim` | optimizer (`paged_adamw_8bit` saves memory with QLoRA) | paged_adamw_8bit | adamw_torch_fused |

- **Train on some GPUs only** (e.g. 4 while the other 4 serve):
  ```bash
  CUDA_VISIBLE_DEVICES=4,5,6,7 ./ocrfix train sft data/real/sft/train.jsonl runs/sft --set hw.n_gpus=4
  ```
- **Full fine-tuning** of the 9B (H200s only, ~150 GB with Adam, split by FSDP):
  `--set hw.train.mode=full`.
- **Out of memory:** lower `per_device_batch_size` (accumulation goes up, so the effective
  batch stays the same), turn `gradient_checkpointing` on, or use `qlora`.
- DPO uses the same `hw.train` settings; its own (`beta`, `lr`, `epochs`) are `train.dpo.*`
  in `dpo.yaml`.
- Training needs the GPUs to itself: stop `serve` first (or train on other GPUs, above).
- Check what a run will use: `./ocrfix config --key hw.train` (with the same `--profile` /
  `--set`).

---

## 12. Test data, tests, project layout

These are developer tools, not pipeline steps, so they call the venv's Python directly:
`.venv\Scripts\python.exe` on Windows (shown), `.venv/bin/python` on Linux.

**Generated test data** (template text with injected OCR errors; math and tables are never
damaged):

```bash
.venv\Scripts\python.exe -X utf8 scripts/make_test_corpus.py data/gen/gold --ocr-mb 2 --clean-mb 0.1 --seed 11
.venv\Scripts\python.exe -X utf8 scripts/make_test_corpus.py data/gen/corpus --ocr-mb 10 --clean-mb 5 --seed 12
```

Each writes `ocr/` (damaged), `truth/` (the answer key), `clean/` and `ocr_errors.jsonl`. Copy
them into the `data/real/` layout (gold ocr/truth from the first, corpus = `ocr/` and clean =
`clean/` from the second). Results on it are optimistic; the real measure is your gold.

**Tests and lint** (about a minute, CPU, a tiny random model, no network):

```bash
.venv\Scripts\python.exe -X utf8 -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
```

**Layout:**

```
ocrfix, ocrfix.cmd          the launchers: ./ocrfix <command> (Linux, Windows)
run.py                      what they run: UTF-8 mode, the profile, then the command
configs/default.yaml        seed, work folder, language rule
configs/stages/*.yaml       one settings file per step
configs/profiles/*.yaml     one file per machine
src/ocrfix/cli.py           every command
src/ocrfix/markup/          markup (LaTeX, tables)
src/ocrfix/chunks/          chunk, reconstruct
src/ocrfix/stages/          score, correct, corrupt, data sft|dpo, eval, report, compare
src/ocrfix/gold/ errors/ corrupt/ calibrate/ train/   gold, error tables, damage, calibration, training
src/ocrfix/scorer/ llm/ verify/ eval/ data/ corpus/   scoring, model client, checks, metrics, text analysis
scripts/                    make_wheelhouse.py, install_offline.py, make_test_corpus.py
docs/                       real_data.md, annotation_guideline.md, runbook.md
data/  runs/                your data and training output (not in git)
```
