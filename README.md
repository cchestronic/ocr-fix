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
| B5 | `data sft` | CPU | damaged / gold / clean chunks → `train.jsonl`, `val.jsonl` |
| B6 | `train sft`, `train merge` | GPU | → LoRA adapter → merged model |
| B7 | `data dpo`, `train dpo` | GPU | optional second training round |
| B8 | `correct`, `eval` | GPU, CPU | gold eval docs → CER, false fixes, recall per error type |
| C1 | `calibrate-noise-chunk` | GPU | gold + corpus scores → limit for whole-garbage chunks |
| C2 | `correct` | GPU | scored chunks → corrected chunks |
| C3 | `report`, `summary` | CPU | quality before / after, accept rates |
| C4 | `reconstruct` | CPU | chunks → `.md` files |

Every step is resumable: if it stops, run the same command again and it continues.

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

### Which machine runs what

- **CPU steps** (`markup`, `gold chunks`, `errors table`, `chunk`, `corrupt`, `data sft`,
  `calibrate`, `eval`, `report`, `summary`, `reconstruct`, `bpc-hist`) run anywhere, this PC
  included.
- **GPU steps** (`serve`, `score`, `correct`, `calibrate-noise`, `calibrate-noise-chunk`,
  `data dpo`, `train ...`) need a GPU machine (or a GPU machine's model servers, section 8).

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
only). No real data yet? Generate test data (section 9).

---

## 5. Track A: prepare (CPU, this PC is fine)

### A1. Repair LaTeX and tables: `markup`

```bash
./ocrfix markup data/real/corpus      data/real/corpus_fixed
./ocrfix markup data/real/gold/ocr    data/real/gold_fixed/ocr
./ocrfix markup data/real/gold/truth  data/real/gold_fixed/truth
./ocrfix markup data/real/clean       data/real/clean_fixed
```

- Fixes unclosed `$$` / `\[` / `\begin{equation}`, unbalanced braces and `\left`/`\right`,
  MinerU spacing (`x _ { i }` → `x_{i}`), broken HTML tables (missing `</td>` `</tr>`
  `</table>`, short rows), Markdown tables without a `|---|` line or with short rows.
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

## 6. Track B: learn (GPU)

Run these on the GPU machine.

### B1. Start the model servers: `serve`

In a terminal of its own, leave it running:

```bash
./ocrfix serve                       # the base model; wait until it prints "ready"
```

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

Two servers, each in a terminal of its own, each with about half the GPU:

```bash
./ocrfix serve --memory 0.44
./ocrfix serve --correct --memory 0.42 --set correct.model=runs/sft_merged
```

Then:

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

Servers as in B7 (base + `--correct`), then:

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

## 7. Track C: correct the corpus (GPU)

Servers as in B7 (base + `--correct`), then:

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

---

## 8. Settings

### Where they live

| file | what you set there |
|---|---|
| `configs/default.yaml` | `seed` (every random choice), `work_dir`, language rules |
| `configs/profiles/<machine>.yaml` | **the machine:** model path, GPUs, training mode and batch, vLLM servers (host, ports, memory). Only `model` and `hw` |
| `configs/stages/markup.yaml` | which LaTeX / table repairs `markup` makes; `workers` |
| `configs/stages/chunk.yaml` | chunk size (`min_tokens` 512, `max_tokens` 768), files (`glob`), what is masked (`protect`) |
| `configs/stages/gold.yaml` | gold split (`split.eval` 0.2, `split.dpo` 0.2), error-table options |
| `configs/stages/corrupt.yaml` | the fake-damage generator (`rate_scale`, `variants`, header phrases) |
| `configs/stages/score.yaml` | `backend` (vllm / transformers), noise removal (`noise.remove`, `noise.whole_chunks`, `noise.core` / `extend` / `min_chars`), `calibration` file |
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
| `score.backend` / `correct.backend` | vllm | `transformers` to run without servers (slow) |
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

### Parallelism: CPU processes, GPUs, vLLM servers, training

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

**vLLM servers (`serve`): which GPUs, how many copies of the model**

`./ocrfix serve` starts `hw.infer.engines` copies of the model ("engines"), each on
`hw.infer.tensor_parallel_size` GPUs, on ports `base_port`, `base_port + 1`, ... `score`,
`correct`, `data dpo` spread their requests over all engines.

| setting (`hw.infer.`) | one run | what it does | 5090 | H200 ×8 |
|---|---|---|---|---|
| `engines` | `serve --engines N` | copies of the model; engine i uses the next `tensor_parallel_size` GPUs | 1 | 8 |
| `tensor_parallel_size` | `--set hw.infer.tensor_parallel_size=2` | GPUs per engine; 1 = fastest when the model fits on one GPU (4B, 9B) | 1 | 1 |
| — | `serve --gpus 4,5,6,7` | which GPUs (default: 0 .. engines × tp − 1) | | |
| `gpu_memory_utilization` | `serve --memory 0.44` | share of each GPU's memory an engine takes | 0.90 | 0.45 |
| `max_num_seqs` | `--set hw.infer.max_num_seqs=128` | requests one engine works on at once | 64 | 256 |
| `max_model_len` | | longest request in tokens (a chunk + its answer fits easily) | 4096 | 8192 |
| `concurrency` | `--set hw.infer.concurrency=256` | requests `score` / `correct` keep in flight in total; about engines × max_num_seqs | 64 | 1024 |
| `base_port`, `correct_port` | `serve --port P` | first port of the base / the trained (`--correct`) engines | 8100 / 8101 | 8000 / 8100 |
| `host` | | address the engines listen on (a LAN address to serve another machine) | 127.0.0.1 | LAN |

- **Engines × tensor_parallel_size ≤ GPUs** (the profile is refused otherwise).
- **Base + trained model at once** (B7, B8, track C need both): start both, each with about
  half the memory, on the same GPUs:
  ```bash
  ./ocrfix serve --memory 0.44
  ./ocrfix serve --correct --memory 0.42 --set correct.model=runs/sft_merged
  ```
  The H200 ×8 profile already sets 0.45 for that; with the whole GPU for one model, `--memory 0.9`.
- **Some GPUs for serving, others for something else:**
  `./ocrfix serve --engines 4 --gpus 0,1,2,3`.
- **Check before starting:** `./ocrfix serve --dry-run ...` prints each engine's command
  with its GPUs (`CUDA_VISIBLE_DEVICES=4 vllm serve ... --port 8000 ...`) and starts nothing.
- Logs per engine: `runs/vllm/<name>-<port>.log`.
- Without servers, `--set score.backend=transformers` / `--set correct.backend=transformers`
  run the model in the command's own process on one GPU: slower (measured on the 5090:
  scoring about 2×, correction 4–8×). Their batch sizes:
  `hw.scorer.max_batch_tokens` (tokens per forward pass) and `correct.hf.batch_size`
  (sequences per generate call, halved automatically when memory runs out).

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

### A CPU server + a GPU server

Keep the project and data on the CPU server; the GPU server only runs `serve` and `train`.
Set `hw.infer.host` in the profile to the GPU server's LAN address on both machines; the
commands that send requests (`score`, `correct`, `data dpo`, `calibrate-noise`,
`calibrate-noise-chunk`) then run on the CPU server. The CPU server needs only the model's
small files (`config.json`, tokenizer files) at the same path. Details:
[docs/runbook.md](docs/runbook.md). The servers have no password: keep them on the internal
network.

---

## 9. Test data, tests, project layout

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
