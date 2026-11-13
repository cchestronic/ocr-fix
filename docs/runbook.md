# ocrfix runbook

Two tracks. Do Track 1 on the single RTX 5090 now. Track 2 repeats the same commands on
4x H200 later, with `--profile h200x4`. Every command runs offline: nothing in ocrfix
downloads, and `scripts/env.sh` sets the HF/vLLM offline variables.

**About the timings.** Only scoring has been measured on the 5090: about 6.4k tokens/s, with
random weights in the real Qwen3.5-9B shape. Every other duration below is an estimate,
marked *est.* Each pipeline run records the real seconds per stage in
`<work_dir>/pipeline.json`. After the 2% sample run, multiply those numbers by 50 to plan the
full corpus.

Durations are given for a corpus of **1B tokens** (Qwen tokenizer); scale them linearly.
A 50 GB Korean/English markdown corpus is roughly **12–17B tokens**: multiply by 12–17. At
that size a 2% sample is too slow on one 5090 (about 12 h per scoring pass), so start with
`sample_frac: 0.003` (or `--limit 200` documents).
To measure your corpus, the score log prints tokens/s and paragraph counts. As a rough guide,
a 10-page paper is about 6–10k tokens.

---

## Track 1: one RTX 5090 (now)

### 1.1 Machine prerequisites

| item | version | why |
|---|---|---|
| NVIDIA driver | **≥ 580** | the locked torch and vLLM wheels are CUDA 13.0 builds |
| GPU | RTX 5090, 32 GB, Blackwell **sm_120** | |
| OS / Python | Linux x86_64, glibc ≥ 2.28, CPython **3.12** | the wheelhouse is built for `manylinux_2_28_x86_64`, cp312 |
| disk | ~20 GB for the model, ~5 GB for the wheelhouse, plus 2–3× the corpus size for the run outputs | |

No system CUDA toolkit is needed: CUDA runtime, cuDNN and NCCL come in as pip wheels.

### 1.2 Offline install from the wheelhouse

On an **online** machine with the same Python and architecture:

```bash
scripts/make_wheelhouse.sh wheelhouse                  # base install
scripts/make_wheelhouse.sh --with-kernels wheelhouse   # + optional training kernels (~10 min compile)
```

Copy `wheelhouse/` to the air-gapped machine, then:

```bash
scripts/install_offline.sh wheelhouse .venv
scripts/install_offline.sh --with-kernels wheelhouse .venv   # if built with kernels
source .venv/bin/activate && source scripts/env.sh
```

`install_offline.sh` checks `SHA256SUMS`, then installs with `--no-index --require-hashes`.

**Exact versions** (`requirements.lock`, `requirements-kernels.lock`):

| package | version | notes for sm_120 |
|---|---|---|
| torch | 2.14.0 (CUDA 13.0 build) | sm_120 kernels included |
| nvidia-cuda-runtime / nvidia-cudnn-cu13 | 13.0.96 / 9.24.0.43 | pulled in by torch |
| triton | 3.8.0 | |
| transformers | 5.20.2 | Qwen3.5 (`qwen3_5_text`) support |
| vllm | 0.33.0 | CUDA 13.0 build; `Qwen3_5ForCausalLM` registered |
| flashinfer-python | 0.7.2.post1 | vLLM attention backend on the 5090 (`FLASHINFER`) |
| bitsandbytes | 0.50.2 | QLoRA nf4 + paged 8-bit AdamW |
| trl / peft / accelerate / datasets | 1.17.1 / 0.21.2 / 1.15.0 / 5.1.0 | |
| flash-linear-attention, fla-core (optional) | 0.6.0 | Gated DeltaNet Triton kernels |
| causal-conv1d (optional) | 1.7.0, built locally for torch 2.14.0+cu130, sm_90 + sm_120 | `scripts/build_causal_conv1d.sh` |
| **flash-attn** | **not shipped** | no reliable sm_120 wheels; see below |

**Attention on the 5090 (the fallbacks):**
- **transformers (scorer, training):** PyTorch SDPA (`hw.attn_impl: sdpa`). The scorer runs
  in padded mode, which is exact and was the faster mode in the measurement.
- **vLLM:** `FLASHINFER`. If FlashInfer fails on your driver, try
  `OCRFIX_ATTENTION_BACKEND=TRITON_ATTN scripts/serve_vllm.sh ...`, or set it empty to let
  vLLM choose.
- **Qwen3.5 linear-attention layers:** vLLM ships its own Triton kernels. transformers uses
  `fla` and `causal-conv1d` when installed, else a pure-PyTorch fallback, which is correct
  but slower. That fallback is why the rtx5090 profile trains without packing.
- **With `fla` installed:** Qwen3.5 in transformers runs **only on CUDA**, and the first
  forward pass JIT-compiles for about a minute (then it is cached).

Check the install:

```bash
ocrfix --help && ocrfix config --profile rtx5090 --key hw.name
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 2.14.0+cu130 13.0 (12, 0)
pytest -q          # ~3 min; tiny random model, no network
```

### 1.3 Copy the model weights

Copy the Hugging Face directory of Qwen3.5-9B (config, `*.safetensors`, tokenizer files,
`generation_config.json`, chat template) by hand to the default location, `/models/Qwen3.5-9B`.
If you put it elsewhere, point `paths.base_model` at it (`--set paths.base_model=/path`, or edit
`configs/default.yaml`).

```bash
# on the source machine
cd /models && find Qwen3.5-9B -type f -print0 | sort -z | xargs -0 sha256sum > Qwen3.5-9B.sha256
# on the air-gapped machine, after copying both
cd /models && sha256sum -c --quiet Qwen3.5-9B.sha256
du -sh /models/Qwen3.5-9B     # ~18 GB in bf16
```

- **Where it's used:** `paths.base_model` is the scorer model, the tokenizer, the training
  base and, by default, the served model (`llm.model`).
- **Pointing vLLM elsewhere:** set `llm.model` to a merged checkpoint, or set `llm.adapter`
  to serve a LoRA on top of the base.
- **Never use a hub id:** offline mode would refuse it anyway.

### 1.4 First run on a ~2% sample

Before any real run, check the plumbing in about 3 minutes, with the echo LLM and the test model:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "tests")
from pathlib import Path; from conftest import build_tiny_model
build_tiny_model(Path("runs/tiny_model"), ["linear_attention", "full_attention"])
EOF
ocrfix pipeline configs/pipeline/smoke.yaml --profile rtx5090 --set paths.base_model=runs/tiny_model
```

All seven stages should run. The echo server's `Tab1e` → `Table` fix should be accepted by
verify, and eval should show CER 0 for `verified`. A second run should report every stage
`cached`.

**The 2% sample.** Edit `configs/pipeline/sample.yaml` (`input_dir`, `work_dir`) and run:

```bash
ocrfix pipeline configs/pipeline/sample.yaml --profile rtx5090 --dry-run   # the 7 commands
ocrfix pipeline configs/pipeline/sample.yaml --profile rtx5090
```

`sample_frac: 0.02` keeps 2% of the documents, chosen by a hash of the path. The sample
spreads over the whole corpus, and a rerun picks the same documents. `--limit N`
additionally caps the number of documents, which is useful for a 10-document check first.

**What happens on the first run:**
- **Without a calibration:** `correct`, `verify` and `eval` are *skipped* with a note,
  because nothing is flagged yet. Ingest, score, fix and rescore run, giving the scores you
  need to sample gold (1.5) and the scored text for calibration (1.6).
- **After calibration:** rerun the same command. The finished stages report `cached`, and
  only correct, verify and eval run. The score stages never store flags, so a new or
  changed calibration needs no rescore: correct applies the current one.
- **LLM server:** with `serve: auto`, the pipeline starts `scripts/serve_vllm.sh` for the
  correct stage and stops it before verify. On one GPU, vLLM's 90% memory reservation and the
  scorer can't share the card.
- **If port 8000 is in use** by another program, the pipeline stops and says so. Move the
  engine with `--set hw.infer.base_port=8100 --set 'hw.infer.urls=[http://127.0.0.1:8100]'`.

**Resume:**
- **Interrupted run:** a run killed mid-stage resumes on the next start. Ingest, score and
  correct continue where they stopped.
- **Changed setting:** if you change an input or a stage's settings (for example
  `--set fix.line_join.enabled=true`), that stage and everything after it is rebuilt.
  Earlier stages stay `cached`.
- **Rebuild by hand:** `--force <stage>` (or `--force all`).
- **One stage only:** `--only <stage>` runs just that stage.

**Expected wall-clock on one 5090 for a 2% sample** (20M tokens of a 1B-token corpus):

| stage | what bounds it | duration |
|---|---|---|
| ingest | CPU, disk | < 5 min *est.* |
| score | GPU, 6.4k tokens/s measured (before windows and masking; masking removes the math and table tokens) | 1 h |
| fix | CPU, one process | < 5 min *est.* |
| rescore | same as score | 1 h |
| correct | vLLM, flagged share f (often 0.1–0.3) of the text; about 2–3k generated tokens/s *est.* on 9B bf16, 64 sequences | 0.3–1 h *est.* (+ 2–5 min server start) |
| verify | re-scores only the changed paragraphs | 10–30 min *est.* |
| eval | CPU | < 1 min |
| **total** | | **≈ 2.5–4 h** |

### What the models see: masking (`llm.protect`)

Score, correct, verify and the training data all read each paragraph in the same masked
form, built by one rule (`src/ocrfix/corpus/protect.py`):

| placeholder | replaces | detected by |
|---|---|---|
| `⟨M#⟩` | every math span, inline `$…$` and display `$$…$$` / `\begin{align}…` | the tagger (ingest spans) |
| `⟨T#⟩` | every table, HTML or pipe | the tagger |
| `⟨P#⟩` | the kinds listed in **`llm.protect`** in `configs/default.yaml`: `[images, code, urls, emails, dois, citations]` | regex patterns in `PROTECT_PATTERNS` |

**Why `⟨P#⟩`:** an LLM "fixing" a URL, DOI, email, code identifier or image path is guessing
and can break it. These strings also look random to the scorer and would flag clean
paragraphs.

**The detection is approximate:**
- **Damaged items aren't matched.** An OCR-damaged URL (`https: //www. x. com`) or citation
  (`[l2]`) stays prose, so it is scored and can be corrected.
- **Narrow formats only:** citations are numeric `[12]` only, and code is inline backticks
  only. Fenced code blocks are not detected and are treated as prose.
- **Nothing masked is lost:** a wrongly matched item (e.g. `[1]` used as list numbering) is
  just left unchanged.

**To change it:** edit the list. Remove a kind to have it scored and corrected like ordinary
prose. The change applies to every stage at once, so afterwards rescore, re-sample the gold
set and recalibrate.

Garbled Korean that OCR put inside `$…$` is masked like any math, so the LLM does not fix
it. Eval reports it as missed `hangul_garbled`.

### 1.5 Build the gold set

```bash
W=runs/pipeline/sample
ocrfix gold sample $W/scores.jsonl $W/corpus.jsonl data/gold/to_annotate.jsonl --profile rtx5090
```

This writes the stratified rows, plus `to_annotate.labelstudio.json` (tasks) and
`.labelstudio.xml` (labeling config).

1. Set `gold.page_image` to your page-image path template so annotators can see the PDF page.
   Import both Label Studio files into an offline Label Studio, then annotate following
   `docs/annotation_guideline_paragraphs.md`. About 1,000 paragraphs with the default quotas takes
   roughly 15–25 annotator-hours *est.*
2. Export the annotations as JSON, then import them:

   ```bash
   ocrfix gold import export.json data/gold --profile rtx5090 --strict
   ```

   This writes `gold_train.jsonl`, `gold_eval.jsonl` (with its sha256 manifest),
   `split.json`, `violations.jsonl` and `import_report.json`. Fix the violations and
   re-import; the eval documents stay pinned on re-import.
3. Error statistics (train half only):

   ```bash
   ocrfix errors table data/gold/gold_train.jsonl data/errors
   ```

   Read `data/errors/summary.md`. If `header_footer_insert`, `newline_in_sentence` or
   `hyphen_wrap` are common, enable `fix.layout_junk` and `fix.line_join` in
   `configs/default.yaml`. The next pipeline run then rebuilds fix and the stages after it.

### 1.6 Calibration

```bash
C=data/calib; E=data/errors; G=data/gold/gold_train.jsonl
ocrfix gold records $G $C/gold_src.jsonl                 # OCR side, with its CER
ocrfix gold records $G $C/clean.jsonl --side gold        # clean side
for r in 0.01 0.03 0.1; do
  ocrfix corrupt $C/clean.jsonl $E/error_table.json $E/confusion_map.json $C/corrupt_r$r.jsonl --rate $r
done
ocrfix corrupt $C/clean.jsonl $E/error_table.json $E/confusion_map.json $C/corrupt_table.jsonl
cat $C/corrupt_*.jsonl > $C/corrupted.jsonl
S="--profile rtx5090 --set scorer.calibration=null"
for f in clean corrupted gold_src; do ocrfix score $C/$f.jsonl $C/${f}_scores.jsonl $S; done
ocrfix calibrate $C/clean_scores.jsonl $C/corrupted_scores.jsonl data/calibration \
  --gold-scores $C/gold_src_scores.jsonl
```

- **Output:** `data/calibration/calibration.yaml`. This is the default `scorer.calibration`,
  so every later stage picks it up.
- **What to check** in `summary.md`: `target_met` per language, and the gold
  precision/recall.
- **Duration:** about 20–40 min *est.* on the 5090, dominated by scoring roughly 5× the gold
  text.
- **Then:** rerun the sample pipeline (1.4). Correct, verify and eval now run, and eval
  compares `source`, `fixed` (rules only) and `verified` (rules + base model) on the gold
  eval set.

This sequence (1.5 and 1.6) was run end to end on the test fixtures: `gold sample` →
annotate → `gold import` → `errors table` → `corrupt` → `score` → `calibrate`, then the
pipeline with the calibration. The fixtures are too small for a meaningful rule; that needs
the real gold set.

### 1.7 QLoRA SFT and DPO

```bash
# synthetic pairs for training: corrupt the clean side at the table rates, 2 variants each
ocrfix corrupt data/calib/clean.jsonl data/errors/error_table.json data/errors/confusion_map.json data/train/synthetic.jsonl
ocrfix data sft data/sft --synthetic data/train/synthetic.jsonl \
  --verified runs/pipeline/sample/verified.jsonl --gold data/gold/gold_train.jsonl \
  --identity runs/pipeline/sample/fixed.jsonl
ocrfix data stats data/sft/train.jsonl

scripts/train.sh sft data/sft/train.jsonl runs/sft --profile rtx5090 --eval data/sft/val.jsonl
```

- **Eval documents stay out:** `data.exclude_split` (default `data/gold/split.json`) drops
  the gold eval documents from every source, so they never become training data.
- **For more synthetic data:** corrupt a few thousand clean-looking corpus paragraphs (low
  `spike_frac`) in the same way.
- **Memory:** QLoRA keeps the nf4 base at about 5 GB, with LoRA r/alpha from `train.lora`,
  sequence length 2048, micro-batch 2 and paged 8-bit AdamW. Peak VRAM is logged at every
  logging step and saved in `runs/sft/train_summary.json`. **The full 9B QLoRA run has not
  been measured on the 5090 yet;** if it runs out of memory, use
  `--set hw.train.per_device_batch_size=1` (gradient accumulation adjusts).
- **Rate:** 1.5–3k trained tokens/s *est.* without the kernels; about 1.5–2× that with
  `--with-kernels` *est.*
- **Duration:** an SFT set of 20k examples × 800 tokens × 2 epochs (32M tokens) takes
  **3–6 h** *est.*

DPO samples from the SFT model, so serve it first (base + adapter), then train:

```bash
scripts/serve_vllm.sh --profile rtx5090 --set llm.adapter=runs/sft &      # wait for "ready"
ocrfix data dpo data/dpo.jsonl --gold data/gold/gold_train.jsonl \
  --verified runs/pipeline/sample/verified.jsonl --model ocrfix-lora --set llm.adapter=runs/sft
kill %1                                                                  # free the GPU
scripts/train.sh dpo data/dpo.jsonl runs/sft runs/dpo --profile rtx5090
scripts/train.sh merge runs/dpo runs/merged --profile rtx5090
```

- **DPO duration:** 1 epoch, the policy and reference forward passes share one base (no
  second weight copy). 10k pairs take **2–4 h** *est.*
- **Merge:** about 10 min. It dequantizes the nf4 base, merges the adapter and writes bf16
  safetensors to `runs/merged`. It warns that the merged weights differ slightly from the
  training-time model; that is expected.

### 1.8 Evaluation: base-prompted vs SFT vs SFT+DPO

Run the gold eval sources through the pipeline once per model, each into its own work dir
(`--work-dir`). The eval set is small, so each run takes minutes. Then compare the three:

```bash
ocrfix gold records data/gold/gold_eval.jsonl data/evalset/src.jsonl
cat > configs/pipeline/evalset.yaml <<'YAML'
input_records: data/evalset/src.jsonl     # Records instead of markdown: no ingest
work_dir: runs/evalset/base
gold_eval: data/gold/gold_eval.jsonl
stages: [fix, rescore, correct, verify, eval]
YAML
P="ocrfix pipeline configs/pipeline/evalset.yaml --profile rtx5090"
$P --work-dir runs/evalset/base                                  # base model, prompted
$P --work-dir runs/evalset/sft     --set llm.adapter=runs/sft    # base + SFT adapter
$P --work-dir runs/evalset/sft_dpo --set llm.model=runs/merged   # merged SFT+DPO
ocrfix eval data/gold/gold_eval.jsonl base=runs/evalset/base/verified.jsonl \
  sft=runs/evalset/sft/verified.jsonl sft_dpo=runs/evalset/sft_dpo/verified.jsonl runs/evalset/report
```

- **One model, one set of answers:** each run keeps its LLM responses in a file keyed by
  the LLM settings, so switching models never reuses another model's answers.
- **Flagged paragraphs only:** each work dir's own `eval/` compares source, fixed and
  verified for that model. Add `correct_all: true` to the YAML to correct every paragraph.

`runs/evalset/report/report.md` covers:
- CER/WER (en) and CER/spacing error rate (ko);
- false-fix rate, identity preservation and placeholder integrity;
- recall per error type and structure accuracy;
- a paired Wilcoxon test per document between systems;
- the 20 worst paragraphs per system.

Eval takes seconds. For a gold set of ~500 eval paragraphs, each system run takes about
5–10 min *est.*, mostly model loading and vLLM start-up.

---

## Track 2: 4x H200 (later)

The commands are the same with `--profile h200x4`. The profile does the rest:
- **Scoring:** `ocrfix score` (and the pipeline's score stages) re-execs under torchrun with
  4 ranks, the full model on each.
- **Training:** `scripts/train.sh` uses torchrun.
- **Serving:** `serve_vllm.sh` starts **4 data-parallel engines**, GPU *i* on port 8000+*i*
  with max-model-len 8192 and no tensor parallelism. The client round-robins over
  `hw.infer.urls` with 512 requests in flight.

### 2.1 Install

- **Software:** the same wheelhouse and versions. The locked torch/vLLM wheels cover sm_90.
  Build the wheelhouse **with kernels**: the profile trains with packing, and `ocrfix train`
  refuses packing without them.
- **Model:** copy it the same way as in 1.3.
- **Driver:** ≥ 580.

```bash
scripts/install_offline.sh --with-kernels wheelhouse .venv
scripts/serve_vllm.sh --profile h200x4 --dry-run    # 4 vllm serve commands, ports 8000-8003
```

- **vLLM attention:** `FLASH_ATTN`, vLLM's bundled FlashAttention, which supports sm_90.
- **transformers:** the profile asks for `flash_attention_2`, but flash-attn isn't in the
  lock. The scorer then uses padded SDPA (it logs the reason), which is the measured faster
  path anyway.
- **Training attention:** training doesn't read `hw.attn_impl`; it uses transformers'
  default for Qwen3.5.

### 2.2 Full-corpus schedule

```bash
ocrfix pipeline configs/pipeline/full.yaml --profile h200x4
```

`full.yaml` serves the merged model (`llm.model=runs/merged`). Copy `runs/merged` and
`data/calibration/` over from the 5090 work, or redo 1.5–1.7 on the H200s.

Estimated durations for a 1B-token corpus, all *est.* Replace them with
`pipeline.json` × 50 from the sample run, scaled by the GPU ratio:

| stage | parallelism | duration |
|---|---|---|
| ingest | 1 CPU process | 0.5–1 h |
| score | 4 ranks × ~15–25k tokens/s, one pass over the masked text | 3–5 h |
| fix | 1 CPU process | 0.5–1 h |
| rescore | as score | 3–5 h |
| correct | 4 engines, ~8–12k generated tokens/s each at 256 sequences; f = 0.1–0.3 | 2–6 h |
| verify | **one GPU** (`InProcessScorer`); re-scores changed paragraphs only | 2–5 h |
| eval | CPU | minutes |
| **total** | | **≈ 11–23 h** |

Every stage resumes, so a run can span several sessions. Running
`ocrfix pipeline ... --only score` lets you schedule stages in separate jobs.

- **verify:** it is the stage that doesn't scale out yet. To use all 4 GPUs, split
  `corrected.jsonl` by document into 4 parts, run `ocrfix verify` on each with
  `CUDA_VISIBLE_DEVICES=i`, and concatenate the results.
- **fix and ingest:** single CPU processes. For more speed, split the input directory and run
  several pipelines into separate work dirs.

### 2.3 Training on 4x H200

```bash
scripts/train.sh sft data/sft/train.jsonl runs/sft --profile h200x4                          # bf16 LoRA, DDP x4, packing, seq 8192
scripts/train.sh sft data/sft/train.jsonl runs/full --profile h200x4 --set hw.train.mode=full  # full fine-tune, FSDP
scripts/train.sh dpo data/dpo.jsonl runs/sft runs/dpo --profile h200x4
scripts/train.sh merge runs/dpo runs/merged --profile h200x4
```

| run | throughput *est.* | 32M-token SFT | 10k-pair DPO |
|---|---|---|---|
| bf16 LoRA, DDP ×4, packed | 40–80k tokens/s total | 10–20 min | 15–30 min |
| full FT, FSDP ×4 | ~⅓ of LoRA | 30–60 min | 45–90 min (`--set hw.train.mode=full`; SFT must be full too) |

**Choosing the mode** (`hw.train.mode`, the same for SFT and DPO): `qlora` (4-bit base +
LoRA, the 5090 default), `lora` (bf16 base + LoRA, the H200 default) or `full` (no LoRA, every
weight trains). Full DPO needs a full SFT model: a full SFT run, or a LoRA SFT merged with
`train merge`. A LoRA DPO can also start from a full SFT model (a new LoRA on top); `train
merge` then merges into that model automatically. A full SFT/DPO result is already a
complete model: serve it with `--set llm.model=runs/full`, no merge.

For full fine-tuning, the effective batch, learning rate and epochs come from
`train.*`. Lower `train.lr` (1e-5 is a common start) with `--set train.lr=1e-5`.

---

## Adding a hardware profile

Hardware differences live only in `configs/profiles/<name>.yaml`, as the `hw` block. Code
never branches on GPU model, so a new machine means a new file, never an `if`.

1. Copy the closest profile: `cp configs/profiles/rtx5090.yaml configs/profiles/a100x2.yaml`.
2. Edit the `hw` block:
   - `name`: must equal the file name.
   - `n_gpus`: `> 1` makes score and train re-exec under torchrun.
   - `dtype` and `attn_impl` (`sdpa`, or `flash_attention_2` if flash-attn is installed and
     works); `vllm_attention_backend` (`FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, or empty
     for auto).
   - `train`:
     - `mode` (`qlora` when the bf16 model + LoRA + optimizer state doesn't fit; else
       `lora` or `full`);
     - `per_device_batch_size`, `max_seq_len`, `gradient_checkpointing`, `optim`;
     - `packing` (needs the kernels);
     - `distributed` (`auto`/`none`/`ddp`/`fsdp`), plus `fsdp` for full fine-tuning.
   - `infer`:
     - `data_parallel_size`: the number of engines, ≤ `n_gpus`;
     - `tensor_parallel_size`: 1 unless the model doesn't fit on one GPU;
     - `max_model_len`: ≥ `train.max_seq_len`;
     - `gpu_memory_utilization`, `max_num_seqs`;
     - `base_port`, and `urls`, which **must** list `base_port + i` for every engine;
     - `client_concurrency`: about `max_num_seqs × engines`.
   - `scorer`: `max_seq_len`, `max_batch_tokens` (tokens per forward pass; start at about
     6k per 32 GB of free memory), `logits_chunk`, `tokenize_workers`.
3. Check it:

   ```bash
   ocrfix config --profile a100x2                 # resolved config
   scripts/serve_vllm.sh --profile a100x2 --dry-run
   pytest -q tests/test_pipeline.py tests/test_llm.py -k profile   # every profile is tested
   ```

   The tests run on every file in `configs/profiles/`. They check that each required key is
   set, that `urls` match `base_port`, that the engines fit the GPU count, and that
   `max_model_len ≥ train.max_seq_len`.
4. Measure before a long run:
   - `ocrfix score` on a few thousand paragraphs prints tokens/s and peak VRAM; raise
     `max_batch_tokens` until peak VRAM is about 85% of the card.
   - `scripts/train.sh sft ... --max-steps 20` logs `peak_vram_gib`.
5. Use it everywhere with `--profile a100x2`, or `export OCRFIX_PROFILE=a100x2`.

---

## Where things are

| path | content |
|---|---|
| `configs/default.yaml` | all non-hardware settings |
| `configs/profiles/*.yaml` | the `hw` block per machine |
| `configs/pipeline/*.yaml` | pipeline runs: `smoke`, `sample` (2%), `full` |
| `<work_dir>/logs/<stage>.log`, `pipeline.json` | per-stage output and timings |
| `<work_dir>/.stages/<stage>.json` | resume markers (delete one to force that stage) |
| `runs/vllm/engine-<i>.log` | vLLM engine logs (`paths.work_dir`) |
| `docs/annotation_guideline.md` | what gold text is |
