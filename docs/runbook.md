# ocrfix runbook

Install, model copy, the H200 track and adding a hardware profile. **The step-by-step run on
real data (gold, calibration, SFT/DPO, eval, the corpus) is [real_data.md](real_data.md).**
Every command runs offline: nothing in ocrfix downloads, and `import ocrfix` (and `ocrfix serve` for vLLM) sets the
HF/vLLM offline variables.

**Timings.** `score` and `correct` write `score_stats.json` / `correct_stats.json`
(counts, tokens/s) in their output folder. Run a sample first (`ocrfix chunk ... --sample-frac
0.003`, or `--limit` documents) and scale. A 50 GB Korean/English markdown corpus is roughly
12–17B tokens; a 10-page paper is about 6–10k tokens. Measured on the 5090 with the 4B: vLLM
scoring ~20k tokens/s at half the GPU memory, greedy correction ~3.3k generated tokens/s.

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
python3 scripts/make_wheelhouse.py wheelhouse                  # base install
python3 scripts/make_wheelhouse.py --with-kernels wheelhouse   # + optional training kernels (~10 min compile)
```

Copy `wheelhouse/` to the air-gapped machine, then:

```bash
python3 scripts/install_offline.py wheelhouse .venv
python3 scripts/install_offline.py --with-kernels wheelhouse .venv   # if built with kernels
source .venv/bin/activate
```

`install_offline.py` checks `SHA256SUMS`, then installs with `--no-index --require-hashes`.

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
| causal-conv1d (optional) | 1.7.0, built locally for torch 2.14.0+cu130, sm_90 + sm_120 | `scripts/build_causal_conv1d.py` |
| **flash-attn** | **not shipped** | no reliable sm_120 wheels; see below |

**Attention on the 5090 (the fallbacks):**
- **transformers (scorer, training):** PyTorch SDPA (`hw.attn_impl: sdpa`). The scorer runs
  in padded mode, which is exact and was the faster mode in the measurement.
- **vLLM:** `FLASHINFER`. If FlashInfer fails on your driver, try
  `OCRFIX_ATTENTION_BACKEND=TRITON_ATTN ocrfix serve ...`, or set it empty to let
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
If you put it elsewhere, point `model` at it (`--set model=/path`, or edit
`configs/profiles/<machine>.yaml`).

```bash
# on the source machine
cd /models && find Qwen3.5-9B -type f -print0 | sort -z | xargs -0 sha256sum > Qwen3.5-9B.sha256
# on the air-gapped machine, after copying both
cd /models && sha256sum -c --quiet Qwen3.5-9B.sha256
du -sh /models/Qwen3.5-9B     # ~18 GB in bf16
```

- **Where it's used:** `model` is the scorer model, the tokenizer, the training
  base and, by default, the served model (`correct.model`).
- **Pointing vLLM elsewhere:** set `correct.model` to a merged checkpoint, or set `correct.adapter`
  to serve a LoRA on top of the base.
- **Never use a hub id:** offline mode would refuse it anyway.

### 1.4 Run

Follow [real_data.md](real_data.md): gold chunks, error tables, clean chunks, calibration, SFT
data, SFT, DPO, eval, then the corpus.

### What the models see: masking (`chunk.protect`)

Chunking masks every chunk once, by one rule (`src/ocrfix/corpus/protect.py`); scoring,
correction and the training data all read that masked form:

| placeholder | replaces | detected by |
|---|---|---|
| `⟨M#⟩` | every math span, inline `$…$` and display `$$…$$` / `\begin{align}…` | the tagger |
| `⟨T#⟩` | every table, HTML or pipe | the tagger |
| `⟨P#⟩` | the kinds listed in **`chunk.protect`** in `configs/stages/chunk.yaml`: `[images, code, urls, emails, dois, citations]` | regex patterns in `PROTECT_PATTERNS` |

**Why `⟨P#⟩`:** an LLM "fixing" a URL, DOI, email, code identifier or image path is guessing
and can break it. These strings also look random to the scorer and would flag clean
chunks.

**The detection is approximate:**
- **Damaged items aren't matched.** An OCR-damaged URL (`https: //www. x. com`) or citation
  (`[l2]`) stays prose, so it is scored and can be corrected.
- **Narrow formats only:** citations are numeric `[12]` only, and code is inline backticks
  only. Fenced code blocks are not detected and are treated as prose.
- **Nothing masked is lost:** a wrongly matched item (e.g. `[1]` used as list numbering) is
  just left unchanged.

**To change it:** edit the list. Remove a kind to have it scored and corrected like ordinary
prose. The change applies to every stage at once, so afterwards chunk, score and calibrate
again.

Garbled Korean that OCR put inside `$…$` is masked like any math, so the LLM does not fix
it. `eval` reports it as missed `hangul_garbled`.

---

## Track 2: 4x H200 (later)

The commands are the same with `--profile h200x4`. The profile does the rest:
- **Scoring and correction:** `score` / `correct` round-robin over the 4 engines of each model.
- **Training:** `ocrfix train` uses torchrun.
- **Serving:** `hw.servers` lists both models on all 4 GPUs (`127.0.0.1`), so `ocrfix serve`
  starts **4 data-parallel base engines**, GPU *i* on port 8000+*i*, and with `correct.model`
  set **4 trained engines** on 8100+*i*; 0.44 of each GPU's memory per engine (automatic),
  max-model-len 8192, one GPU per engine (`gpus_per_engine: 1`). The client round-robins over
  all engines of a model with engines × `max_num_seqs` (1024) requests in flight.

### 2.1 Install

- **Software:** the same wheelhouse and versions. The locked torch/vLLM wheels cover sm_90.
  Build the wheelhouse **with kernels**: the profile trains with packing, and `ocrfix train`
  refuses packing without them.
- **Model:** copy it the same way as in 1.3.
- **Driver:** ≥ 580.

```bash
python3 scripts/install_offline.py --with-kernels wheelhouse .venv
ocrfix serve --profile h200x4 --dry-run    # 4 base vllm serve commands, ports 8000-8003
ocrfix serve --profile h200x4 --dry-run --set correct.model=runs/sft_merged   # + 4 trained, 8100-8103
```

- **vLLM attention:** `FLASH_ATTN`, vLLM's bundled FlashAttention, which supports sm_90.
- **transformers** (only `score.backend=transformers` / `correct.backend=transformers`): the
  profile asks for `flash_attention_2`, but flash-attn isn't in the lock. The scorer then uses
  padded SDPA (it logs the reason), which is the measured faster path anyway.
- **Training attention:** training doesn't read `hw.attn_impl`; it uses transformers'
  default for Qwen3.5.

### 2.2 The corpus

The commands of [real_data.md](real_data.md) step 10 with `--profile h200x4`: one
`ocrfix serve --set correct.model=runs/sft_merged` starts 4 base-model engines for scoring and
4 engines of the merged model for correction, sharing the GPUs. Several GPU servers: one
profile listing every server's groups, `ocrfix serve` on each (see
`configs/profiles/cluster.yaml.example`). Copy `runs/*_merged` and
`data/calibration/` from the 5090 work, or redo them with the 9B (scores and calibration
belong to one model). Every chunk stage resumes by shard, so a run can span sessions.

### 2.3 Training on 4x H200

```bash
ocrfix train sft data/real/sft/train.jsonl runs/sft --profile h200x4                        # bf16 LoRA, DDP x4, packing, seq 8192
ocrfix train sft data/real/sft/train.jsonl runs/full --profile h200x4 --set hw.train.mode=full  # full fine-tune, FSDP
ocrfix train dpo data/real/dpo/pairs.jsonl runs/sft runs/dpo --profile h200x4
ocrfix train merge runs/dpo runs/merged --profile h200x4
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
complete model: serve it with `--set correct.model=runs/full`, no merge.

For full fine-tuning, the effective batch, learning rate and epochs come from
`train.*`. Lower `train.lr` (1e-5 is a common start) with `--set train.lr=1e-5`.

---

## Adding a hardware profile

Hardware differences live only in `configs/profiles/<name>.yaml`, as the `hw` block. Code
never branches on GPU model, so a new machine means a new file, never an `if`.

1. Copy the closest profile: `cp configs/profiles/rtx5090.yaml configs/profiles/a100x2.yaml`.
2. Edit `model` (the base model folder) and the `hw` block (a profile sets nothing else):
   - `name`: must equal the file name.
   - `n_gpus`: `> 1` makes `train` re-exec under torchrun.
   - `dtype` and `attn_impl` (`sdpa`, or `flash_attention_2` if flash-attn is installed and
     works); `vllm_attention_backend` (`FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, or empty
     for auto).
   - `train`:
     - `mode` (`qlora` when the bf16 model + LoRA + optimizer state doesn't fit; else
       `lora` or `full`);
     - `per_device_batch_size`, `max_seq_len`, `gradient_checkpointing`, `optim`;
     - `packing` (needs the kernels);
     - `distributed` (`auto`/`none`/`ddp`/`fsdp`), plus `fsdp` for full fine-tuning.
   - `servers`: the vLLM server groups, `{model: base|trained, host, gpus, memory?}`:
     - `model`: `base` (the base model) or `trained` (`correct.model`);
     - `host`: `127.0.0.1`, or the machine's LAN address to serve other machines;
     - `gpus`: `all` or a list, e.g. `[0, 1]` (one engine per `infer.gpus_per_engine` GPUs);
     - `memory` (optional): automatic, 0.88 shared by the groups on the same GPUs (two: 0.44);
     - `[]` for a machine without GPU servers (the backends then use transformers).
   - `infer`:
     - `gpus_per_engine`: 1 unless the model doesn't fit on one GPU;
     - `ports: {base, trained}`: engine k of a model on a machine listens on `ports[model] + k`;
     - `max_model_len`: ≥ `train.max_seq_len`;
     - `max_num_seqs`;
     - `concurrency`: `null` (automatic: engines × `max_num_seqs`).
   - `scorer` (transformers scoring only): `max_batch_tokens` (tokens per forward pass; start
     at about 6k per 32 GB of free memory), `logits_chunk`.
3. Check it:

   ```bash
   ocrfix config --profile a100x2                 # resolved config
   ocrfix serve --profile a100x2 --dry-run
   pytest -q tests/test_config.py   # every profile is tested
   ```

   The tests run on every file in `configs/profiles/`. They check that each required key is
   set, that every server group's GPUs exist on the machine, and that
   `max_model_len ≥ train.max_seq_len`.
4. Measure before a long run:
   - `ocrfix score` on a few shards writes tokens/s to `score_stats.json`;
     `ocrfix compare --no-generate` also times the transformers scorer.
   - `ocrfix train sft ... --max-steps 20` logs `peak_vram_gib`.
5. Use it everywhere with `--profile a100x2`, or `export OCRFIX_PROFILE=a100x2`.

---

## Where things are

| path | content |
|---|---|
| `configs/default.yaml`, `configs/stages/*.yaml` | shared settings; one file per step |
| `configs/profiles/*.yaml` | the `hw` block per machine |
| `<out_dir>/score_stats.json`, `correct_stats.json` | counts and tokens/s of a `score` / `correct` run |
| `runs/vllm/engine-<i>.log` | vLLM engine logs (`work_dir`) |
| `docs/annotation_guideline.md` | how to write gold truth files |
| `docs/real_data.md` | the run on real data, step by step |
