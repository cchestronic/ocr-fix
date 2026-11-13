# Real data, step by step

From your gold and clean documents to a trained correction model, an honest measurement and
the corrected corpus. Run the steps in order. Each one is resumable: run the same command again
after an interruption.

Commands are for the RTX 5090 (`--profile rtx5090`, the 4B model). For the H200s, use
`--profile h200x4` and the 9B; everything else is the same.

## 0. What you prepare

```
data/real/
  corpus/        the MinerU .md tree to correct (50 GB), as it is
  gold/ocr/      MinerU .md files picked for gold, copied unchanged
  gold/truth/    the same files, hand-corrected (same relative paths)
  clean/         born-digital markdown in the same style, no OCR
```

- **Gold:** pairs of `.md` files; how to correct them is in
  [annotation_guideline.md](annotation_guideline.md). Lost words are `〈?〉`.
  - **Size:** 5,000 chunks is about 10M characters. Many short documents of different types
    is best.
- **Clean:** as much as you have, ideally 50–100 MB. It must not be copies of gold documents,
  and it must have the same markdown for math (`$...$`) and tables (`<table>`).

The model, scores and calibration belong together. After switching 4B → 9B, redo steps 3–9.

An engine started with `&` takes a minute or two to load. Wait until it answers
(`curl -s http://127.0.0.1:8100/health` prints nothing and exits 0) before the next command.

Port 8000 is taken on this machine, so the vLLM engines here use 8100 (base model) and 8101
(the fine-tuned model). Set these once per shell:

```bash
P=(--profile rtx5090)
BASE=(--set hw.infer.base_port=8100 --set 'hw.infer.urls=[http://127.0.0.1:8100]')
TWO=(--set 'refine.score_urls=[http://127.0.0.1:8100]' --set 'refine.correct_urls=[http://127.0.0.1:8101]'
     --set llm.served_model_name=tuned --set refine.score_model=qwen3.5)
E="data/real/errors/error_table.json data/real/errors/confusion_map.json"
```

## 1. Gold → chunks (CPU)

```bash
ocrfix gold chunks data/real/gold data/real/gold_chunks "${P[@]}"
```

- **What it does:**
  - chunks the `ocr/` files exactly like the corpus;
  - cuts each `truth/` file at the matching places;
  - splits by document into `sft/` (60%), `dpo/` (20%) and `eval/` (20%), set in
    `gold.chunks.split`.
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
ocrfix errors table data/real/gold_chunks/sft data/real/errors "${P[@]}"
```

`summary.md` shows the real error types and rates per language. The training-data generator
copies these.

## 3. Clean chunks (CPU)

```bash
ocrfix chunk data/real/clean data/real/chunks_clean "${P[@]}"
```

## 4. Calibration (GPU: base model)

This decides which chunk scores count as "damaged".

```bash
scripts/serve_vllm.sh "${P[@]}" "${BASE[@]}" --set paths.work_dir=runs/vllm_base &
ocrfix refine corrupt data/real/chunks_clean $E data/real/calib/corrupt/table.jsonl "${P[@]}"
ocrfix refine corrupt data/real/chunks_clean $E data/real/calib/corrupt/r0.01.jsonl --rate 0.01 "${P[@]}"
ocrfix refine corrupt data/real/chunks_clean $E data/real/calib/corrupt/r0.03.jsonl --rate 0.03 "${P[@]}"
S=(--set scorer.calibration=null --set refine.noise.remove=false)
ocrfix refine score data/real/chunks_clean data/real/calib/clean_scored "${P[@]}" "${BASE[@]}" "${S[@]}"
ocrfix refine score data/real/calib/corrupt data/real/calib/corrupt_scored "${P[@]}" "${BASE[@]}" "${S[@]}"
ocrfix calibrate data/real/calib/clean_scored data/real/calib/corrupt_scored data/calibration "${P[@]}"
```

- **A few thousand clean chunks are enough here:** `--limit N` on `refine corrupt` counts
  input shards.
- **Check `data/calibration/summary.md`:** recall per language at ≤ 5% clean chunks flagged.

## 5. SFT data (CPU)

```bash
ocrfix refine corrupt data/real/chunks_clean $E data/real/train/corrupt_table.jsonl "${P[@]}"
ocrfix refine corrupt data/real/chunks_clean $E data/real/train/corrupt_r0.03.jsonl --rate 0.03 "${P[@]}"
ocrfix refine data data/real/sft \
  --synthetic data/real/train/corrupt_table.jsonl --synthetic data/real/train/corrupt_r0.03.jsonl \
  --gold data/real/gold_chunks/sft --identity data/real/chunks_clean "${P[@]}"
```

- **Mix** (`refine.data.mix`):
  - synthetic 60%: clean chunks damaged at the gold rates. 30% of them also get junk, and 10%
    get 1–2 lost words answered with `〈?〉`.
  - gold 20%: each gold chunk twice (`refine.data.gold_repeats`).
  - identity 20%: clean chunks returned unchanged.
- **Size:** the total is the most the mix allows without reusing anything. Cap it with
  `--set refine.data.total=30000`.
- **Check:** the output line and `stats.json` show what was taken and dropped per source and
  language.

## 6. SFT (GPU: stop vLLM first)

```bash
pkill -f "[v]llm serve"
scripts/train.sh sft data/real/sft/train.jsonl runs/sft --eval data/real/sft/val.jsonl "${P[@]}"
scripts/train.sh merge runs/sft runs/sft_merged "${P[@]}"
```

On the 5090, about 1 hour per 4,000 examples (QLoRA, 4B).

## 7. DPO pairs (GPU: base + SFT model)

```bash
scripts/serve_vllm.sh "${P[@]}" "${BASE[@]}" --set hw.infer.gpu_memory_utilization=0.44 --set paths.work_dir=runs/vllm_base &
scripts/serve_vllm.sh "${P[@]}" --set hw.infer.gpu_memory_utilization=0.44 --set paths.work_dir=runs/vllm_tuned \
  --set hw.infer.base_port=8101 --set 'hw.infer.urls=[http://127.0.0.1:8101]' \
  --set llm.model=runs/sft_merged --set llm.served_model_name=tuned &
ocrfix refine corrupt data/real/chunks_clean $E data/real/train/corrupt_dpo.jsonl --set seed=7 "${P[@]}"
ocrfix refine dpo data/real/dpo/pairs.jsonl \
  --synthetic data/real/train/corrupt_dpo.jsonl --gold data/real/gold_chunks/dpo \
  --exclude data/real/sft/train.jsonl "${P[@]}" "${TWO[@]}" --set refine.dpo.model=tuned
```

- **What happens:** the SFT model answers every prompt once greedily and 4 times sampled.
  Every answer that isn't the right one becomes a pair, typed:
  - `missed`, `false_fix`, `wrong_fix`, `mixed`, `placeholders`;
  - `guessed`: a word where `〈?〉` belongs;
  - `overcautious`: `〈?〉` where a fix was possible.
- **Check `pairs.stats.json`:**
  - You want several thousand pairs, most of them `greedy:` or `sampled:` (the model's own
    mistakes).
  - If the model is right on almost every prompt, DPO has little to learn. Use harder
    synthetic prompts (`--rate 0.03` on the DPO corrupt file) or skip DPO.

## 8. DPO (GPU: stop vLLM first)

```bash
pkill -f "[v]llm serve"
scripts/train.sh dpo data/real/dpo/pairs.jsonl runs/sft runs/dpo "${P[@]}" --set train.dpo.reference=sft
scripts/train.sh merge runs/dpo runs/dpo_merged "${P[@]}"
```

## 9. Eval on the gold eval documents (GPU)

Score the eval chunks once with the base model, then correct them with each model:

```bash
scripts/serve_vllm.sh "${P[@]}" "${BASE[@]}" --set hw.infer.gpu_memory_utilization=0.44 --set paths.work_dir=runs/vllm_base &
ocrfix refine score data/real/gold_chunks/eval data/real/eval/scored "${P[@]}" "${BASE[@]}"
# the base model as the corrector (same engine)
ocrfix refine correct data/real/eval/scored data/real/eval/base "${P[@]}" "${BASE[@]}"
# the SFT model: start the second engine as in step 7 (llm.model=runs/sft_merged), then
ocrfix refine correct data/real/eval/scored data/real/eval/sft "${P[@]}" "${TWO[@]}"
ocrfix refine correct data/real/eval/scored data/real/eval/sft_all --all "${P[@]}" "${TWO[@]}"
# the DPO model: stop the second engine, start it again with llm.model=runs/dpo_merged, then
ocrfix refine correct data/real/eval/scored data/real/eval/dpo "${P[@]}" "${TWO[@]}"

ocrfix refine eval data/real/gold/truth \
  base=data/real/eval/base sft=data/real/eval/sft sft_all=data/real/eval/sft_all dpo=data/real/eval/dpo \
  data/real/eval/report "${P[@]}"
```

- **Stop one engine:** `pkill -f "[v]llm serve.*port 8101"`.
- **`report.md`** compares whole documents with the truth files. System `ocr` is the text
  before correction.
  - **CER, WER and SER**, overall and per language.
  - **False fixes:** edits where the truth has none.
  - **Recall per error type:** the same types as `errors table`.
  - **`〈?〉` per lost word:** marked (`〈?〉`, right), left (garbage kept), invented (a word
    written: the worst). Plus `false 〈?〉`: `〈?〉` where the truth has a word.
  - **Paired Wilcoxon tests** between systems.
- **Pick the corrector and the mode:**
  - correct only flagged chunks, or every chunk with `--all`;
  - SFT or DPO.

## Optional: transformers instead of vLLM for the small jobs

DPO sampling (step 7) and eval corrections (step 9) can run in-process with transformers
(`--set refine.generator=transformers`, model `llm.model`), with no second engine and no merge
needed for an adapter (`--set llm.adapter=runs/sft`). The corpus (step 10) stays on vLLM. Before
trusting it, check that both backends give the same results, one engine at a time (leave room
for transformers with `--set hw.infer.gpu_memory_utilization=0.45` on the engine):

```bash
# scores: the base engine on 8100 vs transformers (paths.scorer_model)
ocrfix refine compare data/real/eval/scored data/real/compare_score.json --no-generate "${P[@]}" "${BASE[@]}"
# greedy answers: the tuned engine on 8101 (llm.model=runs/sft_merged) vs transformers loading the same dir
ocrfix refine compare data/real/eval/scored data/real/compare_sft.json --no-score "${P[@]}" "${TWO[@]}" \
  --set llm.model=runs/sft_merged
```

- **`pass`** needs flags to agree on ≥ 99% of chunks and ≥ 95% of greedy answers to be
  identical.
- **Small differences are normal:** bf16 with different kernels, and a greedy answer can split at
  a near-tie. The report shows the first differing answers.
- **Calibration** should come from the same scorer as the corpus (vLLM), unless the score check
  passes.
- **Speed:** transformers generation is much slower than vLLM without the optional kernels
  (`flash-linear-attention`, `causal-conv1d`). The report gives tokens/s for both.

Then, for example, DPO pairs without the second engine:
```bash
ocrfix refine dpo data/real/dpo/pairs.jsonl --synthetic data/real/train/corrupt_dpo.jsonl \
  --gold data/real/gold_chunks/dpo --exclude data/real/sft/train.jsonl "${P[@]}" \
  --set refine.generator=transformers --set llm.model=runs/sft_merged
```

## 10. The corpus (GPU)

With the base engine on 8100 and the chosen model on 8101:

```bash
ocrfix chunk data/real/corpus data/real/chunks "${P[@]}"
ocrfix refine score data/real/chunks data/real/scored "${P[@]}" "${BASE[@]}"
ocrfix refine correct data/real/scored data/real/refined "${P[@]}" "${TWO[@]}"     # add --all if step 9 says so
ocrfix refine report data/real/refined --clean data/real/calib/clean_scored "${P[@]}"
ocrfix reconstruct data/real/refined data/real/corrected --source current "${P[@]}"
```

- **`refine report`:** bpc and flag rates before and after, per language.
- **`refine summary data/real/refined`:** accept rate and the reasons for rejections.
- **Output:** `data/real/corrected/` holds the corrected `.md` files. `data/real/corpus/` is
  never changed.
