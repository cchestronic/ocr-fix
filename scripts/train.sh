#!/usr/bin/env bash
# Training launcher. GPU count, mode and parallelism come from the profile, never from here.
#
#   rtx5090 (1 GPU, QLoRA nf4 + paged 8-bit AdamW, max_seq_len 2048):
#     scripts/train.sh sft data/sft/train.jsonl runs/sft --profile rtx5090
#     scripts/train.sh dpo data/dpo.jsonl runs/sft runs/dpo --profile rtx5090
#   h200x4 (4 GPUs, torchrun; bf16 LoRA with DDP + packing, max_seq_len 8192):
#     scripts/train.sh sft data/sft/train.jsonl runs/sft --profile h200x4
#     scripts/train.sh sft data/sft/train.jsonl runs/full --profile h200x4 --set hw.train.mode=full  # FSDP
#     scripts/train.sh dpo data/dpo.jsonl runs/sft runs/dpo --profile h200x4
#   merge (single process, any profile):
#     scripts/train.sh merge runs/dpo runs/merged --profile rtx5090
#
# Extra args (--set k=v, --max-steps N, --limit N, --eval FILE) go to `ocrfix train <cmd>`.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
# fewer OOMs from fragmentation on the 32 GB card; harmless elsewhere
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD="${1:?usage: scripts/train.sh sft|dpo|merge ARGS... --profile NAME}"
PROFILE="${OCRFIX_PROFILE:-rtx5090}"
SETS=()
for ((i = 1; i <= $#; i++)); do
  case "${!i}" in
    --profile) j=$((i + 1)); PROFILE="${!j}" ;;
    --set) j=$((i + 1)); SETS+=(--set "${!j}") ;;
  esac
done
N_GPUS="$(ocrfix config --profile "$PROFILE" "${SETS[@]}" --key hw.n_gpus)"

if [[ "$CMD" == "merge" || "$N_GPUS" -le 1 ]]; then
  exec ocrfix train "$@"
fi
exec torchrun --standalone --nproc_per_node "$N_GPUS" -m ocrfix.cli train "$@"
