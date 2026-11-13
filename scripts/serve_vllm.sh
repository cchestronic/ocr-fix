#!/usr/bin/env bash
# Serve the local Qwen3.5 (paths.base_model) with vLLM's OpenAI-compatible server, one engine per
# data-parallel replica, all settings from the profile (`hw.infer`, `hw.dtype`, `llm.*`):
#
#   scripts/serve_vllm.sh --profile rtx5090            # 1 engine on :8000, max-model-len 4096
#   scripts/serve_vllm.sh --profile h200x4             # 4 engines on :8000-8003 (GPU i -> port 8000+i)
#   scripts/serve_vllm.sh --profile h200x4 --dry-run   # print the commands only
#   scripts/serve_vllm.sh --profile rtx5090 -- --enforce-eager   # extra args for every engine
#   scripts/serve_vllm.sh --profile rtx5090 --set llm.model=/models/ocrfix-merged   # overrides
#
# `ocrfix correct` round-robins over `hw.infer.urls`, which must match base_port + i (a test
# checks the profiles). No tensor parallelism: 9B fits on one GPU, replicas scale better.
#
# Requirements
#   - vLLM 0.33.0 from requirements.lock (torch 2.14.0+cu130). Blackwell (RTX 5090, sm_120)
#     needs a vLLM built against CUDA >= 12.8 with sm_120 kernels; any vLLM >= 0.10 CUDA 12.8+
#     wheel works, the locked cu130 build does. CUDA 13.0 needs NVIDIA driver >= 580.
#     H200 (sm_90) works with the same wheel.
#   - Attention backend: `hw.vllm_attention_backend` -> `--attention-backend` (vLLM 0.33 has
#     no VLLM_ATTENTION_BACKEND env var any more). rtx5090 uses FLASHINFER; h200x4 uses
#     FLASH_ATTN. If FlashAttention is unavailable or fails on your GPU, fall back with
#         OCRFIX_ATTENTION_BACKEND=FLASHINFER scripts/serve_vllm.sh ...   (or TRITON_ATTN)
#     i.e. `--attention-backend FLASHINFER`; an empty value lets vLLM choose.
#     Qwen3.5's linear-attention (Gated DeltaNet) layers use vLLM's built-in Triton kernels
#     either way; the optional fla / causal-conv1d packages are not needed for serving.
#   - Offline: scripts/env.sh (sourced below) exports HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#     VLLM_NO_USAGE_STATS=1 (and friends); the model is a local directory (`llm.model`).
#
# Greedy decoding and thinking-off are set per request by the client, not here. Logs go to
# <paths.work_dir>/vllm/engine-<i>.log. Ctrl-C stops every engine; if one dies, all stop.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

PROFILE="${OCRFIX_PROFILE:-rtx5090}"
DRY_RUN=0
EXTRA=()
SETS=()
while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --set) SETS+=(--set "$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1 (vLLM args go after --)" >&2; exit 2 ;;
  esac
done

OCRFIX="${OCRFIX:-ocrfix}"
if ! command -v "$OCRFIX" >/dev/null; then
  [[ -x .venv/bin/ocrfix ]] || { echo "ocrfix not found (activate the venv or set OCRFIX)" >&2; exit 2; }
  OCRFIX=.venv/bin/ocrfix
  export PATH="$PWD/.venv/bin:$PATH"  # for vllm too
fi
cfg() {
  local v
  v="$("$OCRFIX" config --profile "$PROFILE" "${SETS[@]}" --key "$1")"
  [[ "$v" == "None" || "$v" == "null" ]] && v=""
  printf '%s' "$v"
}

MODEL="$(cfg llm.model)"
NAME="$(cfg llm.served_model_name)"
ADAPTER="$(cfg llm.adapter)"
ADAPTER_NAME="$(cfg llm.adapter_name)"
DTYPE="$(cfg hw.dtype)"
N_ENGINES="$(cfg hw.infer.data_parallel_size)"
TP="$(cfg hw.infer.tensor_parallel_size)"
BASE_PORT="$(cfg hw.infer.base_port)"
MAX_LEN="$(cfg hw.infer.max_model_len)"
GPU_UTIL="$(cfg hw.infer.gpu_memory_utilization)"
MAX_SEQS="$(cfg hw.infer.max_num_seqs)"
SEED="$(cfg llm.seed)"
BACKEND="${OCRFIX_ATTENTION_BACKEND-$(cfg hw.vllm_attention_backend)}"
LOG_DIR="$(cfg paths.work_dir)/vllm"

if ((DRY_RUN == 0)) && [[ ! -d "$MODEL" ]]; then
  echo "model directory not found: $MODEL (llm.model)" >&2
  exit 1
fi

PIDS=()
stop() { ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null || true; }
trap stop EXIT INT TERM

((DRY_RUN)) || mkdir -p "$LOG_DIR"
echo "profile=$PROFILE engines=$N_ENGINES tp=$TP max_model_len=$MAX_LEN backend=${BACKEND:-auto}"
for ((i = 0; i < N_ENGINES; i++)); do
  gpus="$(seq -s, $((i * TP)) $((i * TP + TP - 1)))"
  port=$((BASE_PORT + i))
  cmd=(vllm serve "$MODEL"
    --served-model-name "$NAME"
    --host 127.0.0.1 --port "$port"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs "$MAX_SEQS"
    --enable-prefix-caching
    --seed "$SEED")
  [[ -n "$BACKEND" ]] && cmd+=(--attention-backend "$BACKEND")
  if [[ -n "$ADAPTER" ]]; then
    # vLLM refuses adapters above --max-lora-rank (default 16); ours are train.lora.r (32)
    RANK="$(grep -o '"r": *[0-9]*' "$ADAPTER/adapter_config.json" 2>/dev/null | grep -o '[0-9]*$' || true)"
    RANK="${RANK:-$(cfg train.lora.r)}"
    cmd+=(--enable-lora --lora-modules "$ADAPTER_NAME=$ADAPTER" --max-lora-rank "$RANK")
  fi
  cmd+=("${EXTRA[@]}")
  if ((DRY_RUN)); then
    echo "CUDA_VISIBLE_DEVICES=$gpus ${cmd[*]}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpus" "${cmd[@]}" >"$LOG_DIR/engine-$i.log" 2>&1 &
  PIDS+=($!)
  echo "engine $i: GPU $gpus, http://127.0.0.1:$port (log $LOG_DIR/engine-$i.log)"
done
((DRY_RUN)) && exit 0

# Wait until every engine answers /health (model load + compile takes a few minutes).
for ((i = 0; i < N_ENGINES; i++)); do
  port=$((BASE_PORT + i))
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "engine $i died; see $LOG_DIR/engine-$i.log" >&2; exit 1; }
    sleep 5
  done
  echo "engine $i ready"
done
echo "all $N_ENGINES engine(s) ready"
wait -n "${PIDS[@]}"
echo "an engine exited; stopping the rest" >&2
exit 1
