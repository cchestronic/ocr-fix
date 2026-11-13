#!/usr/bin/env bash
# Build a causal-conv1d wheel against the LOCKED torch (run on the ONLINE machine; called by
# make_wheelhouse.sh --with-kernels). PyPI ships only an sdist and upstream has no prebuilt
# wheel for our torch, whose C++ ABI changes every minor release, so we compile it ourselves.
#   scripts/build_causal_conv1d.sh WHEELHOUSE_DIR
# Needs: gcc/g++, network. Does NOT need a system CUDA toolkit: nvcc comes from pip, pinned
# to the same CUDA minor as torch (nvcc/headers from different minors refuse to compile).
# Output wheel covers sm_75..sm_121, including sm_90 (H200) and sm_120 (RTX 5090).
set -euo pipefail
cd "$(dirname "$0")/.."
WH="$(realpath "$1")"
CC1D_VERSION=1.7.0
CUDA_MINOR=13.0          # must match torch's CUDA build (torch.version.cuda)
NVCC_VERSION=13.0.88     # nvidia-cuda-nvcc / -crt / nvvm
CCCL_VERSION=13.0.85

if ls "$WH"/causal_conv1d-"$CC1D_VERSION"-*.whl >/dev/null 2>&1; then
  echo "causal-conv1d $CC1D_VERSION already in $WH"; exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install -q --upgrade pip
TORCH_VERSION="$(grep -oP '^torch==\K[^ ]+' requirements.lock)"
"$PY" -m pip install -q --no-index --find-links "$WH" "torch==$TORCH_VERSION"
"$PY" -m pip install -q setuptools wheel packaging ninja \
  "nvidia-cuda-nvcc==$NVCC_VERSION" "nvidia-cuda-crt==$NVCC_VERSION" \
  "nvidia-nvvm==$NVCC_VERSION" "nvidia-cuda-cccl==$CCCL_VERSION"
"$PY" - "$CUDA_MINOR" <<'PYEOF'
import sys, torch
assert torch.version.cuda.startswith(sys.argv[1]), (torch.version.cuda, sys.argv[1])
PYEOF

CUDA="$("$PY" -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')"
# pip CUDA ships only libcudart.so.13; the linker wants libcudart.so.
mkdir -p "$WORK/lib" && ln -s "$CUDA/lib/libcudart.so.13" "$WORK/lib/libcudart.so"

mkdir -p "$WORK/src"
"$PY" - "$CC1D_VERSION" "$WORK/src/sdist.tar.gz" <<'PYEOF'
import json, sys, urllib.request
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/causal-conv1d/{sys.argv[1]}/json"))
urllib.request.urlretrieve(next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist"), sys.argv[2])
PYEOF
tar -xzf "$WORK/src/sdist.tar.gz" -C "$WORK/src"

(
  cd "$WORK"/src/causal_conv1d-"$CC1D_VERSION"
  export CUDA_HOME="$CUDA" PATH="$CUDA/bin:$PATH"
  export CPATH="$CUDA/include:$CUDA/include/cccl" LIBRARY_PATH="$WORK/lib:$CUDA/lib"
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS=2
  "$PY" -m pip wheel -q --no-build-isolation --no-deps -w "$WH" .
)
ls "$WH"/causal_conv1d-"$CC1D_VERSION"-*.whl
