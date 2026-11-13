#!/usr/bin/env bash
# Run on an ONLINE machine. Produces a wheelhouse to carry onto the air-gapped network.
#   scripts/make_wheelhouse.sh [--with-kernels] [WHEELHOUSE_DIR]
# --with-kernels adds the optional Qwen3.5 training kernels (requirements-kernels.lock and a
# locally compiled causal-conv1d, ~10 min build). Without them transformers falls back to
# pure PyTorch: correct, slower training, no sequence packing.
set -euo pipefail
cd "$(dirname "$0")/.."
WITH_KERNELS=0
if [[ "${1:-}" == "--with-kernels" ]]; then WITH_KERNELS=1; shift; fi
OUT="${1:-wheelhouse}"
PY_VERSION="${PY_VERSION:-3.12}"
mkdir -p "$OUT"

download() {  # binary wheels for the target (Linux x86_64, CPython $PY_VERSION), hash-checked
  pip download --require-hashes -r "$1" --dest "$OUT" --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --python-version "$PY_VERSION" --implementation cp
}

download requirements.lock
cp requirements.lock "$OUT/"

if ((WITH_KERNELS)); then
  download requirements-kernels.lock
  cp requirements-kernels.lock "$OUT/"
  # No wheel on PyPI: compile against the locked torch. A locally built wheel has no stable
  # hash, so it is outside the locks and covered by SHA256SUMS below instead.
  scripts/build_causal_conv1d.sh "$OUT"
fi

# Build the project wheel itself so the target needs no build step.
pip wheel --no-deps --wheel-dir "$OUT" .

( cd "$OUT" && sha256sum ./*.whl > SHA256SUMS )
echo "wheelhouse ready: $OUT ($(ls "$OUT"/*.whl | wc -l) wheels, kernels=$WITH_KERNELS)"
