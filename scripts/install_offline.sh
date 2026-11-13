#!/usr/bin/env bash
# Run on the AIR-GAPPED machine. Never touches the network.
#   scripts/install_offline.sh [--with-kernels] [WHEELHOUSE_DIR] [VENV_DIR]
# --with-kernels also installs the optional Qwen3.5 training kernels (the wheelhouse must have
# been made with make_wheelhouse.sh --with-kernels).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
WITH_KERNELS=0
if [[ "${1:-}" == "--with-kernels" ]]; then WITH_KERNELS=1; shift; fi
WH="$(realpath "${1:-wheelhouse}")"
VENV="${2:-.venv}"

( cd "$WH" && sha256sum --quiet -c SHA256SUMS )

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --find-links "$WH" --require-hashes -r requirements.lock
if ((WITH_KERNELS)); then
  pip install --no-index --find-links "$WH" --require-hashes -r requirements-kernels.lock
  # Built by build_causal_conv1d.sh, outside the hashed locks (checked via SHA256SUMS above).
  pip install --no-index --find-links "$WH" --no-deps causal-conv1d
fi
# Editable install for development; --no-build-isolation uses the locked hatchling.
pip install --no-index --find-links "$WH" --no-build-isolation --no-deps -e .
ocrfix --help >/dev/null && echo "ocrfix installed into $VENV (kernels=$WITH_KERNELS)"
