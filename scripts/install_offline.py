#!/usr/bin/env python3
"""Run on the AIR-GAPPED machine: install from the wheelhouse. Never touches the network.

    python3 scripts/install_offline.py [--with-kernels] [WHEELHOUSE_DIR] [VENV_DIR]

Checks SHA256SUMS, creates the venv, installs requirements.lock (hash-checked), with
``--with-kernels`` also the optional Qwen3.5 training kernels (the wheelhouse must have been made
with make_wheelhouse.py --with-kernels), then ocrfix itself (editable).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = {"PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("wheelhouse", nargs="?", default="wheelhouse", type=Path)
    ap.add_argument("venv", nargs="?", default=".venv", type=Path)
    ap.add_argument("--with-kernels", action="store_true")
    a = ap.parse_args()
    wh = a.wheelhouse.resolve()

    for line in (wh / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((wh / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"checksum mismatch: {name}")

    venv.create(a.venv, with_pip=True)
    py = str((a.venv / "bin" / "python").resolve())
    env = os.environ | OFFLINE

    def pip(*args: str) -> None:
        subprocess.run([py, "-m", "pip", "install", "--no-index", "--find-links", str(wh), *args],
                       check=True, cwd=ROOT, env=env)  # fmt: skip

    pip("--require-hashes", "-r", "requirements.lock")
    if a.with_kernels:
        pip("--require-hashes", "-r", "requirements-kernels.lock")
        pip("--no-deps", "causal-conv1d")  # built locally, checked by SHA256SUMS above
    pip("--no-build-isolation", "--no-deps", "-e", ".")  # uses the locked hatchling
    subprocess.run([str(a.venv / "bin" / "ocrfix"), "--help"], check=True, env=env,
                   stdout=subprocess.DEVNULL)  # fmt: skip
    print(f"ocrfix installed into {a.venv} (kernels={a.with_kernels})")


if __name__ == "__main__":
    main()
