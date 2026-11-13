#!/usr/bin/env python3
"""Run on an ONLINE machine: download every wheel the air-gapped machine needs.

    python scripts/make_wheelhouse.py [--with-kernels] [WHEELHOUSE_DIR]

Binary wheels for Linux x86_64 / CPython 3.12 (``--python-version``), hash-checked against
requirements.lock, plus the project wheel itself and a SHA256SUMS file. ``--with-kernels`` adds
the optional Qwen3.5 training kernels: requirements-kernels.lock and a locally compiled
causal-conv1d (scripts/build_causal_conv1d.py, ~10 min). Without them transformers falls back to
pure PyTorch: correct, slower training, no sequence packing.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("out", nargs="?", default="wheelhouse", type=Path)
    ap.add_argument("--with-kernels", action="store_true")
    ap.add_argument("--python-version", default="3.12")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def download(lock: str) -> None:
        pip("download", "--require-hashes", "-r", lock, "--dest", str(out), "--only-binary=:all:",
            "--platform", "manylinux_2_28_x86_64", "--python-version", a.python_version,
            "--implementation", "cp")  # fmt: skip
        shutil.copy(ROOT / lock, out / lock)

    download("requirements.lock")
    if a.with_kernels:
        download("requirements-kernels.lock")
        # no wheel on PyPI: compiled against the locked torch; covered by SHA256SUMS, not a lock
        subprocess.run([sys.executable, str(ROOT / "scripts/build_causal_conv1d.py"), str(out)],
                       check=True)  # fmt: skip
    pip("wheel", "--no-deps", "--wheel-dir", str(out), ".")  # the project, so no build step later

    wheels = sorted(out.glob("*.whl"))
    lines = [f"{hashlib.sha256(w.read_bytes()).hexdigest()}  ./{w.name}" for w in wheels]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wheelhouse ready: {out} ({len(wheels)} wheels, kernels={a.with_kernels})")


if __name__ == "__main__":
    main()
