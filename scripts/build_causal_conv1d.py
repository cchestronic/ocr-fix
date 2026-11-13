#!/usr/bin/env python3
"""Build a causal-conv1d wheel against the LOCKED torch (ONLINE machine; called by
make_wheelhouse.py --with-kernels).

    python scripts/build_causal_conv1d.py WHEELHOUSE_DIR

PyPI ships only an sdist and upstream has no prebuilt wheel for our torch, whose C++ ABI changes
every minor release, so it is compiled here. Needs gcc/g++ and network; not a system CUDA
toolkit: nvcc comes from pip, pinned to torch's CUDA minor (nvcc and headers from different
minors refuse to compile). The wheel covers sm_75..sm_121, including sm_90 (H200) and sm_120
(RTX 5090).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC1D_VERSION = "1.7.0"
CUDA_MINOR = "13.0"  # must match torch's CUDA build (torch.version.cuda)
NVCC_VERSION = "13.0.88"  # nvidia-cuda-nvcc / -crt / nvvm
CCCL_VERSION = "13.0.85"


def main() -> None:
    wh = Path(sys.argv[1]).resolve()
    if list(wh.glob(f"causal_conv1d-{CC1D_VERSION}-*.whl")):
        print(f"causal-conv1d {CC1D_VERSION} already in {wh}")
        return
    torch_version = re.search(r"^torch==(\S+)", (ROOT / "requirements.lock").read_text(), re.M)[1]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        venv.create(work / "venv", with_pip=True)
        py = str(work / "venv" / "bin" / "python")

        def run(*args: str, **kw) -> str:
            return subprocess.run(args, check=True, text=True, capture_output=True, **kw).stdout

        run(py, "-m", "pip", "install", "-q", "--upgrade", "pip")
        run(py, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wh),
            f"torch=={torch_version}")  # fmt: skip
        run(py, "-m", "pip", "install", "-q", "setuptools", "wheel", "packaging", "ninja",
            f"nvidia-cuda-nvcc=={NVCC_VERSION}", f"nvidia-cuda-crt=={NVCC_VERSION}",
            f"nvidia-nvvm=={NVCC_VERSION}", f"nvidia-cuda-cccl=={CCCL_VERSION}")  # fmt: skip
        cuda_ver = run(py, "-c", "import torch; print(torch.version.cuda)").strip()
        if not cuda_ver.startswith(CUDA_MINOR):
            raise SystemExit(f"torch CUDA {cuda_ver} != {CUDA_MINOR}: update the pins")
        cuda = Path(
            run(
                py, "-c", "import nvidia, os; print(os.path.join(nvidia.__path__[0], 'cu13'))"
            ).strip()
        )
        lib = work / "lib"  # pip CUDA ships only libcudart.so.13; the linker wants libcudart.so
        lib.mkdir()
        (lib / "libcudart.so").symlink_to(cuda / "lib" / "libcudart.so.13")

        meta = json.load(
            urllib.request.urlopen(f"https://pypi.org/pypi/causal-conv1d/{CC1D_VERSION}/json")
        )
        sdist = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
        archive = work / "sdist.tar.gz"
        urllib.request.urlretrieve(sdist, archive)
        with tarfile.open(archive) as t:
            t.extractall(work / "src", filter="data")
        env = os.environ | {
            "CUDA_HOME": str(cuda), "PATH": f"{cuda}/bin:{os.environ['PATH']}",
            "CPATH": f"{cuda}/include:{cuda}/include/cccl", "LIBRARY_PATH": f"{lib}:{cuda}/lib",
            "CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
            "NVCC_THREADS": "2",
        }  # fmt: skip
        subprocess.run([py, "-m", "pip", "wheel", "-q", "--no-build-isolation", "--no-deps", "-w",
                        str(wh), "."], check=True, env=env,
                       cwd=work / "src" / f"causal_conv1d-{CC1D_VERSION}")  # fmt: skip
    print(*wh.glob(f"causal_conv1d-{CC1D_VERSION}-*.whl"))


if __name__ == "__main__":
    main()
