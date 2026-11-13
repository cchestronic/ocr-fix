"""`ocrfix serve`: start the vLLM engines the chunk commands talk to.

    ocrfix serve                      # the base model, one engine per GPU (port base_port + i)
    ocrfix serve --correct            # the correction model (correct.model), on correct_port + i
    ocrfix serve --dry-run            # print the commands only
    ocrfix serve -- --enforce-eager   # extra vLLM arguments for every engine

Every setting comes from the config: the model paths (``model``, ``correct.model``,
``correct.adapter``), ``hw.dtype``, ``hw.vllm_attention_backend`` and ``hw.infer`` (engines,
host, base_port, max_model_len, gpu_memory_utilization, max_num_seqs, tensor_parallel_size);
``--port``, ``--engines``, ``--gpus`` and ``--memory`` override them for one run (e.g. a base
engine and a tuned engine sharing one GPU at ``--memory 0.44`` each).

The base model is served as ``base``; a LoRA adapter (``correct.adapter``) rides on the base
engines as ``adapter``; ``--correct`` serves ``correct.model`` as ``tuned``. Requests ask for
these names (``config.served_name``), so the engines and the commands agree without settings.

Engine logs go to ``<work_dir>/vllm/<name>-<port>.log``. The command waits until every engine
answers ``/health``, then until one exits or Ctrl-C; then it stops them all.

Blackwell (RTX 5090, sm_120) needs a vLLM built against CUDA >= 12.8 (the locked cu130 build
works; CUDA 13 needs driver >= 580). Override the attention backend with
``OCRFIX_ATTENTION_BACKEND=TRITON_ATTN`` (an empty value lets vLLM choose).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ocrfix.config import ADAPTER_NAME, BASE_NAME, correct_model, served_name
from ocrfix.offline import OFFLINE_ENV


@dataclass
class Engine:
    name: str
    gpus: str
    port: int
    cmd: list[str]

    def url(self, host: str) -> str:
        host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return f"http://{host}:{self.port}"


def _vllm() -> str:
    found = shutil.which("vllm")
    return found or str(Path(sys.executable).parent / "vllm")


def _lora_rank(adapter: str, default: int) -> int:
    """vLLM refuses adapters above --max-lora-rank (default 16); ours are train.lora.r (32)."""
    try:
        return int(json.loads((Path(adapter) / "adapter_config.json").read_text())["r"])
    except (OSError, KeyError, ValueError):
        return default


def engines(
    cfg, *, correct: bool = False, port: int | None = None, n: int | None = None,
    gpus: list[int] | None = None, memory: float | None = None, extra: list[str] | None = None,
) -> list[Engine]:  # fmt: skip
    inf = cfg.hw.infer
    if correct:
        port = port or int(inf.correct_port)
        model = correct_model(cfg)
        if model == str(cfg.model):
            raise ValueError(
                "--correct serves correct.model, which is not set (the base engines already "
                "correct; a correct.adapter rides on them)"
            )
        name, lora = served_name(cfg, "correct"), None
    else:
        model, name, lora = str(cfg.model), BASE_NAME, cfg.correct.adapter
    if not Path(model).is_dir():
        raise ValueError(f"model directory not found: {model}")
    n = int(n or inf.engines)
    tp = int(inf.tensor_parallel_size)
    gpus = list(gpus) if gpus else list(range(n * tp))
    if len(gpus) < n * tp:
        raise ValueError(f"{n} engines x {tp} GPUs need {n * tp} GPUs, got {gpus}")
    backend = os.environ.get("OCRFIX_ATTENTION_BACKEND", cfg.hw.vllm_attention_backend or "")
    out = []
    for i in range(n):
        p = int(port or inf.base_port) + i
        cmd = [
            _vllm(), "serve", model, "--served-model-name", name,
            "--host", str(inf.host), "--port", str(p), "--dtype", str(cfg.hw.dtype),
            "--tensor-parallel-size", str(tp), "--max-model-len", str(inf.max_model_len),
            "--gpu-memory-utilization", str(memory or inf.gpu_memory_utilization),
            "--max-num-seqs", str(inf.max_num_seqs), "--enable-prefix-caching",
            "--seed", str(cfg.seed),
        ]  # fmt: skip
        if backend:
            cmd += ["--attention-backend", backend]
        if lora:
            rank = _lora_rank(str(lora), int(cfg.train.lora.r))
            cmd += ["--enable-lora", "--lora-modules", f"{ADAPTER_NAME}={lora}",
                    "--max-lora-rank", str(rank)]  # fmt: skip
        cmd += list(extra or [])
        out.append(Engine(name, ",".join(map(str, gpus[i * tp : (i + 1) * tp])), p, cmd))
    return out


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def run(cfg, engs: list[Engine], echo=print) -> int:
    """Start the engines, wait for /health, then until one exits or Ctrl-C; stop them all."""
    log_dir = Path(str(cfg.work_dir)) / "vllm"
    log_dir.mkdir(parents=True, exist_ok=True)
    host = str(cfg.hw.infer.host)
    procs: list[subprocess.Popen] = []

    def stop(*_) -> None:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))
    try:
        for e in engs:
            log = log_dir / f"{e.name}-{e.port}.log"
            # the venv's bin first: vLLM's compiler runs `ninja` (a pip package) from PATH
            path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
            env = os.environ | OFFLINE_ENV | {"CUDA_VISIBLE_DEVICES": e.gpus, "PATH": path}
            with open(log, "w") as fh:
                procs.append(subprocess.Popen(e.cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))
            echo(f"{e.name} engine: GPU {e.gpus}, {e.url(host)} (log {log})")
        for e, p in zip(engs, procs, strict=True):
            while not _healthy(e.url(host)):
                if p.poll() is not None:
                    echo(f"{e.name} engine on port {e.port} died; see {log_dir}")
                    return 1
                time.sleep(5)
            echo(f"{e.name} engine on port {e.port} ready")
        echo(f"all {len(engs)} engine(s) ready; Ctrl-C stops them")
        while all(p.poll() is None for p in procs):
            time.sleep(2)
        echo("an engine exited; stopping the rest")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stop()
