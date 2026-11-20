"""`ocrfix serve`: start the vLLM engines of this machine (hw.servers in the profile).

    ./ocrfix serve                    # every server group whose host is this machine
    ./ocrfix serve --only base        # only the base model's groups
    ./ocrfix serve --only trained     # only the trained model's (needs correct.model)
    ./ocrfix serve --dry-run          # print the engines' commands only
    ./ocrfix serve -- --enforce-eager # extra vLLM arguments for every engine

`hw.servers` lists groups `{model: base|trained, host, gpus, memory?}` (config.server_plan):
one engine per `hw.infer.gpus_per_engine` GPUs, ports counted per machine from
`hw.infer.ports`, memory shared automatically when groups use the same GPUs. A group belongs
to this machine when its host is a loopback address or one of this machine's addresses
(`--host` names this machine's address if that detection fails). Groups on other machines are
started there, with the same command and the same profile.

The base model is served as ``base``; a LoRA adapter (``correct.adapter``) rides on the base
engines as ``adapter``; trained groups serve ``correct.model`` as ``tuned``. Requests ask for
these names (``config.served_name``), so the engines and the commands agree without settings.
Trained groups are skipped (with a note) when correct.model is not set.

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
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ocrfix.config import (
    ADAPTER_NAME,
    BASE_NAME,
    LOOPBACK,
    EngineSpec,
    correct_model,
    served_name,
    server_plan,
)
from ocrfix.offline import OFFLINE_ENV


@dataclass
class Engine:
    name: str
    gpus: str
    port: int
    cmd: list[str]
    url: str


def my_addresses() -> set[str]:
    """This machine's names and addresses (loopback included)."""
    out = set(LOOPBACK)
    try:
        name = socket.gethostname()
        out |= {name, socket.getfqdn()}
        out |= {a[4][0] for a in socket.getaddrinfo(name, None)}
    except OSError:
        pass
    try:  # the address of the default route (a UDP "connect" sends nothing)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            out.add(sk.getsockname()[0])
    except Exception:
        pass
    return out


def local_specs(cfg, *, only: str | None = None, me: str | None = None) -> list[EngineSpec]:
    """The engines of hw.servers on this machine (``me``: its address, when detection fails)."""
    mine = my_addresses() | ({me} if me else set())
    return [e for e in server_plan(cfg) if e.host in mine and (only is None or e.model == only)]


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
    cfg, *, only: str | None = None, me: str | None = None, extra: list[str] | None = None,
    echo=print,
) -> list[Engine]:  # fmt: skip
    """The vLLM commands for this machine's server groups (``only``: base or trained)."""
    inf = cfg.hw.infer
    if only not in (None, "base", "trained"):
        raise ValueError(f"--only must be base or trained, got {only!r}")
    specs = local_specs(cfg, only=only, me=me)
    tuned = correct_model(cfg) != str(cfg.model)
    if any(e.model == "trained" for e in specs) and not tuned:
        if only == "trained":
            raise ValueError(
                "the trained servers serve correct.model, which is not set: "
                "--set correct.model=runs/sft_merged (or set it in configs/stages/correct.yaml)"
            )
        echo("note: trained groups skipped (correct.model is not set); starting base only")
        specs = [e for e in specs if e.model != "trained"]
    if not specs:
        hosts = sorted({e.host for e in server_plan(cfg)})
        if hosts:
            why = (
                f" (hosts in the list: {hosts}; this machine: {sorted(my_addresses())}; "
                "add --host <this machine's address> if it is one of them)"
            )
        else:
            why = " (hw.servers is empty in this profile)"
        raise ValueError("no server group of hw.servers is on this machine" + why)
    backend = os.environ.get("OCRFIX_ATTENTION_BACKEND", cfg.hw.vllm_attention_backend or "")
    out = []
    for e in specs:
        if e.model == "trained":
            model, name, lora = correct_model(cfg), served_name(cfg, "correct"), None
        else:
            model, name, lora = str(cfg.model), BASE_NAME, cfg.correct.adapter
        if not Path(model).is_dir():
            raise ValueError(f"model directory not found: {model}")
        cmd = [
            _vllm(), "serve", model, "--served-model-name", name,
            "--host", e.host, "--port", str(e.port), "--dtype", str(cfg.hw.dtype),
            "--tensor-parallel-size", str(len(e.gpus)),
            "--max-model-len", str(inf.max_model_len),
            "--gpu-memory-utilization", str(e.memory),
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
        out.append(Engine(name, ",".join(map(str, e.gpus)), e.port, cmd, e.url))
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
            echo(f"{e.name} engine: GPU {e.gpus}, {e.url} (log {log})")
        for e, p in zip(engs, procs, strict=True):
            while not _healthy(e.url):
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
