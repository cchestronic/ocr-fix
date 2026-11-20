"""Config loading: configs/default.yaml + configs/stages/*.yaml (one file per step) +
configs/profiles/<profile>.yaml (the machine: `model` and `hw` only) + --set overrides, merged
in that order. Each setting lives in exactly one file.

Derived after the merge, so they are never set twice:
  hw.servers             the vLLM server groups (`server_plan`): each group's engines, ports
                         (per machine, from hw.infer.ports) and memory share (automatic when
                         groups share GPUs). Gives the internal address lists:
  hw.infer.score_urls    every base engine (an explicit list overrides it: advanced)
  hw.infer.correct_urls  every trained engine when correct.model is a trained model, else the
                         base engines (an explicit list overrides it: advanced)
  hw.infer.concurrency   null -> (engines of the bigger list) x max_num_seqs
  model, correct.model   a relative path resolves against the current folder, then the project
  score.backend,         auto -> vllm when this profile has servers for it (score_urls /
  correct.backend        correct_urls not empty: engines > 0, or addresses listed), else
                         transformers (in the command itself); vllm / transformers as set
The names vLLM serves the models under follow from the paths (`served_name`)."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_PROFILE = "rtx5090"
PROFILE_KEYS = ("model", "hw")
TRAIN_MODES = ("qlora", "lora", "full")
# Optional Qwen3.5 kernels (requirements-kernels.in); packing is impossible without them.
KERNEL_MODULES = ("fla", "causal_conv1d")


def config_dir() -> Path:
    """OCRFIX_CONFIG_DIR, else <repo>/configs (src layout, editable install)."""
    env = os.environ.get("OCRFIX_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "configs"


def available_profiles(root: Path | None = None) -> list[str]:
    root = root or config_dir()
    return sorted(p.stem for p in (root / "profiles").glob("*.yaml"))


def load_config(
    profile: str | None = None,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    root = root or config_dir()
    profile = profile or os.environ.get("OCRFIX_PROFILE") or DEFAULT_PROFILE
    profile_path = root / "profiles" / f"{profile}.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"unknown profile {profile!r}; available: {available_profiles(root)}"
        )
    prof = OmegaConf.load(profile_path)
    extra = sorted(set(prof.keys()) - set(PROFILE_KEYS))
    if extra:
        raise ValueError(
            f"{profile_path.name}: a profile sets only {list(PROFILE_KEYS)}; move {extra} to "
            "configs/stages/ (or pass --set)"
        )
    cfg = OmegaConf.merge(*load_base(root), prof, OmegaConf.from_dotlist(overrides or []))
    assert isinstance(cfg, DictConfig)
    _resolve_model_dirs(cfg)
    _derive_servers(cfg)
    _resolve_backends(cfg)
    _validate(cfg)
    return cfg


def _leaves(node: object, prefix: str = "") -> list[str]:
    if not isinstance(node, dict) or not node:
        return [prefix]
    return [x for k, v in node.items() for x in _leaves(v, f"{prefix}.{k}" if prefix else str(k))]


def load_base(root: Path | None = None) -> list[DictConfig]:
    """default.yaml, then every configs/stages/*.yaml in name order. A setting may live in only
    one of these files, so a value set in the wrong file can't silently lose to another."""
    root = root or config_dir()
    files = [root / "default.yaml", *sorted((root / "stages").glob("*.yaml"))]
    confs, seen = [], {}
    for f in files:
        c = OmegaConf.load(f)
        for key in _leaves(OmegaConf.to_container(c, resolve=False)):
            if key in seen:
                raise ValueError(f"config: {key} is set in both {seen[key]} and {f.name}")
            seen[key] = f.name
        confs.append(c)
    return confs


def _local_dir(path: str) -> str:
    """A relative model path: the current directory first, then the project root (transformers
    and vLLM would take a relative path that doesn't exist for a hub id)."""
    p = Path(path)
    in_root = config_dir().parent / p
    if not p.is_absolute() and not p.exists() and in_root.exists():
        return str(in_root.resolve())
    return path


def _resolve_model_dirs(cfg: DictConfig) -> None:
    if not OmegaConf.is_missing(cfg, "model"):
        cfg.model = _local_dir(str(cfg.model))
    for k in ("model", "adapter"):
        if cfg.correct.get(k):
            cfg.correct[k] = _local_dir(str(cfg.correct[k]))


# --- vLLM servers ------------------------------------------------------------------------------

# Settings replaced by hw.servers (a profile still using them gets a clear error).
OLD_INFER = ("engines", "tensor_parallel_size", "gpu_memory_utilization", "host", "base_port",
             "correct_port")  # fmt: skip
SERVER_MODELS = ("base", "trained")
LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::")
MEMORY_TOTAL = 0.88  # GPU memory share all engines on one GPU take together


@dataclass(frozen=True)
class EngineSpec:
    """One vLLM engine: a model (base / trained) on some GPUs of one machine."""

    model: str  # base | trained
    host: str  # the machine's address; the engine listens on it
    gpus: tuple[int, ...]
    port: int
    memory: float  # GPU memory share (vLLM --gpu-memory-utilization)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"


def server_plan(cfg: DictConfig) -> list[EngineSpec]:
    """Every engine of ``hw.servers``, in list order.

    A group ``{model, host, gpus, memory?}`` runs one engine per ``hw.infer.gpus_per_engine``
    GPUs. Ports count per machine and model from ``hw.infer.ports`` (base 8000, 8001, ...;
    trained 8100, ...). Memory: a group's ``memory``, else ``MEMORY_TOTAL`` divided by the
    number of groups on that machine sharing its busiest GPU (two groups on the same GPUs:
    0.44 each; a group alone: 0.88)."""
    hw = cfg.hw
    inf = hw.infer
    groups = list(hw.get("servers") or [])
    tp = int(inf.gpus_per_engine)
    ports = {m: int(inf.ports[m]) for m in SERVER_MODELS}
    parsed = []
    for i, g in enumerate(groups):
        where = f"hw.servers[{i}]"
        unknown = set(g) - {"model", "host", "gpus", "memory"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)} (model, host, gpus, memory)")
        model = str(g.get("model"))
        if model not in SERVER_MODELS:
            raise ValueError(f"{where}: model must be base or trained, got {model!r}")
        host = str(g.get("host") or "127.0.0.1")
        gpus = g.get("gpus", "all")
        if gpus == "all":
            if int(hw.n_gpus) < 1:
                raise ValueError(f"{where}: gpus: all needs hw.n_gpus > 0, and this profile has "
                                 "no GPUs; list the GPUs, e.g. gpus: [0, 1, 2, 3]")  # fmt: skip
            gpus = list(range(int(hw.n_gpus)))
        gpus = [int(x) for x in gpus]
        if not gpus or len(gpus) % tp:
            raise ValueError(f"{where}: {len(gpus)} GPUs is not a multiple of "
                             f"hw.infer.gpus_per_engine ({tp})")  # fmt: skip
        parsed.append((model, host, gpus, g.get("memory")))
    use: dict[tuple[str, int], int] = {}
    for _, host, gpus, _ in parsed:
        for gpu in gpus:
            use[(host, gpu)] = use.get((host, gpu), 0) + 1
    out, next_port = [], {}
    for model, host, gpus, memory in parsed:
        share = max(use[(host, gpu)] for gpu in gpus)
        mem = float(memory) if memory is not None else round(MEMORY_TOTAL / share, 2)
        for k in range(0, len(gpus), tp):
            key = (host, model)
            port = next_port.get(key, ports[model])
            next_port[key] = port + 1
            out.append(EngineSpec(model, host, tuple(gpus[k : k + tp]), port, mem))
    return out


def _derive_servers(cfg: DictConfig) -> None:
    """hw.servers -> hw.infer.score_urls / correct_urls / concurrency (internal; never set)."""
    inf = cfg.hw.infer
    old = [k for k in OLD_INFER if k in inf]
    if old:
        raise ValueError(
            f"hw.infer.{old} are replaced by one list, hw.servers: e.g. "
            "servers: [{model: base, host: 127.0.0.1, gpus: all}, {model: trained, host: "
            "127.0.0.1, gpus: all}] (ports from hw.infer.ports, memory automatic; "
            "configs/profiles/rtx5090.yaml explains it)"
        )
    plan = server_plan(cfg)
    base = [e.url for e in plan if e.model == "base"]
    trained = [e.url for e in plan if e.model == "trained"]
    tuned = cfg.correct.model and str(cfg.correct.model) != str(cfg.model)
    # Advanced: an explicit list replaces the one built from hw.servers (e.g. servers behind a
    # load balancer, or the tests' fake servers). Not used in the profiles.
    if inf.get("score_urls") is None:
        inf.score_urls = base
    if inf.get("correct_urls") is None:
        tuned_urls = trained if (tuned and not cfg.correct.adapter) else None
        inf.correct_urls = tuned_urls if tuned_urls is not None else list(inf.score_urls)
    if inf.get("concurrency") is None:
        n = max(len(inf.score_urls), len(inf.correct_urls), 1)
        inf.concurrency = n * int(inf.max_num_seqs)


BACKENDS = ("vllm", "transformers")


def _resolve_backends(cfg: DictConfig) -> None:
    """``auto`` -> vllm if the profile has servers for that use (the urls, after
    `_derive_urls`), else transformers. So a GPU machine with engines uses vLLM for everything
    and a machine without (the cpu profile) uses transformers for everything: scores always
    come from one engine per machine."""
    for section, urls in (("score", "score_urls"), ("correct", "correct_urls")):
        b = str(cfg[section].backend)
        if b == "auto":
            cfg[section].backend = "vllm" if cfg.hw.infer[urls] else "transformers"
        elif b not in BACKENDS:
            raise ValueError(f"{section}.backend must be auto, vllm or transformers, got {b!r}")


# The names vLLM serves the models under (`ocrfix serve`); requests ask for these.
BASE_NAME, TUNED_NAME, ADAPTER_NAME = "base", "tuned", "adapter"


def correct_model(cfg: DictConfig) -> str:
    """The model directory that corrects: correct.model, else the base."""
    return str(cfg.correct.model or cfg.model)


def served_name(cfg: DictConfig, role: str = "correct") -> str:
    """``score`` -> the base model's name; ``correct`` -> the adapter's, the tuned model's, or
    the base's (whichever correct.* selects)."""
    if role == "score":
        return BASE_NAME
    if cfg.correct.adapter:
        return ADAPTER_NAME
    return TUNED_NAME if correct_model(cfg) != str(cfg.model) else BASE_NAME


def _validate(cfg: DictConfig) -> None:
    mode = cfg.hw.train.mode
    if mode not in TRAIN_MODES:
        raise ValueError(f"hw.train.mode must be one of {TRAIN_MODES}, got {mode!r}")
    # Resolve interpolations / mandatory values early so typos fail before work starts.
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def missing_kernels() -> list[str]:
    return [m for m in KERNEL_MODULES if importlib.util.find_spec(m) is None]
