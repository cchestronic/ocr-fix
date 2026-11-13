"""Config loading: configs/default.yaml + configs/stages/*.yaml (one file per step) +
configs/profiles/<profile>.yaml (the machine: `model` and `hw` only) + --set overrides, merged
in that order. Each setting lives in exactly one file.

Derived after the merge, so they are never set twice:
  hw.infer.score_urls    null -> this machine's engines, http://host:base_port+i
  hw.infer.correct_urls  null -> score_urls when the base corrects, else the tuned engines
                         (http://host:correct_port+i, `ocrfix serve --correct`)
  model, correct.model   a relative path resolves against the current folder, then the project
The names vLLM serves the models under follow from the paths (`served_name`)."""

from __future__ import annotations

import importlib.util
import os
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
    _derive_urls(cfg)
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


def _derive_urls(cfg: DictConfig) -> None:
    """score_urls null -> the base engines, http://host:base_port+i. correct_urls null -> the
    base engines too when the base (or an adapter on it) corrects, else the tuned engines that
    `ocrfix serve --correct` starts, http://host:correct_port+i."""
    inf = cfg.hw.infer
    host = "127.0.0.1" if str(inf.host) in ("0.0.0.0", "::") else str(inf.host)
    n = int(inf.engines)
    if inf.score_urls is None:
        inf.score_urls = [f"http://{host}:{int(inf.base_port) + i}" for i in range(n)]
    if inf.correct_urls is None:
        tuned = cfg.correct.model and str(cfg.correct.model) != str(cfg.model)
        if tuned and not cfg.correct.adapter:
            inf.correct_urls = [f"http://{host}:{int(inf.correct_port) + i}" for i in range(n)]
        else:
            inf.correct_urls = list(inf.score_urls)


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
