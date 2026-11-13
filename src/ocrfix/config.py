"""Config loading: configs/default.yaml + configs/profiles/<profile>.yaml + --set overrides."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_PROFILE = "rtx5090"
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
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "default.yaml"),
        OmegaConf.load(profile_path),
        OmegaConf.from_dotlist(overrides or []),
    )
    assert isinstance(cfg, DictConfig)
    _resolve_model_dir(cfg)
    _validate(cfg)
    return cfg


def _resolve_model_dir(cfg: DictConfig) -> None:
    """A relative ``paths.base_model``: the current directory first, then the project root
    (transformers and vLLM would take a relative path that doesn't exist for a hub id)."""
    if OmegaConf.is_missing(cfg.paths, "base_model"):
        return
    p = Path(str(cfg.paths.base_model))
    in_root = config_dir().parent / p
    if not p.is_absolute() and not p.exists() and in_root.exists():
        cfg.paths.base_model = str(in_root.resolve())


def _validate(cfg: DictConfig) -> None:
    mode = cfg.hw.train.mode
    if mode not in TRAIN_MODES:
        raise ValueError(f"hw.train.mode must be one of {TRAIN_MODES}, got {mode!r}")
    # Resolve interpolations / mandatory values early so typos fail before work starts.
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def missing_kernels() -> list[str]:
    return [m for m in KERNEL_MODULES if importlib.util.find_spec(m) is None]
