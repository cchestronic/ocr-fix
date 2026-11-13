"""calibration.yaml: the chosen tau and per-language flag rules. Read by score and verify.

A paragraph (or chunk) is flagged (likely has OCR damage) when any set part holds:

    spike_frac > rule.spike_frac   or   p90 > rule.p90   or   bpc > rule.bpc

A part that is null is off (files without ``bpc`` are the older spike_frac / p90 rules).

Languages without their own rule use ``"*"``. A null feature never flags, and neither does a
``low_confidence`` row (too few scored tokens). ``variant`` is always ``raw`` now (the
newline-collapsed ``flat`` scoring pass was removed); files with ``flat`` still load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ocrfix.scorer.features import spike_key

VERSION = 1


@dataclass
class Rule:
    spike_frac: float | None
    p90: float | None = None
    bpc: float | None = None
    recall: float | None = None
    false_flag: float | None = None
    target_met: bool | None = None
    n_clean: int = 0
    n_corrupted: int = 0


@dataclass
class Calibration:
    tau: float
    variant: str = "raw"  # raw | flat
    rules: dict[str, Rule] = field(default_factory=dict)
    target_recall: float | None = None
    max_false_flag: float | None = None

    @property
    def prefix(self) -> str:
        return "flat_" if self.variant == "flat" else ""

    def rule(self, lang: str | None) -> Rule:
        return self.rules.get(lang or "*") or self.rules["*"]

    def spike(self, feats: dict[str, Any]) -> float | None:
        """spike_frac at this tau: the ``spike_frac@tau`` column, else ``spike_frac``."""
        p = self.prefix
        v = feats.get(p + spike_key(self.tau))
        return feats.get(p + "spike_frac") if v is None else v

    def flag(self, feats: dict[str, Any], lang: str | None) -> bool:
        if feats.get("low_confidence"):
            return False  # too few scored tokens for a stable rate
        r = self.rule(lang)
        s = self.spike(feats)
        if r.spike_frac is not None and s is not None and s > r.spike_frac:
            return True
        for name, t in (("p90", r.p90), ("bpc", r.bpc)):
            v = feats.get(self.prefix + name)
            if t is not None and v is not None and v > t:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["version"] = VERSION
        return d


def load_calibration(path: str | Path) -> Calibration:
    d = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(d, dict)
    if d.get("version") != VERSION:
        raise ValueError(f"{path}: calibration version {d.get('version')!r}, expected {VERSION}")
    rules = {str(k): Rule(**v) for k, v in d["rules"].items()}
    if "*" not in rules:
        raise ValueError(f"{path}: no '*' rule")
    return Calibration(
        tau=float(d["tau"]),
        variant=d.get("variant", "raw"),
        rules=rules,
        target_recall=d.get("target_recall"),
        max_false_flag=d.get("max_false_flag"),
    )


def save_calibration(cal: Calibration, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(OmegaConf.to_yaml(OmegaConf.create(cal.to_dict())), encoding="utf-8")
    tmp.replace(path)
