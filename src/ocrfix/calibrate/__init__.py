"""Choose tau and per-language flag rules from scores of clean vs corrupted paragraphs.

``ocrfix calibrate <clean_scores> <corrupted_scores> <out_dir> [--gold-scores ...]`` writes
``calibration.yaml``; ``load_calibration(path).flag(features, lang)`` applies it.
"""

from ocrfix.calibrate.rules import Calibration, Rule, load_calibration, save_calibration

__all__ = ["Calibration", "Rule", "load_calibration", "save_calibration"]
