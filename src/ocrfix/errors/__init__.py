"""OCR error analysis on gold pairs: char alignment, op types, rate tables.

``ocrfix errors table <gold_chunks/sft> <out_dir>`` -> error_table.json, confusion_map.json,
summary.md (never the eval side: the tables feed calibration and training data).
"""

from ocrfix.errors.align import TYPES, EditOp, align, classify, diff_ops
from ocrfix.errors.table import build, run_table, to_jamo, write_outputs

__all__ = [
    "TYPES",
    "EditOp",
    "align",
    "build",
    "classify",
    "diff_ops",
    "run_table",
    "to_jamo",
    "write_outputs",
]
