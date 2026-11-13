"""OCR error analysis on gold pairs: char alignment, op types, rate tables.

``ocrfix errors table <gold.jsonl> <out_dir> [--fixed fixed.jsonl]`` -> error_table.json,
confusion_map.json, summary.md. Use ``gold_train.jsonl`` (not the eval half) when the
tables feed ``calibrate``.
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
