"""Synthetic OCR corruption of clean paragraphs at the rates measured by ``errors table``.

``ocrfix corrupt <clean.jsonl> <error_table.json> <confusion_map.json> <out.jsonl>``.
"""

from ocrfix.corrupt.ops import GENERATORS, Edit
from ocrfix.corrupt.run import build_models, corrupt_file, corrupt_text, rng_for, run_corrupt

__all__ = [
    "GENERATORS",
    "Edit",
    "build_models",
    "corrupt_file",
    "corrupt_text",
    "rng_for",
    "run_corrupt",
]
