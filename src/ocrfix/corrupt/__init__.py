"""Synthetic OCR damage at the rates measured by ``errors table`` (``ocrfix corrupt``)."""

from ocrfix.corrupt.ops import GENERATORS, Edit
from ocrfix.corrupt.run import build_models, corrupt_text, rng_for

__all__ = ["GENERATORS", "Edit", "build_models", "corrupt_text", "rng_for"]
