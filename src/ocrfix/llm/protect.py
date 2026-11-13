"""Masking for the LLM: the shared rule in ``ocrfix.corpus.protect`` (re-exported)."""

from ocrfix.corpus.protect import PROTECT_PATTERNS, mask, protected_spans

__all__ = ["PROTECT_PATTERNS", "mask", "protected_spans"]
