"""Rule-based fixers (and later the LLM fixer), run per document in ``fix.order``.

``ocrfix fix <paragraphs.jsonl> <out.jsonl> --ops <ops.jsonl>``. See ``base`` for the
interface and registry.
"""

from ocrfix.fixers.base import (
    FIXERS,
    Fixer,
    FixerError,
    Op,
    build_fixers,
    check_ops,
    discover,
    register,
)

__all__ = [
    "FIXERS",
    "Fixer",
    "FixerError",
    "Op",
    "build_fixers",
    "check_ops",
    "discover",
    "register",
]
