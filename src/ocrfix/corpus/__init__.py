"""Text analysis shared by every stage: spans (prose / math / table), language, placeholders,
the masking rule (protect.py) and the corpus files (files.py)."""

from ocrfix.corpus.files import doc_id_for, find_markdown, in_sample
from ocrfix.corpus.lang import detect_lang
from ocrfix.corpus.placeholders import PlaceholderError, restore
from ocrfix.corpus.tagger import tag_spans

__all__ = [
    "PlaceholderError",
    "detect_lang",
    "doc_id_for",
    "find_markdown",
    "in_sample",
    "restore",
    "tag_spans",
]
