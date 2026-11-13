"""Corpus construction: MinerU markdown -> paragraph Records with prose/math/table spans.

Input:  MinerU output dir (``*.md``)
Output: ``data/corpus.jsonl`` Record JSONL
"""

from ocrfix.corpus.ingest import doc_id_for, doc_records, ingest, split_paragraphs
from ocrfix.corpus.lang import detect_lang
from ocrfix.corpus.placeholders import PlaceholderError, restore, substitute
from ocrfix.corpus.tagger import tag_spans

__all__ = [
    "PlaceholderError",
    "detect_lang",
    "doc_id_for",
    "doc_records",
    "ingest",
    "restore",
    "split_paragraphs",
    "substitute",
    "tag_spans",
]
