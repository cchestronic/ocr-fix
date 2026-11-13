"""Per-paragraph LM surprisal features (Qwen3.5) for OCR-error detection.

Input:  paragraphs JSONL (Records from ``ingest``)
Output: scores JSONL: doc_id, para_id + features (see ``features.py``)
"""
