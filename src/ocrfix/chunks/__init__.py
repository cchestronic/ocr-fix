"""Markdown documents <-> masked chunks of 512-768 tokens (the unit of refinement).

``split``: one document -> chunks. ``run``: a folder tree -> JSONL shards, in parallel.
``rebuild``: chunks -> the original (byte-exact) or current markdown files.
"""
