"""The correction request for one whole chunk: a fixed system instruction, the masked chunk
as the user message, and the corrected chunk as the answer (no markers)."""

from __future__ import annotations

import math
import re
from typing import Any

SYSTEM = """You correct OCR errors in a passage from a scientific document (Korean and English).

Rules:
- Output the passage as printed in the original document. Fix only OCR damage: misrecognized
  characters, broken or extra spaces, line breaks inside sentences, line-break hyphenation,
  stray page numbers and symbols.
- Do not rephrase, translate, summarize, complete sentences, or fix the author's own spelling,
  grammar or spacing.
- Keep the paragraph structure: blank lines between paragraphs stay where they are.
- Placeholders ⟨M1⟩ ⟨T1⟩ ⟨P1⟩ ... stand for math, tables and protected text. Copy every
  placeholder exactly once, unchanged and in the same order.
- Output only the corrected passage, nothing else."""

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def messages(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def request_body(text: str, n_tokens: int, *, model: str, llm: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages(text),
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": int(llm.get("seed", 0)),
        "max_tokens": math.ceil(float(llm["max_tokens_factor"]) * n_tokens)
        + int(llm["max_tokens_extra"]),
        "chat_template_kwargs": {"enable_thinking": bool(llm["enable_thinking"])},
    }


def parse_answer(output: str) -> str:
    """Drop a <think> block and the surrounding whitespace."""
    return _THINK.sub("", output).strip()
