"""Chat messages for one chunk: fixed system instruction + numbered paragraphs.

Each paragraph is introduced by a marker line ``<<<n>>>`` (1-based within the chunk); the
model must answer with the same markers, each followed by the corrected paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ocrfix.schema import UNCERTAIN

MARKER = "<<<{}>>>"

SYSTEM = f"""You correct OCR errors in paragraphs from scientific documents (Korean and English).

Rules:
- Output the text as printed in the original document. Fix only OCR damage: misrecognized
  characters, broken or extra spaces, line-break hyphenation, stray symbols.
- Do not rephrase, translate, summarize, complete sentences, or fix the author's own spelling,
  grammar or spacing.
- Placeholders ⟨M1⟩ ⟨T1⟩ ⟨P1⟩ ... stand for math, tables and protected text. Copy every
  placeholder exactly once, unchanged and in the same order.
- Where the text is unreadable and you cannot decide, write {UNCERTAIN}. Never guess.
- Each paragraph starts with a marker line such as {MARKER.format(1)}. Answer with the same
  markers in the same order, each on its own line, followed by the corrected paragraph.
  Output nothing else."""


@dataclass
class ChunkPara:
    doc_id: str
    para_id: str
    masked: str
    mapping: dict[str, str]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    paras: list[ChunkPara] = field(default_factory=list)
    input_tokens: int = 0

    @property
    def para_ids(self) -> list[str]:
        return [p.para_id for p in self.paras]


def user_message(chunk: Chunk) -> str:
    return "\n\n".join(f"{MARKER.format(i)}\n{p.masked}" for i, p in enumerate(chunk.paras, 1))


def messages(chunk: Chunk) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(chunk)}]


def max_tokens(input_tokens: int, factor: float = 1.3, extra: int = 16) -> int:
    import math

    return math.ceil(factor * input_tokens) + extra


def request_body(chunk: Chunk, *, model: str, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible /v1/chat/completions body. Greedy; thinking off via the template."""
    return {
        "model": model,
        "messages": messages(chunk),
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": int(llm_cfg.get("seed", 0)),
        "max_tokens": max_tokens(
            chunk.input_tokens,
            float(llm_cfg["max_tokens_factor"]),
            int(llm_cfg["max_tokens_extra"]),
        ),
        "chat_template_kwargs": {"enable_thinking": bool(llm_cfg["enable_thinking"])},
    }
