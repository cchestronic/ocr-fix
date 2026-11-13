"""Per-token masks for scoring a masked chunk.

The text is scored in the same masked form the LLM sees (math ⟨M#⟩, tables ⟨T#⟩, protected
prose ⟨P#⟩). A token is *prose* iff it overlaps no placeholder; the first ``boundary`` prose
tokens after a placeholder (and at the chunk start) are not scored: the model is still
adjusting to the unusual symbol. Scored tokens = prose and not boundary. Token arrays live
only in memory, never in JSONL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenizedPara:
    key: tuple[str, str]
    group: tuple
    order: int
    text: str  # masked text; offsets below index into it
    ids: np.ndarray  # int32 [n], no special tokens
    starts: np.ndarray  # int32 [n]
    ends: np.ndarray  # int32 [n]
    prose: np.ndarray  # bool [n] overlaps no placeholder
    boundary: np.ndarray  # bool [n] right after a placeholder or a split
    mask_error: bool = False

    @property
    def scored(self) -> np.ndarray:
        return self.prose & ~self.boundary

    def __len__(self) -> int:
        return len(self.ids)


def token_masks(
    starts: np.ndarray, ends: np.ndarray, placeholders: list[tuple[int, int]], boundary: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """prose (overlaps no placeholder, non-empty) and boundary masks."""
    n = len(starts)
    hits = np.zeros(n, dtype=bool)
    for a, b in placeholders:
        hits |= (starts < b) & (ends > a)
    prose = ~hits & (ends > starts)
    bmask = np.zeros(n, dtype=bool)
    left = 0
    for i in range(n):
        if hits[i]:
            left = boundary
        elif prose[i] and left > 0:
            bmask[i] = True
            left -= 1
    return prose, bmask


def mark_boundary(tp: TokenizedPara, start: int, boundary: int) -> None:
    """Don't score the first ``boundary`` prose tokens from ``start`` (a split point)."""
    left = boundary
    for i in range(start, len(tp)):
        if left <= 0:
            break
        if tp.prose[i]:
            tp.boundary[i] = True
            left -= 1
