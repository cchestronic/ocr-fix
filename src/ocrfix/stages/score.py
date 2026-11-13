"""Scoring one chunk: token masks, features, and the two backends that produce the bits.

The chunk's masked text is tokenized here (CPU), so both backends score exactly the same
token ids and the masks line up: a token is prose iff it overlaps no placeholder, and the
first ``score.boundary`` prose tokens after a placeholder, and at the start of the chunk,
are not scored. Every sequence is ``[start token] + ids``, so each token gets a surprisal.

Backends (``score.backend``):
  vllm          POST /v1/completions with the ids as the prompt, ``prompt_logprobs: 0``,
                ``max_tokens: 1`` (vLLM then skips the prefix cache for that request, so every
                token is scored). Many requests in flight, round-robin over ``hw.infer.score_urls``.
  transformers  the local model through ``scorer.model.score_sequences`` (one device).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.scorer.features import paragraph_features
from ocrfix.scorer.tokens import TokenizedPara, mark_boundary, token_masks

LN2 = math.log(2.0)


class Encoder:
    """The model's tokenizer, ids and char offsets, no special tokens."""

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self._fast = self.tok.backend_tokenizer

    def __call__(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        e = self._fast.encode(text, add_special_tokens=False)
        return list(e.ids), list(e.offsets)


def start_token_id(model_path: str) -> int:
    """The token every scored sequence starts with: the model's EOS, as in ``score``."""
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(model_path, local_files_only=True)
    eos = c.get_text_config().eos_token_id
    return int(eos[0] if isinstance(eos, list) else eos)


def tokenize_chunk(text: str, encode: Encoder, boundary: int, key=("", "")) -> TokenizedPara:
    ids, offsets = encode(text)
    starts = np.array([a for a, _ in offsets], dtype=np.int32)
    ends = np.array([b for _, b in offsets], dtype=np.int32)
    phs = [m.span() for m in PLACEHOLDER.finditer(text)]
    prose, bmask = token_masks(starts, ends, phs, boundary)
    tp = TokenizedPara(key, key, 0, text, np.array(ids, dtype=np.int32), starts, ends, prose, bmask)
    mark_boundary(tp, 0, boundary)  # the chunk start: no context yet
    return tp


def features(
    tp: TokenizedPara, bits: np.ndarray, tau: float, taus: Sequence[float], min_scored: int
) -> dict[str, Any]:
    f = paragraph_features(tp, bits, tau, taus, min_scored)
    f["n_tokens"] = len(tp)
    return f


@dataclass
class ScoreError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class VLLMScorer:
    def __init__(self, client, http, model: str, start_id: int, concurrency: int) -> None:
        self.client, self.http, self.model, self.start_id = client, http, model, start_id
        self.sem = asyncio.Semaphore(concurrency)

    async def bits(self, ids: np.ndarray) -> np.ndarray:
        prompt = [self.start_id, *map(int, ids)]
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                "prompt_logprobs": 0}  # fmt: skip
        async with self.sem:
            d, _, err, _ = await self.client.post(self.http, "/v1/completions", body)
        if d is None:
            raise ScoreError(f"score request failed: {err}")
        try:
            plp = d["choices"][0]["prompt_logprobs"]
            return parse_prompt_logprobs(plp, prompt)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ScoreError(f"bad score response: {type(e).__name__}: {e}") from None

    async def bits_batch(self, seqs: Sequence[np.ndarray]) -> list[np.ndarray | ScoreError]:
        return await asyncio.gather(*(self.bits(s) for s in seqs), return_exceptions=True)


def parse_prompt_logprobs(plp: list, prompt: list[int]) -> np.ndarray:
    """vLLM's prompt_logprobs (one entry per prompt token, the first None; each a dict keyed
    by token id, as a string in JSON) -> bits of prompt[1:]."""
    if len(plp) != len(prompt):
        raise ValueError(f"{len(plp)} prompt_logprobs for {len(prompt)} prompt tokens")
    out = np.empty(len(prompt) - 1, dtype=np.float32)
    for i in range(1, len(prompt)):
        entry = plp[i]
        lp = entry.get(str(prompt[i]), entry.get(prompt[i]))
        if lp is None:
            raise ValueError(f"no logprob for prompt token {i}")
        out[i - 1] = -float(lp["logprob"] if isinstance(lp, dict) else lp) / LN2
    return out


class TransformersScorer:
    """The local model on one device; calls are batched one at a time behind a lock."""

    def __init__(self, cfg) -> None:
        import torch

        from ocrfix.scorer.model import load_scorer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.mode, self.start_id = load_scorer(cfg, device)
        hw = cfg.hw.scorer
        self.max_tokens, self.chunk = int(hw.max_batch_tokens), int(hw.logits_chunk)
        self.lock = asyncio.Lock()

    def bits_many(self, seqs: Sequence[np.ndarray]) -> list[np.ndarray]:
        from ocrfix.scorer.model import score_sequences

        return score_sequences(self.model, list(seqs), mode=self.mode, max_tokens=self.max_tokens,
                               logits_chunk=self.chunk, eos_id=self.start_id)  # fmt: skip

    async def bits(self, ids: np.ndarray) -> np.ndarray:
        async with self.lock:
            return (await asyncio.to_thread(self.bits_many, [ids]))[0]

    async def bits_batch(self, seqs: Sequence[np.ndarray]) -> list[np.ndarray | ScoreError]:
        async with self.lock:
            return await asyncio.to_thread(self.bits_many, seqs)
