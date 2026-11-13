"""Generation with transformers in this process, behind vLLM's chat API.

``refine.generator: transformers`` puts an :class:`HFTransport` under the usual
:class:`~ocrfix.llm.client.Client`, so `refine correct` and `refine dpo` send exactly the
requests they send to vLLM (same body, same parsing). The transport:

* renders ``messages`` with the tokenizer's chat template (``add_generation_prompt``,
  ``chat_template_kwargs`` such as ``enable_thinking``), as vLLM does, and tokenizes it
  without special tokens;
* collects concurrent requests for ``refine.hf.wait`` seconds and runs them as one left-padded
  ``generate`` batch of at most ``refine.hf.batch_size`` sequences, grouped by sampling
  settings (greedy, or temperature / top_p / n); ``n`` > 1 is ``num_return_sequences``; out of
  GPU memory, a batch is split in half and retried (answers are the same: batching doesn't
  change greedy outputs);
* stops at the generation config's EOS ids (``<|im_end|>`` included) and at each request's own
  ``max_tokens``: ``finish_reason`` is ``length`` when that limit cut the answer, else
  ``stop``; decodes with special tokens skipped (vLLM's default);
* answers ``/health`` and ``/v1/chat/completions``; anything else is 404.

Greedy answers should match vLLM's up to bf16 kernel differences (``ocrfix refine compare``
measures it). Sampled answers never match token for token: the random streams differ.
The model is ``llm.model`` (+ ``llm.adapter`` through peft, unmerged on ``hw.dtype`` weights;
a QLoRA adapter was trained on nf4 weights, so serve its `train merge` output to match vLLM).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _key(body: dict[str, Any]) -> tuple:
    t = float(body.get("temperature") or 0.0)
    if t <= 0:
        return ("greedy",)
    return ("sample", t, float(body.get("top_p", 1.0)), int(body.get("n", 1)))


class HFGenerator:
    def __init__(self, cfg, model=None, tokenizer_path: str | None = None) -> None:
        """``model``: an already loaded model to reuse (e.g. the in-process scorer's, when it is
        the same checkpoint and no adapter is set)."""
        import torch
        from transformers import AutoTokenizer, GenerationConfig

        llm = cfg.llm
        path = str(llm.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(tokenizer_path or path, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tok = tok
        if model is None:
            from ocrfix.scorer.model import load_model

            model = load_model(path, str(cfg.hw.dtype), str(cfg.hw.attn_impl), self.device)
            if llm.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(llm.adapter)).eval()
        self.model = model
        try:
            gc = GenerationConfig.from_pretrained(path, local_files_only=True)
            eos = gc.eos_token_id
        except OSError:
            eos = model.config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(end, int) and end >= 0 and end != tok.unk_token_id and end not in eos:
            eos.append(end)
        self.eos = [int(e) for e in eos if e is not None]

    def render(self, body: dict[str, Any]) -> list[int]:
        text = self.tok.apply_chat_template(
            body["messages"], tokenize=False, add_generation_prompt=True,
            **(body.get("chat_template_kwargs") or {}),
        )  # fmt: skip
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def generate(self, bodies: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """One batch of requests with the same sampling settings -> vLLM-style responses. Out
        of GPU memory, the batch is split in half and each half tried again (down to one)."""
        import torch

        try:
            return self._generate(bodies)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(bodies) == 1:
                raise
            half = len(bodies) // 2
            log.warning("out of GPU memory with %d requests; splitting the batch", len(bodies))
            return self.generate(bodies[:half]) + self.generate(bodies[half:])

    def _generate(self, bodies: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        import torch

        key = _key(bodies[0])
        prompts = [self.render(b) for b in bodies]
        limits = [int(b.get("max_tokens") or 1024) for b in bodies]
        n = key[3] if key[0] == "sample" else 1
        width = max(map(len, prompts))
        pad = self.tok.pad_token_id
        ids = torch.full((len(prompts), width), pad, dtype=torch.long)
        mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, width - len(p) :] = torch.tensor(p)
            mask[i, width - len(p) :] = 1
        kw: dict[str, Any] = {"max_new_tokens": max(limits), "eos_token_id": self.eos,
                              "pad_token_id": pad}  # fmt: skip
        if key[0] == "greedy":
            kw |= {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        else:
            kw |= {"do_sample": True, "temperature": key[1], "top_p": key[2], "top_k": None,
                   "num_return_sequences": n}  # fmt: skip
            torch.manual_seed(int(bodies[0].get("seed") or 0))
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **kw
            )
        new = out[:, width:].tolist()
        res = []
        for i, b in enumerate(bodies):
            choices, used = [], 0
            for j in range(n):
                toks = new[i * n + j]
                stop = next((k for k, t in enumerate(toks) if t in self.eos), None)
                if stop is not None and stop <= limits[i]:
                    toks, finish = toks[:stop], "stop"
                else:  # cut by this request's own max_tokens
                    toks, finish = toks[: limits[i]], "length"
                used += len(toks)
                choices.append({
                    "index": j, "finish_reason": finish,
                    "message": {"role": "assistant",
                                "content": self.tok.decode(toks, skip_special_tokens=True)},
                })  # fmt: skip
            res.append({"model": b.get("model"), "choices": choices,
                        "usage": {"prompt_tokens": len(prompts[i]),
                                  "completion_tokens": used}})  # fmt: skip
        return res


class HFTransport(httpx.AsyncBaseTransport):
    """vLLM's chat endpoint answered by :class:`HFGenerator`, batching concurrent requests."""

    def __init__(self, gen: HFGenerator, batch_size: int = 16, wait: float = 0.05) -> None:
        self.gen, self.batch_size, self.wait = gen, int(batch_size), float(wait)
        self.pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self.task: asyncio.Task | None = None
        self.tokens = 0
        self.seconds = 0.0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200)
        if not path.endswith("/v1/chat/completions"):
            return httpx.Response(404, text=f"not served: {path}")
        body = json.loads(await request.aread())
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((body, fut))
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self._drain())
        try:
            return httpx.Response(200, json=await fut)
        except Exception as e:
            log.exception("transformers generation failed")
            return httpx.Response(500, text=f"{type(e).__name__}: {e}")

    async def _drain(self) -> None:
        while self.pending:
            await asyncio.sleep(self.wait)  # let concurrent requests arrive
            key = _key(self.pending[0][0])
            n = key[3] if key[0] == "sample" else 1
            size = max(1, self.batch_size // n)
            batch = [p for p in self.pending if _key(p[0]) == key][:size]
            ids = {id(p) for p in batch}
            self.pending = [p for p in self.pending if id(p) not in ids]
            t0 = time.perf_counter()
            try:
                out = await asyncio.to_thread(self.gen.generate, [b for b, _ in batch])
            except Exception as e:
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
                continue
            self.seconds += time.perf_counter() - t0
            for (_, f), r in zip(batch, out, strict=True):
                self.tokens += r["usage"]["completion_tokens"]
                if not f.done():
                    f.set_result(r)


def transport_from_cfg(cfg, model=None) -> HFTransport:
    h = cfg.refine.hf
    return HFTransport(HFGenerator(cfg, model), int(h.batch_size), float(h.wait))
