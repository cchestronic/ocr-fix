"""transformers generation behind vLLM's chat API (`refine.generator: transformers`) and
`refine compare`, on the tiny random Qwen3.5 (hybrid: linear + full attention)."""

from __future__ import annotations

import asyncio
import json
import math
import shutil

import httpx
import numpy as np
import pytest
from tests.test_refine import rows, setup  # noqa: F401  (fixture)

from ocrfix.config import load_config
from ocrfix.llm.client import Client
from ocrfix.llm.hf import HFGenerator, HFTransport
from ocrfix.refine.compare import run_compare
from ocrfix.refine.run import run_correct, run_score
from ocrfix.refine.score import TransformersScorer

LN2 = math.log(2)
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def chat_dir(tiny_model_dir, tmp_path_factory):
    from transformers import AutoTokenizer

    d = tmp_path_factory.mktemp("chat_model")
    shutil.copytree(tiny_model_dir, d, dirs_exist_ok=True)
    tok = AutoTokenizer.from_pretrained(d)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def gen(chat_dir, model_device):
    cfg = load_config("rtx5090", [f"paths.base_model={chat_dir}", "hw.dtype=float32"])
    return HFGenerator(cfg)


def body(text: str, **kw) -> dict:
    return {"model": "m", "messages": [{"role": "system", "content": "Fix OCR."},
                                       {"role": "user", "content": text}],
            "temperature": 0.0, "max_tokens": 12,
            "chat_template_kwargs": {"enable_thinking": False}} | kw  # fmt: skip


def test_greedy_matches_stepwise_argmax(gen):
    """generate() with the hybrid cache gives the tokens a full recompute picks."""
    import torch

    b = body("The model is trained on clean documents.", max_tokens=10)
    ids = gen.render(b)
    assert gen.tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    seq = list(ids)
    ref = []
    with torch.inference_mode():
        for _ in range(10):
            logits = gen.model(torch.tensor([seq], device=gen.device)).logits[0, -1]
            t = int(logits.argmax())
            if t in gen.eos:
                break
            ref.append(t)
            seq.append(t)
    out = gen.generate([b])[0]["choices"][0]
    assert out["message"]["content"] == gen.tok.decode(ref, skip_special_tokens=True)


def test_batched_equals_single(gen):
    texts = ["Short one.", "A much longer input text with several words in it, " * 3, "Mid size."]
    single = [gen.generate([body(t)])[0]["choices"][0]["message"]["content"] for t in texts]
    batch = gen.generate([body(t) for t in texts])
    assert [r["choices"][0]["message"]["content"] for r in batch] == single


def test_max_tokens_and_samples(gen):
    r = gen.generate([body("Some text here.", max_tokens=3)])[0]
    assert r["usage"]["completion_tokens"] <= 3
    if r["usage"]["completion_tokens"] == 3:
        assert r["choices"][0]["finish_reason"] == "length"
    s = gen.generate([body("Some text.", temperature=1.0, top_p=0.9, n=3, seed=1)])[0]
    assert len(s["choices"]) == 3


def test_transport_batches_concurrent_requests(gen):
    calls = []
    real = gen.generate

    def counting(bodies):
        calls.append(len(bodies))
        return real(bodies)

    gen.generate = counting
    try:
        tr = HFTransport(gen, batch_size=8, wait=0.05)
        client = Client(["http://transformers"], concurrency=8, retries=0, transport=tr)

        async def main():
            async with client.http() as http:
                return await asyncio.gather(
                    *(client.complete(http, f"c{i}", body(f"Text number {i}.")) for i in range(6))
                )

        res = asyncio.run(main())
    finally:
        gen.generate = real
    assert all(r.ok or r.finish_reason == "length" for r in res)
    assert sum(calls) == 6 and len(calls) < 6


def test_correct_through_transformers(setup, chat_dir, model_device):  # noqa: F811
    tmp, cfg, fake = setup
    run_score(
        cfg("refine.noise.remove=false"),
        tmp / "chunks",
        tmp / "scored",
        transport=httpx.MockTransport(fake),
    )
    c = cfg("refine.generator=transformers", f"llm.model={chat_dir}", "llm.max_tokens_extra=4",
            "llm.max_tokens_factor=0.1")  # fmt: skip
    run_correct(
        c, tmp / "scored", tmp / "out", select_all=True, transport=httpx.MockTransport(fake)
    )
    out = rows(tmp / "out")
    sent = [r for r in out if r["history"] and r["history"][-1].get("stage") == "correct"]
    assert sent
    for r in sent:
        h = r["history"][-1]
        # answered by the local model (truncated answers are request errors, as with vLLM)
        assert "candidate" in h or "truncated" in h.get("error", "")
    # no chat request reached the fake server
    assert all(p != "/v1/chat/completions" for _, p in fake.calls)


class SameModelVLLM:
    """A 'vLLM' that is the same tiny model: prompt_logprobs from the transformers scorer,
    chat from the transformers generator. compare must find no difference."""

    def __init__(self, cfg) -> None:
        self.scorer = TransformersScorer(cfg)
        self.gen = HFGenerator(cfg, self.scorer.model)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        b = json.loads(request.content)
        if request.url.path == "/v1/completions":
            bits = self.scorer.bits_many([np.asarray(b["prompt"][1:])])[0]
            plp = [None] + [{str(t): {"logprob": -float(x) * LN2}}
                            for t, x in zip(b["prompt"][1:], bits, strict=True)]  # fmt: skip
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": plp}]})
        return httpx.Response(200, json=self.gen.generate([b])[0])


def test_compare_finds_no_difference_with_the_same_model(setup, chat_dir, model_device):  # noqa: F811
    tmp, cfg, fake = setup
    run_score(
        cfg("refine.noise.remove=false"),
        tmp / "chunks",
        tmp / "scored",
        transport=httpx.MockTransport(fake),
    )
    c = cfg(f"paths.base_model={chat_dir}", "llm.max_tokens_extra=4", "llm.max_tokens_factor=0.1")
    rep = run_compare(c, tmp / "scored", tmp / "cmp.json", limit=10,
                      transport=httpx.MockTransport(SameModelVLLM(c)))  # fmt: skip
    s = rep["score"]
    assert s["bits_abs_diff"]["max"] < 1e-3
    assert s["flag_agree"] == 1.0 and s["noise_agree"] in (None, 1.0)
    assert rep["generate"]["exact"] == 1.0 and rep["pass"] == {"flags": True, "greedy": True}
    assert json.loads((tmp / "cmp.json").read_text())["chunks"] == rep["chunks"]
