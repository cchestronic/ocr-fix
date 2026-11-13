"""`ocrfix compare <chunks> <out.json>`: vLLM vs transformers on the same chunks.

Run it before trusting one backend's results for the other: calibration thresholds fit on one
scorer are applied to the other's scores, and eval / DPO answers from transformers stand in for
what vLLM does on the corpus. Both sides use the same checkpoints: vLLM serves the base on
``hw.infer.score_urls`` and the corrector on ``hw.infer.correct_urls``; transformers loads
``model`` and ``correct.model`` (+ ``correct.adapter``). ``/v1/models``
of each server is recorded, so a mismatch shows.

Scoring (every chunk; the current masked text, as `score` sees it):
  bits       |vLLM - transformers| per scored token: mean, p99, max
  features   per chunk: |difference| of bpc, p90, spike_frac (mean, max)
  flags      with a calibration: share of chunks both flag the same; the disagreements
  noise      with score.noise.remove: share of chunks with the same noise spans
Generation (greedy, the `correct` request; ``--generate``):
  exact      share of chunks with identical answers
  distance   char edit distance between the two answers: mean, and per 1000 answer chars
  speed      completion tokens per second of each backend
  examples   the first differing answers, around the first difference
``pass``: flags agree on >= 99% of chunks and >= 95% of greedy answers are identical.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from rapidfuzz.distance import Levenshtein

from ocrfix.io import iter_jsonl
from ocrfix.stages.noise import noise_spans
from ocrfix.stages.prompt import parse_answer, request_body
from ocrfix.stages.run import Context, _has_prose, current, generator_transport
from ocrfix.stages.score import TransformersScorer, tokenize_chunk

FLAG_AGREE = 0.99
GREEDY_EXACT = 0.95


def _stats(x: list[float]) -> dict[str, float | None]:
    if not x:
        return {"mean": None, "p99": None, "max": None}
    a = np.asarray(x, dtype=np.float64)
    return {"mean": round(float(a.mean()), 6), "p99": round(float(np.quantile(a, 0.99)), 6),
            "max": round(float(a.max()), 6)}  # fmt: skip


async def _models(url: str) -> Any:
    try:
        async with httpx.AsyncClient(timeout=10) as h:
            r = await h.get(url.rstrip("/") + "/v1/models")
            return [{"id": m.get("id"), "root": m.get("root")} for m in r.json()["data"]]
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _first_diff(a: str, b: str) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def run_compare(
    cfg, chunks: Path, out: Path, *, limit: int = 200, score: bool = True,
    generate: bool = True, examples: int = 5, transport=None,
) -> dict[str, Any]:  # fmt: skip
    """``transport``: an httpx transport standing in for the vLLM servers (tests)."""
    rows = []
    for r in iter_jsonl(chunks):
        if r.get("status") in ("noise", "score_error") or not _has_prose(current(r)):
            continue
        rows.append(r)
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"{chunks}: no chunk with prose")

    async def main() -> dict[str, Any]:
        rep: dict[str, Any] = {"chunks": len(rows)}
        async with Context(cfg, transport) as ctx:  # score.backend must be vllm: that side
            if ctx.backend != "vllm":
                raise ValueError(
                    "run with score.backend=vllm (the transformers side is loaded here)"
                )
            if transport is None:
                rep["servers"] = {
                    "score": await _models(str(cfg.hw.infer.score_urls[0])),
                    "correct": await _models(str(cfg.hw.infer.correct_urls[0])),
                }
            tps = [tokenize_chunk(current(r), ctx.encode, ctx.boundary) for r in rows]
            hf = None
            if score:
                rep["score"], hf = await _score(cfg, ctx, rows, tps)
            if generate:
                rep["generate"] = await _generate(cfg, ctx, rows, tps, hf, examples)
        sc, g = rep.get("score"), rep.get("generate")
        rep["pass"] = {
            "flags": None
            if sc is None or sc["flag_agree"] is None
            else sc["flag_agree"] >= FLAG_AGREE,
            "greedy": None if g is None else g["exact"] >= GREEDY_EXACT,
        }
        return rep

    rep = asyncio.run(main())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return rep


async def _score(cfg, ctx, rows, tps) -> tuple[dict[str, Any], TransformersScorer]:
    t0 = time.perf_counter()
    vb = await ctx.scorer.bits_batch([tp.ids for tp in tps])
    tv = time.perf_counter() - t0
    bad = [i for i, b in enumerate(vb) if isinstance(b, Exception)]
    if bad:
        raise ValueError(f"vLLM scoring failed for {len(bad)} chunks: {vb[bad[0]]}")
    hf = TransformersScorer(cfg)
    t0 = time.perf_counter()
    hb = await asyncio.to_thread(hf.bits_many, [tp.ids for tp in tps])
    th = time.perf_counter() - t0
    n_tok = sum(len(tp) for tp in tps)
    speed = {
        "vllm": round(n_tok / tv),
        "transformers": round(n_tok / th),
    }

    tok_diff: list[float] = []
    fd: dict[str, list[float]] = {"bpc": [], "p90": [], "spike_frac": []}
    flags = noise_same = 0
    flag_diff = []
    nz = ctx.noise
    for r, tp, a, b in zip(rows, tps, vb, hb, strict=True):
        s = tp.scored
        tok_diff += np.abs(a[s] - b[s]).tolist()
        fa, fb = ctx.feats(tp, a), ctx.feats(tp, b)
        for k in fd:
            if fa.get(k) is not None and fb.get(k) is not None:
                fd[k].append(abs(float(fa[k]) - float(fb[k])))
        if ctx.cal is not None:
            ga, gb = ctx.flag(fa, r.get("lang")), ctx.flag(fb, r.get("lang"))
            flags += ga == gb
            if ga != gb:
                flag_diff.append({"chunk_id": r["chunk_id"], "vllm": ga, "transformers": gb,
                                  "bpc": [fa.get("bpc"), fb.get("bpc")]})  # fmt: skip
        if nz.get("remove"):
            sa = noise_spans(tp, a, nz["core"], nz["extend"], min_chars=int(nz["min_chars"]))
            sb = noise_spans(tp, b, nz["core"], nz["extend"], min_chars=int(nz["min_chars"]))
            noise_same += [x[:2] for x in sa] == [x[:2] for x in sb]
    n = len(rows)
    rep = {
        "bits_abs_diff": _stats(tok_diff),
        "feature_abs_diff": {k: _stats(v) for k, v in fd.items()},
        "flag_agree": round(flags / n, 4) if ctx.cal is not None else None,
        "flag_disagreements": flag_diff[:20],
        "noise_agree": round(noise_same / n, 4) if nz.get("remove") else None,
    }
    rep["tokens_per_s"] = speed
    return rep, hf


async def _generate(cfg, ctx, rows, tps, hf, n_examples: int) -> dict[str, Any]:
    bodies = [
        request_body(current(r), len(tp), model=ctx.correct_model, llm=ctx.llm)
        for r, tp in zip(rows, tps, strict=True)
    ]
    answers: dict[str, list] = {}
    speed, truncated = {}, {}
    for side in ("vllm", "transformers"):
        if side == "vllm":
            client = ctx._client(cfg.hw.infer.correct_urls)
        else:
            client = ctx._client(["http://transformers"], generator_transport(cfg, hf))
        sem = asyncio.Semaphore(client.concurrency)
        async with client.http() as http:

            async def one(i, body, client=client, http=http, sem=sem):
                async with sem:
                    return await client.complete(http, rows[i]["chunk_id"], body)

            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(i, b) for i, b in enumerate(bodies)))
            dt = time.perf_counter() - t0
        toks = sum(int((x.usage or {}).get("completion_tokens") or 0) for x in res)
        speed[side] = round(toks / dt, 1) if dt else None
        # a truncated answer is still the backend's answer: compare it too
        answers[side] = [parse_answer(x.text) if x.text is not None else None for x in res]
        truncated[side] = sum(x.finish_reason == "length" for x in res)
    same = dist = chars = failed = 0
    ex = []
    for r, a, b in zip(rows, answers["vllm"], answers["transformers"], strict=True):
        if a is None or b is None:
            failed += 1
            continue
        if a == b:
            same += 1
            continue
        d = Levenshtein.distance(a, b)
        dist += d
        chars += max(len(a), 1)
        if len(ex) < n_examples:
            i = _first_diff(a, b)
            ex.append({"chunk_id": r["chunk_id"], "distance": d, "vllm": a[max(i - 40, 0) : i + 60],
                       "transformers": b[max(i - 40, 0) : i + 60]})  # fmt: skip
    n = len(rows) - failed
    return {
        "chunks": n, "failed": failed, "exact": round(same / n, 4) if n else 0.0,
        "distance_mean": round(dist / n, 3) if n else None,
        "distance_per_1000_chars_of_differing": round(1000 * dist / chars, 3) if chars else 0.0,
        "completion_tokens_per_s": speed, "truncated": truncated, "examples": ex,
    }  # fmt: skip


def format_compare(rep: dict[str, Any]) -> str:
    lines = [f"{rep['chunks']} chunks"]
    s = rep.get("score")
    if s:
        lines += [
            f"scoring tokens/s: {s['tokens_per_s']}",
            f"bits per token |diff|: {s['bits_abs_diff']}",
            f"bpc |diff|: {s['feature_abs_diff']['bpc']}",
            f"flags agree: {s['flag_agree']}  noise spans agree: {s['noise_agree']}",
        ]
    g = rep.get("generate")
    if g:
        lines += [
            f"greedy answers identical: {g['exact']} ({g['chunks']} chunks, {g['failed']} failed); "
            f"mean char distance {g['distance_mean']}",
            f"completion tokens/s: {g['completion_tokens_per_s']}",
        ]
        for e in g["examples"]:
            lines.append(f"  {e['chunk_id']} (distance {e['distance']}):")
            lines.append(f"    vllm:         {e['vllm']!r}")
            lines.append(f"    transformers: {e['transformers']!r}")
    lines.append(f"pass: {rep['pass']}")
    return "\n".join(lines)
