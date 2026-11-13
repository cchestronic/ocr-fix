"""`ocrfix score` and `ocrfix correct`: chunk shards in -> chunk shards out.

Each input shard gives one output shard of the same name, written via ``.tmp`` + rename, so
an interrupted run resumes at the first missing shard. Rows keep every field from `chunk`.

**score** fills, for the current text (``text``, else ``source_text``):
  score   features (bpc, spike_frac, p90, ..., low_confidence, n_tokens); null on failure
  flag    the calibration's verdict (``score.calibration``), null without one
  status  "scored", or "score_error" with ``error``
With ``score.noise.remove``, random noise found in the scores (`score.noise`) is cut out
first: ``text`` becomes the cleaned text, which is scored again, and a ``history`` entry
``{stage: denoise, removed: [{start, end, bits_per_char, text}], score_before}`` keeps what
was removed. A chunk with no prose left gets status "noise" (never sent to the corrector).

**correct** takes score output. A chunk is sent when it is scored, not low_confidence, and
flagged (``flag``, or the calibration applied now), or with ``--all``. Per chunk, at once:
  1. the correction model rewrites the whole masked chunk (`stages/prompt.py`);
  2. checks: placeholders (every one back, once, in order), markers (no new math/tables
     after restoring), length_ratio, edit_ratio, language, gap_fill (``correct.checks``);
  3. if they pass, re-score the corrected text; ``scorer`` needs bpc to drop by
     ``correct.checks.min_bpc_gain`` and spike_frac not to rise past ``spike_ratio`` x before;
  4. accepted: ``text`` = correction, ``score`` = the new score, status "accepted";
     otherwise the text stays and status is "rejected" ("unchanged" if the model returned
     the input, "error" if the request failed).
Every attempt is appended to ``history``: model, candidate text, checks, score after.
Chunks not sent keep status "scored".

Scoring and correction can use different servers: ``hw.infer.score_urls`` and
``hw.infer.correct_urls`` (both default to this machine's engines).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from ocrfix.calibrate.rules import Calibration, load_calibration
from ocrfix.corpus.placeholders import PLACEHOLDER, PlaceholderError, restore
from ocrfix.corpus.tagger import tag_spans
from ocrfix.io import iter_jsonl, list_shards, open_text
from ocrfix.stages.noise import noise_spans, remove_spans
from ocrfix.stages.prompt import parse_answer, request_body
from ocrfix.stages.score import (
    Encoder,
    ScoreError,
    TransformersScorer,
    VLLMScorer,
    features,
    start_token_id,
    tokenize_chunk,
)
from ocrfix.verify.checks import (
    Check,
    check_edit,
    check_gap_fill,
    check_language,
    check_length,
    check_markers,
    check_scorer,
)


def current(r: dict[str, Any]) -> str:
    return r["text"] if r.get("text") is not None else r["source_text"]


# --- settings and connections ----------------------------------------------------------------


@dataclass
class Context:
    cfg: Any
    transport: Any = None  # httpx transport for tests
    encode: Encoder = field(init=False)
    scorer: Any = field(init=False, default=None)
    cal: Calibration | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        from ocrfix.config import served_name

        c = self.cfg
        sc = c.score
        self.encode = Encoder(str(c.model))
        self.boundary = int(sc.boundary)
        self.min_scored = int(sc.min_scored_tokens)
        self.tau = float(sc.tau)
        if sc.calibration and Path(sc.calibration).is_file():
            self.cal = load_calibration(sc.calibration)
            self.tau = self.cal.tau
        self.taus = [float(t) for t in sc.tau_candidates if float(t) != self.tau]
        # the request settings (stages.prompt.request_body), with the run's seed
        self.llm = OmegaConf.to_container(c.correct, resolve=True) | {"seed": int(c.seed)}
        self.v = self.llm["checks"]
        self.lang_opts = OmegaConf.to_container(c.lang, resolve=True)
        self.noise = OmegaConf.to_container(sc.noise, resolve=True)
        self.backend = str(sc.backend)
        if self.backend not in ("vllm", "transformers"):
            raise ValueError(f"score.backend must be vllm or transformers, got {self.backend!r}")
        self.generator = str(c.correct.backend)
        if self.generator not in ("vllm", "transformers"):
            raise ValueError(
                f"correct.backend must be vllm or transformers, got {self.generator!r}"
            )
        self.correct_model = served_name(c, "correct")
        self.langs = set(c.correct.langs)
        self.score_model = served_name(c, "score")

    def _client(self, urls, transport=None):
        from ocrfix.llm.client import Client

        return Client(list(urls), concurrency=int(self.cfg.hw.infer.concurrency),
                      retries=int(self.llm["retries"]), backoff=float(self.llm["backoff"]),
                      timeout=float(self.llm["timeout"]),
                      transport=transport or self.transport)  # fmt: skip

    async def __aenter__(self) -> Context:
        self._https = []
        if self.backend == "vllm":
            client = self._client(self.cfg.hw.infer.score_urls)
            http = client.http()
            self._https.append(http)
            start = start_token_id(str(self.cfg.model))
            self.scorer = VLLMScorer(client, http, self.score_model, start, client.concurrency)
        else:
            self.scorer = TransformersScorer(self.cfg)
        return self

    async def correct_conn(self):
        urls, transport = self.cfg.hw.infer.correct_urls, None
        if self.generator == "transformers":
            transport = generator_transport(self.cfg, self.scorer)
            urls = ["http://transformers"]
        client = self._client(urls, transport)
        http = client.http()
        self._https.append(http)
        return client, http, asyncio.Semaphore(client.concurrency)

    async def __aexit__(self, *exc) -> None:
        for h in self._https:
            await h.aclose()

    def feats(self, tp, bits) -> dict[str, Any]:
        return features(tp, bits, self.tau, self.taus, self.min_scored)

    def flag(self, feats: dict[str, Any] | None, lang: str | None) -> bool | None:
        if self.cal is None or feats is None:
            return None
        return self.cal.flag(feats, lang)


def generator_transport(cfg, scorer=None):
    """An in-process transformers transport for correct / dpo requests (``correct.backend:
    transformers``); it reuses the in-process scorer's model when that is the base and no
    adapter is set."""
    from ocrfix.config import correct_model
    from ocrfix.llm.hf import transport_from_cfg

    share = None
    if (isinstance(scorer, TransformersScorer) and not cfg.correct.adapter
            and correct_model(cfg) == str(cfg.model)):  # fmt: skip
        share = scorer.model
    return transport_from_cfg(cfg, share)


# --- shards ----------------------------------------------------------------------------------


def _plan(chunks: Path, out_dir: Path, limit: int | None) -> list[tuple[Path, Path]]:
    shards = list_shards(chunks)
    if not shards:
        raise ValueError(f"no chunk shards in {chunks}")
    if out_dir.resolve() in {s.parent.resolve() for s in shards}:
        raise ValueError("the output folder must differ from the input folder")
    return [(s, out_dir / s.name) for s in shards[:limit]]


def _write(out: Path, rows: list[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open_text(tmp, "wt") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(out)


@dataclass
class RunStats:
    shards: int = 0
    shards_skipped: int = 0
    chunks: int = 0
    seconds: float = 0.0
    status: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["status"], d["failed"] = dict(self.status), dict(self.failed)
        d["tokens_per_s"] = round(self.tokens / self.seconds) if self.seconds else None
        return d


async def _for_shards(plan, work, st: RunStats) -> None:
    for src, out in plan:
        st.shards += 1
        if out.is_file():
            st.shards_skipped += 1
            continue
        rows = list(iter_jsonl(src))
        await work(rows)
        _write(out, rows)
        st.chunks += len(rows)
        st.status.update(r.get("status", "new") for r in rows)


# --- score -----------------------------------------------------------------------------------


async def _bits_of(ctx: Context, texts: list[str]) -> list[tuple[Any, Any]]:
    """(tokenized, bits or exception) per text; empty texts get empty bits, no request."""
    tps = [tokenize_chunk(t, ctx.encode, ctx.boundary) for t in texts]
    todo = [i for i, tp in enumerate(tps) if len(tp)]
    results = await ctx.scorer.bits_batch([tps[i].ids for i in todo])
    bits: list[Any] = [np.zeros(0, dtype=np.float32)] * len(tps)
    for i, b in zip(todo, results, strict=True):
        bits[i] = b
    return list(zip(tps, bits, strict=True))


def _has_prose(text: str) -> bool:
    return any(ch.isalnum() for ch in PLACEHOLDER.sub("", text))


async def score_rows(ctx: Context, rows: list[dict[str, Any]], st: RunStats) -> None:
    scored = await _bits_of(ctx, [current(r) for r in rows])
    cleaned: dict[int, str] = {}
    for i, (r, (tp, b)) in enumerate(zip(rows, scored, strict=True)):
        st.tokens += len(tp)
        if isinstance(b, BaseException):
            r.update(score=None, flag=None, status="score_error", error=str(b))
            continue
        f = ctx.feats(tp, b)
        r.update(score=f, flag=ctx.flag(f, r.get("lang")), status="scored")
        r.pop("error", None)
        if ctx.noise["remove"]:
            spans = noise_spans(tp, b, ctx.noise["core"], ctx.noise["extend"],
                                min_chars=int(ctx.noise["min_chars"]))  # fmt: skip
            if spans:
                text = current(r)
                cleaned[i] = remove_spans(text, spans)
                r.setdefault("history", []).append({
                    "stage": "denoise", "time": time.time(), "score_before": f,
                    "removed": [{"start": a, "end": e, "bits_per_char": bpc, "text": text[a:e]}
                                for a, e, bpc in spans],
                })  # fmt: skip
                st.failed["noise_spans"] += len(spans)
    if cleaned:
        await _rescore_cleaned(ctx, rows, cleaned, st)
    if ctx.noise["whole_chunks"] and ctx.cal is not None:
        for r in rows:
            _drop_noise_chunk(ctx, r, st)


def _drop_noise_chunk(ctx: Context, r: dict[str, Any], st: RunStats) -> None:
    """A chunk whose bpc (after word-level noise removal) is above the calibration's
    noise-chunk threshold for its language is noise as a whole: its prose is dropped (math,
    tables and other placeholders stay), status ``noise``; the original stays in source_text."""
    f = r.get("score")
    if r.get("status") != "scored" or not f or not ctx.cal.is_noise_chunk(f, r.get("lang")):
        return
    r.setdefault("history", []).append({
        "stage": "noise_chunk", "time": time.time(), "bpc": f.get("bpc"),
        "threshold": ctx.cal.noise_chunk_threshold(r.get("lang")),
    })  # fmt: skip
    r.update(text=" ".join(PLACEHOLDER.findall(current(r))), status="noise", flag=None)
    st.failed["noise_chunks"] += 1


async def _rescore_cleaned(
    ctx: Context, rows: list[dict[str, Any]], cleaned: dict[int, str], st: RunStats
) -> None:
    """Re-score what noise removal left, so the flag and the corrector see the cleaned text."""
    rescored = await _bits_of(ctx, list(cleaned.values()))
    for (i, text), (tp, b) in zip(cleaned.items(), rescored, strict=True):
        r = rows[i]
        st.tokens += len(tp)
        r["text"] = text
        if isinstance(b, BaseException):
            r.update(score=None, flag=None, status="score_error", error=str(b))
            continue
        f = ctx.feats(tp, b)
        r.update(score=f, flag=ctx.flag(f, r.get("lang")))
        r["status"] = "scored" if _has_prose(text) else "noise"


def run_score(cfg, chunks: Path, out_dir: Path, *, limit: int | None = None,
              transport=None) -> RunStats:  # fmt: skip
    plan = _plan(chunks, out_dir, limit)
    st = RunStats()

    async def main() -> None:
        async with Context(cfg, transport) as ctx:
            await _for_shards(plan, lambda rows: score_rows(ctx, rows, st), st)

    t0 = time.perf_counter()
    asyncio.run(main())
    st.seconds = round(time.perf_counter() - t0, 1)
    _save_stats(out_dir, "score", st)
    return st


# --- correct -> re-score -> verify -------------------------------------------------------------


def text_checks(
    src: str, out: str, mapping: dict[str, str], v: dict, lang_opts: dict
) -> list[Check]:
    """Checks on the masked source and correction; placeholders first (the rest need them)."""
    try:
        raw_src = restore(src, mapping)
        raw_out = restore(out, mapping)
    except PlaceholderError as e:
        return [Check("placeholders", False, str(e))]
    src_spans = tag_spans(raw_src)
    lo, hi = v["length_ratio"]
    return [
        Check("placeholders", True, f"{len(mapping)} intact"),
        check_markers(raw_src, src_spans, raw_out),
        check_length(src, out, float(lo), float(hi)),
        check_edit(src, out, float(v["max_edit_ratio"])),
        check_language(raw_src, src_spans, raw_out, float(v["max_script_shift"]), lang_opts),
        check_gap_fill(src, out, int(v["gap_fill_max_syllables"])),
    ]


def selected(ctx: Context, r: dict[str, Any], select_all: bool) -> bool:
    """Scored, enough prose, in a language the model corrects (``correct.langs``; Chinese /
    Japanese ``cjk`` and ``other`` chunks pass through unchanged), and flagged (or --all)."""
    s = r.get("score")
    if r.get("status") != "scored" or not s or s.get("low_confidence"):
        return False
    if (r.get("lang") or "other") not in ctx.langs:
        return False
    if select_all:
        return True
    flag = r.get("flag")
    if flag is None:
        flag = ctx.flag(s, r.get("lang"))
    if flag is None:
        raise ValueError(
            "chunks have no `flag` and score.calibration is missing: calibrate first, or pass --all"
        )
    return bool(flag)


async def correct_one(ctx: Context, conn, r: dict[str, Any], st: RunStats) -> None:
    client, http, sem = conn
    src = current(r)
    body = request_body(src, int(r["score"]["n_tokens"]), model=ctx.correct_model, llm=ctx.llm)
    async with sem:
        res = await client.complete(http, r["chunk_id"], body)
    entry: dict[str, Any] = {"stage": "correct", "model": ctx.correct_model, "time": time.time()}
    r.setdefault("history", []).append(entry)
    if not res.ok:
        r["status"], entry["error"] = "error", res.error
        st.failed["request"] += 1
        return
    out = parse_answer(res.text or "")
    entry["candidate"] = out
    if out == src:
        r["status"] = "unchanged"
        return
    checks = text_checks(src, out, r["placeholders"], ctx.v, ctx.lang_opts)
    after = None
    if all(c.passed for c in checks) or ctx.v["rescore_failed"]:
        try:
            tp = tokenize_chunk(out, ctx.encode, ctx.boundary)
            after = ctx.feats(tp, await ctx.scorer.bits(tp.ids))
            st.tokens += len(tp)
            checks.append(check_scorer(r["score"], after, min_bpc_gain=float(ctx.v["min_bpc_gain"]),
                                       spike_ratio=float(ctx.v["spike_ratio"])))  # fmt: skip
        except ScoreError as e:
            checks.append(Check("scorer", False, f"re-score failed: {e}"))
    else:
        checks.append(Check("scorer", False, "skipped: a text check already failed"))
    failed = [c for c in checks if not c.passed]
    entry.update(accepted=not failed, checks=[c.to_dict() for c in checks], score_after=after,
                 score_before=r["score"])  # fmt: skip
    for c in failed:
        st.failed[c.name] += 1
    if failed:
        r["status"] = "rejected"
        return
    r.update(text=out, score=after, flag=ctx.flag(after, r.get("lang")), status="accepted")


def run_correct(cfg, chunks: Path, out_dir: Path, *, select_all: bool = False,
                limit: int | None = None, transport=None) -> RunStats:  # fmt: skip
    plan = _plan(chunks, out_dir, limit)
    st = RunStats()

    async def main() -> None:
        async with Context(cfg, transport) as ctx:
            conn = await ctx.correct_conn()

            async def work(rows: list[dict[str, Any]]) -> None:
                todo = [r for r in rows if selected(ctx, r, select_all)]
                st.status["skipped_lang"] += sum(
                    r.get("status") == "scored" and (r.get("lang") or "other") not in ctx.langs
                    for r in rows
                )
                await asyncio.gather(*(correct_one(ctx, conn, r, st) for r in todo))

            await _for_shards(plan, work, st)

    t0 = time.perf_counter()
    asyncio.run(main())
    st.seconds = round(time.perf_counter() - t0, 1)
    _save_stats(out_dir, "correct", st)
    return st


def _save_stats(out_dir: Path, name: str, st: RunStats) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_stats.json").write_text(json.dumps(st.to_dict(), indent=1) + "\n")


# --- summary ---------------------------------------------------------------------------------


def summarize(path: Path) -> dict[str, Any]:
    status: Counter = Counter()
    by_lang: dict[str, Counter] = {}
    failed: Counter = Counter()
    gain = []
    for r in iter_jsonl(path):
        s = r.get("status", "new")
        status[s] += 1
        by_lang.setdefault(r.get("lang") or "other", Counter())[s] += 1
        h = (r.get("history") or [None])[-1]
        if h and h.get("stage") == "correct":
            failed.update(c["check"] for c in h.get("checks", []) if not c["passed"])
            b, a = (h.get("score_before") or {}).get("bpc"), (h.get("score_after") or {}).get("bpc")
            if s == "accepted" and a is not None and b is not None:
                gain.append(b - a)
    return {
        "status": dict(status),
        "by_lang": {k: dict(v) for k, v in by_lang.items()},
        "failed_checks": dict(failed.most_common()),
        "mean_bpc_gain_accepted": round(sum(gain) / len(gain), 4) if gain else None,
    }


def format_summary(s: dict[str, Any]) -> str:
    st = s["status"]
    sent = sum(st.get(k, 0) for k in ("accepted", "rejected", "unchanged", "error"))
    lines = [f"{sum(st.values())} chunks: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items()))]
    if sent:
        lines.append(f"sent to the corrector: {sent}, accepted {st.get('accepted', 0)} "
                     f"({100 * st.get('accepted', 0) / sent:.1f}%)")  # fmt: skip
    if s["mean_bpc_gain_accepted"] is not None:
        lines.append(f"mean bpc gain of accepted corrections: {s['mean_bpc_gain_accepted']}")
    for k, v in s["failed_checks"].items():
        lines.append(f"  failed {k:<13} {v}")
    for lang, c in sorted(s["by_lang"].items()):
        lines.append(f"  {lang:<6} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    return "\n".join(lines)
