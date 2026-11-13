"""`ocrfix correct`: LLM correction of flagged paragraphs, chunked per document.

1. Select: paragraphs whose score row has ``flag: true`` (scored with a calibration), else
   flagged by ``scorer.calibration`` applied to the score row; ``--all`` takes every one.
2. Mask math / tables / protected runs (``llm.protect``) and chunk each document's selected
   paragraphs in order: at most ``llm.chunk.max_paragraphs`` and ``max_input_tokens`` (user
   message tokens; default derived so input + ceil(1.3 x input) + 16 + system prompt fit
   ``hw.infer.max_model_len``). A paragraph too long for one request is skipped
   (``llm_error: too_long``).
   chunk_id = ``<doc_id>:<sha1 of the chunk's para_ids and masked texts>`` (stable across
   reruns; changes when the input does, so stale responses are never reused).
3. Send through ``Client`` (round-robin over ``hw.infer.urls``), responses appended to
   ``--responses`` (resumable).
4. Parse and write every input paragraph to ``out``: corrected ones get the new ``text``
   (re-tagged spans), ``fixer`` + ``llm``, ``llm_chunk_id``, ``llm_source_text`` (the text
   sent to the LLM); failed ones keep their text and
   get ``llm_error``. ``original_text`` is kept from `fix` output, else set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ocrfix.corpus import tag_spans
from ocrfix.corpus.placeholders import PlaceholderError
from ocrfix.io import iter_jsonl, iter_records, open_text
from ocrfix.llm.client import Client, run_requests
from ocrfix.llm.parse import ParaResult, parse_chunk
from ocrfix.llm.prompt import SYSTEM, Chunk, ChunkPara, request_body, user_message
from ocrfix.llm.protect import mask
from ocrfix.schema import Record, nfc

log = logging.getLogger(__name__)
Key = tuple[str, str]
TokenCounter = Callable[[str], int]


@dataclass
class CorrectStats:
    selected: int = 0
    chunks: int = 0
    sent: int = 0
    corrected: int = 0
    errors: Counter[str] = field(default_factory=Counter)


def hf_counter(tokenizer_path: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def max_input_budget(
    max_model_len: int, system_tokens: int, factor: float, extra: int, overhead: int
) -> int:
    """Largest input n with n + ceil(factor n) + extra + system + overhead <= max_model_len."""
    return max(0, math.floor((max_model_len - system_tokens - extra - overhead - 1) / (1 + factor)))


def selected_keys(scores: Path, *, select_all: bool, calibration: str | None) -> set[Key] | None:
    """Keys to correct; None means every paragraph."""
    if select_all:
        return None
    cal = None
    if calibration and Path(calibration).is_file():
        from ocrfix.calibrate.rules import load_calibration

        cal = load_calibration(calibration)
    keys: set[Key] = set()
    for d in iter_jsonl(scores):
        flag = d.get("flag")
        if flag is None:
            if cal is None:
                raise ValueError(
                    f"{scores}: rows have no `flag` and scorer.calibration is missing; "
                    "score with a calibration, or pass --all"
                )
            flag = cal.flag(d, d.get("lang"))
        if flag:
            keys.add((str(d["doc_id"]), str(d["para_id"])))
    return keys


def build_chunks(
    paras: list[Record],
    *,
    count: TokenCounter,
    max_input_tokens: int,
    max_paragraphs: int,
    protect: list[str],
) -> tuple[list[Chunk], dict[Key, str]]:
    """Chunks for one document's selected paragraphs, plus errors for skipped ones."""
    chunks: list[Chunk] = []
    errors: dict[Key, str] = {}
    cur: list[tuple[ChunkPara, int]] = []

    def close() -> None:
        if not cur:
            return
        doc = cur[0][0].doc_id
        h = hashlib.sha1()
        for p, _ in cur:
            h.update(f"{p.para_id}\x00{p.masked}\x01".encode())
        ch = Chunk(f"{doc}:{h.hexdigest()[:12]}", doc, [p for p, _ in cur])
        ch.input_tokens = count(user_message(ch))
        chunks.append(ch)
        cur.clear()

    for rec in paras:
        try:
            masked, mapping = mask(rec.text, rec.spans, protect)
        except PlaceholderError as e:
            errors[rec.key] = f"unmaskable: {e}"
            continue
        cp = ChunkPara(rec.doc_id, rec.para_id, masked, mapping)
        n = count(f"<<<{max_paragraphs}>>>\n{masked}") + 2
        if n > max_input_tokens:
            errors[rec.key] = "too_long"
            continue
        if cur and (len(cur) >= max_paragraphs or sum(t for _, t in cur) + n > max_input_tokens):
            close()
        cur.append((cp, n))
    close()
    return chunks, errors


def correct(
    paragraphs: Path,
    scores: Path,
    out: Path,
    responses: Path,
    *,
    client: Client,
    llm_cfg: dict[str, Any],
    count: TokenCounter,
    max_model_len: int,
    select_all: bool = False,
    calibration: str | None = None,
    limit: int | None = None,
) -> CorrectStats:
    stats = CorrectStats()
    keys = selected_keys(scores, select_all=select_all, calibration=calibration)
    ck = llm_cfg["chunk"]
    budget = ck.get("max_input_tokens") or max_input_budget(
        max_model_len,
        count(SYSTEM),
        float(llm_cfg["max_tokens_factor"]),
        int(llm_cfg["max_tokens_extra"]),
        int(ck["template_overhead"]),
    )
    by_doc: dict[str, list[Record]] = defaultdict(list)
    for rec in iter_records(paragraphs):
        if keys is None or rec.key in keys:
            by_doc[rec.doc_id].append(rec)
    chunks: list[Chunk] = []
    errors: dict[Key, str] = {}
    for doc_id in by_doc:  # input order
        cs, es = build_chunks(
            sorted(by_doc[doc_id], key=lambda r: r.order),
            count=count, max_input_tokens=budget,
            max_paragraphs=int(ck["max_paragraphs"]), protect=list(llm_cfg["protect"]),
        )  # fmt: skip
        chunks += cs
        errors |= es
    stats.selected = sum(len(v) for v in by_doc.values())
    if limit is not None:
        chunks = chunks[:limit]
    stats.chunks = len(chunks)
    log.info(
        "%d paragraphs selected, %d chunks (<= %d input tokens)",
        stats.selected,
        len(chunks),
        budget,
    )

    model = llm_cfg["adapter_name"] if llm_cfg.get("adapter") else llm_cfg["served_model_name"]
    reqs = [(c.chunk_id, request_body(c, model=model, llm_cfg=llm_cfg)) for c in chunks]
    got, sent = run_requests(client, reqs, responses)
    stats.sent = sent["ok"] + sent["failed"]

    results: dict[Key, ParaResult] = {}
    for c in chunks:
        r = got.get(c.chunk_id) or {}
        for pr in parse_chunk(
            c, r.get("text"), None if r.get("ok") else r.get("error") or "no response"
        ):
            results[(pr.doc_id, pr.para_id)] = pr

    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_text(tmp, "wt") as f:
        for rec in iter_records(paragraphs):
            rec = _apply(rec, results.get(rec.key), errors.get(rec.key), stats)
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(out)
    return stats


def _apply(rec: Record, res: ParaResult | None, skip: str | None, stats: CorrectStats) -> Record:
    extra = dict(rec.extra)
    if res is None and skip is None:
        return rec
    if res is not None and res.ok and res.text is not None:
        text = nfc(res.text)
        extra.setdefault("original_text", rec.text)
        extra["llm_source_text"] = rec.text  # what the LLM saw (verify compares to this)
        extra["fixer"] = [*extra.get("fixer", []), "llm"]
        extra["llm_chunk_id"] = res.chunk_id
        extra.pop("llm_error", None)
        stats.corrected += text != rec.text
        return replace(rec, text=text, spans=tag_spans(text), extra=extra)
    err = skip or (res.error if res else None) or "unknown"
    extra["llm_error"] = err
    if res is not None:
        extra["llm_chunk_id"] = res.chunk_id
    stats.errors[err.split(":")[0]] += 1
    return replace(rec, extra=extra)


def run_correct(
    cfg, paragraphs: Path, scores: Path, out: Path, responses: Path | None = None,
    *, select_all: bool = False, limit: int | None = None,
    count: TokenCounter | None = None, transport=None,
) -> CorrectStats:  # fmt: skip
    from omegaconf import OmegaConf

    llm = OmegaConf.to_container(cfg.llm, resolve=True)
    assert isinstance(llm, dict)
    inf = cfg.hw.infer
    client = Client(
        list(inf.urls),
        concurrency=int(inf.client_concurrency),
        retries=int(llm["retries"]),
        backoff=float(llm["backoff"]),
        timeout=float(llm["timeout"]),
        transport=transport,
    )
    if responses is None:
        responses = out.with_name(out.name.split(".")[0] + ".responses.jsonl")
    return correct(
        paragraphs, scores, out, responses,
        client=client, llm_cfg=llm, count=count or hf_counter(cfg.paths.tokenizer),
        max_model_len=int(inf.max_model_len), select_all=select_all,
        calibration=cfg.scorer.calibration, limit=limit,
    )  # fmt: skip
