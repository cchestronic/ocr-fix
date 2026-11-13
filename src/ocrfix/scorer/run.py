"""``ocrfix score``: paragraphs JSONL -> per-paragraph surprisal features JSONL.

Each paragraph is masked like the LLM input (``corpus.protect``) and scored inside a window
of consecutive paragraphs of its document (``tokens``), so short paragraphs have context and
batches are nearly uniform in length. One output row per paragraph, from its own tokens.

Runs under torchrun with hw.n_gpus processes (the CLI re-execs itself). Each rank owns the
documents whose doc_id hashes to it (windows never cross documents), loads the full model on
its own GPU, and appends results to ``<out>.parts/rank-NNNNN.jsonl`` one shard at a time.
Shards hold whole documents. When all ranks finish, rank 0 appends the parts to ``<out>``
and removes them. Resume = skip keys already in ``<out>`` or any part; a document with some
keys left is re-windowed whole, and only the missing rows are written.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import time
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ocrfix.calibrate.rules import Calibration, load_calibration
from ocrfix.corpus import tag_spans
from ocrfix.io import iter_records, read_ids, repair_jsonl_tail
from ocrfix.schema import Record
from ocrfix.scorer.features import paragraph_features
from ocrfix.scorer.model import choose_mode, load_model, score_sequences
from ocrfix.scorer.tokens import (
    TokenizedPara,
    Tokenizer,
    Window,
    build_window,
    para_bits,
    plan_windows,
)

log = logging.getLogger("ocrfix.score")
Key = tuple[str, str]


def rank_of(doc_id: str, world: int) -> int:
    return zlib.crc32(doc_id.encode()) % world


def _dist_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _docs(records: Iterator[Record]) -> Iterator[list[Record]]:
    """Runs of consecutive records with the same doc_id."""
    cur: list[Record] = []
    for rec in records:
        if cur and rec.doc_id != cur[0].doc_id:
            yield cur
            cur = []
        cur.append(rec)
    if cur:
        yield cur


def _todo(
    paragraphs: Path, done: set, rank: int, world: int, limit: int | None, shard_size: int
) -> Iterator[list[Record]]:
    """Shards of whole documents this rank owns and has rows left in. ``limit`` counts
    paragraphs of the input, across all ranks."""
    shard: list[Record] = []
    for doc in _docs(islice(iter_records(paragraphs), limit)):
        if rank_of(doc[0].doc_id, world) != rank or all(r.key in done for r in doc):
            continue
        shard += doc
        if len(shard) >= shard_size:
            yield shard
            shard = []
    if shard:
        yield shard


def window_settings(cfg: DictConfig) -> tuple[int, int]:
    w = cfg.scorer.window
    target, max_tokens = int(w.target_tokens), int(w.max_tokens)
    if not 0 < target <= max_tokens:
        raise ValueError(f"scorer.window: need 0 < target_tokens <= max_tokens, got {w}")
    if max_tokens >= int(cfg.hw.scorer.max_seq_len):
        raise ValueError(
            f"scorer.window.max_tokens ({max_tokens}) must be < hw.scorer.max_seq_len "
            f"({cfg.hw.scorer.max_seq_len})"
        )
    return target, max_tokens


class Engine:
    """Tokenizer + model + window settings: scores Records in windows on one device."""

    def __init__(self, cfg: DictConfig, device) -> None:
        hw, sc = cfg.hw.scorer, cfg.scorer
        self.cfg = cfg
        self.target, self.max_tokens = window_settings(cfg)
        self.boundary = int(sc.boundary)
        self.min_scored = int(sc.min_scored_tokens)
        self.max_batch_tokens = int(hw.max_batch_tokens)
        self.logits_chunk = int(hw.logits_chunk)
        self.tau, self.taus, self.cal = resolve_tau(cfg)
        self.model, self.mode, self.eos_id = _load_scorer(cfg, device)
        self.tok = Tokenizer(
            cfg.paths.tokenizer, int(hw.tokenize_workers), self.boundary, list(cfg.llm.protect)
        )

    def _bits(self, windows: Sequence[Window]) -> list[np.ndarray]:
        if not windows:
            return []
        return score_sequences(
            self.model, [w.ids for w in windows], mode=self.mode,
            max_tokens=self.max_batch_tokens, logits_chunk=self.logits_chunk, eos_id=self.eos_id,
        )  # fmt: skip

    def plan(self, tps: Sequence[TokenizedPara]) -> list[list[tuple[int, int, int]]]:
        return plan_windows(
            tps, len(self.tok.separator), self.target, self.max_tokens, self.boundary
        )

    def score(self, records: Sequence[Record]) -> tuple[list[TokenizedPara], list[np.ndarray]]:
        """Tokenized paragraphs and their bits, windowed in the given order."""
        tps = self.tok(records)
        windows = [build_window(tps, p, self.tok.separator) for p in self.plan(tps)]
        return tps, para_bits(tps, windows, self._bits(windows))

    def features(self, tp: TokenizedPara, bits: np.ndarray) -> dict[str, Any]:
        return paragraph_features(tp, bits, self.tau, self.taus, self.min_scored)

    def rescore(self, records: Sequence[Record], corrected: dict[Key, str]) -> dict[Key, dict]:
        """Features of the corrected text of each key in ``corrected``, in the window context
        ``score`` gave the source: ``records`` are the source rows in their scoring order.
        A causal model only needs each window up to its last corrected paragraph; earlier
        corrected paragraphs of the window are in their corrected form."""
        src = self.tok(records)
        plans = self.plan(src)
        idx = {tp.key: i for i, tp in enumerate(src)}
        cand_recs = [
            replace(records[idx[k]], text=t, spans=tag_spans(t))
            for k, t in corrected.items()
            if k in idx
        ]
        after = self.tok(cand_recs)
        paras = [*src, *after]
        aft = {tp.key: len(src) + j for j, tp in enumerate(after)}
        split = {tp.key for tp in src if len(tp) > self.max_tokens}
        new_plans: list[list[tuple[int, int, int]]] = []

        def swap(i: int, lo: int, hi: int) -> tuple[int, int, int]:
            j = aft.get(src[i].key)
            return (j, 0, len(paras[j])) if j is not None else (i, lo, hi)

        for plan in plans:
            hits = [k for k, (i, _, _) in enumerate(plan) if src[i].key in aft]
            hits = [k for k in hits if src[plan[k][0]].key not in split]
            if hits:  # up to the last corrected paragraph: later tokens can't affect it
                new_plans.append([swap(*piece) for piece in plan[: hits[-1] + 1]])
        for key in split & set(aft):  # a long paragraph: its own windows, re-split
            j = aft[key]
            sub = plan_windows([paras[j]], len(self.tok.separator), self.target,
                               self.max_tokens, self.boundary)  # fmt: skip
            new_plans += [[(j, lo, hi) for _, lo, hi in p] for p in sub]
        windows = [build_window(paras, p, self.tok.separator) for p in new_plans]
        bits = para_bits(paras, windows, self._bits(windows))
        return {k: self.features(paras[j], bits[j]) for k, j in aft.items()}

    def close(self) -> None:
        self.tok.close()


def _dump(dump_dir: Path, tp: TokenizedPara, bits: np.ndarray) -> None:
    arr = np.zeros(
        len(tp),
        dtype=[("id", "i4"), ("start", "i4"), ("end", "i4"), ("bits", "f4"),
               ("prose", "?"), ("boundary", "?")],
    )  # fmt: skip
    arr["id"], arr["start"], arr["end"], arr["bits"] = tp.ids, tp.starts, tp.ends, bits
    arr["prose"], arr["boundary"] = tp.prose, tp.boundary
    d = dump_dir / tp.key[0]
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tp.key[1]}.npy", arr)


def score(
    cfg: DictConfig,
    paragraphs: Path,
    out: Path,
    *,
    limit: int | None = None,
    dump_tokens: Path | None = None,
) -> int:
    """Returns the number of paragraphs this rank scored."""
    import torch
    import torch.distributed as dist

    rank, world, local_rank = _dist_env()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank {rank}] %(message)s")
    cuda = torch.cuda.is_available()
    if world > 1:
        backend = "nccl" if cuda else "gloo"
        dist.init_process_group(backend, timeout=datetime.timedelta(hours=12))
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    keep = list(cfg.scorer.passthrough)
    parts = out.with_name(out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    part = parts / f"rank-{rank:05d}.jsonl"
    done = read_ids(out) if out.exists() else set()
    for p in parts.glob("rank-*.jsonl"):
        if p == part:
            repair_jsonl_tail(p)
        done |= read_ids(p)

    eng = Engine(cfg, device)
    log.info(
        "mode=%s device=%s world=%d windows %d-%d tokens",
        eng.mode, device, world, eng.target, eng.max_tokens,
    )  # fmt: skip

    n_done = n_tok = 0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shards = _todo(paragraphs, done, rank, world, limit, int(cfg.scorer.shard_size))
    try:
        with open(part, "a", encoding="utf-8") as fh:
            for shard in shards:
                tps, bits = eng.score(shard)
                lines = []
                for rec, tp, b in zip(shard, tps, bits, strict=True):
                    if rec.key in done:
                        continue  # context for its window only
                    if dump_tokens is not None:
                        _dump(dump_tokens, tp, b)
                    lines.append(_result_line(rec, eng.features(tp, b), keep, eng.cal))
                fh.write("".join(lines))
                fh.flush()
                n_done += len(lines)
                n_tok += sum(len(tp) for tp in tps)
                _log_progress(n_done, n_tok, t0, device)
    finally:
        eng.close()

    if world > 1:
        dist.barrier()
    if rank == 0:
        _merge(parts, out)
    if world > 1:
        dist.destroy_process_group()
    return n_done


def resolve_tau(cfg: DictConfig) -> tuple[float, list[float], Calibration | None]:
    """(tau, other tau candidates, calibration): a calibration file's tau wins."""
    sc = cfg.scorer
    tau, cal = float(sc.tau), None
    if sc.calibration and Path(sc.calibration).is_file():
        cal = load_calibration(sc.calibration)
        tau = cal.tau
        log.info("calibration %s: tau=%g, writing `flag`", sc.calibration, tau)
    return tau, [float(t) for t in sc.tau_candidates if float(t) != tau], cal


def _load_scorer(cfg: DictConfig, device):
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(cfg.paths.scorer_model, local_files_only=True)
    mode, attn, why = choose_mode(cfg.hw.attn_impl, model_cfg)
    log.info("scorer mode=%s attn=%s (%s)", mode, attn, why)
    model = load_model(cfg.paths.scorer_model, cfg.hw.dtype, attn, device)
    eos_id = model.config.eos_token_id
    eos_id = eos_id[0] if isinstance(eos_id, list) else eos_id
    return model, mode, eos_id


class InProcessScorer:
    """Score Records in this process on one device; same features and windows as
    ``ocrfix score``.

    ``scorer(records) -> [features]`` scores the records as one input file would be.
    ``scorer.rescore(records, corrected) -> {key: features}`` is what ``verify`` uses: the
    corrected text of each key, in the window context its source had.
    """

    def __init__(self, cfg: DictConfig, device=None) -> None:
        import torch

        self.cfg = cfg
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = Engine(cfg, device)

    def __call__(self, records: list[Record]) -> list[dict]:
        out: list[dict] = []
        shard_size = int(self.cfg.scorer.shard_size)
        for shard in _shards_by_doc(records, shard_size):
            tps, bits = self.engine.score(shard)
            out += [
                json.loads(_result_line(r, self.engine.features(tp, b), [], self.engine.cal))
                for r, tp, b in zip(shard, tps, bits, strict=True)
            ]
        return out

    def rescore(self, records: list[Record], corrected: dict[Key, str]) -> dict[Key, dict]:
        out: dict[Key, dict] = {}
        for shard in _shards_by_doc(records, int(self.cfg.scorer.shard_size)):
            keys = {r.key for r in shard}
            sub = {k: v for k, v in corrected.items() if k in keys}
            if sub:
                out |= self.engine.rescore(shard, sub)
        return out

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> InProcessScorer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _shards_by_doc(records: Sequence[Record], shard_size: int) -> Iterator[list[Record]]:
    shard: list[Record] = []
    for doc in _docs(iter(records)):
        shard += doc
        if len(shard) >= shard_size:
            yield shard
            shard = []
    if shard:
        yield shard


def _result_line(
    rec: Record,
    feats: dict[str, Any],
    passthrough: list[str] | None = None,
    cal: Calibration | None = None,
) -> str:
    d = {"doc_id": rec.doc_id, "para_id": rec.para_id, "lang": rec.lang}
    d |= {k: rec.extra[k] for k in passthrough or () if k in rec.extra}
    d |= feats
    if cal is not None:
        d["flag"] = cal.flag(feats, rec.lang)
    return json.dumps(d, ensure_ascii=False) + "\n"


def _log_progress(n_done: int, n_tok: int, t0: float, device) -> None:
    import torch

    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    log.info("%d paragraphs, %.0f tokens/s, peak VRAM %.2f GiB", n_done, n_tok / dt, peak)


def _merge(parts: Path, out: Path) -> None:
    """Append part files to ``out``, skipping keys already there (safe to rerun after a crash)."""
    if out.exists():
        repair_jsonl_tail(out)
    have = read_ids(out) if out.exists() else set()
    with open(out, "a", encoding="utf-8") as dst:
        for p in sorted(parts.glob("rank-*.jsonl")):
            repair_jsonl_tail(p)
            with open(p, encoding="utf-8") as src:
                for line in src:
                    d = json.loads(line)
                    if (d["doc_id"], d["para_id"]) not in have:
                        dst.write(line)
    shutil.rmtree(parts)
