"""`ocrfix data sft`: chunk SFT examples in exactly the form `correct` sends.

Each example is ``messages`` = [system ``stages.prompt.SYSTEM``, user = the masked chunk,
assistant = the corrected masked chunk], the format `train sft` renders. The chat template
adds the empty ``<think>`` block of non-thinking mode itself (``sft.empty_think`` puts one in
the content as well, for templates that don't). Sources:

  synthetic  `corrupt` rows: input = ``source_text`` (damaged), target = ``gold_text``.
             The damaged chunk must hold the same masked items as the clean one
             (``placeholders == gold_placeholders``), else the pair is dropped
             (``placeholders_changed``: e.g. Hangul garbled into ``$..$``, which at inference
             is a placeholder the model must copy, so it must never learn to rewrite one).
             ``sft.junk``: a share of them also get short junk runs (symbols, Latin
             ``lI1``-like strings, random Hangul) after sentence ends or as their own
             paragraph, which the target drops: the junk `score` leaves (under
             ``score.noise.min_chars``, or predictable).
  gold       `gold chunks` rows (the sft side): input = the OCR chunk, target = the
             hand-corrected text (``gold_text``); only ``gold_status: ok`` rows;
             ``sft.gold_repeats`` copies each (SFT only); no junk added (the real damage
             is there already).
  identity   clean chunks (`chunk` or `score` output): target = input, so the model
             learns to leave correct text alone.

Gold rows of the eval split are refused (never trained on). Quotas: ``sft.mix``
x ``lang_balance`` as in `data sft` (largest total without reuse, capped by ``total``), cells
filled by a seeded shuffle. The split is by document (``sft.val_frac``). Two passes over the
inputs (count, then write), so memory holds only ids.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.data.common import EMPTY_THINK, in_val, quotas
from ocrfix.io import iter_jsonl
from ocrfix.stages.prompt import messages

# junk alphabets; no markdown, math, table, code or placeholder syntax ($ | ` \ < > [ ] ⟨ ⟩ 〈 〉)
_RANGES = [(0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF), (0x2500, 0x257F),
           (0x25A0, 0x25FF), (0x2E80, 0x2EF3), (0x2100, 0x214F)]  # fmt: skip
_NOT = {0x2329, 0x232A}  # angle brackets that look like 〈 〉
SYMBOLS = "".join(chr(c) for a, b in _RANGES for c in range(a, b + 1) if c not in _NOT)
LATIN = "lI1!i.,:;'rnmvwuoc"
_SPOT = re.compile(r"[.!?。](?= \S)")


def junk_run(rng: random.Random, lo: int, hi: int) -> str:
    """One junk string with lo..hi non-space chars."""
    n = rng.randint(lo, hi)
    kind = rng.choices(["symbols", "latin", "hangul"], weights=[0.4, 0.3, 0.3])[0]
    out: list[str] = []
    k = 0
    while k < n:
        if kind == "latin":
            w = "".join(rng.choice(LATIN) for _ in range(min(n - k, rng.randint(2, 5))))
        elif kind == "symbols":
            w = "".join(rng.choice(SYMBOLS) for _ in range(min(n - k, rng.randint(1, 6))))
        else:
            m = min(n - k, rng.randint(1, 4))
            w = "".join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(m))
        out.append(w)
        k += len(w)
    return " ".join(out)


def add_junk(text: str, rng: random.Random, opts: dict[str, Any]) -> tuple[str, list[str]]:
    """``text`` with 1..max_runs junk runs inserted; (new text, the runs)."""
    lo, hi = (int(x) for x in opts["chars"])
    spots = [m.end() for m in _SPOT.finditer(text)]
    paras = [m.start() for m in re.finditer(r"\n\n", text)]
    inserts: list[tuple[int, str]] = []
    for _ in range(rng.randint(1, int(opts["max_runs"]))):
        run = junk_run(rng, lo, hi)
        if paras and rng.random() < float(opts["own_line"]):
            inserts.append((rng.choice(paras), "\n\n" + run))
        elif spots:
            inserts.append((rng.choice(spots), " " + run))
        else:  # one sentence, no break: at the end
            inserts.append((len(text), " " + run))
    out, pos = [], 0
    for at, s in sorted(inserts, key=lambda x: x[0]):
        out += [text[pos:at], s]
        pos = at
    out.append(text[pos:])
    return "".join(out), [s.strip() for _, s in sorted(inserts, key=lambda x: x[0])]


@dataclass
class DataStats:
    rows: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    pool: dict[str, dict[str, int]] = field(default_factory=dict)
    taken: dict[str, dict[str, int]] = field(default_factory=dict)
    with_junk: int = 0
    train: int = 0
    val: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["dropped"] = {k: dict(v) for k, v in self.dropped.items()}
        return d


def _id(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def _example(
    source: str, r: dict[str, Any], inp: str, target: str, empty_think: bool
) -> dict[str, Any]:
    msgs = messages(inp)
    msgs.append({"role": "assistant", "content": (EMPTY_THINK if empty_think else "") + target})
    ex = {
        "id": _id(source, r["chunk_id"]),
        "source": source,
        "lang": r.get("lang") or "other",
        "doc_id": r["doc_id"],
        "chunk_id": r["chunk_id"],
        "placeholders": r.get("placeholders") or {},
        "messages": msgs,
    }
    for k in ("corruption", "error_rate"):
        if r.get(k) is not None:
            ex[k] = r[k]
    return ex


def examples(
    source: str, paths: list[Path], opts: dict[str, Any], seed: int, st: DataStats | None
) -> Iterator[dict[str, Any]]:
    """Examples of one source, in input order (``st`` counts drops; None on the 2nd pass)."""
    junk = opts["junk"]
    for p in paths:
        for r in iter_jsonl(p):
            why = None
            if r.get("split") == "eval":
                why = "eval_doc"
            elif source == "gold":
                if "gold_status" not in r:
                    raise ValueError(f"{p}: gold rows need gold_status (`gold chunks` output)")
                if r["gold_status"] != "ok":
                    why = r["gold_status"]
                inp, target = r["source_text"], r["gold_text"]
            elif source == "synthetic":
                if r.get("gold_text") is None:
                    raise ValueError(f"{p}: synthetic rows need gold_text (`corrupt` output)")
                if r.get("placeholders") != r.get("gold_placeholders"):
                    why = "placeholders_changed"
                inp, target = r["source_text"], r["gold_text"]
            else:
                inp = target = r["text"] if r.get("text") is not None else r["source_text"]
                if r.get("status") in ("noise", "score_error"):
                    why = r["status"]
            if why is None and not PLACEHOLDER.sub("", target).strip():
                why = "no_prose"
            if why:
                if st is not None:
                    st.dropped[source][why] += 1
                continue
            ex = _example(source, r, inp, target, opts["empty_think"])
            if source == "synthetic" and float(junk["prob"]) > 0:
                rng = random.Random(_id(str(seed), "junk", r["chunk_id"]))
                if rng.random() < float(junk["prob"]):
                    noisy, runs = add_junk(inp, rng, junk)
                    ex["messages"][1]["content"] = noisy
                    ex["junk"] = runs
            if source == "gold" and int(opts.get("gold_repeats", 1)) > 1:
                for k in range(int(opts["gold_repeats"])):
                    yield {**ex, "id": f"{ex['id']}#{k}", "repeat": k}
                continue
            yield ex


def build_sft_data(
    sources: dict[str, list[Path]], out_dir: Path, *, opts: dict[str, Any], seed: int
) -> DataStats:
    """``opts``: the ``sft`` config section."""
    st = DataStats()
    sources = {s: ps for s, ps in sources.items() if ps}
    pools: dict[str, dict[str, list[str]]] = {}
    for s, paths in sources.items():
        by_lang: dict[str, list[str]] = defaultdict(list)
        n = 0
        for ex in examples(s, paths, opts, seed, st):
            by_lang[ex["lang"]].append(ex["id"])
            n += 1
        st.rows[s] = n + sum(st.dropped[s].values())
        pools[s] = by_lang
        st.pool[s] = {k: len(v) for k, v in sorted(by_lang.items())}
    q = quotas(pools, dict(opts["mix"]), opts.get("lang_balance"), opts.get("total"))
    chosen: set[str] = set()
    for s, langs in q.items():
        st.taken[s] = {}
        for lang, k in langs.items():
            ids = sorted(pools[s][lang])
            random.Random(f"{seed}:{s}:{lang}").shuffle(ids)
            chosen.update(ids[:k])
            st.taken[s][lang] = min(k, len(ids))
    del pools

    out_dir.mkdir(parents=True, exist_ok=True)
    val_frac = float(opts["val_frac"])
    files = {k: out_dir / f"{k}.jsonl" for k in ("train", "val")}
    tmps = {k: f.with_name(f.name + ".tmp") for k, f in files.items()}
    handles = {k: t.open("w", encoding="utf-8") for k, t in tmps.items()}
    try:
        for s, paths in sources.items():
            for ex in examples(s, paths, opts, seed, None):
                if ex["id"] not in chosen:
                    continue
                side = "val" if in_val(ex["doc_id"], seed, val_frac) else "train"
                handles[side].write(json.dumps(ex, ensure_ascii=False) + "\n")
                setattr(st, side, getattr(st, side) + 1)
                st.with_junk += "junk" in ex
    finally:
        for h in handles.values():
            h.close()
    for k in files:
        tmps[k].replace(files[k])
    (out_dir / "stats.json").write_text(json.dumps(st.to_dict(), indent=1) + "\n", "utf-8")
    return st
