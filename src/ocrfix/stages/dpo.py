"""`ocrfix data dpo`: chunk preference pairs, rejected answers mostly from the model itself.

Prompts are made exactly like `data sft` (``stages/data.py`` ``examples``: synthetic = damaged
chunk -> clean chunk, with the same junk insertion; gold = OCR chunk ->
hand-corrected chunk (`gold chunks` dpo side); identity = clean chunk -> itself), so chosen
is always the known right text. Prompts whose input or target appears in an
``--exclude`` file (the SFT training data) are skipped: the SFT model has learned those
answers, so it would make no mistakes on them.

**On-policy** (the main source): the correction model (``correct.model`` / ``correct.adapter``,
else the base) answers every prompt once greedily, as `correct` does, and
``n_samples`` times at ``temperature`` / ``top_p``, in non-thinking mode. Every distinct
answer that differs from chosen is wrong (chosen is exact), and is typed:
  placeholders  a placeholder lost, duplicated or reordered (``restore`` fails)
  truncated     cut at max_tokens
  false_fix     changed text that was already right (identity prompts: any change)
  missed        left damage that chosen fixes, changed nothing else
  wrong_fix     edited the damaged places, but not into chosen
  mixed         more than one of false_fix / missed / wrong_fix
Up to ``max_pairs`` per prompt: the greedy answer first (what `correct` will output),
then sampled ones closest to chosen (near misses: the pair differs only where it matters).

**Constructed** (off-policy, few): when the model got a prompt right in every answer, a share
``constructed_share`` of those prompts gets one constructed rejected answer, taking the kinds
in ``constructed`` in turn: ``unchanged`` (the damaged input as is), ``half_fixed`` (half of
the input -> chosen edits undone), ``placeholder_dropped`` (chosen minus one placeholder).
They only keep the model from drifting on what it already does right; DPO pushes rejected
answers down relative to the SFT reference, so easy ones teach little.

Rows: TRL conversational ``{id, prompt, chosen, rejected}`` + ``rejected_type`` (``greedy:`` /
``sampled:`` / ``constructed:`` + kind), ``source, lang, doc_id, chunk_id``. Answers are cached
in ``<out stem>.responses.jsonl`` (rerun resumes).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from ocrfix.corpus.placeholders import PLACEHOLDER, PlaceholderError, restore
from ocrfix.io import iter_jsonl
from ocrfix.llm.client import Client, run_requests
from ocrfix.stages.data import examples
from ocrfix.stages.prompt import messages, parse_answer, request_body


def _h(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def _regions(a: str, b: str) -> list[tuple[int, int]]:
    """Ranges of ``a`` that an a -> b alignment changes (an insertion is a 0-width range)."""
    return [(o.src_start, o.src_end) for o in Levenshtein.opcodes(a, b) if o.tag != "equal"]


def _touch(r: tuple[int, int], rs: list[tuple[int, int]], slack: int = 1) -> bool:
    return any(r[0] <= b + slack and a <= r[1] + slack for a, b in rs)


def classify(inp: str, chosen: str, answer: str, mapping: dict[str, str]) -> str:
    """Why ``answer`` (!= chosen) is wrong for ``inp`` (see the module docstring)."""
    try:
        restore(answer, mapping)
    except PlaceholderError:
        return "placeholders"
    if PLACEHOLDER.findall(answer) != PLACEHOLDER.findall(inp):
        return "placeholders"
    gold = _regions(inp, chosen)
    edits = _regions(inp, answer)
    if not gold:
        return "false_fix"
    false_fix = any(not _touch(e, gold) for e in edits)
    missed = any(not _touch(g, edits) for g in gold)
    if false_fix and missed:
        return "mixed"
    # neither: every damaged place was edited and nothing else, yet the answer isn't chosen
    return "false_fix" if false_fix else "missed" if missed else "wrong_fix"


def half_fixed(inp: str, chosen: str, rng: random.Random) -> str | None:
    """``chosen`` with a random half of the input -> chosen edits undone."""
    ops = [o for o in Levenshtein.opcodes(inp, chosen) if o.tag != "equal"]
    if len(ops) < 2:
        return None
    undo = set(rng.sample(range(len(ops)), len(ops) // 2))
    out, k = [], 0
    for o in Levenshtein.opcodes(inp, chosen):
        if o.tag == "equal":
            out.append(chosen[o.dest_start : o.dest_end])
            continue
        out.append(inp[o.src_start : o.src_end] if k in undo else chosen[o.dest_start : o.dest_end])
        k += 1
    return "".join(out)


def constructed(kind: str, inp: str, chosen: str, rng: random.Random) -> str | None:
    if kind == "unchanged":
        return inp if inp != chosen else None
    if kind == "half_fixed":
        return half_fixed(inp, chosen, rng)
    if kind == "placeholder_dropped":
        phs = list(PLACEHOLDER.finditer(chosen))
        if not phs:
            return None
        m = rng.choice(phs)
        a, b = m.start(), m.end()
        if a > 0 and chosen[a - 1] == " " and b < len(chosen) and chosen[b] in " .,":
            a -= 1  # don't leave a double space
        return chosen[:a] + chosen[b:]
    raise ValueError(f"dpo.constructed: unknown kind {kind!r}")


@dataclass
class DpoStats:
    prompts: dict[str, int] = field(default_factory=dict)
    excluded: int = 0
    requests: dict[str, int] = field(default_factory=dict)
    all_right: int = 0  # prompts where every answer equals chosen
    greedy_right: int = 0
    pairs: int = 0
    types: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["types"] = dict(self.types.most_common())
        return d


def _fingerprints(paths: list[Path]) -> set[str]:
    out = set()
    for p in paths:
        for r in iter_jsonl(p):
            for m in r.get("messages", [])[1:]:
                out.add(_h(m["content"]))
    return out


def prompts(
    sources: dict[str, list[Path]], opts: dict[str, Any], seed: int, exclude: set[str],
    st: DpoStats,
) -> Iterator[dict[str, Any]]:  # fmt: skip
    """`data sft` examples not in ``exclude``; at most ``max_prompts[source]`` per source
    (a seeded hash order, so the pick is spread over the input)."""
    cap = opts["dpo"].get("max_prompts") or {}
    for s, paths in sources.items():
        rows = []
        for ex in examples(s, paths, opts, seed, None):
            user, target = ex["messages"][1]["content"], ex["messages"][2]["content"]
            if _h(user) in exclude or _h(target) in exclude:
                st.excluded += 1
                continue
            rows.append((_h(str(seed), "dpo", ex["id"]), ex))
        rows.sort(key=lambda x: x[0])
        k = cap.get(s)
        rows = rows if k is None else rows[: int(k)]
        st.prompts[s] = len(rows)
        yield from (ex for _, ex in rows)


def build_dpo_pairs(
    sources: dict[str, list[Path]], out: Path, *, opts: dict[str, Any], seed: int,
    exclude: list[Path], encode, transport=None,
) -> DpoStats:  # fmt: skip
    """``opts``: as for `data sft` (``empty_think`` must be off: chosen is compared with the
    answers as is), plus ``dpo`` (the dpo section), ``llm`` (correct + seed), ``model`` (the
    served name), ``urls``,
    ``concurrency``.
    ``encode(text) -> (ids, offsets)`` sizes max_tokens."""
    st = DpoStats()
    opts = {**opts, "gold_repeats": 1}
    d, llm = opts["dpo"], opts["llm"]
    model = opts["model"]
    items = list(
        prompts({k: v for k, v in sources.items() if v}, opts, seed, _fingerprints(exclude), st)
    )

    reqs = []
    for ex in items:
        user = ex["messages"][1]["content"]
        body = request_body(user, len(encode(user)[0]), model=model, llm=llm)
        reqs.append((f"{ex['id']}:g", body))
        if int(d["n_samples"]) > 0:
            reqs.append((f"{ex['id']}:s", body | {
                "n": int(d["n_samples"]), "temperature": float(d["temperature"]),
                "top_p": float(d["top_p"]), "seed": int(seed),
            }))  # fmt: skip
    client = Client(list(opts["urls"]), concurrency=int(opts["concurrency"]),
                    retries=int(llm["retries"]), backoff=float(llm["backoff"]),
                    timeout=float(llm["timeout"]), transport=transport)  # fmt: skip
    responses, st.requests = run_requests(
        client, reqs, out.with_name(out.stem + ".responses.jsonl")
    )

    kinds = list(d["constructed"])
    n_solved = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ex in items:
            user, chosen = ex["messages"][1]["content"], ex["messages"][2]["content"]
            mapping = ex.get("placeholders") or {}
            cands: list[tuple[str, str, int]] = []  # (answer, origin, distance to chosen)
            g = responses.get(f"{ex['id']}:g") or {}
            s = responses.get(f"{ex['id']}:s") or {}
            answers = [("greedy", g)] if g.get("ok") or g.get("finish_reason") else []
            answers += [("sampled", c) for c in (s.get("choices") or [])]
            any_wrong = False
            for origin, a in answers:
                text = parse_answer(a.get("text") or "")
                if text == chosen and a.get("finish_reason") != "length":
                    if origin == "greedy":
                        st.greedy_right += 1
                    continue
                any_wrong = True
                cands.append((text, origin + ":" + (
                    "truncated" if a.get("finish_reason") == "length"
                    else classify(user, chosen, text, mapping)
                ), 0 if origin == "greedy" else Levenshtein.distance(text, chosen)))  # fmt: skip
            picked, seen = [], {chosen}
            for text, typ, _ in sorted(cands, key=lambda c: c[2]):
                if text in seen or not text:
                    continue
                seen.add(text)
                picked.append((text, typ))
                if len(picked) >= int(d["max_pairs"]):
                    break
            if answers and not any_wrong:
                st.all_right += 1
                rng = random.Random(_h(str(seed), "constructed", ex["id"]))
                if kinds and rng.random() < float(d["constructed_share"]):
                    order = [kinds[(n_solved + j) % len(kinds)] for j in range(len(kinds))]
                    for kind in order:
                        text = constructed(kind, user, chosen, rng)
                        if text and text != chosen:
                            picked.append((text, "constructed:" + kind))
                            break
                    n_solved += 1
            for text, typ in picked:
                row = {
                    "id": _h(ex["id"], typ, text),
                    "prompt": messages(user),
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": text}],
                    "rejected_type": typ,
                    **{k: ex[k] for k in ("source", "lang", "doc_id", "chunk_id")},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                st.pairs += 1
                st.types[typ] += 1
    tmp.replace(out)
    out.with_name(out.stem + ".stats.json").write_text(json.dumps(st.to_dict(), indent=1) + "\n")
    return st
