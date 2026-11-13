"""`ocrfix calibrate-noise <gold_chunks>`: fit the noise-removal thresholds on gold OCR text.

Noise removal (stages/noise.py) cuts runs of whitespace-separated words whose bits per char
are high: a word above ``core[script]`` starts a run, neighbours above ``extend[script]`` join
it, runs of at least ``min_chars`` non-space chars are removed. Here those three are fitted on
the OCR side of the gold chunks (`gold chunks` sft/ and dpo/; never eval/), which holds real OCR
errors next to real junk, with the truth saying which is which:

1. **Label** each character of the masked OCR text by aligning it with the truth:
     text     what the truth keeps: as is, corrected (an OCR-damaged word must be corrected,
              not deleted) or replaced by 〈?〉 (the model must write 〈?〉 there);
     deleted  an alignment region (adjacent edits merged) whose truth side has no letter or
              digit: the annotator deleted it (page numbers, headers, junk). Ignored: removing
              it can't hurt, and it isn't counted as noise either.
2. **Insert** junk into a copy of every chunk: the noise to find (``calibrate.noise.
   insert_runs`` runs of ``insert_chars`` non-space chars: symbols, ``lI1``-like Latin, random
   Hangul, loose jamo, digit/symbol table leftovers; at whitespace, 20% as their own
   paragraph). The rest of the copy keeps its real OCR text.
3. **Score** the originals and the copies with the base model (``score.backend``), as
   `ocrfix score` does.
4. **Search** core (per script), extend (core minus a margin) and ``calibrate.noise.min_chars``
   (20 and up: only long runs), applying exactly the rule `score` uses (``noise.word_values``
   / ``noise.level``, vectorized; a test checks the two agree). text_removed = removed text
   chars / all text chars; recall = inserted noise chars removed. The best recall with
   text_removed <= ``calibrate.noise.max_text_removed`` (1 in 10,000) wins.
5. **Write** the three values into configs/stages/score.yaml (``--write``, the default), with a
   comment line giving the date, the model, the noise found and the text removed; and
   ``<out_dir>/noise_report.json`` (the old values' result, the best settings, counts, and up
   to 30 examples of real text the chosen setting removes, to check by eye).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from rapidfuzz.distance import Levenshtein

from ocrfix.io import iter_jsonl
from ocrfix.schema import UNCERTAIN
from ocrfix.stages.noise import COLD, HOT, STOP, WARM, char_bits, word_values
from ocrfix.stages.run import Context
from ocrfix.stages.score import tokenize_chunk

_LETTER = re.compile(r"[^\W_]")  # any letter or digit, any script
CORE = {"hangul": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0],
        "other": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]}  # fmt: skip
MARGIN = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
KINDS = {"stop": 0, "free": 1, "hangul": 2, "other": 3}
REAL, INSERTED = 1, 2  # per char: 0 = text, REAL = deleted by the annotator, INSERTED = junk

# junk alphabets; no markdown, math, table or placeholder syntax ($ | ` \ < > [ ] # * _ ⟨⟩ 〈〉)
_SYMBOLS = "■□▪▫▲△▼▽◆◇○●◎★☆※¶§†‡∴∵≈≡⊕⊗⌘⌂▯▢◘◙♦♣♠♥¤ʘ⁂÷×±°·•‥…∷∶⁘⁙"
_LATIN = "lI1!i.,:;'rnmvwuoc"
_JAMO = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"
_TABLE = "0123456789.,:;-=+%/()"


def junk(rng: random.Random, lo: int, hi: int) -> str:
    """One junk run with lo..hi non-space chars, in short space-separated pieces."""
    n = rng.randint(lo, hi)
    kind = rng.choice(["symbols", "latin", "hangul", "jamo", "table"])
    out, k = [], 0
    while k < n:
        m = min(n - k, rng.randint(1, 6))
        if kind == "hangul":
            w = "".join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(m))
        else:
            alphabet = {"symbols": _SYMBOLS, "latin": _LATIN, "jamo": _JAMO, "table": _TABLE}[kind]
            w = "".join(rng.choice(alphabet) for _ in range(m))
        out.append(w)
        k += m
    return " ".join(out)


def noise_labels(ocr: str, gold: str) -> np.ndarray:
    """REAL for OCR chars the annotator deleted (alignment regions, adjacent edits merged,
    whose truth side has no letter or digit and no 〈?〉), else 0."""
    lab = np.zeros(len(ocr), dtype=np.int8)
    region: list[int] | None = None
    for o in [*Levenshtein.opcodes(ocr, gold), None]:
        if o is not None and o.tag != "equal":
            if region is None:
                region = [o.src_start, o.src_end, o.dest_start, o.dest_end]
            else:
                region[1], region[3] = o.src_end, o.dest_end
            continue
        if region is not None:
            a, b, c, d = region
            g = gold[c:d]
            if b > a and not _LETTER.search(g) and UNCERTAIN not in g:
                lab[a:b] = REAL
            region = None
    return lab


def insert_junk(text: str, lab: np.ndarray, rng: random.Random, runs: tuple[int, int],
                chars: tuple[int, int]) -> tuple[str, np.ndarray]:  # fmt: skip
    """``text`` with junk runs inserted at whitespace (never inside a word or placeholder);
    the labels follow, the inserted chars are INSERTED."""
    spots = [m.start() for m in re.finditer(r"(?<=\S)\s", text)]
    if not spots:
        spots = [len(text)]
    picks = sorted(rng.sample(spots, min(len(spots), rng.randint(*runs))))
    parts, labs, pos = [], [], 0
    for at in picks:
        sep = "\n\n" if rng.random() < 0.2 else " "
        run = sep + junk(rng, *chars)
        parts += [text[pos:at], run]
        labs += [lab[pos:at], np.full(len(run), INSERTED, dtype=np.int8)]
        labs[-1][: len(sep)] = 0  # the separator itself is whitespace
        pos = at
    parts.append(text[pos:])
    labs.append(lab[pos:])
    return "".join(parts), np.concatenate(labs)


@dataclass
class Words:
    """Every word of every chunk, flat (chunk boundaries break runs)."""

    chunk: np.ndarray
    start: np.ndarray  # char offsets in the chunk text
    end: np.ndarray
    kind: np.ndarray  # KINDS code
    x: np.ndarray  # bits per char (nan for stop / free)
    size: np.ndarray  # non-space chars (a word is \\S+)
    real: np.ndarray  # chars the annotator deleted (ignored)
    inserted: np.ndarray  # inserted junk chars (the noise)

    @property
    def text(self) -> np.ndarray:
        """Real text chars per word: what the truth keeps."""
        return self.size - self.real - self.inserted


def collect(texts: list[str], labels: list[np.ndarray], tps, bits) -> Words:
    keys = ("chunk", "start", "end", "kind", "x", "size", "real", "inserted")
    cols: dict[str, list] = {k: [] for k in keys}
    for i, (lab, tp, b) in enumerate(zip(labels, tps, bits, strict=True)):
        for a, e, kind, x in word_values(tp.text, char_bits(tp, b)):
            cols["chunk"].append(i)
            cols["start"].append(a)
            cols["end"].append(e)
            cols["kind"].append(KINDS[kind])
            cols["x"].append(x)
            cols["size"].append(e - a)
            cols["real"].append(int((lab[a:e] == REAL).sum()))
            cols["inserted"].append(int((lab[a:e] == INSERTED).sum()))
    return Words(*(np.asarray(cols[k], dtype=float if k == "x" else np.int64) for k in keys))


def removed_words(w: Words, core: dict[str, float], extend: dict[str, float],
                  min_chars: list[int]) -> list[np.ndarray]:  # fmt: skip
    """Per min_chars, a mask of the words the rule removes: a run = a maximal stretch of
    WARM-or-HOT words in one chunk holding a HOT word; removed when its non-space chars >=
    min_chars."""
    lvl = np.full(len(w.kind), STOP)
    lvl[w.kind == KINDS["free"]] = WARM
    for name in ("hangul", "other"):
        sel = w.kind == KINDS[name]
        x = w.x[sel]
        lvl[sel] = np.where(x > core[name], HOT, np.where(x > extend[name], WARM, COLD))
    on = lvl >= WARM
    same = np.r_[False, w.chunk[1:] == w.chunk[:-1]]
    seg = np.cumsum(on & ~(np.r_[False, on[:-1]] & same)) - 1
    if not on.any():
        return [np.zeros(len(on), dtype=bool) for _ in min_chars]
    s = np.where(on, seg, 0)
    n = int(seg[on].max()) + 1
    size = np.bincount(s[on], weights=w.size[on], minlength=n)
    hot = np.bincount(s[on], weights=(lvl[on] == HOT), minlength=n) > 0
    return [on & (hot & (size >= m))[s] for m in min_chars]


def _score(w: Words, mask: np.ndarray) -> dict[str, Any]:
    text_all, ins_all = int(w.text.sum()), int(w.inserted.sum())
    text, ins = int(w.text[mask].sum()), int(w.inserted[mask].sum())
    return {
        "recall": round(ins / ins_all, 4) if ins_all else 0.0,
        "text_removed": round(text / text_all, 6) if text_all else 0.0,
        "text_removed_chars": text,
        "deleted_removed_chars": int(w.real[mask].sum()),  # page numbers etc.: harmless
        "removed_chars": int(w.size[mask].sum()),
    }


def search(w: Words, max_text: float, min_chars: list[int]) -> tuple[dict | None, list[dict]]:
    results = []
    for ch in CORE["hangul"]:
        for co in CORE["other"]:
            for margin in MARGIN:
                core = {"hangul": ch, "other": co}
                ext = {"hangul": ch - margin, "other": co - margin}
                for m, mask in zip(min_chars, removed_words(w, core, ext, min_chars), strict=True):
                    results.append({"core": core, "extend": ext, "min_chars": m} | _score(w, mask))
    results.sort(key=lambda r: (r["text_removed"] > max_text, -r["recall"], r["text_removed"]))
    best = results[0] if results and results[0]["text_removed"] <= max_text else None
    return best, results


def text_examples(w: Words, texts: list[str], mask: np.ndarray, n: int = 30) -> list[str]:
    """Real text the setting removes, with 30 chars of context: [[removed]]."""
    out = []
    for i in np.flatnonzero(mask & (w.text > 0))[:n]:
        t, a, e = texts[int(w.chunk[i])], int(w.start[i]), int(w.end[i])
        out.append(f"{t[max(a - 30, 0) : a]}[[{t[a:e]}]]{t[e : e + 30]}".replace("\n", " / "))
    return out


def write_score_yaml(path: Path, fit: dict[str, Any]) -> None:
    """Replace core / extend / min_chars under score.noise, keeping every other line, and the
    comment line just above min_chars."""
    lines = path.read_text().splitlines(keepends=True)
    fmt = lambda d: "{" + ", ".join(f"{k}: {d[k]}" for k in ("hangul", "other")) + "}"  # noqa: E731
    model = Path(fit["model"]).name
    stamp = (
        f"    # fitted by `ocrfix calibrate-noise` {datetime.date.today()}, {model}, "
        f"{fit['chunks']} gold chunks: noise found {fit['recall']}, text removed "
        f"{fit['text_removed']}\n"
    )
    new = {
        "min_chars": f"    min_chars: {fit['min_chars']}\n",
        "core": f"    core: {fmt(fit['core'])}\n",
        "extend": f"    extend: {fmt(fit['extend'])}\n",
    }
    out, in_noise, seen = [], False, set()
    for line in lines:
        if line.startswith("  noise:"):
            in_noise = True
        elif in_noise and line.strip() and not line.startswith("    "):
            in_noise = False
        key = line.strip().split(":")[0] if in_noise else None
        if in_noise and key in new:
            if key == "min_chars":
                if out and out[-1].lstrip().startswith(("# fitted by", "# set by hand")):
                    out.pop()
                out.append(stamp)
            out.append(new[key])
            seen.add(key)
        else:
            out.append(line)
    if seen != set(new):
        raise ValueError(f"{path}: score.noise.{sorted(set(new) - seen)} not found")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(out))
    tmp.replace(path)


def run_calibrate_noise(cfg, gold_dir: Path, out_dir: Path, *, score_yaml: Path | None,
                        transport=None, limit: int | None = None) -> dict[str, Any]:  # fmt: skip
    """``score_yaml``: the file to rewrite (None: report only)."""
    opts = cfg.calibrate.noise
    rows = [r for side in ("sft", "dpo") if (gold_dir / side).is_dir()
            for r in iter_jsonl(gold_dir / side) if r.get("gold_status") == "ok"]  # fmt: skip
    rows = rows[:limit] if limit else rows
    if not rows:
        raise ValueError(f"{gold_dir}: no `gold chunks` rows with gold_status ok in sft/ or dpo/")
    t0 = time.perf_counter()
    texts, labels = [], []
    for r in rows:
        lab = noise_labels(r["source_text"], r["gold_text"])
        texts.append(r["source_text"])
        labels.append(lab)
        rng = random.Random(f"{cfg.seed}:noise:{r['chunk_id']}")
        t, ll = insert_junk(r["source_text"], lab, rng, tuple(opts.insert_runs),
                            tuple(opts.insert_chars))  # fmt: skip
        texts.append(t)
        labels.append(ll)

    async def score() -> tuple[list, list]:
        async with Context(cfg, transport) as ctx:
            tps = [tokenize_chunk(t, ctx.encode, ctx.boundary) for t in texts]
            bits = await ctx.scorer.bits_batch([tp.ids for tp in tps])
        bad = [b for b in bits if isinstance(b, Exception)]
        if bad:
            raise ValueError(f"scoring failed for {len(bad)} chunks: {bad[0]}")
        return tps, bits

    tps, bits = asyncio.run(score())
    words = collect(texts, labels, tps, bits)
    min_chars = [int(m) for m in opts.min_chars]
    best, results = search(words, float(opts.max_text_removed), min_chars)
    old = OmegaConf.to_container(cfg.score.noise, resolve=True)
    old_mask = removed_words(words, old["core"], old["extend"], [int(old["min_chars"])])[0]
    noise_chars = int(words.inserted.sum())
    enough = noise_chars >= int(opts.min_noise_chars)
    use = bool(best) and enough
    examples = []
    if best:
        mask = removed_words(words, best["core"], best["extend"], [best["min_chars"]])[0]
        examples = text_examples(words, texts, mask)
    fit = {
        "use": use,
        "reason": None if use else (
            f"only {noise_chars} inserted noise chars (< {opts.min_noise_chars})" if not enough
            else f"no setting removes <= {opts.max_text_removed} of the text"),
        "model": str(cfg.model), "chunks": len(rows),
        "noise_chars": noise_chars, "deleted_chars": int(words.real.sum()),
        "text_chars": int(words.text.sum()),
        "max_text_removed": float(opts.max_text_removed),
        **(best or {}),
        "previous": {"core": old["core"], "extend": old["extend"], "min_chars": old["min_chars"]}
        | _score(words, old_mask),
        "text_removed_examples": examples,
        "seconds": round(time.perf_counter() - t0, 1),
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noise_report.json").write_text(
        json.dumps(fit | {"top": results[:20]}, indent=1, ensure_ascii=False) + "\n"
    )
    if use and score_yaml is not None:
        write_score_yaml(score_yaml, fit)
        fit["written"] = str(score_yaml)
    return fit
