"""`ocrfix gold sample`: pick paragraphs for human gold annotation.

Strata, filled in this order, each paragraph in at most one:
  1. ``feature:<name>``: fixed quotas (``gold.sample.features``) for paragraphs with inline or
     display math, tables, suspected garbled Hangul, mixed ko/en, hyphenated line ends.
  2. ``<lang>:low|mid|high``: per-language quotas (``gold.sample.per_lang``) split over the
     ``spike_frac`` tertiles of that language (``gold.sample.tertiles``, default 40/30/30).
Within a stratum candidates are taken in order of sha256(seed, doc_id, para_id), so the
sample depends on the seed and the data, never on file order. At most
``gold.sample.max_per_doc`` paragraphs come from one document.

Reads ``paragraphs`` twice (features, then text of the picks and their neighbours); only
small per-paragraph tuples are held in memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ocrfix.corpus.placeholders import PlaceholderError, substitute
from ocrfix.gold.features import paragraph_features
from ocrfix.gold.labelstudio import labeling_config, tasks
from ocrfix.io import iter_jsonl, iter_records, open_text, read_ids
from ocrfix.schema import Record

log = logging.getLogger(__name__)

TERTILES = ("low", "mid", "high")
Key = tuple[str, str]


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    para_id: str
    order: int
    lang: str
    score: float | None
    features: tuple[str, ...]
    prio: str

    @property
    def key(self) -> Key:
        return (self.doc_id, self.para_id)


@dataclass
class SampleResult:
    rows: list[dict[str, Any]]
    wanted: dict[str, int] = field(default_factory=dict)
    got: dict[str, int] = field(default_factory=dict)
    n_paragraphs: int = 0
    n_unscored: int = 0

    @property
    def shortfalls(self) -> dict[str, tuple[int, int]]:
        got = self.got
        return {s: (w, got.get(s, 0)) for s, w in self.wanted.items() if got.get(s, 0) < w}


def priority(seed: int, key: Key) -> str:
    return hashlib.sha256(f"{seed}\x00{key[0]}\x00{key[1]}".encode()).hexdigest()


def split_quota(n: int, fracs: dict[str, float]) -> dict[str, int]:
    """Integer quotas summing to n (largest remainder)."""
    total = sum(fracs.values())
    raw = {k: n * v / total for k, v in fracs.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: out[k] - raw[k])[: n - sum(out.values())]:
        out[k] += 1
    return out


def select(
    cands: list[Candidate],
    *,
    per_lang: dict[str, int],
    tertiles: dict[str, float],
    features: dict[str, int],
    max_per_doc: int,
) -> tuple[dict[Key, str], dict[str, int], dict[str, int]]:
    """Return ({key: stratum}, wanted per stratum, got per stratum)."""
    chosen: dict[Key, str] = {}
    per_doc = Counter[str]()
    wanted: dict[str, int] = {}
    got: dict[str, int] = {}

    def take(pool: Iterable[Candidate], n: int, stratum: str) -> None:
        wanted[stratum] = n
        k = 0
        for c in pool:
            if k >= n:
                break
            if c.key in chosen or per_doc[c.doc_id] >= max_per_doc:
                continue
            chosen[c.key] = stratum
            per_doc[c.doc_id] += 1
            k += 1
        got[stratum] = k

    by_prio = sorted(cands, key=lambda c: c.prio)
    for feat, n in features.items():
        if n > 0:
            take((c for c in by_prio if feat in c.features), n, f"feature:{feat}")

    for lang, n in per_lang.items():
        if n <= 0:
            continue
        # tertiles by rank over every scored paragraph of the language (ties broken by prio),
        # so heavy ties at spike_frac == 0 still give three equal-sized bins
        ranked = sorted(
            (c for c in cands if c.lang == lang and c.score is not None),
            key=lambda c: (c.score, c.prio),
        )
        cut = [len(ranked) * i // 3 for i in range(4)]
        for i, (t, q) in enumerate(split_quota(n, {t: tertiles[t] for t in TERTILES}).items()):
            pool = sorted(ranked[cut[i] : cut[i + 1]], key=lambda c: c.prio)
            take(pool, q, f"{lang}:{t}")
    return chosen, wanted, got


def page_image_path(template: str | None, rec: Record) -> str | None:
    if not template:
        return None
    src = PurePosixPath(rec.source_path)
    return template.format(
        doc_id=rec.doc_id,
        para_id=rec.para_id,
        order=rec.order,
        source_path=rec.source_path,
        source_stem=src.stem,
        source_dir=str(src.parent),
    )


def sample(
    scores: Path,
    paragraphs: Path,
    *,
    seed: int,
    per_lang: dict[str, int],
    tertiles: dict[str, float],
    features: dict[str, int],
    max_per_doc: int = 8,
    score_key: str = "spike_frac",
    page_image: str | None = None,
    exclude: set[Key] | None = None,
    limit: int | None = None,
    feature_opts: dict[str, Any] | None = None,
) -> SampleResult:
    exclude = exclude or set()
    feature_opts = feature_opts or {}
    score_of = {(str(d["doc_id"]), str(d["para_id"])): d.get(score_key) for d in iter_jsonl(scores)}

    cands: list[Candidate] = []
    doc_orders: dict[str, list[tuple[int, str]]] = defaultdict(list)
    n_paragraphs = n_unscored = 0
    for rec in iter_records(paragraphs, limit):
        n_paragraphs += 1
        doc_orders[rec.doc_id].append((rec.order, rec.para_id))
        if rec.key in exclude:
            continue
        try:
            substitute(rec.text, rec.spans)
        except PlaceholderError:  # literal ⟨M1⟩ in the source: can't round-trip, skip
            log.warning("%s/%s: text contains a placeholder; not sampled", *rec.key)
            continue
        score = score_of.get(rec.key)
        n_unscored += score is None
        cands.append(
            Candidate(
                rec.doc_id,
                rec.para_id,
                rec.order,
                rec.lang,
                None if score is None else float(score),
                paragraph_features(rec, **feature_opts),
                priority(seed, rec.key),
            )
        )
    feats = {c.key: c.features for c in cands}
    chosen, wanted, got = select(
        cands, per_lang=per_lang, tertiles=tertiles, features=features, max_per_doc=max_per_doc
    )
    del cands

    neighbours: dict[Key, tuple[Key | None, Key | None]] = {}
    for doc_id in {k[0] for k in chosen}:
        order = [(doc_id, pid) for _, pid in sorted(doc_orders[doc_id])]
        for i, key in enumerate(order):
            if key in chosen:
                prev = order[i - 1] if i > 0 else None
                nxt = order[i + 1] if i + 1 < len(order) else None
                neighbours[key] = (prev, nxt)
    needed = set(chosen) | {k for pair in neighbours.values() for k in pair if k}
    recs = {r.key: r for r in iter_records(paragraphs, limit) if r.key in needed}

    rows = []
    for key in sorted(chosen, key=lambda k: (k[0], recs[k].order, k[1])):
        rec = recs[key]
        prev, nxt = neighbours[key]
        masked, mapping = substitute(rec.text, rec.spans)
        rows.append(
            {
                "doc_id": rec.doc_id,
                "para_id": rec.para_id,
                "order": rec.order,
                "stratum": chosen[key],
                "lang": rec.lang,
                "features": list(feats[key]),
                score_key: score_of.get(key),
                "source_path": rec.source_path,
                "source_text": masked,
                "placeholders": mapping,
                "context_prev": recs[prev].text if prev else None,
                "context_next": recs[nxt].text if nxt else None,
                "page_image_path": page_image_path(page_image, rec),
                "gold_text": "",
                "span_labels": {},
                "structure_label": "",
                "unrecoverable_count": None,
                "annotator_id": "",
                "notes": "",
            }
        )
    return SampleResult(rows, wanted, got, n_paragraphs, n_unscored)


def _stem(out: Path) -> Path:
    name = out.name
    for suffix in (".jsonl.gz", ".jsonl", ".json"):
        if name.endswith(suffix):
            return out.with_name(name[: -len(suffix)])
    return out


def write_sample(result: SampleResult, out: Path, image_url_prefix: str = "") -> list[Path]:
    """Write the annotation JSONL plus a Label Studio import file and labeling config."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = _stem(out)
    ls_json = stem.with_name(stem.name + ".labelstudio.json")
    ls_xml = stem.with_name(stem.name + ".labelstudio.xml")
    has_images = any(r["page_image_path"] for r in result.rows)
    outputs = {
        out: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.rows),
        ls_json: json.dumps(tasks(result.rows, image_url_prefix), ensure_ascii=False, indent=1),
        ls_xml: labeling_config(with_image=has_images),
    }
    for path, content in outputs.items():
        tmp = path.with_name(path.name + ".tmp")
        with open_text(tmp, "wt") as f:
            f.write(content)
        tmp.replace(path)
    return list(outputs)


def run_sample(
    cfg, scores: Path, paragraphs: Path, out: Path, *, exclude=(), limit=None
) -> SampleResult:
    sc = cfg.gold.sample
    excluded: set[Key] = set()
    for p in exclude:
        excluded |= read_ids(p)
    result = sample(
        scores,
        paragraphs,
        seed=int(cfg.seed),
        per_lang=dict(sc.per_lang),
        tertiles=dict(sc.tertiles),
        features=dict(sc.features),
        max_per_doc=int(sc.max_per_doc),
        score_key=sc.score_key,
        page_image=cfg.gold.page_image,
        exclude=excluded,
        limit=limit,
        feature_opts={
            "garbled_max_len": int(sc.garbled_max_len),
            "mixed_min": float(cfg.corpus.lang.mixed_min),
            "hangul_weight": float(cfg.corpus.lang.hangul_weight),
        },
    )
    write_sample(result, out, cfg.gold.label_studio.image_url_prefix or "")
    return result
