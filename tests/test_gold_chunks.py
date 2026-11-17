"""`ocrfix gold chunks` (md pairs -> chunk gold) and `ocrfix eval` (documents vs truth)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.test_chunks import chunk_cfg  # noqa: F401  (fixture)

from ocrfix.corpus.placeholders import restore
from ocrfix.gold.chunks import TruthMap, build_gold_chunks
from ocrfix.io import iter_jsonl
from ocrfix.stages.eval import run_eval

OPTS = {"header_footer_min_chars": 8, "confusion_max_len": 4}


def test_cuts_follow_the_alignment():
    # an insertion right at a chunk's start belongs to that chunk, at its end to that chunk
    tm = TruthMap("ab\n\ncd", "ab\n\nXcd")
    assert tm.at(4, "start") == (4, False) and tm.at(2, "end") == (2, False)
    tm = TruthMap("ab\n\ncd", "abY\n\ncd")
    assert tm.at(2, "end") == (3, False) and tm.at(4, "start") == (5, False)
    # OCR text deleted across a cut vanishes on both sides
    tm = TruthMap("ab JUNK\n\ncd", "ab\n\ncd")
    assert not tm.at(5, "end")[1] and not tm.at(5, "start")[1]
    # a replacement across a cut can't be split
    assert TruthMap("ab WXYZ cd", "ab Q cd").at(5, "end")[1]


SENT = [
    "The proposed method reduces the character error rate on scanned pages.",
    "제안한 방법은 스캔 문서의 문자 오류율을 효과적으로 줄였다.",
    "Each layer preserves the spatial resolution, where $x_t$ denotes the input.",
    "검증 데이터에서 신경망은 출력 품질을 안정적으로 유지하였다.",
]


def _doc(i: int) -> str:
    paras = []
    for p in range(6):
        paras.append(" ".join(SENT[(i + p + k) % 4] for k in range(3)))
    return "\n\n".join(paras) + "\n"


def _damage(text: str) -> str:
    return (text.replace("character", "charactor").replace("효과적으로", "흐괴젹ㅇ로")
            .replace("preserves the", "prese rves the").replace("resolution", "qzxvw"))  # fmt: skip


def _truth(text: str) -> str:
    return text


@pytest.fixture
def gold(tmp_path):
    root = tmp_path / "gold"
    docs = {}
    for i in range(8):
        clean = _doc(i)
        rel = f"v{i % 2}/d{i}.md"
        for side, text in (("ocr", _damage(clean)), ("truth", _truth(clean))):
            p = root / side / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, "utf-8")
        docs[rel] = _truth(clean)
    # the annotator edited math in one document: those chunks can't be used
    p = root / "truth" / "v1/d7.md"
    p.write_text(p.read_text("utf-8").replace("$x_t$", "$x_{t}$"), "utf-8")
    return root, docs


def _rows(d: Path) -> list[dict]:
    return list(iter_jsonl(d))


def test_gold_chunks_end_to_end(tmp_path, gold, chunk_cfg):  # noqa: F811
    root, docs = gold
    out = tmp_path / "out"
    cfg = chunk_cfg("gold.chunks.split.eval=0.3", "gold.chunks.split.dpo=0.3")
    st = build_gold_chunks(cfg, root, out, workers=0)
    assert sum(st["docs"].values()) == 8 and not st["no_truth"]
    rows = {s: _rows(out / s) for s in ("sft", "dpo", "eval")}
    assert sum(bool(v) for v in rows.values()) >= 2
    seen: dict[str, str] = {}
    for side, rs in rows.items():
        for r in rs:
            assert seen.setdefault(r["doc_id"], side) == side == r["split"]
    split = json.loads((out / "split.json").read_text())
    assert {d for s in ("sft", "dpo", "eval") for d in split[f"{s}_docs"]} == set(seen)

    by_doc: dict[str, list[dict]] = {}
    for rs in rows.values():
        for r in rs:
            by_doc.setdefault(r["source_path"], []).append(r)
    for path, cs in by_doc.items():
        cs.sort(key=lambda c: c["chunk_index"])
        assert len(cs) > 1
        if path == "v1/d7.md":
            assert {c["gold_status"] for c in cs} >= {"placeholders_changed"}
            continue
        assert {c["gold_status"] for c in cs} == {"ok"}
        # lossless: the pieces with the OCR separators give back the truth file
        rebuilt = "".join(
            c["prefix"] + restore(c["gold_text"], c["gold_placeholders"]) + c["suffix"] for c in cs
        )
        assert rebuilt == docs[path]
        for c in cs:
            assert "charactor" not in c["gold_text"] and "qzxvw" not in c["gold_text"]
            assert c["gold_text"].count("resolution") == c["source_text"].count("qzxvw")

    # a rerun with other shares keeps every document on its side
    cfg2 = chunk_cfg("gold.chunks.split.eval=0.6", "gold.chunks.split.dpo=0.1")
    build_gold_chunks(cfg2, root, out, workers=0)
    assert json.loads((out / "split.json").read_text()) == split


def _system(tmp_path: Path, name: str, rows: list[dict], fn) -> str:
    d = tmp_path / name
    d.mkdir()
    with (d / "part-00000.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r | {"text": fn(r)}, ensure_ascii=False) + "\n")
    return f"{name}={d}"


def test_eval_documents(tmp_path, gold, chunk_cfg):  # noqa: F811
    root, _ = gold
    out = tmp_path / "out"
    build_gold_chunks(chunk_cfg(), root, out, workers=0)
    rows = [r for s in ("sft", "dpo", "eval") for r in _rows(out / s)]
    rows = [r for r in rows if r["source_path"] != "v1/d7.md"]
    perfect = _system(tmp_path, "perfect", rows, lambda r: r["gold_text"])
    guess = _system(tmp_path, "guess", rows,
                    lambda r: r["gold_text"].replace("resolution", "accuracy"))  # fmt: skip
    over = _system(tmp_path, "over", rows,
                   lambda r: r["gold_text"].replace("method", "methd"))  # fmt: skip
    res = run_eval(root / "truth", [perfect, guess, over], tmp_path / "ev", opts=OPTS)
    m = {k: v["all"] for k, v in res["metrics"].items()}
    assert res["docs"] == 7
    assert m["ocr"]["cer"] > 0
    assert m["perfect"]["cer"] == 0 and m["perfect"]["false_fix_rate"] == 0
    assert 0 < m["guess"]["cer"] < m["ocr"]["cer"]
    assert m["over"]["false_fix_rate"] > 0
    assert (tmp_path / "ev" / "report.md").read_text().startswith("# Eval: 7 documents")
    names = {(c["a"], c["b"]) for c in res["comparisons"]}
    assert ("ocr", "perfect") in names and len(names) == 6
