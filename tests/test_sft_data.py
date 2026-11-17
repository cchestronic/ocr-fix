"""`ocrfix data sft`: chunk SFT examples in the form `correct` sends."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.stages.data import add_junk, build_sft_data, junk_run
from ocrfix.stages.prompt import messages

FORBIDDEN = re.compile(r"[$|`\\<>\[\]⟨⟩〈〉〈〉#*_]")
JUNK = {"prob": 1.0, "max_runs": 3, "chars": [3, 30], "own_line": 0.3}


def opts(**kw):
    return {
        "mix": {"synthetic": 0.75, "identity": 0.25},
        "lang_balance": None,
        "total": None,
        "junk": JUNK,
        "val_frac": 0.0,
        "empty_think": False,
    } | kw


def test_junk_runs_are_plain_text():
    rng = random.Random(0)
    for _ in range(500):
        run = junk_run(rng, 3, 30)
        assert 3 <= sum(not c.isspace() for c in run) <= 30
        assert not FORBIDDEN.search(run) and run == run.strip()


def test_add_junk_only_inserts():
    text = "First sentence. Second one ⟨M1⟩ here.\n\nNext paragraph! End."
    for seed in range(50):
        noisy, runs = add_junk(text, random.Random(seed), JUNK)
        assert 1 <= len(runs) <= 3
        rest = noisy
        for r in runs:
            assert r in rest
            rest = (
                rest.replace(" " + r, "", 1) if " " + r in rest else rest.replace("\n\n" + r, "", 1)
            )
        assert rest == text
        assert PLACEHOLDER.findall(noisy) == ["⟨M1⟩"]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def _clean(doc: str, i: int, lang: str = "en") -> dict:
    text = f"Clean text {doc} {i} with ⟨M1⟩ math. Another sentence."
    return {"chunk_id": f"{doc}-c{i:05d}", "doc_id": doc, "lang": lang, "text": None,
            "source_text": text, "placeholders": {"⟨M1⟩": "$x$"}}  # fmt: skip


def _corrupt(row: dict, *, same: bool = True) -> dict:
    gold = row["source_text"]
    return {
        "chunk_id": row["chunk_id"] + "~table~v0", "doc_id": row["doc_id"], "lang": row["lang"],
        "source_text": gold.replace("Clean", "C1ean").replace("Another", "Anoth er"),
        "placeholders": row["placeholders"] if same else {"⟨M1⟩": "$x$", "⟨M2⟩": "$이$"},
        "gold_text": gold, "gold_placeholders": row["placeholders"],
        "corruption": "table", "error_rate": 2.0,
    }  # fmt: skip


def test_build_examples(tmp_path):
    clean = [_clean(f"d{d}", i, "ko" if d % 2 else "en") for d in range(6) for i in range(4)]
    bad = [_corrupt(r) for r in clean] + [_corrupt(_clean("d9", 0), same=False)]
    # rows of a gold eval document are never training data
    bad = [r | {"split": "eval"} if r["doc_id"] == "d5" else r for r in bad]
    clean = [r | {"split": "eval"} if r["doc_id"] == "d5" else r for r in clean]
    syn = _write(tmp_path / "corrupt.jsonl", bad)
    ids = tmp_path / "clean"
    ids.mkdir()
    _write(ids / "part-00000.jsonl", clean)
    o = opts(junk=JUNK | {"prob": 0.5})
    st = build_sft_data({"synthetic": [syn], "identity": [ids]}, tmp_path / "out", opts=o, seed=1)

    assert st.dropped["synthetic"] == {"placeholders_changed": 1, "eval_doc": 4}
    assert st.dropped["identity"] == {"eval_doc": 4}
    assert st.pool == {"synthetic": {"en": 12, "ko": 8}, "identity": {"en": 12, "ko": 8}}
    # 75 / 25: synthetic is the binding pool (total 26, each cell rounded down)
    assert st.taken["synthetic"] == {"en": 11, "ko": 7}
    assert st.taken["identity"] == {"en": 3, "ko": 2}
    rows = [json.loads(x) for x in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert len(rows) == st.train == 23 and st.val == 0
    assert {r["doc_id"] for r in rows} <= {f"d{d}" for d in range(5)}
    gold = {r["chunk_id"]: r for r in bad}
    for r in rows:
        sys_user = r["messages"][:2]
        answer = r["messages"][2]
        assert answer["role"] == "assistant" and "<think>" not in answer["content"]
        target = answer["content"]
        user = sys_user[1]["content"]
        assert sys_user == messages(user)
        if r["source"] == "identity":
            assert user == target and "junk" not in r
        else:
            g = gold[r["chunk_id"]]
            assert target == g["gold_text"] and r["corruption"] == "table"
            for run in r.get("junk", []):
                assert run in user and run not in target
            if "junk" not in r:
                assert user == g["source_text"]
    assert st.with_junk == sum("junk" in r for r in rows) > 0

    # deterministic
    build_sft_data({"synthetic": [syn], "identity": [ids]}, tmp_path / "again", opts=o, seed=1)
    assert (tmp_path / "again" / "train.jsonl").read_text() == (
        tmp_path / "out" / "train.jsonl"
    ).read_text()


def test_split_by_document(tmp_path):
    clean = [_clean(f"d{d:02d}", i) for d in range(40) for i in range(3)]
    ids = tmp_path / "clean.jsonl"
    _write(ids, clean)
    st = build_sft_data({"identity": [ids]}, tmp_path / "out", opts=opts(val_frac=0.3), seed=0)
    out = {}
    for side in ("train", "val"):
        rows = [
            json.loads(x) for x in (tmp_path / "out" / f"{side}.jsonl").read_text().splitlines()
        ]
        out[side] = {r["doc_id"] for r in rows}
    assert out["train"] and out["val"] and not out["train"] & out["val"]
    assert st.train + st.val == 120


def _gold(row: dict, status: str = "ok", split: str = "sft") -> dict:
    gold = row["source_text"]
    return {**row, "source_text": row["source_text"].replace("Clean", "C1ean"),
            "gold_text": gold, "gold_placeholders": row["placeholders"],
            "gold_status": status, "split": split}  # fmt: skip


def test_gold_source(tmp_path):
    clean = [_clean(f"d{d}", i) for d in range(3) for i in range(2)]
    rows = [_gold(r) for r in clean] + [_gold(_clean("d7", 0), "cut_in_edit"),
                                        _gold(_clean("d8", 0), split="eval")]  # fmt: skip
    g = _write(tmp_path / "gold.jsonl", rows)
    o = opts(mix={"gold": 1.0}, gold_repeats=2)
    st = build_sft_data({"gold": [g]}, tmp_path / "out", opts=o, seed=0)
    assert st.dropped["gold"] == {"cut_in_edit": 1, "eval_doc": 1}
    out = [json.loads(x) for x in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert len(out) == 12 and sorted({r["repeat"] for r in out}) == [0, 1]
    for r in out:
        assert r["source"] == "gold" and "junk" not in r
        assert "Clean" in r["messages"][2]["content"]
        assert "C1ean" in r["messages"][1]["content"]
