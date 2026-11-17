"""`ocrfix data dpo`: chunk pairs, rejected = the served model's own wrong answers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

from ocrfix.stages.dpo import build_dpo_pairs, classify, constructed, half_fixed

MAP = {"⟨M1⟩": "$x$"}


def test_classify():
    inp = "Tbe model ⟨M1⟩ is goodd."
    gold = "The model ⟨M1⟩ is good."
    assert classify(inp, gold, "Tbe model ⟨M1⟩ is good.", MAP) == "missed"
    assert classify(inp, gold, "The modell ⟨M1⟩ is good.", MAP) == "false_fix"
    assert classify(inp, gold, "Tbe modell ⟨M1⟩ is good.", MAP) == "mixed"
    assert classify(inp, gold, "Thy model ⟨M1⟩ is gold.", MAP) == "wrong_fix"
    assert classify(inp, gold, "The model is good.", MAP) == "placeholders"
    assert classify(inp, gold, "The model ⟨M1⟩ ⟨M1⟩ is good.", MAP) == "placeholders"
    assert classify(gold, gold, "The model ⟨M1⟩ is god.", MAP) == "false_fix"  # identity


def test_constructed_kinds():
    inp, gold = "Tbe modei ⟨M1⟩ is goodd.", "The model ⟨M1⟩ is good."
    rng = random.Random(0)
    h = half_fixed(inp, gold, rng)
    assert h not in (inp, gold) and "⟨M1⟩" in h
    assert constructed("unchanged", inp, gold, rng) == inp
    assert constructed("unchanged", gold, gold, rng) is None
    assert constructed("half_fixed", gold, gold, rng) is None
    assert constructed("placeholder_dropped", inp, gold, rng) == "The model is good."


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def _clean(doc: str, i: int) -> dict:
    return {"chunk_id": f"{doc}-c{i}", "doc_id": doc, "lang": "en", "text": None,
            "source_text": f"Clean text {doc} {i} with ⟨M1⟩ math. Another sentence.",
            "placeholders": MAP}  # fmt: skip


def _bad(row: dict) -> dict:
    gold = row["source_text"]
    return {"chunk_id": row["chunk_id"] + "~table~v0", "doc_id": row["doc_id"], "lang": "en",
            "source_text": gold.replace("Clean", "C1ean"), "placeholders": MAP,
            "gold_text": gold, "gold_placeholders": MAP, "corruption": "table"}  # fmt: skip


class FakeServer:
    """Greedy: fixes d0's chunks, returns d1's unchanged. Sampled: right, a false fix,
    a dropped placeholder, right again. Identity prompts: always the input."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        user = body["messages"][1]["content"]
        fixed = user.replace("C1ean", "Clean")
        if "C1ean" not in user:
            texts = [user] * body.get("n", 1)
        elif body["temperature"] == 0:
            texts = [fixed if " d0 " in user else user]
        else:
            texts = [fixed, fixed.replace("text", "txt"), fixed.replace(" ⟨M1⟩", ""), fixed]
        choices = [{"message": {"content": t}, "finish_reason": "stop"} for t in texts]
        return httpx.Response(200, json={"choices": choices, "usage": {}})


def opts(**kw):
    return {
        "junk": {"prob": 0.0, "max_runs": 1, "chars": [3, 5], "own_line": 0.0},
        "val_frac": 0.0, "empty_think": False, "model": "tuned",
        "dpo": {"n_samples": 4, "temperature": 0.8, "top_p": 0.95, "max_pairs": 2,
                "constructed": ["unchanged", "half_fixed", "placeholder_dropped"],
                "constructed_share": 1.0, "max_prompts": {}},
        "llm": {"seed": 0, "max_tokens_factor": 1.3, "max_tokens_extra": 16,
                "enable_thinking": False, "retries": 0, "backoff": 0.0, "timeout": 5.0},
        "urls": ["http://127.0.0.1:8100"], "concurrency": 4,
    } | kw  # fmt: skip


def test_build_pairs(tmp_path):
    clean = [_clean(f"d{d}", i) for d in range(3) for i in range(2)]
    syn = _write(tmp_path / "corrupt.jsonl", [_bad(r) for r in clean])
    ide = _write(tmp_path / "clean.jsonl", clean[:2])
    # an SFT file that already holds d2's damaged chunks
    sft = _write(tmp_path / "sft.jsonl", [
        {"messages": [{"role": "system", "content": "S"},
                      {"role": "user", "content": _bad(r)["source_text"]},
                      {"role": "assistant", "content": r["source_text"]}]}
        for r in clean if r["doc_id"] == "d2"
    ])  # fmt: skip
    out = tmp_path / "dpo.jsonl"
    fake = FakeServer()
    encode = lambda t: (list(range(len(t) // 4 + 1)), [])  # noqa: E731

    def run():
        return build_dpo_pairs(
            {"synthetic": [syn], "identity": [ide]},
            out,
            opts=opts(),
            seed=0,
            exclude=[sft],
            encode=encode,
            transport=httpx.MockTransport(fake),
        )

    st = run()
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    # d2's 2 damaged chunks are in the SFT file
    assert st.prompts == {"synthetic": 4, "identity": 2} and st.excluded == 2
    assert st.greedy_right == 2 + 2  # d0 synthetic + both identity prompts
    by_chunk: dict[str, list[dict]] = {}
    for r in rows:
        assert r["rejected"][0]["content"] != r["chosen"][0]["content"]
        assert r["prompt"][0]["role"] == "system" and r["prompt"][1]["role"] == "user"
        by_chunk.setdefault(r["chunk_id"], []).append(r)
    # d1: greedy left the damage -> first pair; then the nearest sampled mistake
    d1 = by_chunk["d1-c0~table~v0"]
    assert [r["rejected_type"] for r in d1] == ["greedy:missed", "sampled:false_fix"]
    # d0: greedy right; the two sampled mistakes
    d0 = by_chunk["d0-c0~table~v0"]
    assert sorted(r["rejected_type"] for r in d0) == ["sampled:false_fix", "sampled:placeholders"]
    # identity prompts: every answer right -> one constructed pair (share 1.0)
    ide_rows = [r for r in rows if r["source"] == "identity"]
    assert [r["rejected_type"] for r in ide_rows] == ["constructed:placeholder_dropped"] * 2
    assert st.all_right == 2 and st.pairs == len(rows) == 2 * 2 + 2 * 2 + 2

    # rerun: answers come from the responses file, no new requests
    calls = fake.calls
    st2 = run()
    assert fake.calls == calls and st2.pairs == st.pairs
