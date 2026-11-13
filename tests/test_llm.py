from __future__ import annotations

import json
import math
from collections import Counter

import httpx
import pytest

from ocrfix.config import load_config
from ocrfix.corpus import tag_spans
from ocrfix.corpus.placeholders import restore
from ocrfix.llm import SYSTEM, Chunk, ChunkPara, Client, mask, messages, parse_chunk, request_body
from ocrfix.llm.client import load_responses, run_requests
from ocrfix.llm.correct import build_chunks, max_input_budget, run_correct
from ocrfix.llm.prompt import MARKER, user_message
from ocrfix.schema import Record

PROTECT = ["images", "code", "urls", "emails", "dois", "citations"]
LLM_CFG = {
    "max_tokens_factor": 1.3, "max_tokens_extra": 16, "enable_thinking": False, "seed": 0,
}  # fmt: skip


def count(s: str) -> int:
    """Stand-in tokenizer: whitespace tokens."""
    return len(s.split())


def rec(text: str, i: int = 0, doc: str = "d") -> Record:
    return Record(doc, f"p{i:05d}", i, "en", text, tag_spans(text), "x.md")


# --- masking and prompt ----------------------------------------------------------------------


def test_mask_protects_and_round_trips():
    text = (
        "See [12, 13] and https://example.org/a_b. Run `pip install x` with $E=mc^2$, "
        "mail a@b.co, doi 10.1000/xyz123, ![fig](img/1.png).\n<table><tr><td>1</td></tr></table>"
    )
    masked, mapping = mask(text, tag_spans(text), PROTECT)
    assert masked == ("See ⟨P1⟩ and ⟨P2⟩. Run ⟨P3⟩ with ⟨M1⟩, mail ⟨P4⟩, doi ⟨P5⟩, ⟨P6⟩.\n⟨T1⟩")
    assert mapping["⟨P2⟩"] == "https://example.org/a_b"  # trailing period stays prose
    assert mapping["⟨P3⟩"] == "`pip install x`" and mapping["⟨T1⟩"].startswith("<table>")
    assert list(mapping) == ["⟨P1⟩", "⟨P2⟩", "⟨P3⟩", "⟨M1⟩", "⟨P4⟩", "⟨P5⟩", "⟨P6⟩", "⟨T1⟩"]
    assert restore(masked, mapping) == text
    # a URL inside code is one ⟨P#⟩ (the code); only selected kinds are masked
    assert mask("`http://x.org/y`", tag_spans("`http://x.org/y`"), PROTECT)[0] == "⟨P1⟩"
    assert mask("see [3]", tag_spans("see [3]"), ["urls"])[0] == "see [3]"


def chunk_of(*texts: str, doc: str = "d") -> Chunk:
    paras = []
    for i, t in enumerate(texts):
        m, mp = mask(t, tag_spans(t), PROTECT)
        paras.append(ChunkPara(doc, f"p{i:05d}", m, mp))
    ch = Chunk(f"{doc}:test", doc, paras)
    ch.input_tokens = count(user_message(ch))
    return ch


def test_prompt_rendering():
    ch = chunk_of("Tab1e 1 shows $x$.", "본 연구 에서는 [3] 을 사용하였다.")
    msgs = messages(ch)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM
    assert "⟨M1⟩ ⟨T1⟩ ⟨P1⟩" in SYSTEM and "〈?〉" in SYSTEM and MARKER.format(1) in SYSTEM
    assert (
        msgs[1]["content"]
        == "<<<1>>>\nTab1e 1 shows ⟨M1⟩.\n\n<<<2>>>\n본 연구 에서는 ⟨P1⟩ 을 사용하였다."
    )
    body = request_body(ch, model="qwen3.5-9b", llm_cfg=LLM_CFG)
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["n"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == math.ceil(1.3 * ch.input_tokens) + 16
    assert body["messages"] == msgs and body["model"] == "qwen3.5-9b"
    on = request_body(ch, model="m", llm_cfg=LLM_CFG | {"enable_thinking": True})
    assert on["chat_template_kwargs"]["enable_thinking"] is True


def test_prompt_renders_with_chat_template(tiny_model_dir):
    """The no-think flag reaches the template (Qwen templates read enable_thinking)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    tok.chat_template = (
        "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
        "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}"
    )
    body = request_body(chunk_of("a b"), model="m", llm_cfg=LLM_CFG)
    text = tok.apply_chat_template(body["messages"], tokenize=False, **body["chat_template_kwargs"])
    assert text.startswith("<|system|>You correct OCR errors") and text.endswith("<think></think>")
    assert "<|user|><<<1>>>\na b" in text


# --- chunking --------------------------------------------------------------------------------


def test_budget_fits_model_len():
    n = max_input_budget(4096, 300, 1.3, 16, 64)
    assert n + math.ceil(1.3 * n) + 16 + 300 + 64 <= 4096
    m = n + 1
    assert m + math.ceil(1.3 * m) + 16 + 300 + 64 > 4096 - 1


def test_chunking():
    paras = [rec(" ".join(["w"] * 10), i) for i in range(7)] + [rec(" ".join(["w"] * 80), 7)]
    chunks, errors = build_chunks(
        paras, count=count, max_input_tokens=40, max_paragraphs=3, protect=PROTECT
    )
    # 12 tokens per paragraph (10 words + marker + 2): 3 fit in 40; the 80-word one can't
    assert [c.para_ids for c in chunks] == [
        ["p00000", "p00001", "p00002"], ["p00003", "p00004", "p00005"], ["p00006"],
    ]  # fmt: skip
    assert errors == {("d", "p00007"): "too_long"}
    assert all(c.input_tokens == count(user_message(c)) for c in chunks)
    # max_paragraphs caps too
    chunks2, _ = build_chunks(
        paras[:7], count=count, max_input_tokens=1000, max_paragraphs=2, protect=PROTECT
    )
    assert [len(c.paras) for c in chunks2] == [2, 2, 2, 1]
    # ids are stable, and change when the text changes
    again, _ = build_chunks(
        paras, count=count, max_input_tokens=40, max_paragraphs=3, protect=PROTECT
    )
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]
    edited = [rec("x " * 10, 0), *paras[1:]]
    other, _ = build_chunks(
        edited, count=count, max_input_tokens=40, max_paragraphs=3, protect=PROTECT
    )
    assert other[0].chunk_id != chunks[0].chunk_id and other[1].chunk_id == chunks[1].chunk_id
    assert chunks[0].chunk_id.startswith("d:")


# --- parsing ---------------------------------------------------------------------------------


def test_parse_ok_and_restores():
    ch = chunk_of("Tab1e 1 shows $x$ [3].", "Second")
    out = "<<<1>>>\nTable 1 shows ⟨M1⟩ ⟨P1⟩.\n\n<<<2>>>\nSecond."
    r1, r2 = parse_chunk(ch, out)
    assert r1.ok and r1.text == "Table 1 shows $x$ [3]." and r2.ok and r2.text == "Second."
    assert r1.notes == []


@pytest.mark.parametrize(
    ("output", "errors", "note"),
    [
        # lost marker 2: paragraph 1 swallowed paragraph 2, so both fail
        ("<<<1>>>\nA ⟨M1⟩\nB\n<<<3>>>\nC", ["marker_missing_next", "marker_missing", None], None),
        ("<<<1>>>\nA ⟨M1⟩\n<<<2>>>\nB\nC", [None, "marker_missing_next", "marker_missing"], None),
        ("A ⟨M1⟩ B C", ["marker_missing"] * 3, "text before first marker"),  # all lost
        (
            "<<<1>>>\nA ⟨M1⟩\n<<<1>>>\nB\n<<<2>>>\nB\n<<<3>>>\nC",
            ["marker_duplicate", None, None],
            None,
        ),
        ("<<<1>>>\nA\n<<<2>>>\nB\n<<<3>>>\nC", ["placeholders", None, None], None),  # ⟨M1⟩ lost
        ("<<<1>>>\nA ⟨M2⟩\n<<<2>>>\nB\n<<<3>>>\nC", ["placeholders", None, None], None),
        ("<<<1>>>\nA ⟨M1⟩\n<<<2>>>\n\n<<<3>>>\nC", [None, "empty", None], None),
        ("<<<1>>>\nA ⟨M1⟩\n<<<2>>>\n<<<3>>>\nC", [None, "empty", None], None),
        ("<<<2>>>\nB\n<<<1>>>\nA ⟨M1⟩\n<<<3>>>\nC", [None, None, None], "markers out of order"),
        (
            "<<<1>>>\nA ⟨M1⟩\n<<<2>>>\nB\n<<<3>>>\nC\n<<<4>>>\nD",
            [None, None, None],
            "unknown markers [4]",
        ),
        ("<think>hmm</think>\n<<<1>>>\nA ⟨M1⟩\n<<<2>>>\nB\n<<<3>>>\nC", [None, None, None], None),
    ],
)
def test_parse_errors(output, errors, note):
    ch = chunk_of("A $x$", "B", "C")
    res = parse_chunk(ch, output)
    got = [None if r.ok else r.error.split(":")[0] for r in res]
    assert got == errors
    if note:
        assert note in res[0].notes


def test_parse_single_paragraph_without_marker_and_request_error():
    (r,) = parse_chunk(chunk_of("Tab1e"), "Table")
    assert r.ok and r.text == "Table" and "recovered: no marker" in r.notes
    res = parse_chunk(chunk_of("a", "b"), None, "HTTP 500: boom")
    assert [x.error for x in res] == ["HTTP 500: boom"] * 2


# --- client (mocked server) ------------------------------------------------------------------

URLS = [f"http://127.0.0.1:{8000 + i}" for i in range(4)]


def completion(content: str, finish: str = "stop") -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def fixing_server(hits: Counter, fail_first: dict | None = None):
    """Echo every paragraph with 'Tab1e' -> 'Table'. Scripted failures: 503 for the first
    n requests per URL, 400 on 'FAIL400', truncation on 'TRUNCATE', marker 2 dropped on
    'DROP2'."""
    fail_first = dict(fail_first or {})

    def handler(request: httpx.Request) -> httpx.Response:
        url = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        hits[url] += 1
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        if fail_first.get(url, 0) > 0:
            fail_first[url] -= 1
            return httpx.Response(503, text="overloaded")
        if "FAIL400" in user:
            return httpx.Response(400, text="bad request")
        if "TRUNCATE" in user:
            return httpx.Response(200, json=completion("<<<1>>>\npart", "length"))
        content = user.replace("Tab1e", "Table")
        if "DROP2" in user:
            content = content.replace("<<<2>>>\n", "")
        return httpx.Response(200, json=completion(content))

    return httpx.MockTransport(handler)


def client(transport, **kw) -> Client:
    return Client(URLS, concurrency=3, retries=2, backoff=0.0, transport=transport, **kw)


def test_round_robin_over_urls(tmp_path):
    hits: Counter = Counter()
    reqs = [(f"c{i}", {"messages": [{}, {"content": f"<<<1>>>\nTab1e {i}"}]}) for i in range(8)]
    got, stats = run_requests(client(fixing_server(hits)), reqs, tmp_path / "r.jsonl")
    assert stats == {"skipped": 0, "ok": 8, "failed": 0}
    assert hits == {u: 2 for u in URLS}
    assert {got[f"c{i}"]["text"] for i in range(8)} == {f"<<<1>>>\nTable {i}" for i in range(8)}
    assert Counter(r["url"] for r in got.values()) == {u: 2 for u in URLS}


def test_retry_resume_and_final_errors(tmp_path):
    hits: Counter = Counter()
    transport = fixing_server(hits, fail_first={URLS[0]: 1})
    path = tmp_path / "r.jsonl"
    reqs = [
        ("ok", {"messages": [{}, {"content": "<<<1>>>\nTab1e"}]}),
        ("bad", {"messages": [{}, {"content": "FAIL400"}]}),
        ("cut", {"messages": [{}, {"content": "TRUNCATE"}]}),
    ]
    got, stats = run_requests(client(transport), reqs, path)
    assert got["ok"]["ok"] and got["ok"]["attempts"] == 2  # 503 on the first URL, then next one
    assert (
        not got["bad"]["ok"] and got["bad"]["attempts"] == 1 and "HTTP 400" in got["bad"]["error"]
    )
    assert not got["cut"]["ok"] and got["cut"]["finish_reason"] == "length"
    assert stats["ok"] == 1 and stats["failed"] == 2

    # rerun: ok and truncated are final; the 400 is retried (and fails again)
    before = sum(hits.values())
    got, stats = run_requests(client(transport), reqs, path)
    assert stats["skipped"] == 2 and sum(hits.values()) == before + 1
    # a torn last line (crash mid-write) is dropped, not fatal
    with open(path, "a") as f:
        f.write('{"chunk_id": "ok", "ok": tr')
    assert set(load_responses(path)) == {"ok", "bad", "cut"}


def test_all_retries_fail(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")

    got, stats = run_requests(
        client(httpx.MockTransport(down)),
        [("x", {"messages": [{}, {"content": ""}]})],
        tmp_path / "r.jsonl",
    )
    assert (
        stats["failed"] == 1 and got["x"]["attempts"] == 3 and "ConnectError" in got["x"]["error"]
    )


# --- ocrfix correct end to end ---------------------------------------------------------------


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


@pytest.fixture
def corpus(tmp_path):
    texts = {
        "a": ["Tab1e 1 has $x_1$ and [4].", "Clean paragraph.", "Another Tab1e here."],
        "b": ["Tab1e in doc b with `code`.", "Fine.", "Tab1e again", "DROP2 Tab1e last"],
        "c": ["FAIL400 Tab1e"],
    }
    recs, scores = [], []
    for doc, ts in texts.items():
        for i, t in enumerate(ts):
            r = rec(t, i, doc)
            recs.append(r.to_dict() | {"original_text": t + " (pre-fix)", "fixer": ["unicode"]})
            scores.append({"doc_id": doc, "para_id": r.para_id, "lang": "en", "flag": "Tab1e" in t})
    write_jsonl(tmp_path / "paras.jsonl", recs)
    write_jsonl(tmp_path / "scores.jsonl", scores)
    return tmp_path


def test_correct_end_to_end(corpus, tiny_model_dir):
    cfg = load_config(
        "h200x4", ["llm.retries=0", "llm.backoff=0", f"paths.base_model={tiny_model_dir}"]
    )
    hits: Counter = Counter()
    out = corpus / "corrected.jsonl"

    def go():
        return run_correct(
            cfg, corpus / "paras.jsonl", corpus / "scores.jsonl", out, count=count,
            transport=fixing_server(hits),
        )  # fmt: skip

    st = go()
    assert st.selected == 6 and st.chunks == 3 and st.sent == 3
    rows = {
        (r["doc_id"], r["para_id"]): r for r in map(json.loads, out.read_text("utf-8").splitlines())
    }
    assert len(rows) == 8
    a0 = rows[("a", "p00000")]
    assert a0["text"] == "Table 1 has $x_1$ and [4]." and a0["fixer"] == ["unicode", "llm"]
    assert a0["original_text"] == "Tab1e 1 has $x_1$ and [4]. (pre-fix)"  # kept from fix
    assert [s["kind"] for s in a0["spans"]] == ["prose", "math", "prose"]
    assert rows[("a", "p00002")]["text"] == "Another Table here."
    # unselected paragraphs pass through untouched
    assert rows[("a", "p00001")]["text"] == "Clean paragraph."
    assert "llm_error" not in rows[("a", "p00001")] and rows[("a", "p00001")]["fixer"] == [
        "unicode"
    ]
    # doc b: marker 2 lost -> paragraph 1 swallowed it; both keep their text; 3 is fine
    b = {k[1]: v for k, v in rows.items() if k[0] == "b"}
    assert b["p00000"]["llm_error"] == "marker_missing_next"
    assert b["p00002"]["llm_error"] == "marker_missing"
    assert (
        b["p00000"]["text"] == "Tab1e in doc b with `code`."
        and b["p00002"]["text"] == "Tab1e again"
    )
    assert b["p00003"]["text"] == "DROP2 Table last"
    assert rows[("c", "p00000")]["llm_error"].startswith("HTTP 400")
    assert st.corrected == 3
    assert st.errors == {"marker_missing_next": 1, "marker_missing": 1, "HTTP 400": 1}
    assert sum(hits.values()) == 3 and len(hits) == 3  # round robin across engines

    # resume: successful chunks are not sent again; the failed (HTTP 400) one is
    st2 = go()
    assert st2.sent == 1 and sum(hits.values()) == 4


def test_correct_selection(corpus, tmp_path):
    cfg = load_config("rtx5090", ["llm.retries=0", "llm.backoff=0", "scorer.calibration=null"])
    hits: Counter = Counter()
    st = run_correct(
        cfg, corpus / "paras.jsonl", corpus / "scores.jsonl", tmp_path / "all.jsonl",
        select_all=True, count=count, transport=fixing_server(hits),
    )  # fmt: skip
    assert st.selected == 8 and set(hits) == {URLS[0]}
    # no flag column and no calibration file -> refuse
    write_jsonl(tmp_path / "noflag.jsonl", [{"doc_id": "a", "para_id": "p00000", "lang": "en"}])
    with pytest.raises(ValueError, match="no `flag`"):
        run_correct(cfg, corpus / "paras.jsonl", tmp_path / "noflag.jsonl", tmp_path / "x.jsonl",
                    count=count, transport=fixing_server(hits))  # fmt: skip


PROFILES = sorted(
    p.stem
    for p in (__import__("pathlib").Path(__file__).parents[1] / "configs/profiles").glob("*.yaml")
)


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_urls_match_engines(profile):
    inf = load_config(profile).hw.infer
    assert list(inf.urls) == [
        f"http://127.0.0.1:{inf.base_port + i}" for i in range(inf.data_parallel_size)
    ]
    assert inf.tensor_parallel_size == 1


@pytest.mark.parametrize(("profile", "n", "max_len"), [("rtx5090", 1, 4096), ("h200x4", 4, 8192)])
def test_serve_script_dry_run(profile, n, max_len):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = os.environ | {"PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}
    env.pop("OCRFIX_ATTENTION_BACKEND", None)
    res = subprocess.run(
        ["bash", str(root / "scripts/serve_vllm.sh"), "--profile", profile, "--dry-run"],
        capture_output=True, text=True, env=env, check=True,
    )  # fmt: skip
    cmds = [x for x in res.stdout.splitlines() if "vllm serve" in x]
    assert len(cmds) == n
    for i, c in enumerate(cmds):
        assert c.startswith(f"CUDA_VISIBLE_DEVICES={i} vllm serve ")
        assert f"--port {8000 + i}" in c and f"--max-model-len {max_len}" in c
        assert "--enable-prefix-caching" in c and "--tensor-parallel-size 1" in c
        assert "--dtype bfloat16" in c
        backend = "FLASHINFER" if profile == "rtx5090" else "FLASH_ATTN"
        assert c.endswith(f"--attention-backend {backend}")
    if profile == "rtx5090":
        assert "--gpu-memory-utilization 0.9 " in cmds[0]
