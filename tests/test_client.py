"""The vLLM HTTP client: round-robin over URLs, retries, resumable response files."""

from __future__ import annotations

import json
from collections import Counter

import httpx

from ocrfix.llm.client import Client, load_responses, run_requests

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
