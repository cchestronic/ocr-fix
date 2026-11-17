"""Async client for vLLM's OpenAI-compatible server(s).

Requests go round-robin over ``urls`` (one per data-parallel engine), at most
``concurrency`` in flight. Connection errors, timeouts, 429 and 5xx are retried (next URL,
exponential backoff); other 4xx are not. Each result is appended to a responses JSONL as
soon as it arrives, keyed by ``chunk_id``; a rerun skips chunk_ids that already succeeded
(the last line per chunk_id wins), so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ocrfix.io import iter_jsonl, repair_jsonl_tail

log = logging.getLogger(__name__)
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class Result:
    chunk_id: str
    ok: bool
    text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    url: str | None = None
    attempts: int = 0
    error: str | None = None
    # every choice when the request asked for n > 1: [{"text", "finish_reason"}]
    choices: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Client:
    def __init__(
        self,
        urls: list[str],
        *,
        concurrency: int = 16,
        retries: int = 3,
        backoff: float = 2.0,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not urls and transport is not None:
            # an in-process transport (correct.backend: transformers) answers every request:
            # no server is contacted, so a machine without engines (hw.infer.engines: 0) works
            urls = ["http://in-process"]
        if not urls:
            raise ValueError("no server URLs (hw.infer.score_urls / correct_urls)")
        self.urls = [u.rstrip("/") for u in urls]
        self._next = itertools.cycle(range(len(self.urls)))
        self.concurrency = concurrency
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.transport = transport

    def next_url(self) -> str:
        return self.urls[next(self._next)]

    async def post(
        self, http: httpx.AsyncClient, path: str, body: dict
    ) -> tuple[dict[str, Any] | None, str | None, str | None, int]:
        """(json, url, error, attempts): one request with retries on the next URL."""
        err = ""
        url: str | None = None
        for attempt in range(1, self.retries + 2):
            url = self.next_url()
            try:
                r = await http.post(f"{url}{path}", json=body)
                if r.status_code == 200:
                    return r.json(), url, None, attempt
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in RETRY_STATUS:
                    return None, url, err, attempt
            except (httpx.TransportError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"
            if attempt <= self.retries:
                log.warning("%s: attempt %d on %s failed (%s); retrying", path, attempt, url, err)
                await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
        return None, None, err, self.retries + 1

    async def complete(self, http: httpx.AsyncClient, chunk_id: str, body: dict) -> Result:
        d, url, err, attempts = await self.post(http, "/v1/chat/completions", body)
        if d is None:
            return Result(chunk_id, ok=False, url=url, attempts=attempts, error=err)
        try:
            choice = d["choices"][0]
            finish = choice.get("finish_reason")
            choices = [
                {"text": c["message"].get("content") or "", "finish_reason": c.get("finish_reason")}
                for c in d["choices"]
            ]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            return Result(chunk_id, ok=False, url=url, attempts=attempts,
                          error=f"bad response: {type(e).__name__}: {e}")  # fmt: skip
        return Result(
            chunk_id,
            ok=finish != "length",
            text=text,
            finish_reason=finish,
            usage=d.get("usage"),
            url=url,
            attempts=attempts,
            error="truncated (finish_reason=length)" if finish == "length" else None,
            choices=choices if len(choices) > 1 else None,
        )

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout, transport=self.transport,
            limits=httpx.Limits(max_connections=self.concurrency * 2),
        )  # fmt: skip

    async def run(self, requests: Iterable[tuple[str, dict]], responses: Path) -> dict[str, int]:
        sem = asyncio.Semaphore(self.concurrency)
        stats = {"ok": 0, "failed": 0}
        t0 = time.perf_counter()
        limits = httpx.Limits(max_connections=self.concurrency * 2)
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self.transport, limits=limits
        ) as http:
            with open(responses, "a", encoding="utf-8") as fh:

                async def one(cid: str, body: dict) -> None:
                    async with sem:
                        res = await self.complete(http, cid, body)
                    fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                    fh.flush()
                    stats["ok" if res.ok else "failed"] += 1
                    n = stats["ok"] + stats["failed"]
                    if n % 100 == 0:
                        log.info("%d chunks done (%.1f/s)", n, n / (time.perf_counter() - t0))

                await asyncio.gather(*(one(cid, body) for cid, body in requests))
        return stats


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    """Last response per chunk_id (a retried chunk appends a newer line)."""
    if not path.exists():
        return {}
    repair_jsonl_tail(path)
    return {d["chunk_id"]: d for d in iter_jsonl(path)}


def run_requests(
    client: Client, requests: list[tuple[str, dict]], responses: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Send the requests whose chunk_id has no final response yet; return all responses.

    Final = succeeded, or truncated (greedy decoding would truncate again)."""
    responses.parent.mkdir(parents=True, exist_ok=True)
    done = {
        k for k, v in load_responses(responses).items()
        if v.get("ok") or v.get("finish_reason") == "length"
    }  # fmt: skip
    todo = [(cid, body) for cid, body in requests if cid not in done]
    stats = {"skipped": len(requests) - len(todo), "ok": 0, "failed": 0}
    if todo:
        stats |= asyncio.run(client.run(todo, responses))
    return load_responses(responses), stats
