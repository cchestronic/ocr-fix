"""Stand-in for vLLM in smoke runs: an OpenAI-compatible server on 127.0.0.1 that answers
every chat completion with the last user message (the ``<<<n>>>`` paragraphs), after a few
harmless rule-based edits so `verify` has corrections to check:

  - a hyphen at a line or word break inside a lowercase word is joined (``pro- posed``)
  - runs of spaces become one
  - a few common OCR confusions, as whole words (``Tab1e`` -> ``Table``, ``rnodel`` -> ``model``)

It never loads a model, so the whole pipeline can run on a machine without a GPU or vLLM.
Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions`` (``n`` copies).

    python -m ocrfix.llm.echo_server --port 8000 [--port 8001 ...] [--identity]
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_HYPHEN = re.compile(r"(?<=[a-z])-(?:[ \t]*\n[ \t]*|[ \t]+)(?=[a-z])")
_SPACES = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
CONFUSIONS = {"Tab1e": "Table", "rnodel": "model", "tbe": "the", "0f": "of", "Fig.l": "Fig. 1"}
_CONFUSION = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, CONFUSIONS)) + r")(?![\w])")


def edit(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if not line.startswith("<<<"):
            line = _CONFUSION.sub(lambda m: CONFUSIONS[m.group(1)], _SPACES.sub(" ", line))
        out.append(line)
    return _HYPHEN.sub("", "\n".join(out))


def answer(body: dict[str, Any], identity: bool = False) -> dict[str, Any]:
    msgs = body.get("messages") or []
    user = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
    content = user if identity else edit(user)
    n = int(body.get("n") or 1)
    toks = len(user.split())
    return {
        "id": "echo",
        "object": "chat.completion",
        "model": body.get("model", "echo"),
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for i in range(n)
        ],
        "usage": {
            "prompt_tokens": toks,
            "completion_tokens": toks * n,
            "total_tokens": toks * (n + 1),
        },
    }


def make_handler(identity: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": "echo", "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except ValueError:
                self._send(400, {"error": "bad json"})
                return
            self._send(200, answer(body, identity))

        def log_message(self, *args: Any) -> None:  # quiet
            pass

    return Handler


def serve(ports: list[int], identity: bool = False) -> list[ThreadingHTTPServer]:
    """Start one server per port in daemon threads (loopback only); returns them."""
    servers = []
    for port in ports:
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(identity))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    return servers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, action="append", required=True)
    ap.add_argument("--identity", action="store_true", help="Answer the input unchanged.")
    args = ap.parse_args()
    servers = serve(args.port, args.identity)
    print(f"echo LLM on {', '.join(f'127.0.0.1:{p}' for p in args.port)}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    for s in servers:
        s.shutdown()


if __name__ == "__main__":
    main()
