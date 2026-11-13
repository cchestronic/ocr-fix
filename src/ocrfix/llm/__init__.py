"""LLM correction through vLLM's OpenAI-compatible server(s) (``scripts/serve_vllm.sh``).

protect.py (⟨M#⟩ ⟨T#⟩ ⟨P#⟩ masking), prompt.py (system instruction, markers, request body),
client.py (round-robin async client, retries, resumable responses), parse.py (markers ->
paragraphs, placeholders restored), correct.py (``ocrfix correct``).
"""

from ocrfix.llm.client import Client, Result, load_responses, run_requests
from ocrfix.llm.parse import ParaResult, parse_chunk
from ocrfix.llm.prompt import SYSTEM, Chunk, ChunkPara, max_tokens, messages, request_body
from ocrfix.llm.protect import mask

__all__ = [
    "SYSTEM",
    "Chunk",
    "ChunkPara",
    "Client",
    "ParaResult",
    "Result",
    "load_responses",
    "mask",
    "max_tokens",
    "messages",
    "parse_chunk",
    "request_body",
    "run_requests",
]
