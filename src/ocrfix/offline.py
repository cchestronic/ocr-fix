"""Force offline mode for every HF / vLLM library. Import before any of them."""

from __future__ import annotations

import os

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def enforce_offline() -> None:
    """Set offline env vars unconditionally (not setdefault: never allow opting back in)."""
    os.environ.update(OFFLINE_ENV)
