"""Token masks and the scoring model (batching modes give the same bits)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ocrfix.scorer.model import is_hybrid, load_model, score_sequences  # noqa: E402
from ocrfix.scorer.tokens import token_masks  # noqa: E402

# --- masks ------------------------------------------------------------------------------


def test_token_masks():
    # masked text: "ab ⟨M1⟩ cd ef gh ij ⟨T1⟩ kl"; placeholders at [3, 7) and [20, 24)
    phs = [(3, 7), (20, 24)]
    tokens = [
        (0, 2), (2, 3),  # "ab", " "
        (3, 5), (5, 7),  # placeholder pieces
        (7, 10), (10, 13), (13, 16), (16, 19),  # 4 prose tokens after it
        (19, 21),  # straddles into the table placeholder
        (21, 24),  # inside it
        (24, 27),  # prose after the table
        (27, 27),  # empty offsets: not prose
    ]  # fmt: skip
    starts, ends = (np.array(x, dtype=np.int32) for x in zip(*tokens, strict=True))
    prose, boundary = token_masks(starts, ends, phs, boundary=3)
    assert prose.tolist() == [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
    assert boundary.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    assert (prose & ~boundary).tolist() == [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


# --- packed vs padded -------------------------------------------------------------------


def _seqs(vocab: int, lengths: list[int], seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(4, vocab, size=n).astype(np.int32) for n in lengths]


LENGTHS = [1, 5, 17, 3, 64, 40, 2, 33, 9, 120]


def _bits(model, seqs, mode, max_tokens=160, chunk=1024):
    return score_sequences(
        model, seqs, mode=mode, max_tokens=max_tokens, logits_chunk=chunk,
        eos_id=model.config.eos_token_id,
    )  # fmt: skip


def _max_diff(a, b) -> float:
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b, strict=True))


def test_packed_equals_padded(tiny_attn_model_dir):
    model = load_model(str(tiny_attn_model_dir), "float32", "sdpa", torch.device("cpu"))
    assert not is_hybrid(model.config)
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    packed = _bits(model, seqs, "packed")
    padded = _bits(model, seqs, "padded")
    solo = [_bits(model, [s], "padded")[0] for s in seqs]  # batch of one: no packing, no pads
    assert [len(b) for b in packed] == LENGTHS
    assert _max_diff(packed, padded) < 1e-3
    assert _max_diff(padded, solo) < 1e-3
    # chunked lm_head/cross-entropy gives the same numbers
    assert _max_diff(_bits(model, seqs, "padded", chunk=7), padded) < 1e-4
    assert all((b > 0).all() for b in padded)


def test_hybrid_padded_equals_solo(tiny_model_dir, model_device):
    """Right padding is exact for linear-attention (recurrent + conv) layers too."""
    model = load_model(str(tiny_model_dir), "float32", "sdpa", torch.device(model_device))
    assert is_hybrid(model.config)
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    padded = _bits(model, seqs, "padded")
    solo = [_bits(model, [s], "padded")[0] for s in seqs]
    assert _max_diff(padded, solo) < 1e-3


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not all(importlib.util.find_spec(m) for m in ("fla", "causal_conv1d")),
    reason="packing a hybrid model needs CUDA + the optional kernels (fla, causal_conv1d)",
)
def test_hybrid_packed_equals_padded(tiny_model_dir):
    model = load_model(str(tiny_model_dir), "float32", "sdpa", torch.device("cuda"))
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    assert _max_diff(_bits(model, seqs, "packed"), _bits(model, seqs, "padded")) < 1e-3
