"""GPU side of scoring: load the LM, pick packed vs padded batching, per-token NLL in bits.

Every sequence (a window of paragraphs, ``tokens``) is ``[EOS] + token ids``; hidden state
``i`` predicts token ``i``, so every token gets a surprisal and no sequence sees another's
tokens.

* packed (flash_attention_2): sequences concatenated into one row of <= max_batch_tokens,
  position_ids reset per sequence, ``cu_seq_lens_*`` for block-diagonal varlen attention.
  Hybrid models (Qwen3.5 linear-attention layers) also get ``cu_seq_lens_q`` (fla) and
  ``seq_idx`` (causal-conv1d) so recurrent and conv state reset at every boundary; without
  those kernels transformers cannot keep packed sequences apart, so we don't pack.
* padded (sdpa): sequences sorted by length, right-padded, rows*max_len <= max_batch_tokens.
  Right padding is exact for causal layers: real tokens never see the pads after them.

NLL is computed from final hidden states through ``lm_head`` in chunks of ``logits_chunk``
tokens, so fp32 logits never exist for the whole batch.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

LN2 = math.log(2.0)
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def is_hybrid(config) -> bool:
    return "linear_attention" in (getattr(config, "layer_types", None) or [])


def choose_mode(attn_impl: str, config) -> tuple[str, str, str]:
    """(mode, attn_implementation, reason). Packed only when it is exact and fast."""
    if attn_impl != "flash_attention_2":
        return "padded", "sdpa", f"profile attn_impl={attn_impl}"
    if not _installed("flash_attn"):
        return "padded", "sdpa", "flash_attn not importable"
    if is_hybrid(config) and not _installed("fla", "causal_conv1d"):
        return "padded", "sdpa", "linear-attention layers need fla + causal_conv1d to pack"
    return "packed", "flash_attention_2", "flash_attn varlen"


def load_model(path: str, dtype: str, attn_implementation: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=_DTYPES[dtype],
        attn_implementation=attn_implementation,
        local_files_only=True,
    )
    return model.to(device).eval()


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, T]
    kwargs: dict  # attention_mask / position_ids / cu_seq_lens / seq_idx
    rows: list[tuple[int, int, int]]  # (seq index, row, start col) of each sequence's hidden 0
    lengths: list[int]  # number of paragraph tokens per sequence (targets)


def _plan_padded(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] + 1)
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in order:  # ascending: the newest element is always the longest in the group
        width = lengths[i] + 1
        if cur and width * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _plan_packed(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """First-fit decreasing into bins of max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        need = lengths[i] + 1
        for b, r in enumerate(room):
            if need <= r:
                bins[b].append(i)
                room[b] -= need
                break
        else:
            bins.append([i])
            room.append(max_tokens - need)
    return bins


def make_batches(
    seqs: Sequence[np.ndarray], mode: str, max_tokens: int, eos_id: int, hybrid: bool = False
) -> Iterator[Batch]:
    lengths = [len(s) for s in seqs]
    if max(lengths, default=0) + 1 > max_tokens:
        raise ValueError(f"sequence of {max(lengths)} tokens exceeds max_batch_tokens")
    plan = _plan_packed if mode == "packed" else _plan_padded
    for group in plan(lengths, max_tokens):
        if mode == "packed":
            yield _packed_batch(seqs, group, eos_id, hybrid)
        else:
            yield _padded_batch(seqs, group, eos_id)


def _packed_batch(seqs, group, eos_id, hybrid) -> Batch:
    ids, pos, seq_idx, rows, cu = [], [], [], [], [0]
    for k, i in enumerate(group):
        s = seqs[i]
        rows.append((i, 0, cu[-1]))
        ids += [eos_id, *s.tolist()]
        pos += range(len(s) + 1)
        seq_idx += [k] * (len(s) + 1)
        cu.append(cu[-1] + len(s) + 1)
    cu_t = torch.tensor(cu, dtype=torch.int32)
    max_len = int((cu_t[1:] - cu_t[:-1]).max())
    kwargs = {
        "position_ids": torch.tensor([pos]),
        "cu_seq_lens_q": cu_t,
        "cu_seq_lens_k": cu_t,
        "max_length_q": max_len,
        "max_length_k": max_len,
    }
    if hybrid:
        kwargs["seq_idx"] = torch.tensor([seq_idx], dtype=torch.int32)
    return Batch(torch.tensor([ids]), kwargs, rows, [len(seqs[i]) for i in group])


def _padded_batch(seqs, group, eos_id) -> Batch:
    width = max(len(seqs[i]) for i in group) + 1
    ids = torch.full((len(group), width), eos_id, dtype=torch.long)
    mask = torch.zeros((len(group), width), dtype=torch.long)
    rows = []
    for r, i in enumerate(group):
        s = torch.from_numpy(seqs[i].astype(np.int64))
        ids[r, 1 : len(s) + 1] = s
        mask[r, : len(s) + 1] = 1
        rows.append((i, r, 0))
    return Batch(ids, {"attention_mask": mask}, rows, [len(seqs[i]) for i in group])


def _backbone_and_head(model):
    backbone = getattr(model, "model", None) or model.get_decoder()
    return backbone, model.get_output_embeddings()


@torch.inference_mode()
def batch_bits(model, batch: Batch, logits_chunk: int) -> list[np.ndarray]:
    """Per-token surprisal (bits) for every sequence in the batch, in ``batch.rows`` order."""
    device = next(model.parameters()).device
    backbone, head = _backbone_and_head(model)
    kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.kwargs.items()}
    input_ids = batch.input_ids.to(device)
    hidden = backbone(input_ids=input_ids, use_cache=False, **kwargs).last_hidden_state

    sel_rows, sel_cols, sel_tgt = [], [], []
    for (_, r, c), n in zip(batch.rows, batch.lengths, strict=True):
        sel_rows.append(torch.full((n,), r, dtype=torch.long))
        sel_cols.append(torch.arange(c, c + n))
        sel_tgt.append(batch.input_ids[r, c + 1 : c + 1 + n])
    rows_t = torch.cat(sel_rows).to(device)
    cols_t = torch.cat(sel_cols).to(device)
    targets = torch.cat(sel_tgt).to(device)
    h = hidden[rows_t, cols_t]  # [N, H]
    del hidden

    out = torch.empty(len(targets), dtype=torch.float32, device=device)
    for a in range(0, len(targets), logits_chunk):
        b = min(a + logits_chunk, len(targets))
        logits = head(h[a:b]).float()
        out[a:b] = F.cross_entropy(logits, targets[a:b], reduction="none")
        del logits
    bits = (out / LN2).cpu().numpy()
    return list(np.split(bits, np.cumsum(batch.lengths)[:-1]))


def score_sequences(
    model,
    seqs: Sequence[np.ndarray],
    *,
    mode: str,
    max_tokens: int,
    logits_chunk: int,
    eos_id: int,
) -> list[np.ndarray]:
    """Bits per token for each sequence, in input order."""
    out: list[np.ndarray | None] = [None] * len(seqs)
    hybrid = is_hybrid(model.config)
    for batch in make_batches(seqs, mode, max_tokens, eos_id, hybrid):
        for (i, _, _), bits in zip(batch.rows, batch_bits(model, batch, logits_chunk), strict=True):
            out[i] = bits
    return out  # type: ignore[return-value]


def load_scorer(cfg, device):
    """(model, mode, start-token id) for the base model (``model``) on ``device``."""
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(cfg.model, local_files_only=True)
    mode, attn, why = choose_mode(cfg.hw.attn_impl, model_cfg)
    log.info("scorer mode=%s attn=%s (%s)", mode, attn, why)
    model = load_model(cfg.model, cfg.hw.dtype, attn, device)
    eos_id = model.config.eos_token_id
    eos_id = eos_id[0] if isinstance(eos_id, list) else eos_id
    return model, mode, eos_id
