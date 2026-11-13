"""`ocrfix train dpo`: TRL DPOTrainer on top of the SFT result, in any ``hw.train.mode``.

The SFT result (``sft``) is either a LoRA adapter (from qlora / lora SFT) or a full model
directory (from full SFT, or ``ocrfix train merge``). The policy starts as the SFT model:

  mode   sft = adapter                           sft = full model dir
  qlora  nf4 base + the adapter, trainable        that model in nf4 + a new LoRA
  lora   hw.dtype base + the adapter, trainable   that model + a new LoRA
  full   refused: merge first (`train merge`)     that model, all weights trainable

Reference (``train.dpo.reference``):
  LoRA modes share the base weights, no second copy:
    base  the adapter disabled: the base model, or the full SFT model when ``sft`` is one
    sft   a frozen copy of the SFT adapter (TRL's ``ref`` adapter); with a new LoRA on a
          full SFT model this is the same as ``base``
  full mode, with ``train.dpo.precompute_ref`` (default) the reference log-probs are computed
  once before training, so no reference copy stays in memory (under FSDP TRL loads one for
  that pass and frees it):
    sft   the starting policy (the SFT model)
    base  ``paths.base_model``, loaded as a separate frozen model

Rows are ``ocrfix data dpo`` output, rendered like SFT: prompt = served generation prompt,
chosen / rejected = the assistant turn ending with the template's end-of-turn token, which
is TRL's EOS here, so nothing is appended to it.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from ocrfix.io import iter_jsonl
from ocrfix.train.common import (
    checkpoint_kind,
    is_main,
    load_base,
    load_tokenizer,
    lora_config,
    read_meta,
    render,
    resolve,
    settings_dict,
    training_kwargs,
    turn_end,
    vram_callback,
    write_meta,
)
from ocrfix.train.sft import held_out

log = logging.getLogger("ocrfix.train")
REFERENCES = ("base", "sft")


def rendered_pairs(
    path: Path, tok, template_kwargs: dict[str, Any], max_len: int, limit: int | None
):
    rows, dropped = [], 0
    for i, d in enumerate(iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        p, chosen = render(tok, d["prompt"], d["chosen"][-1]["content"], template_kwargs)
        _, rejected = render(tok, d["prompt"], d["rejected"][-1]["content"], template_kwargs)
        n = len(tok(p + max(chosen, rejected, key=len), add_special_tokens=False)["input_ids"]) + 1
        if n > max_len:
            dropped += 1
            continue
        rows.append(
            {"prompt": p, "chosen": chosen, "rejected": rejected, "id": str(d.get("id", i))}
        )
    return rows, dropped


def build_policy(cfg, st, sft: Path):
    """(model, base dir the weights come from, sft kind)."""
    kind = checkpoint_kind(sft)
    if st.mode == "full":
        if kind == "adapter":
            raise ValueError(
                f"{sft} is a LoRA adapter; full-mode DPO needs a full model: run "
                f"`ocrfix train merge {sft} <dir>` and pass <dir>, or use hw.train.mode=lora"
            )
        return load_base(str(sft), st), str(sft), kind
    from peft import PeftModel, get_peft_model

    if kind == "adapter":
        meta = read_meta(sft)
        base_dir = str(meta.get("base_model") or cfg.paths.base_model)
        if meta.get("quantized_base") is not None and meta["quantized_base"] != (
            st.mode == "qlora"
        ):
            log.warning(
                "SFT adapter was trained with mode=%s, DPO runs with mode=%s",
                meta.get("mode"), st.mode,
            )  # fmt: skip
        base = load_base(base_dir, st)
        return PeftModel.from_pretrained(base, str(sft), is_trainable=True), base_dir, kind
    base = load_base(str(sft), st)
    return get_peft_model(base, lora_config(cfg, base)), str(sft), kind


def train_dpo(
    cfg,
    data: Path,
    sft: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    max_steps: int = -1,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    st = resolve(cfg)
    d = cfg.train.dpo
    reference = str(d.reference)
    if reference not in REFERENCES:
        raise ValueError(f"train.dpo.reference must be one of {REFERENCES}, got {reference!r}")
    tok = load_tokenizer(cfg.paths.tokenizer)
    tk = {"enable_thinking": bool(cfg.llm.enable_thinking)}
    rows, dropped = rendered_pairs(data, tok, tk, st.max_len, limit)
    frac = float(cfg.train.eval_frac)
    ev = [r for r in rows if held_out(r["id"], frac)][: int(cfg.train.max_eval)]
    ids = {r["id"] for r in ev}
    train = [r for r in rows if r["id"] not in ids]
    if not train:
        raise ValueError(f"{data}: no training pairs left (dropped {dropped} over max_seq_len)")
    log.info(
        "dpo: %d train, %d eval, %d dropped; %s", len(train), len(ev), dropped, settings_dict(st)
    )

    model, base_dir, kind = build_policy(cfg, st, sft)
    full = st.mode == "full"
    ref_model = None
    if full and reference == "base":
        ref_model = load_base(str(cfg.paths.base_model), st)
        ref_model.requires_grad_(False)

    # TRL appends tok.eos_token to chosen/rejected unless they end with it; make the
    # template's end-of-turn token the EOS for training (the saved tokenizer is unchanged)
    train_tok = tok
    end = turn_end(tok, [r[k] for r in train for k in ("chosen", "rejected")])
    if end and end != tok.eos_token:
        train_tok = copy.deepcopy(tok)
        train_tok.eos_token = end

    keys = ("prompt", "chosen", "rejected")
    args = DPOConfig(
        **training_kwargs(
            cfg,
            st,
            out_dir,
            epochs=float(d.epochs),
            lr=float(d.lr),
            max_steps=max_steps,
            has_eval=bool(ev),
        ),
        beta=float(d.beta),
        loss_type=[str(d.loss_type)],
        max_length=st.max_len,
        precompute_ref_log_probs=full and bool(d.get("precompute_ref", True)),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=Dataset.from_list([{k: r[k] for k in keys} for r in train]),
        eval_dataset=Dataset.from_list([{k: r[k] for k in keys} for r in ev]) if ev else None,
        processing_class=train_tok,
        callbacks=[vram := vram_callback()],
    )
    if not full:
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        if reference == "base" and "ref" in unwrapped.peft_config:
            # TRL adds a frozen copy of the adapter as reference; without it, the reference
            # log-probs come from the model with adapters disabled, i.e. the base
            unwrapped.delete_adapter("ref")
            unwrapped.set_adapter("default")
    result = trainer.train()
    metrics = dict(result.metrics) | {"peak_vram_gib": round(vram.peak, 2), "reference": reference}
    if ev:
        metrics |= trainer.evaluate()
    trainer.save_model(str(out_dir))
    if is_main():
        tok.save_pretrained(str(out_dir))
        write_meta(out_dir, {
            "stage": "dpo", "mode": st.mode, "base_model": base_dir,
            "quantized_base": st.mode == "qlora", "sft": str(sft), "sft_kind": kind,
            "profile": str(cfg.hw.name), "settings": settings_dict(st),
            "rows": {"train": len(train), "eval": len(ev), "dropped": dropped}, "metrics": metrics,
        })  # fmt: skip
        (out_dir / "train_summary.json").write_text(
            json.dumps(metrics, indent=1, default=str) + "\n"
        )
    return metrics
