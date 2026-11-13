"""`ocrfix train sft`: TRL SFTTrainer, LoRA / QLoRA / full by ``hw.train.mode``.

Input rows are ``ocrfix data sft`` examples (``messages``: system, user, assistant). They are
rendered to prompt/completion strings (``common.render``) so the loss is on the assistant
turn only (TRL ``completion_only_loss``; a test checks the labelled tokens of real batches
decode to exactly the answer plus the end-of-turn token, which is passed as TRL's
``eos_token`` so nothing is appended after it). Rows longer than ``hw.train.max_seq_len`` tokens
are dropped (counted), never truncated mid-answer. Eval: ``--eval`` file, else a held-out
slice (``train.eval_frac`` by example id, at most ``train.max_eval``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ocrfix.io import iter_jsonl
from ocrfix.train.common import (
    is_main,
    load_base,
    load_tokenizer,
    lora_config,
    render,
    resolve,
    settings_dict,
    training_kwargs,
    turn_end,
    vram_callback,
    write_meta,
)

log = logging.getLogger("ocrfix.train")


def held_out(example_id: str, frac: float) -> bool:
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) / 16**8 < frac


def rendered_rows(
    path: Path, tok, template_kwargs: dict[str, Any], max_len: int, limit: int | None
):
    rows, dropped = [], 0
    for i, d in enumerate(iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        msgs = d["messages"]
        prompt, answer = msgs[:-1], msgs[-1]
        if answer["role"] != "assistant":
            raise ValueError(f"{path}: row {i}: last message must be the assistant turn")
        p, c = render(tok, prompt, answer["content"], template_kwargs)
        n = len(tok(p + c, add_special_tokens=False)["input_ids"]) + 1
        if n > max_len:
            dropped += 1
            continue
        rows.append({"prompt": p, "completion": c, "id": str(d.get("id", i))})
    return rows, dropped


def train_sft(
    cfg,
    data: Path,
    out_dir: Path,
    *,
    eval_data: Path | None = None,
    limit: int | None = None,
    max_steps: int = -1,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    st = resolve(cfg)
    tok = load_tokenizer(cfg.model)
    tk = {"enable_thinking": bool(cfg.correct.enable_thinking)}
    rows, dropped = rendered_rows(data, tok, tk, st.max_len, limit)
    if eval_data is not None:
        ev, _ = rendered_rows(eval_data, tok, tk, st.max_len, None)
        train = rows
    else:
        frac = float(cfg.train.eval_frac)
        ev = [r for r in rows if held_out(r["id"], frac)][: int(cfg.train.max_eval)]
        ids = {r["id"] for r in ev}
        train = [r for r in rows if r["id"] not in ids]
    if not train:
        raise ValueError(f"{data}: no training rows left (dropped {dropped} over max_seq_len)")
    log.info(
        "sft: %d train, %d eval, %d dropped (> %d tokens); %s",
        len(train),
        len(ev),
        dropped,
        st.max_len,
        settings_dict(st),
    )

    model = load_base(cfg.model, st)
    peft_cfg = None if st.mode == "full" else lora_config(cfg, model)
    t = cfg.train
    args = SFTConfig(
        **training_kwargs(
            cfg,
            st,
            out_dir,
            epochs=float(t.sft.epochs),
            lr=float(t.lr),
            max_steps=max_steps,
            has_eval=bool(ev),
        ),
        max_length=st.max_len,
        packing=st.packing,
        completion_only_loss=True,
        eos_token=turn_end(tok, [r["completion"] for r in train]),
        train_sampling_strategy=(
            "group_by_length" if bool(t.group_by_length) and not st.packing else "random"
        ),
        dataset_num_proc=None,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(
            [{k: r[k] for k in ("prompt", "completion")} for r in train]
        ),
        eval_dataset=Dataset.from_list([{k: r[k] for k in ("prompt", "completion")} for r in ev])
        if ev
        else None,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[vram := vram_callback()],
    )
    result = trainer.train()
    metrics = dict(result.metrics) | {"peak_vram_gib": round(vram.peak, 2)}
    if ev:
        metrics |= trainer.evaluate()
    trainer.save_model(str(out_dir))
    if is_main():
        tok.save_pretrained(str(out_dir))
        write_meta(out_dir, {
            "stage": "sft", "mode": st.mode, "base_model": str(cfg.model),
            "quantized_base": st.mode == "qlora", "profile": str(cfg.hw.name),
            "settings": settings_dict(st),
            "rows": {"train": len(train), "eval": len(ev), "dropped": dropped},
            "metrics": metrics,
        })  # fmt: skip
        (out_dir / "train_summary.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics
