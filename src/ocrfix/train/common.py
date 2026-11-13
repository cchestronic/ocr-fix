"""Shared training setup: resolved hardware settings, model + LoRA loading, chat rendering,
peak-VRAM logging. Everything hardware-specific comes from ``cfg.hw.train``.

Chat rendering: examples are rendered with the tokenizer's chat template exactly as vLLM
renders the inference prompt (``add_generation_prompt=True`` plus
``chat_template_kwargs.enable_thinking = correct.enable_thinking``). The training target is the
rest of the full conversation's rendering (assistant content and end-of-turn token), so
loss falls on assistant tokens only and the model learns to stop.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocrfix.config import missing_kernels

log = logging.getLogger("ocrfix.train")
META = "ocrfix_train.json"


@dataclass
class TrainSettings:
    mode: str  # qlora | lora | full
    distributed: str  # none | ddp | fsdp
    world: int
    micro_batch: int
    grad_accum: int
    max_len: int
    packing: bool
    optim: str
    gradient_checkpointing: bool
    dtype: str
    bf16: bool
    fsdp: dict[str, Any] | None

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum * self.world


def resolve(cfg, *, mode: str | None = None) -> TrainSettings:
    import torch

    hw = cfg.hw.train
    mode = mode or str(hw.mode)
    world = int(os.environ.get("WORLD_SIZE", 1))
    dist = str(hw.get("distributed", "none"))
    if dist == "auto":
        dist = "fsdp" if mode == "full" else ("ddp" if world > 1 else "none")
    if world == 1 and dist == "ddp":
        dist = "none"
    if mode == "qlora" and dist == "fsdp":
        raise ValueError("qlora with FSDP is not supported here; use ddp or a single GPU")
    micro = int(hw.per_device_batch_size)
    accum = max(1, math.ceil(int(cfg.train.effective_batch) / (micro * world)))
    packing = bool(hw.get("packing", False))
    if packing and (missing := missing_kernels()):
        raise ValueError(
            f"hw.train.packing needs the optional Qwen3.5 kernels; missing: {', '.join(missing)} "
            "(scripts/install_offline.py --with-kernels, or --set hw.train.packing=false)"
        )
    dtype = str(cfg.hw.dtype)
    fsdp = None
    if dist == "fsdp":
        f = hw.get("fsdp") or {}
        fsdp = {
            "sharding": str(f.get("sharding", "full_shard auto_wrap")),
            "layer_cls": f.get("layer_cls"),
        }
    return TrainSettings(
        mode=mode, distributed=dist, world=world, micro_batch=micro, grad_accum=accum,
        max_len=int(hw.max_seq_len), packing=packing, optim=str(hw.get("optim", "adamw_torch")),
        gradient_checkpointing=bool(hw.gradient_checkpointing), dtype=dtype,
        bf16=dtype == "bfloat16" and torch.cuda.is_available(), fsdp=fsdp,
    )  # fmt: skip


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if not tok.chat_template:
        raise ValueError(f"{path}: tokenizer has no chat template")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(path: str, st: TrainSettings):
    """The base model as training needs it: 4-bit nf4 (qlora) or ``hw.dtype`` weights."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"local_files_only": True, "dtype": torch_dtype(st.dtype)}
    if st.mode == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # bitsandbytes places quantized weights at load time: one device per rank
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = False
    if st.mode == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=st.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )  # fmt: skip
    return model


def turn_end(tok, completions: list[str]) -> str | None:
    """The special token every rendered completion ends with (the chat template's end of
    turn, e.g. ``<|im_end|>``). TRL appends ``eos_token`` to a completion that doesn't end
    with it; passing this token as the EOS keeps it from training an extra token (such as
    ``<|endoftext|>`` after ``<|im_end|>``) that inference never produces."""
    special = {*tok.all_special_tokens, *(t.content for t in tok.added_tokens_decoder.values())}
    ends = [t for t in special if completions and all(c.endswith(t) for c in completions)]
    return max(ends, key=len) if ends else None


def checkpoint_kind(path: Path) -> str:
    """``adapter`` (a LoRA dir) or ``full`` (a model dir: full fine-tune or merged)."""
    path = Path(path)
    if (path / "adapter_config.json").is_file():
        return "adapter"
    if (path / "config.json").is_file():
        return "full"
    raise ValueError(
        f"{path}: neither a LoRA adapter (adapter_config.json) nor a model (config.json)"
    )


def lora_config(cfg, model):
    import torch
    from peft import LoraConfig

    present = {n.split(".")[-1] for n, m in model.named_modules() if _is_linear(m, torch)}
    wanted = list(cfg.train.lora.target_modules)
    targets = [t for t in wanted if t in present]
    if missing := sorted(set(wanted) - present):
        log.info("LoRA targets not in this model (skipped): %s", ", ".join(missing))
    if not targets:
        raise ValueError(f"none of train.lora.target_modules {wanted} are in the model")
    return LoraConfig(
        r=int(cfg.train.lora.r), lora_alpha=int(cfg.train.lora.alpha),
        lora_dropout=float(cfg.train.lora.dropout), target_modules=targets,
        bias="none", task_type="CAUSAL_LM",
    )  # fmt: skip


def _is_linear(m, torch) -> bool:
    if isinstance(m, torch.nn.Linear):
        return True
    return type(m).__name__ in ("Linear4bit", "Linear8bitLt")


def render(
    tok, prompt: list[dict[str, str]], answer: str, template_kwargs: dict[str, Any]
) -> tuple[str, str]:
    """(prompt text as served, completion text = answer + end of turn)."""
    p = tok.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    full = tok.apply_chat_template(
        [*prompt, {"role": "assistant", "content": answer}], tokenize=False, **template_kwargs
    )
    if not full.startswith(p):
        raise ValueError(
            "chat template: the conversation with the answer does not start with the generation "
            "prompt (e.g. the template adds an empty think block at generation time that the "
            "answer lacks); check sft.empty_think against correct.enable_thinking"
        )
    # the template closes the turn with "<end-of-turn>\n"; the newline is not a target
    return p, full[len(p) :].rstrip("\n")


def training_kwargs(
    cfg,
    st: TrainSettings,
    out_dir: Path,
    *,
    epochs: float,
    lr: float,
    max_steps: int,
    has_eval: bool,
) -> dict[str, Any]:
    t = cfg.train
    kw: dict[str, Any] = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": st.micro_batch,
        "per_device_eval_batch_size": st.micro_batch,
        "gradient_accumulation_steps": st.grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": str(t.scheduler),
        "warmup_steps": float(t.warmup_ratio),  # transformers 5: a float < 1 is a ratio
        "weight_decay": float(t.weight_decay),
        "max_grad_norm": float(t.max_grad_norm),
        "optim": st.optim,
        "bf16": st.bf16,
        "logging_steps": int(t.logging_steps),
        "save_steps": int(t.save_steps),
        "save_total_limit": 3,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": int(t.eval_steps) if has_eval else None,
        "report_to": "none",
        "seed": int(cfg.seed),
        "use_cpu": not _cuda(),
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
        "gradient_checkpointing": st.gradient_checkpointing and st.distributed != "fsdp",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    if st.distributed == "ddp":
        kw["ddp_find_unused_parameters"] = False
    if st.distributed == "fsdp" and st.fsdp:
        kw["fsdp"] = st.fsdp["sharding"]
        kw["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": [st.fsdp["layer_cls"]]
            if st.fsdp["layer_cls"]
            else None,
            "activation_checkpointing": st.gradient_checkpointing,
            "use_orig_params": True,
        }
    return kw


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def vram_callback():
    from transformers import TrainerCallback

    class PeakVram(TrainerCallback):
        """Adds peak_vram_gib (this rank) to every log line and the final summary."""

        peak = 0.0

        def on_log(self, args, state, control, logs=None, **kw):
            import torch

            if torch.cuda.is_available() and logs is not None:
                self.peak = max(self.peak, torch.cuda.max_memory_allocated() / 2**30)
                logs["peak_vram_gib"] = round(self.peak, 2)

        def on_train_end(self, args, state, control, **kw):
            import torch

            if torch.cuda.is_available():
                self.peak = max(self.peak, torch.cuda.max_memory_allocated() / 2**30)
                log.info("peak VRAM %.2f GiB (rank-local, allocated)", self.peak)

    return PeakVram()


def write_meta(out_dir: Path, info: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META).write_text(json.dumps(info, indent=1, default=str) + "\n", encoding="utf-8")


def read_meta(path: Path) -> dict[str, Any]:
    p = path / META
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def settings_dict(st: TrainSettings) -> dict[str, Any]:
    return asdict(st) | {"effective_batch": st.effective_batch}


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0
