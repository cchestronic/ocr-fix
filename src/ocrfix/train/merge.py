"""`ocrfix train merge`: fold a LoRA adapter into a bf16 checkpoint vLLM can load.

Base weights for the merge:
  - adapter trained on a 4-bit base (qlora; read from the adapter's ``ocrfix_train.json``):
    the base is loaded in nf4 exactly as in training and dequantized to bf16, then the
    adapter is merged. That reproduces the training-time model up to bf16 rounding, but the
    weights carry the nf4 quantization error relative to the original bf16 checkpoint.
    ``--no-dequantize`` merges into the original bf16 weights instead; then the merged model
    differs slightly from what was trained (the adapter learned against nf4 weights).
    Either way a warning is logged.
  - lora / full: merged into the ``hw.dtype`` base directly.
The output is a plain bf16 safetensors checkpoint (no quantization_config) plus tokenizer
and generation config: ``ocrfix serve --set correct.model=<out>`` serves it (trained groups).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ocrfix.train.common import META, read_meta, torch_dtype, write_meta

log = logging.getLogger("ocrfix.train")
QLORA_WARNING = (
    "adapter was trained on a 4-bit (nf4) base: the merged bf16 weights differ slightly from "
    "the training-time model ({how}). Evaluate the merged checkpoint, not the adapter."
)


def dequantize_linear4bit(model, dtype) -> int:
    """Replace every bitsandbytes Linear4bit with a plain nn.Linear holding its dequantized
    weight. Returns the number of layers replaced."""
    import bitsandbytes as bnb
    import torch

    n = 0
    for _, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                qw = child.weight
                w = bnb.functional.dequantize_4bit(qw.data, qw.quant_state).to(dtype)
                lin = torch.nn.Linear(
                    child.in_features, child.out_features, bias=child.bias is not None,
                    dtype=dtype, device=w.device,
                )  # fmt: skip
                lin.weight.data.copy_(w)
                if child.bias is not None:
                    lin.bias.data.copy_(child.bias.data.to(dtype))
                setattr(mod, child_name, lin)
                n += 1
    return n


def _forget_quantization(model) -> None:
    """After dequantizing, the model is a plain one: drop transformers' bitsandbytes state so
    it can be moved and saved without a quantization_config."""
    for attr, value in (("hf_quantizer", None), ("quantization_method", None),
                        ("is_quantized", False), ("is_loaded_in_4bit", False)):  # fmt: skip
        if hasattr(model, attr):
            setattr(model, attr, value)
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config


def merge(
    base_model: str,
    adapter: Path,
    out: Path,
    *,
    dtype: str = "bfloat16",
    dequantize: bool | None = None,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = read_meta(adapter)
    if meta.get("mode") == "full" or not (Path(adapter) / "adapter_config.json").is_file():
        raise ValueError(
            f"{adapter} is not a LoRA adapter (full fine-tune?): it is already a complete model; "
            "serve it directly: ocrfix serve --set correct.model=<dir>"
        )
    quantized = bool(meta.get("quantized_base", False))
    dequantize = quantized if dequantize is None else dequantize
    td = torch_dtype(dtype)
    if dequantize:
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "dequantizing the nf4 base needs a GPU (bitsandbytes); or pass --no-dequantize"
            )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, local_files_only=True, dtype=td, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )  # fmt: skip
        n = dequantize_linear4bit(base, td)
        _forget_quantization(base)
        log.info("dequantized %d nf4 layers to %s", n, dtype)
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, dtype=td)
    if quantized:
        how = (
            "merged into the dequantized nf4 base"
            if dequantize
            else "merged into the original bf16 base"
        )
        log.warning(QLORA_WARNING.format(how=how))

    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()
    _forget_quantization(model)
    model = model.to("cpu")  # weights are already in `dtype`
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tok_src = adapter if (adapter / "tokenizer_config.json").is_file() else Path(base_model)
    AutoTokenizer.from_pretrained(str(tok_src), local_files_only=True).save_pretrained(str(out))
    for name in ("generation_config.json", "chat_template.jinja", "preprocessor_config.json"):
        if (Path(base_model) / name).is_file() and not (out / name).is_file():
            shutil.copy2(Path(base_model) / name, out / name)
    info = {
        "stage": "merged", "base_model": base_model, "adapter": str(adapter), "dtype": dtype,
        "quantized_base": quantized, "dequantized": bool(dequantize), "adapter_meta": meta,
    }  # fmt: skip
    write_meta(out, info)
    return info


__all__ = ["META", "merge"]
