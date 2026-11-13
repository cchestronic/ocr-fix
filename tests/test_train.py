"""Smoke tests: build each trainer on the tiny random Qwen3.5 and run 2 steps on fixtures.

The 4-bit (QLoRA) path needs bitsandbytes and CUDA and is skipped without them.
"""

from __future__ import annotations

import json
import shutil

import pytest

from ocrfix.config import load_config
from ocrfix.data.common import EMPTY_THINK
from ocrfix.llm.prompt import SYSTEM

pytest.importorskip("trl")
pytest.importorskip("peft")

# Qwen-style template: <|im_start|>role\ncontent<|im_end|>; with enable_thinking=false the
# generation prompt carries the empty think block, as Qwen3 templates do.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def chat_model_dir(tiny_model_dir, tmp_path_factory):
    from transformers import AutoTokenizer

    d = tmp_path_factory.mktemp("chat_model")
    shutil.copytree(tiny_model_dir, d, dirs_exist_ok=True)
    tok = AutoTokenizer.from_pretrained(d)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    return d


def cfg_for(model_dir, *extra):
    return load_config(
        "rtx5090",
        [
            f"paths.base_model={model_dir}",
            "hw.train.mode=lora",
            "hw.dtype=float32",
            "hw.train.optim=adamw_torch",
            "hw.train.max_seq_len=256",
            "hw.train.per_device_batch_size=2",
            "hw.train.gradient_checkpointing=false",
            "train.effective_batch=4",
            "train.lora.r=4",
            "train.lora.alpha=8",
            "train.eval_frac=0.25",
            "train.logging_steps=1",
            "train.eval_steps=1",
            "train.save_steps=1000",
            "scorer.calibration=null",
            *extra,
        ],
    )


def msgs(user: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM[:80]},
        {"role": "user", "content": f"<<<1>>>\n{user}"},
        {"role": "assistant", "content": f"{EMPTY_THINK}<<<1>>>\n{answer}"},
    ]


@pytest.fixture(scope="module")
def sft_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("sft") / "train.jsonl"
    rows = [
        {
            "id": f"e{i}",
            "messages": msgs(f"Tab1e {i} shows the rnodel.", f"Table {i} shows the model."),
        }
        for i in range(16)
    ]
    rows.append({"id": "long", "messages": msgs("x " * 400, "x " * 400)})  # over max_seq_len
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture(scope="module")
def dpo_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("dpo") / "pairs.jsonl"
    rows = []
    for i in range(12):
        m = msgs(f"Tab1e {i} shows the rnodel.", f"Table {i} shows the model.")
        rows.append({
            "id": f"d{i}", "prompt": m[:2], "chosen": [m[2]],
            "rejected": [{"role": "assistant", "content": EMPTY_THINK + m[1]["content"]}],
            "rejected_type": "unchanged",
        })  # fmt: skip
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_render_targets_assistant_only(chat_model_dir):
    from ocrfix.train.common import load_tokenizer, render

    tok = load_tokenizer(str(chat_model_dir))
    m = msgs("Tab1e", "Table")
    p, c = render(tok, m[:2], m[2]["content"], {"enable_thinking": False})
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert c == "<<<1>>>\nTable<|im_end|>"
    with pytest.raises(ValueError, match="generation prompt"):
        render(tok, m[:2], "<<<1>>>\nTable", {"enable_thinking": False})  # think block missing


def test_resolve_settings():
    from ocrfix.train.common import resolve

    st = resolve(load_config("rtx5090", ["hw.train.packing=false"]))
    assert (st.mode, st.distributed, st.max_len, st.optim) == (
        "qlora",
        "none",
        2048,
        "paged_adamw_8bit",
    )
    assert st.micro_batch * st.grad_accum == 32 and st.gradient_checkpointing
    h = load_config("h200x4", ["hw.train.packing=false"])
    assert resolve(h).distributed == "none"  # one process outside torchrun
    full = resolve(load_config("h200x4", ["hw.train.packing=false", "hw.train.mode=full"]))
    assert full.distributed == "fsdp" and full.fsdp["layer_cls"] == "Qwen3_5DecoderLayer"


def test_packing_needs_kernels(monkeypatch):
    import ocrfix.train.common as common

    monkeypatch.setattr(common, "missing_kernels", lambda: ["fla"])
    with pytest.raises(ValueError, match=r"optional Qwen3\.5 kernels"):
        common.resolve(load_config("h200x4"))


def test_sft_two_steps(chat_model_dir, sft_file, tmp_path):
    from ocrfix.train.sft import train_sft

    out = tmp_path / "sft"
    m = train_sft(cfg_for(chat_model_dir), sft_file, out, max_steps=2)
    assert m["train_loss"] > 0 and "eval_loss" in m
    assert (out / "adapter_config.json").is_file() and (out / "adapter_model.safetensors").is_file()
    meta = json.loads((out / "ocrfix_train.json").read_text())
    assert meta["rows"]["dropped"] == 1 and meta["mode"] == "lora" and not meta["quantized_base"]
    assert meta["settings"]["grad_accum"] == 2 and meta["settings"]["effective_batch"] == 4


@pytest.fixture(scope="module")
def sft_adapter(chat_model_dir, sft_file, tmp_path_factory):
    from ocrfix.train.sft import train_sft

    out = tmp_path_factory.mktemp("adapter")
    train_sft(cfg_for(chat_model_dir), sft_file, out, max_steps=1)
    return out


@pytest.mark.parametrize("reference", ["base", "sft"])
def test_dpo_two_steps(chat_model_dir, dpo_file, sft_adapter, tmp_path, reference):
    from ocrfix.train.dpo import train_dpo

    out = tmp_path / "dpo"
    m = train_dpo(
        cfg_for(chat_model_dir, f"train.dpo.reference={reference}"),
        dpo_file,
        sft_adapter,
        out,
        max_steps=2,
    )
    assert m["reference"] == reference and "train_loss" in m
    assert (out / "adapter_model.safetensors").is_file()


def test_dpo_reference_is_shared_base(chat_model_dir, dpo_file, sft_adapter, monkeypatch):
    """No second model: the trainer gets ref_model=None and a PEFT model."""
    import trl

    seen = {}
    orig = trl.DPOTrainer.__init__

    def spy(self, model, ref_model=None, **kw):
        seen["ref_model"], seen["peft"] = ref_model, type(model).__name__
        kw["args"].max_steps = 1
        orig(self, model, ref_model, **kw)
        seen["adapters"] = sorted(self.accelerator.unwrap_model(self.model).peft_config)

    monkeypatch.setattr(trl.DPOTrainer, "__init__", spy)
    from ocrfix.train.dpo import train_dpo

    train_dpo(
        cfg_for(chat_model_dir), dpo_file, sft_adapter, sft_adapter.parent / "dpo_spy", max_steps=1
    )
    assert seen["ref_model"] is None and seen["peft"].startswith("Peft")
    assert "default" in seen["adapters"]


def test_merge(chat_model_dir, sft_adapter, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from ocrfix.train.merge import merge

    out = tmp_path / "merged"
    info = merge(str(chat_model_dir), sft_adapter, out, dtype="float32")
    assert not info["dequantized"] and (out / "config.json").is_file()
    cfg = json.loads((out / "config.json").read_text())
    assert "quantization_config" not in cfg
    assert any(p.suffix == ".safetensors" for p in out.iterdir())
    merged = AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in n for n, _ in merged.named_parameters())
    # merged weights = base + adapter delta on a LoRA target
    from peft import PeftModel

    ref = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(chat_model_dir), str(sft_adapter)
    )
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        assert torch.allclose(merged(x).logits, ref(x).logits, atol=1e-4)


qlora = pytest.mark.skipif(
    not __import__("torch").cuda.is_available()
    or __import__("importlib").util.find_spec("bitsandbytes") is None,
    reason="QLoRA needs CUDA + bitsandbytes",
)


@qlora
def test_qlora_sft_and_dequantized_merge(chat_model_dir, sft_file, tmp_path):
    from ocrfix.train.merge import merge
    from ocrfix.train.sft import train_sft

    cfg = cfg_for(
        chat_model_dir, "hw.train.mode=qlora", "hw.dtype=bfloat16",
        "hw.train.optim=paged_adamw_8bit", "hw.train.gradient_checkpointing=true",
    )  # fmt: skip
    out = tmp_path / "qlora"
    m = train_sft(cfg, sft_file, out, max_steps=2)
    assert m["peak_vram_gib"] > 0
    assert json.loads((out / "ocrfix_train.json").read_text())["quantized_base"]
    info = merge(str(chat_model_dir), out, tmp_path / "merged")
    assert info["dequantized"] and info["quantized_base"]
    assert "quantization_config" not in json.loads(
        (tmp_path / "merged" / "config.json").read_text()
    )

    # the dequantized merge reproduces the training-time model (nf4 base + adapter)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )  # fmt: skip
    trained = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            chat_model_dir, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0}
        ),
        str(out),
    )
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.bfloat16).cuda()
    x = torch.randint(0, 100, (1, 8), device="cuda")
    with torch.no_grad():
        a, b = trained(x).logits.float(), merged(x).logits.float()
    assert (a - b).abs().max() < 0.1 * a.abs().max()


def test_sft_loss_only_on_assistant_tokens(chat_model_dir, sft_file, tmp_path, monkeypatch):
    """The batches SFTTrainer trains on: loss (labels != -100) covers exactly the assistant
    turn (answer + end of turn), never the system/user prompt or the generation prompt."""
    import trl

    from ocrfix.train.sft import train_sft

    seen = {}
    orig_train = trl.SFTTrainer.train

    def inspect(self, *a, **kw):
        batch = next(iter(self.get_train_dataloader()))
        seen["batch"] = {k: v.cpu() for k, v in batch.items() if hasattr(v, "cpu")}
        seen["tok"] = self.processing_class
        return orig_train(self, *a, **kw)

    monkeypatch.setattr(trl.SFTTrainer, "train", inspect)
    train_sft(cfg_for(chat_model_dir), sft_file, tmp_path / "o", max_steps=1)
    b, tok = seen["batch"], seen["tok"]
    labels, ids = b["labels"], b["input_ids"]
    for row in range(len(ids)):
        mask = labels[row] != -100
        assert mask.any()
        trained = tok.decode(ids[row][mask])
        prompt = tok.decode(ids[row][~mask & (b["attention_mask"][row] == 1)])
        assert trained.startswith("<<<1>>>\nTable ") and trained.endswith(
            "shows the model.<|im_end|>"
        )
        assert "Tab1e" not in trained and "rnodel" not in trained  # the input is not trained on
        assert "<|im_start|>user" in prompt and prompt.endswith("<think>\n\n</think>\n\n")
        # the labelled tokens are one contiguous block at the end of the sequence
        idx = mask.nonzero().flatten()
        assert (idx[1:] - idx[:-1] == 1).all()


@pytest.fixture(scope="module")
def full_sft(chat_model_dir, sft_file, tmp_path_factory):
    from ocrfix.train.sft import train_sft

    out = tmp_path_factory.mktemp("full_sft")
    train_sft(cfg_for(chat_model_dir, "hw.train.mode=full"), sft_file, out, max_steps=1)
    return out


def test_full_sft_writes_a_model(full_sft):
    from ocrfix.train.common import checkpoint_kind

    assert checkpoint_kind(full_sft) == "full"
    assert json.loads((full_sft / "ocrfix_train.json").read_text())["mode"] == "full"


@pytest.mark.parametrize("reference", ["sft", "base"])
def test_dpo_full_mode(chat_model_dir, dpo_file, full_sft, tmp_path, reference):
    import torch
    from transformers import AutoModelForCausalLM

    from ocrfix.train.dpo import train_dpo

    out = tmp_path / "dpo_full"
    cfg = cfg_for(chat_model_dir, "hw.train.mode=full", f"train.dpo.reference={reference}",
                  "train.dpo.lr=1e-2")  # fmt: skip
    m = train_dpo(cfg, dpo_file, full_sft, out, max_steps=2)
    assert m["reference"] == reference and "train_loss" in m
    meta = json.loads((out / "ocrfix_train.json").read_text())
    assert meta["mode"] == "full" and meta["sft_kind"] == "full"
    assert not (out / "adapter_config.json").exists() and (out / "config.json").is_file()
    a = AutoModelForCausalLM.from_pretrained(full_sft)
    b = AutoModelForCausalLM.from_pretrained(out)
    changed = any(
        not torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )
    assert changed  # all weights were trainable


def test_dpo_full_mode_refuses_an_adapter(chat_model_dir, dpo_file, sft_adapter, tmp_path):
    from ocrfix.train.dpo import train_dpo

    with pytest.raises(ValueError, match="train merge"):
        train_dpo(cfg_for(chat_model_dir, "hw.train.mode=full"), dpo_file, sft_adapter,
                  tmp_path / "x", max_steps=1)  # fmt: skip


def test_lora_dpo_on_full_sft_then_merge(chat_model_dir, dpo_file, full_sft, tmp_path):
    """A new LoRA on the full SFT model; merge finds that model as the base."""
    from typer.testing import CliRunner

    from ocrfix.cli import app
    from ocrfix.train.dpo import train_dpo

    out = tmp_path / "dpo_lora"
    train_dpo(cfg_for(chat_model_dir), dpo_file, full_sft, out, max_steps=1)
    meta = json.loads((out / "ocrfix_train.json").read_text())
    assert meta["base_model"] == str(full_sft) and meta["sft_kind"] == "full"
    assert (out / "adapter_config.json").is_file()
    r = CliRunner().invoke(app, [
        "train", "merge", str(out), str(tmp_path / "merged"), "--profile", "rtx5090",
        "--set", f"paths.base_model={chat_model_dir}", "--set", "hw.train.mode=lora",
    ])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert f"base: {full_sft}" in r.output
    r = CliRunner().invoke(app, ["train", "merge", str(full_sft), str(tmp_path / "m2")])
    assert r.exit_code != 0 and "not a LoRA adapter" in r.output


def test_dpo_trains_answer_and_end_of_turn_only(chat_model_dir, dpo_file, sft_adapter, tmp_path,
                                                monkeypatch):  # fmt: skip
    """chosen/rejected end with the template's <|im_end|>; TRL appends nothing after it."""
    import trl

    from ocrfix.train.dpo import train_dpo

    seen = {}
    orig = trl.DPOTrainer.train

    def inspect(self, *a, **kw):
        seen["row"] = self.train_dataset[0]
        seen["tok"] = self.processing_class
        return orig(self, *a, **kw)

    monkeypatch.setattr(trl.DPOTrainer, "train", inspect)
    train_dpo(cfg_for(chat_model_dir), dpo_file, sft_adapter, tmp_path / "d", max_steps=1)
    tok, row = seen["tok"], seen["row"]
    for k in ("chosen_ids", "rejected_ids"):
        ids = row.get(k) or row.get(k.replace("_ids", "_input_ids"))
        if ids is None:
            pytest.skip(f"TRL row has no {k}: {sorted(row)}")
        text = tok.decode(ids)
        assert text.endswith("<|im_end|>") and "<|endoftext|>" not in text
    from transformers import AutoTokenizer

    saved = AutoTokenizer.from_pretrained(tmp_path / "d")
    assert saved.eos_token == "<|endoftext|>"  # the saved tokenizer is unchanged
