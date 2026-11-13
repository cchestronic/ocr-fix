"""Test-wide guards: offline env, no network, tiny random models only."""

from __future__ import annotations

import math
import socket
from pathlib import Path

import pytest

from ocrfix.offline import enforce_offline

enforce_offline()

FIXTURES = Path(__file__).parent / "fixtures"
# Text-only Qwen3.5 (hybrid: Gated DeltaNet linear-attention + full-attention layers).
TINY_MODEL_TYPE = "qwen3_5_text"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost") or self.family == socket.AF_UNIX:
            return real_connect(self, address)
        raise RuntimeError(f"network access attempted in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="session")
def model_device() -> str:
    """Device for tiny-model tests. With flash-linear-attention installed, transformers routes
    Qwen3.5's linear-attention layers to Triton kernels regardless of device, so CPU fails."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        return "cuda"
    try:
        import fla  # noqa: F401
    except ImportError:
        return "cpu"
    pytest.skip("flash-linear-attention is installed: Qwen3.5 needs CUDA")


def build_tiny_model(out: Path, layer_types: list[str]) -> Path:
    """A tiny randomly initialized Qwen3.5 + byte-level BPE tokenizer saved to ``out``."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch = pytest.importorskip("torch")

    corpus = [
        line
        for p in sorted((FIXTURES / "sample_mineru").rglob("*.md"))
        for line in p.read_text(encoding="utf-8").splitlines()
    ]
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        tokenizers.trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    hf_tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    config = transformers.AutoConfig.for_model(
        TINY_MODEL_TYPE,
        vocab_size=len(hf_tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=4096,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # Random init decays linear-attention state almost fully every token, which would hide
    # state leaking across packed paragraphs. Trained models keep state long; mimic that.
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "A_log"):
                module.A_log.fill_(math.log(1e-3))
    model.save_pretrained(out)
    hf_tok.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Hybrid like the real model: one linear-attention and one full-attention layer."""
    return build_tiny_model(
        tmp_path_factory.mktemp("tiny_model"), ["linear_attention", "full_attention"]
    )


@pytest.fixture(scope="session")
def tiny_attn_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Full-attention layers only: packable without the optional linear-attention kernels."""
    return build_tiny_model(
        tmp_path_factory.mktemp("tiny_attn_model"), ["full_attention", "full_attention"]
    )
