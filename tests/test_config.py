"""configs/default.yaml + configs/stages/*.yaml + profile + --set; derived values; serve."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.config import config_dir, correct_model, load_base, load_config, served_name

# what every hardware profile must define (README, "Adding a hardware profile")
PROFILE_KEYS = [
    "model", "hw.name", "hw.n_gpus", "hw.dtype", "hw.attn_impl", "hw.vllm_attention_backend",
    "hw.train.mode", "hw.train.per_device_batch_size", "hw.train.max_seq_len",
    "hw.train.gradient_checkpointing", "hw.train.optim", "hw.train.packing",
    "hw.train.distributed", "hw.infer.engines", "hw.infer.tensor_parallel_size",
    "hw.infer.gpu_memory_utilization", "hw.infer.max_model_len", "hw.infer.max_num_seqs",
    "hw.infer.host", "hw.infer.base_port", "hw.infer.correct_port", "hw.infer.concurrency",
    "hw.scorer.max_batch_tokens", "hw.scorer.logits_chunk",
]  # fmt: skip
PROFILES = sorted(p.stem for p in (Path(__file__).parents[1] / "configs/profiles").glob("*.yaml"))


def test_every_stage_file_loads_and_overrides_work():
    stages = sorted((config_dir() / "stages").glob("*.yaml"))
    names = {p.stem for p in stages}
    assert names == {"chunk", "score", "gold", "corrupt", "calibrate", "sft", "dpo", "correct",
                     "eval", "reconstruct", "markup", "pipeline"}  # fmt: skip
    assert len(load_base()) == len(stages) + 1
    cfg = load_config("rtx5090", ["train.dpo.beta=0.2", "correct.checks.min_bpc_gain=0.05"])
    assert cfg.train.dpo.beta == 0.2 and cfg.correct.checks.min_bpc_gain == 0.05
    assert cfg.train.lr == 1.0e-4 and "lost" not in cfg.sft
    # each step's file holds the section of its name (sft/dpo also hold their `train` part)
    for p in stages:
        assert p.stem in __import__("yaml").safe_load(p.read_text())


def _copy(tmp_path) -> Path:
    root = tmp_path / "configs"
    shutil.copytree(config_dir(), root)
    return root


def test_a_setting_in_two_files_is_refused(tmp_path):
    root = _copy(tmp_path)
    (root / "stages" / "zz_extra.yaml").write_text("train:\n  lr: 0.5\n")
    with pytest.raises(ValueError, match=r"train\.lr is set in both sft\.yaml and zz_extra\.yaml"):
        load_config("rtx5090", root=root)
    (root / "stages" / "zz_extra.yaml").write_text("train:\n  my_new_key: 1\n")
    assert load_config("rtx5090", root=root).train.my_new_key == 1


def test_a_profile_sets_only_the_machine(tmp_path):
    root = _copy(tmp_path)
    prof = root / "profiles" / "rtx5090.yaml"
    prof.write_text(prof.read_text() + "\nscore:\n  tau: 3\n")
    with pytest.raises(ValueError, match=r"sets only \['model', 'hw'\]; move \['score'\]"):
        load_config("rtx5090", root=root)


def test_derived_urls_and_names():
    cfg = load_config("h200x4")
    assert list(cfg.hw.infer.score_urls) == [f"http://127.0.0.1:{8000 + i}" for i in range(4)]
    assert list(cfg.hw.infer.correct_urls) == list(cfg.hw.infer.score_urls)
    assert served_name(cfg, "score") == served_name(cfg) == "base"
    cfg = load_config("rtx5090", ["hw.infer.host=0.0.0.0", "hw.infer.engines=2",
                                  "correct.model=/m/tuned"])  # fmt: skip
    assert list(cfg.hw.infer.score_urls) == ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]
    assert correct_model(cfg) == "/m/tuned" and served_name(cfg) == "tuned"
    # a tuned model is served by `serve --correct` on correct_port
    assert list(cfg.hw.infer.correct_urls) == ["http://127.0.0.1:8101", "http://127.0.0.1:8102"]
    assert served_name(load_config("rtx5090", ["correct.adapter=/a"])) == "adapter"
    cfg = load_config("rtx5090", ["hw.infer.correct_urls=[http://10.0.0.2:8000]"])
    assert list(cfg.hw.infer.correct_urls) == ["http://10.0.0.2:8000"]


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_is_complete(profile):
    from omegaconf import OmegaConf

    cfg = load_config(profile, ["hw.train.packing=false"])
    assert cfg.hw.name == profile
    missing = [k for k in PROFILE_KEYS if OmegaConf.select(cfg, k, default=None) is None]
    assert not missing, f"{profile}: missing {missing}"
    assert cfg.hw.train.mode in ("qlora", "lora", "full")
    assert cfg.hw.infer.engines * cfg.hw.infer.tensor_parallel_size <= cfg.hw.n_gpus
    assert cfg.hw.infer.max_model_len >= cfg.hw.train.max_seq_len


@pytest.mark.parametrize(("profile", "n", "port", "max_len"),
                         [("rtx5090", 1, 8100, 4096), ("h200x4", 4, 8000, 8192)])  # fmt: skip
def test_serve_dry_run(profile, n, port, max_len, tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    res = CliRunner().invoke(app, ["serve", "--dry-run", "--profile", profile,
                                   "--set", f"model={model}"])  # fmt: skip
    assert res.exit_code == 0, res.output
    cmds = [x for x in res.output.splitlines() if " serve " in x]
    assert len(cmds) == n
    backend = "FLASHINFER" if profile == "rtx5090" else "FLASH_ATTN"
    for i, c in enumerate(cmds):
        assert c.startswith(f"CUDA_VISIBLE_DEVICES={i} ") and f" serve {model} " in c
        assert "--served-model-name base" in c and f"--port {port + i}" in c
        assert f"--max-model-len {max_len}" in c and "--enable-prefix-caching" in c
        assert f"--attention-backend {backend}" in c and "--enable-lora" not in c


def test_serve_correct_adapter_and_overrides(tmp_path):
    base, tuned, adapter = (tmp_path / x for x in ("base", "tuned", "adapter"))
    for d in (base, tuned, adapter):
        d.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    run = lambda *a: CliRunner().invoke(app, ["serve", "--dry-run", "--set", f"model={base}", *a])  # noqa: E731
    res = run("--correct", "--set", f"correct.model={tuned}", "--port", "8101", "--memory", "0.44")
    assert res.exit_code == 0, res.output
    assert "--served-model-name tuned" in res.output and "--port 8101" in res.output
    assert f"serve {tuned} " in res.output and "--gpu-memory-utilization 0.44" in res.output
    res = run("--set", f"correct.adapter={adapter}")
    assert f"--lora-modules adapter={adapter}" in res.output and "--max-lora-rank 32" in res.output
    res = run("--correct")  # no correct.model: nothing to serve
    assert res.exit_code != 0 and "correct.model" in res.output
    res = run("--gpus", "3", "--", "--enforce-eager")
    assert res.output.startswith("CUDA_VISIBLE_DEVICES=3 ") and res.output.rstrip().endswith(
        "--enforce-eager"
    )


def test_cluster_example_profile(tmp_path):
    """configs/profiles/cluster.yaml.example, copied to cluster.yaml, is a complete profile that
    sends scoring and correction to other machines' engines."""
    from omegaconf import OmegaConf

    root = _copy(tmp_path)
    shutil.copy(root / "profiles" / "cluster.yaml.example", root / "profiles" / "cluster.yaml")
    cfg = load_config("cluster", ["hw.train.packing=false"], root=root)
    assert cfg.hw.name == "cluster"
    assert not [k for k in PROFILE_KEYS if OmegaConf.select(cfg, k, default=None) is None]
    assert [u.split(":")[1] for u in cfg.hw.infer.score_urls] == ["//10.0.0.11"] * 4
    assert len(cfg.hw.infer.correct_urls) == 8
    assert "cluster" not in PROFILES  # the example is not a profile as shipped


def test_auto_backend_follows_the_machine():
    """auto: vllm where the profile has servers, transformers where it has none (cpu)."""
    import pytest

    def be(profile, *sets):
        c = load_config(profile, list(sets))
        return str(c.score.backend), str(c.correct.backend)

    assert be("cpu") == ("transformers", "transformers")
    assert be("rtx5090") == ("vllm", "vllm")
    assert be("h200x8") == ("vllm", "vllm")
    # servers elsewhere listed on a machine without its own engines
    assert be("cpu", "hw.infer.score_urls=[http://10.0.0.11:8000]") == ("vllm", "vllm")
    # forcing one still wins
    assert be("rtx5090", "score.backend=transformers") == ("transformers", "vllm")
    with pytest.raises(ValueError, match="auto, vllm or transformers"):
        load_config("cpu", ["score.backend=hf"])
