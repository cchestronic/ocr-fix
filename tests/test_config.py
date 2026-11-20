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
    "hw.train.distributed", "hw.servers", "hw.infer.gpus_per_engine", "hw.infer.ports.base",
    "hw.infer.ports.trained", "hw.infer.max_model_len", "hw.infer.max_num_seqs",
    "hw.infer.concurrency", "hw.scorer.max_batch_tokens", "hw.scorer.logits_chunk",
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
    assert list(cfg.hw.infer.correct_urls) == list(cfg.hw.infer.score_urls)  # base corrects
    assert served_name(cfg, "score") == served_name(cfg) == "base"
    assert cfg.hw.infer.concurrency == 4 * 256  # automatic: engines x max_num_seqs
    cfg = load_config("h200x4", ["correct.model=/m/tuned"])
    assert correct_model(cfg) == "/m/tuned" and served_name(cfg) == "tuned"
    # a trained model: the trained groups' engines
    assert list(cfg.hw.infer.correct_urls) == [f"http://127.0.0.1:{8100 + i}" for i in range(4)]
    assert served_name(load_config("rtx5090", ["correct.adapter=/a"])) == "adapter"
    # advanced: an explicit list overrides the one built from hw.servers
    cfg = load_config("rtx5090", ["hw.infer.correct_urls=[http://10.0.0.2:8000]"])
    assert list(cfg.hw.infer.correct_urls) == ["http://10.0.0.2:8000"]


def test_server_plan_ports_memory_and_hosts():
    from ocrfix.config import server_plan

    sv = ("hw.servers=[{model: base, host: 10.0.0.11, gpus: [0, 1]}, "
          "{model: trained, host: 10.0.0.11, gpus: [2, 3, 4, 5, 6, 7]}, "
          "{model: trained, host: 10.0.0.12, gpus: all}, "
          "{model: base, host: 10.0.0.12, gpus: [0, 1, 2, 3], memory: 0.3}]")  # fmt: skip
    cfg = load_config("h200x8", [sv, "correct.model=/m/tuned"])
    plan = server_plan(cfg)
    got = [(e.model, e.host, e.gpus, e.port, e.memory) for e in plan]
    assert got[:2] == [("base", "10.0.0.11", (0,), 8000, 0.88),
                       ("base", "10.0.0.11", (1,), 8001, 0.88)]  # fmt: skip
    assert got[2] == ("trained", "10.0.0.11", (2,), 8100, 0.88)
    # gpu2: trained on all 8 GPUs shares GPUs 0-3 with a base group -> 0.44; base set 0.3
    assert got[8] == ("trained", "10.0.0.12", (0,), 8100, 0.44)
    assert got[-1] == ("base", "10.0.0.12", (3,), 8003, 0.3)
    assert len(cfg.hw.infer.score_urls) == 6 and len(cfg.hw.infer.correct_urls) == 14
    assert cfg.hw.infer.concurrency == 14 * 256
    # two engines of 2 GPUs each
    cfg = load_config("h200x4", ["hw.infer.gpus_per_engine=2"])
    assert [e.gpus for e in server_plan(cfg) if e.model == "base"] == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="not a multiple"):
        load_config("h200x4", ["hw.infer.gpus_per_engine=3"])
    with pytest.raises(ValueError, match="base or trained"):
        load_config("h200x4", ["hw.servers=[{model: tuned, gpus: all}]"])


def test_old_server_settings_are_refused_with_the_new_form():
    with pytest.raises(ValueError, match=r"replaced by one list, hw.servers"):
        load_config("rtx5090", ["hw.infer.engines=2"])


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_is_complete(profile):
    from omegaconf import OmegaConf

    cfg = load_config(profile, ["hw.train.packing=false"])
    assert cfg.hw.name == profile
    missing = [k for k in PROFILE_KEYS if OmegaConf.select(cfg, k, default=None) is None]
    assert not missing, f"{profile}: missing {missing}"
    assert cfg.hw.train.mode in ("qlora", "lora", "full")
    from ocrfix.config import server_plan

    for e in server_plan(cfg):
        assert all(g < cfg.hw.n_gpus for g in e.gpus) and 0 < e.memory <= 1
    assert cfg.hw.infer.max_model_len >= cfg.hw.train.max_seq_len


@pytest.mark.parametrize(("profile", "n", "port", "max_len"),
                         [("rtx5090", 1, 8100, 4096), ("h200x4", 4, 8000, 8192)])  # fmt: skip
def test_serve_dry_run(profile, n, port, max_len, tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    res = CliRunner().invoke(app, ["serve", "--dry-run", "--profile", profile,
                                   "--set", f"model={model}"])  # fmt: skip
    assert res.exit_code == 0, res.output
    # no correct.model: the trained groups are skipped with a note, base only
    assert "trained groups skipped" in res.output
    cmds = [x for x in res.output.splitlines() if " serve " in x]
    assert len(cmds) == n
    backend = "FLASHINFER" if profile == "rtx5090" else "FLASH_ATTN"
    for i, c in enumerate(cmds):
        assert c.startswith(f"CUDA_VISIBLE_DEVICES={i} ") and f" serve {model} " in c
        assert "--served-model-name base" in c and f"--port {port + i}" in c
        assert f"--max-model-len {max_len}" in c and "--enable-prefix-caching" in c
        assert f"--attention-backend {backend}" in c and "--enable-lora" not in c
        # base and trained groups share the GPUs: 0.44 each
        assert "--gpu-memory-utilization 0.44" in c


def test_serve_both_kinds_only_adapter_and_extra_args(tmp_path):
    base, tuned, adapter = (tmp_path / x for x in ("base", "tuned", "adapter"))
    for d in (base, tuned, adapter):
        d.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 32}')

    def run(*a):
        args = ["serve", "--dry-run", "--profile", "rtx5090", "--set", f"model={base}", *a]
        return CliRunner().invoke(app, args)

    res = run("--set", f"correct.model={tuned}")  # one command, both kinds
    assert res.exit_code == 0, res.output
    assert "--served-model-name base" in res.output and "--port 8100" in res.output
    assert "--served-model-name tuned" in res.output and "--port 8101" in res.output
    assert f"serve {tuned} " in res.output
    res = run("--only", "trained", "--set", f"correct.model={tuned}")
    assert "--served-model-name base" not in res.output and "--port 8101" in res.output
    res = run("--only", "trained")  # no correct.model: nothing to serve
    assert res.exit_code != 0 and "correct.model" in res.output
    res = run("--set", f"correct.adapter={adapter}")
    assert f"--lora-modules adapter={adapter}" in res.output and "--max-lora-rank 32" in res.output
    res = run("--only", "base", "--", "--enforce-eager")
    assert res.output.rstrip().endswith("--enforce-eager")


def test_serve_starts_only_this_machines_groups(tmp_path):
    base, tuned = tmp_path / "base", tmp_path / "tuned"
    base.mkdir()
    tuned.mkdir()
    sv = ("hw.servers=[{model: base, host: 10.9.9.1, gpus: [0, 1]}, "
          "{model: trained, host: 10.9.9.2, gpus: all}]")  # fmt: skip

    def run(*a):
        args = ["serve", "--dry-run", "--profile", "h200x4", "--set", f"model={base}",
                "--set", f"correct.model={tuned}", "--set", sv, *a]  # fmt: skip
        return CliRunner().invoke(app, args)

    res = run()  # this test machine is neither
    assert res.exit_code != 0 and "no server group of hw.servers is on this machine" in res.output
    res = run("--host", "10.9.9.2")
    assert res.exit_code == 0, res.output
    assert "--served-model-name tuned" in res.output
    assert "--served-model-name base" not in res.output
    assert res.output.count(" serve ") == 4 and "--host 10.9.9.2" in res.output


def test_cluster_example_profile(tmp_path):
    """configs/profiles/cluster.yaml.example, copied to cluster.yaml, is a complete profile that
    sends scoring and correction to other machines' engines."""
    from omegaconf import OmegaConf

    root = _copy(tmp_path)
    shutil.copy(root / "profiles" / "cluster.yaml.example", root / "profiles" / "cluster.yaml")
    cfg = load_config("cluster", ["hw.train.packing=false"], root=root)
    assert cfg.hw.name == "cluster"
    assert not [k for k in PROFILE_KEYS if OmegaConf.select(cfg, k, default=None) is None]
    assert [u.split(":")[1] for u in cfg.hw.infer.score_urls] == ["//10.0.0.11"] * 2
    cfg = load_config("cluster", ["hw.train.packing=false", "correct.model=/m/t"], root=root)
    assert len(cfg.hw.infer.correct_urls) == 6 + 8 + 8
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
    assert be("cpu", "hw.servers=[{model: base, host: 10.0.0.11, gpus: [0]}]") == ("vllm", "vllm")
    # forcing one still wins
    assert be("rtx5090", "score.backend=transformers") == ("transformers", "vllm")
    with pytest.raises(ValueError, match="auto, vllm or transformers"):
        load_config("cpu", ["score.backend=hf"])
