"""`ocrfix pipeline`: config validation, per-stage resume, the echo LLM server, and one
end-to-end smoke run on tests/fixtures with the tiny model (scorer) and the echo server."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest

from ocrfix.corpus.ingest import in_sample, ingest
from ocrfix.llm.echo_server import answer, edit, serve
from ocrfix.pipeline import (
    STAGES,
    Pipeline,
    PipelineConfig,
    PipelineError,
    llm_server,
    load_pipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"
SMOKE_GOLD = FIXTURES / "sample_mineru_gold_eval.jsonl"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_yaml(path: Path, **kw) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(kw))
    return path


def smoke_config(tmp_path: Path, **kw) -> PipelineConfig:
    port = free_port()
    base = {
        "input_dir": str(FIXTURES / "sample_mineru"),
        "work_dir": str(tmp_path / "work"),
        "gold_eval": str(SMOKE_GOLD),
        "correct_all": True,
        "serve": "echo",
        "set": [
            "scorer.calibration=null",
            f"hw.infer.base_port={port}",
            f"hw.infer.urls=[http://127.0.0.1:{port}]",
        ],
    }
    return load_pipeline(write_yaml(tmp_path / "p.yaml", **(base | kw)))


# ---------------------------------------------------------------- config


def test_load_pipeline_defaults_and_validation(tmp_path):
    pc = load_pipeline(write_yaml(tmp_path / "a.yaml", input_dir="in", work_dir="w"))
    assert pc.stages == list(STAGES) and pc.serve == "auto" and pc.gold_eval is None
    with pytest.raises(PipelineError, match="unknown keys"):
        load_pipeline(write_yaml(tmp_path / "b.yaml", input_dir="in", work_dir="w", foo=1))
    with pytest.raises(PipelineError, match="work_dir is required"):
        load_pipeline(write_yaml(tmp_path / "c.yaml", input_dir="in"))
    with pytest.raises(PipelineError, match="exactly one of"):
        load_pipeline(
            write_yaml(tmp_path / "c2.yaml", input_dir="i", input_records="r", work_dir="w")
        )
    with pytest.raises(PipelineError, match="unknown stages"):
        load_pipeline(write_yaml(tmp_path / "d.yaml", input_dir="i", work_dir="w", stages=["x"]))
    with pytest.raises(PipelineError, match="serve"):
        load_pipeline(write_yaml(tmp_path / "e.yaml", input_dir="i", work_dir="w", serve="no"))
    with pytest.raises(PipelineError, match="sample_frac"):
        load_pipeline(write_yaml(tmp_path / "f.yaml", input_dir="i", work_dir="w", sample_frac=2))
    # stages run in pipeline order whatever order the file lists them in
    pc = load_pipeline(
        write_yaml(tmp_path / "g.yaml", input_dir="i", work_dir="w", stages=["fix", "ingest"])
    )
    assert pc.stages == ["ingest", "fix"]


PROFILE_KEYS = [
    "hw.name", "hw.n_gpus", "hw.dtype", "hw.attn_impl", "hw.vllm_attention_backend",
    "hw.train.mode", "hw.train.per_device_batch_size", "hw.train.max_seq_len",
    "hw.train.gradient_checkpointing", "hw.train.optim", "hw.train.packing",
    "hw.train.distributed", "hw.infer.tensor_parallel_size", "hw.infer.data_parallel_size",
    "hw.infer.gpu_memory_utilization", "hw.infer.max_model_len", "hw.infer.max_num_seqs",
    "hw.infer.base_port", "hw.infer.urls", "hw.infer.client_concurrency",
    "hw.scorer.max_seq_len", "hw.scorer.max_batch_tokens", "hw.scorer.logits_chunk",
    "hw.scorer.tokenize_workers",
]  # fmt: skip


@pytest.mark.parametrize(
    "profile", sorted(p.stem for p in (FIXTURES.parent.parent / "configs/profiles").glob("*.yaml"))
)
def test_every_profile_is_complete(profile):
    """What a new profile must define (docs/runbook.md, "Adding a profile")."""
    from omegaconf import OmegaConf

    from ocrfix.config import load_config

    cfg = load_config(profile, ["hw.train.packing=false"])
    assert cfg.hw.name == profile
    missing = [k for k in PROFILE_KEYS if OmegaConf.select(cfg, k, default=None) is None]
    assert not missing, f"{profile}: missing {missing}"
    assert cfg.hw.train.mode in ("qlora", "lora", "full")
    assert cfg.hw.infer.data_parallel_size * cfg.hw.infer.tensor_parallel_size <= cfg.hw.n_gpus
    assert cfg.hw.infer.max_model_len >= cfg.hw.train.max_seq_len


def test_repo_pipeline_configs_load():
    for p in sorted((Path(__file__).parents[1] / "configs" / "pipeline").glob("*.yaml")):
        load_pipeline(p)


def test_dry_run_commands(tmp_path):
    pl = Pipeline(smoke_config(tmp_path, sample_frac=0.5), profile="rtx5090", limit=1)
    cmds = pl.dry_run()
    assert len(cmds) == len(STAGES)
    assert "ingest" in cmds[0] and "--sample-frac 0.5" in cmds[0] and "--limit 1" in cmds[0]
    assert " correct " in cmds[4] and "--all" in cmds[4] and "corrected.responses." in cmds[4]
    assert all("--profile rtx5090" in c and "scorer.calibration=null" in c for c in cmds)


def test_correct_skipped_without_calibration(tmp_path):
    pc = smoke_config(tmp_path, correct_all=False)
    pl = Pipeline(pc, profile="rtx5090", sets=["scorer.calibration=" + str(tmp_path / "no.yaml")])
    assert "no calibration" in (pl.plan("correct").skip or "")


def test_responses_file_per_llm_config(tmp_path):
    pc = smoke_config(tmp_path)
    a = Pipeline(pc, profile="rtx5090").responses_path()
    b = Pipeline(pc, profile="rtx5090", sets=["llm.adapter=/x"]).responses_path()
    assert a != b and a == Pipeline(pc, profile="rtx5090").responses_path()


# ---------------------------------------------------------------- sampling


def test_in_sample_is_deterministic_and_proportional():
    paths = [f"dir/doc{i}.md" for i in range(4000)]
    picked = [p for p in paths if in_sample(p, 0.02, seed=7)]
    assert picked == [p for p in paths if in_sample(p, 0.02, seed=7)]
    assert 40 <= len(picked) <= 125
    assert picked != [p for p in paths if in_sample(p, 0.02, seed=8)]


def test_ingest_sample_frac(tmp_path):
    out = tmp_path / "c.jsonl"
    st = ingest(FIXTURES / "sample_mineru", out, sample_frac=1e-9)
    assert st.docs_written == 0 and st.docs_sampled_out == 3
    st = ingest(FIXTURES / "sample_mineru", tmp_path / "d.jsonl", sample_frac=1.0)
    assert st.docs_written == 3


# ---------------------------------------------------------------- echo server


def test_echo_edit_keeps_markers():
    text = "<<<1>>>\nas shown in Tab1e 1. The  rnodel is pro-\nposed.\n<<<2>>>\nTab1eau stays"
    assert (
        edit(text) == "<<<1>>>\nas shown in Table 1. The model is proposed.\n<<<2>>>\nTab1eau stays"
    )
    r = answer({"messages": [{"role": "user", "content": "<<<1>>>\nx"}], "n": 3}, identity=True)
    assert [c["message"]["content"] for c in r["choices"]] == ["<<<1>>>\nx"] * 3


def test_echo_server_http():
    port = free_port()
    (srv,) = serve([port])
    try:
        base = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{base}/health").status_code == 200
        body = {"model": "m", "messages": [{"role": "user", "content": "<<<1>>>\nTab1e"}]}
        r = httpx.post(f"{base}/v1/chat/completions", json=body).json()
        assert r["choices"][0]["message"]["content"] == "<<<1>>>\nTable"
        assert r["choices"][0]["finish_reason"] == "stop"
    finally:
        srv.shutdown()


def test_llm_server_refuses_busy_port(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        url = f"http://127.0.0.1:{s.getsockname()[1]}"
        with (
            pytest.raises(PipelineError, match="in use by another process"),
            llm_server("echo", [url], ["true"], tmp_path / "log", 5),
        ):
            pass


def test_llm_server_external_needs_running_server(tmp_path):
    url = f"http://127.0.0.1:{free_port()}"
    with (
        pytest.raises(PipelineError, match="no LLM server"),
        llm_server("external", [url], ["true"], tmp_path / "log", 5),
    ):
        pass


# ---------------------------------------------------------------- resume (no model)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def statuses(results) -> dict[str, str]:
    return {r.stage: r.status for r in results}


def test_resume_cached_and_rebuilt(tmp_path):
    """ingest + fix only (no model): cached on rerun, fix rebuilt on a fixer change while
    ingest stays cached, and --force rebuilds."""
    pc = smoke_config(tmp_path, stages=["ingest", "fix"])
    logs: list[str] = []
    r1 = Pipeline(pc, profile="rtx5090", echo=logs.append).run()
    assert statuses(r1) == {"ingest": "ran", "fix": "ran"}
    work = pc.work_dir
    assert len(lines(work / "corpus.jsonl")) == 14
    r2 = Pipeline(pc, profile="rtx5090", echo=logs.append).run()
    assert statuses(r2) == {"ingest": "cached", "fix": "cached"}
    # an unrelated override does not rerun anything
    r3 = Pipeline(pc, profile="rtx5090", sets=["eval.worst_n=3"], echo=logs.append).run()
    assert statuses(r3) == {"ingest": "cached", "fix": "cached"}
    logs.clear()
    r4 = Pipeline(
        pc, profile="rtx5090", sets=["fix.line_join.enabled=true"], echo=logs.append
    ).run()
    assert statuses(r4) == {"ingest": "cached", "fix": "ran"}
    assert any("rebuilt" in x for x in logs)
    ops = lines(work / "fixed.ops.jsonl")
    assert any(o["rule"].startswith("line_join") for o in ops)
    r5 = Pipeline(pc, profile="rtx5090", force=["ingest"], echo=logs.append).run()
    assert statuses(r5)["ingest"] == "ran"
    summary = json.loads((work / "pipeline.json").read_text())
    assert set(summary["stages"]) == {"ingest", "fix"}


def test_input_records_skip_ingest(tmp_path):
    """Gold sources as the corpus (`gold records`): ingest skipped, fix runs on them."""
    from ocrfix.gold.records import gold_records

    src = tmp_path / "eval_src.jsonl"
    gold_records(SMOKE_GOLD, src)
    pc = smoke_config(tmp_path, input_dir=None, input_records=str(src), stages=["ingest", "fix"])
    res = Pipeline(pc, profile="rtx5090", echo=lambda _: None).run()
    assert statuses(res) == {"ingest": "skipped", "fix": "ran"}
    assert len(lines(pc.work_dir / "fixed.jsonl")) == 13


def test_gold_records_gold_side(tmp_path):
    from ocrfix.gold.records import gold_records

    out = tmp_path / "clean.jsonl"
    assert gold_records(SMOKE_GOLD, out, side="gold") == (13, 0)
    rows = lines(out)
    assert any(r["text"].startswith("The energy") and "Table 1" in r["text"] for r in rows)
    assert all(r["cer"] == 0 for r in rows)


def test_smoke_gold_feeds_errors_table(tmp_path):
    """The smoke gold is real `gold import` output: errors table reads it."""
    from ocrfix.config import load_config
    from ocrfix.errors import run_table

    res = run_table(load_config("rtx5090"), SMOKE_GOLD, tmp_path / "err")
    assert res.n_rows == 13
    assert sum(res.table.groups[("*", "*")].counts.values()) == 1  # Tab1e -> Table


def test_cli_work_dir_override(tmp_path):
    from typer.testing import CliRunner

    from ocrfix.cli import app

    cfg = write_yaml(tmp_path / "p.yaml", input_dir=str(FIXTURES / "sample_mineru"),
                     work_dir=str(tmp_path / "a"), stages=["ingest"])  # fmt: skip
    r = CliRunner().invoke(
        app, ["pipeline", str(cfg), "--work-dir", str(tmp_path / "b"), "--profile", "rtx5090"]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "b" / "corpus.jsonl").is_file() and not (tmp_path / "a").exists()
    assert "ingest   ran" in r.output


def test_limit_and_interrupted_stage_resumes(tmp_path):
    pc = smoke_config(tmp_path, stages=["ingest"])
    Pipeline(pc, profile="rtx5090", limit=1, echo=lambda _: None).run()
    corpus = pc.work_dir / "corpus.jsonl"
    docs = {d["doc_id"] for d in lines(corpus)}
    assert len(docs) == 1
    # simulate an interruption: marker says incomplete with the same fingerprints
    marker = pc.work_dir / ".stages" / "ingest.json"
    m = json.loads(marker.read_text())
    marker.write_text(json.dumps(m | {"complete": False}))

    def decide(pl: Pipeline) -> str:
        plan = pl.plan("ingest")
        return pl.decide("ingest", pl.fingerprint("ingest", plan), plan)

    assert decide(Pipeline(pc, profile="rtx5090", limit=1)) == "resume"  # same: interrupted
    marker.write_text(json.dumps(m))  # completed again
    assert decide(Pipeline(pc, profile="rtx5090", limit=1)) == "cached"
    # dropping the limit = more documents: append to the corpus, don't rebuild it
    assert decide(Pipeline(pc, profile="rtx5090")) == "resume"
    res = Pipeline(pc, profile="rtx5090", echo=lambda _: None).run()
    assert res[0].status == "ran"
    assert len({d["doc_id"] for d in lines(corpus)}) == 3
    # another sample is another corpus: rebuilt
    pc2 = smoke_config(tmp_path, stages=["ingest"], sample_frac=0.5)
    assert decide(Pipeline(pc2, profile="rtx5090")) == "clean"


def test_failed_stage_stops_and_reports(tmp_path):
    pc = smoke_config(tmp_path, stages=["ingest", "score"])
    pl = Pipeline(pc, profile="rtx5090", sets=["paths.base_model=/nonexistent/model"],
                  echo=lambda _: None)  # fmt: skip
    with pytest.raises(PipelineError, match="stage score failed"):
        pl.run()
    assert (pc.work_dir / "logs" / "score.log").is_file()
    marker = json.loads((pc.work_dir / ".stages" / "score.json").read_text())
    assert marker["complete"] is False
    last = json.loads((pc.work_dir / "pipeline.json").read_text())["last_run"]
    assert [r["stage"] for r in last] == ["ingest"]


# ---------------------------------------------------------------- end to end


def test_smoke_pipeline_end_to_end(tmp_path, tiny_model_dir, model_device):
    """Every stage on tests/fixtures: tiny model as scorer, echo server as the LLM."""
    pc = smoke_config(tmp_path)
    sets = [f"paths.base_model={tiny_model_dir}"]
    if model_device == "cpu":
        pytest.skip("the scorer stages need CUDA in a subprocess here")
    res = Pipeline(pc, profile="rtx5090", sets=sets, echo=lambda _: None).run()
    assert statuses(res) == dict.fromkeys(STAGES, "ran")
    w = pc.work_dir
    verified = lines(w / "verified.jsonl")
    assert len(verified) == 14
    fixed_table = [r for r in verified if "Table 1" in r["text"]]
    assert len(fixed_table) == 1 and fixed_table[0]["accepted"]
    m = json.loads((w / "eval" / "metrics.json").read_text())
    assert m["systems"]["fixed"]["overall"]["cer"] > 0
    assert m["systems"]["verified"]["overall"]["cer"] == 0
    assert (w / "eval" / "report.md").is_file()
    again = Pipeline(pc, profile="rtx5090", sets=sets, echo=lambda _: None).run()
    assert statuses(again) == dict.fromkeys(STAGES, "cached")
