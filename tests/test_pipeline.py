"""`ocrfix pipeline`: the stage plans and when model servers start and stop."""

from __future__ import annotations

from pathlib import Path

from ocrfix.config import load_config
from ocrfix.pipeline import BASE, BOTH, CPU, FREE, STAGES, Servers, Step, plan, run_pipeline


def cfg(*sets: str):
    return load_config("cpu", list(sets))


def test_every_stage_has_steps_in_readme_order():
    c = cfg()
    names = {st: [s.name for s in plan(st, c)] for st in STAGES}
    assert names["prepare"][:4] == ["markup corpus", "markup gold/ocr", "markup gold/truth",
                                    "markup clean"]  # fmt: skip
    assert names["calibrate"][-1] == "calibrate (+ gold check)"
    assert names["train"][:3] == ["data sft", "train sft", "merge sft"]
    assert names["train"][-1] == "eval" and "data dpo" not in names["train"]
    assert names["corpus"][-1] == "reconstruct"


def test_dpo_and_model_choice():
    with_dpo = plan("train", cfg("pipeline.dpo=true"))
    assert [s.name for s in with_dpo if "dpo" in s.name][:2] == ["corrupt (dpo)", "data dpo"]
    assert "dpo=" in " ".join(next(s for s in with_dpo if s.name == "eval").args)
    corpus = {s.name: s for s in plan("corpus", cfg())}
    assert Path(corpus["correct"].tuned) == Path("runs/sft_merged")
    corpus = {s.name: s for s in plan("corpus", cfg("pipeline.dpo=true"))}
    assert Path(corpus["correct"].tuned) == Path("runs/dpo_merged")
    corpus = {s.name: s for s in plan("corpus", cfg("pipeline.model=/m/x"))}
    assert corpus["correct"].tuned == "/m/x"


def test_step_settings_come_after_the_users():
    s = Step("x", ["score", "a", "b"], BASE, sets=["score.noise.remove=false"], tuned="m")
    cmd = s.command("rtx5090", ["score.noise.remove=true", "seed=3"])
    assert cmd[5:] == ["score", "a", "b", "--profile", "rtx5090", "--set",
                       "score.noise.remove=true", "--set", "seed=3", "--set",
                       "score.noise.remove=false", "--set", "correct.model=m"]  # fmt: skip


def test_servers_follow_the_steps(tmp_path):
    """vLLM backends (the GPU machines): which server kinds run before each step."""
    sv = Servers(cfg("hw.servers=[{model: base, gpus: [0]}, {model: trained, gpus: [0]}]",
                     "score.backend=vllm", "correct.backend=vllm"), None, [], tmp_path)  # fmt: skip
    calls = []

    def start(kind, tuned):
        calls.append(("start", kind, tuned))
        sv.procs[kind] = (object(), None, tuned, [])

    def stop(kind):
        calls.append(("stop", kind))
        sv.procs.pop(kind)

    sv._start, sv._stop = start, stop
    for needs, tuned in [(BASE, None), (CPU, None), (BOTH, "sft"), (BOTH, "sft"), (FREE, None),
                         (BOTH, "dpo"), (BASE, None)]:  # fmt: skip
        sv.need(Step("s", [], needs, tuned=tuned))
    assert calls == [
        ("start", "base", None),               # scoring: base
        ("start", "trained", "sft"),           # + the trained model (memory set per group)
        ("stop", "base"), ("stop", "trained"),  # training: no servers
        ("start", "base", None),
        ("start", "trained", "dpo"),
        ("stop", "trained"),                   # base only again: the trained model goes
    ]  # fmt: skip


def test_no_servers_with_transformers_backends(tmp_path):
    c = cfg("score.backend=transformers", "correct.backend=transformers")
    sv = Servers(c, None, [], tmp_path)
    sv._start = lambda *a: (_ for _ in ()).throw(AssertionError("started a server"))
    for needs in (BASE, BOTH, FREE):
        sv.need(Step("s", [], needs, tuned="m"))


def test_dry_run_marks_done_steps(tmp_path, capsys):
    d = tmp_path / "real"
    (d / "errors").mkdir(parents=True)
    (d / "errors" / "error_table.json").write_text("{}")
    c = cfg(f"pipeline.data_dir={d}")
    assert run_pipeline("prepare", c, profile=None, user_sets=[], dry_run=True) == 0
    out = capsys.readouterr().out
    assert "errors table" in out and "done, skipped" in out
    assert out.count("done, skipped") == 1 and not (d / "pipeline_logs").exists()


def test_remote_engines_are_never_started_here(tmp_path):
    """Several GPU servers: groups elsewhere must already run; the stage says what to start."""
    import pytest

    sets = ["hw.servers=[{model: base, host: 10.9.9.9, gpus: [0]}]", "score.backend=vllm"]
    sv = Servers(cfg(*sets), None, sets, tmp_path)
    with pytest.raises(RuntimeError, match=r"other machines don't answer: http://10\.9\.9\.9:8100"):
        sv.need(Step("score", [], BASE))
    assert not sv.procs and not list(tmp_path.glob("serve-*.log"))  # nothing started here


def test_bpc_stages_counts_every_corrected_status(tmp_path):
    """The corrected stage keeps accepted / rejected / unchanged rows; noise is counted apart."""
    import json

    from ocrfix.stages.bpc_stages import compare_stages, format_table

    def write(name, rows):
        d = tmp_path / name
        d.mkdir()
        (d / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return d

    def row(bpc, status, lang="ko", n=100):
        return {"lang": lang, "status": status,
                "score": None if bpc is None else {"bpc": bpc, "n_chars": n}}  # fmt: skip

    raw = write("raw", [row(2.0, "scored"), row(4.0, "scored"), row(9.0, "scored")])
    den = write("den", [row(2.0, "scored"), row(3.0, "scored"), row(None, "noise")])
    cor = write("cor", [row(1.5, "accepted"), row(3.0, "rejected"), row(None, "noise")])
    res = compare_stages([("raw", raw), ("denoised", den), ("corrected", cor)])
    assert res["raw"]["*"]["chunks"] == 3 and res["raw"]["*"]["mean"] == 5.0
    assert res["denoised"]["*"] | {} and res["denoised"]["*"]["noise"] == 1
    assert res["corrected"]["*"]["chunks"] == 2 and res["corrected"]["*"]["mean"] == 2.25
    assert res["corrected"]["ko"]["wmean"] == 2.25
    table = "\n".join(format_table(res))
    assert "corrected" in table and "noise" in table


def test_corpus_stage_ends_with_the_stage_comparison():
    names = [s.name for s in plan("corpus", cfg())]
    assert names[-2:] == ["bpc by stage (raw / denoised / corrected)", "reconstruct"]
