"""`ocrfix pipeline <pipeline.yaml>`: ingest -> score -> fix -> rescore -> correct -> verify -> eval
over one directory of MinerU markdown, resumable per stage.

Each stage runs as its own ``ocrfix`` subprocess with the same ``--profile`` and ``--set``
overrides, so GPU memory is released between stages and multi-GPU stages re-exec under
torchrun as they do on their own. Output goes to ``work_dir``:

  corpus.jsonl            ingest (``sample_frac`` / ``--limit`` documents); with
                          ``input_records`` that file is the corpus and ingest is skipped
  scores.jsonl            score of the corpus
  fixed.jsonl             fix (+ fixed.ops.jsonl)
  scores_fixed.jsonl      rescore: score of the fixed text (what correct and verify compare to)
  corrected.jsonl         correct (+ corrected.responses.<llm-config hash>.jsonl, resumable)
  verified.jsonl          verify
  eval/                   eval of fixed and verified against ``gold_eval`` (skipped if unset)
  logs/<stage>.log        each stage's output; pipeline.json: status and seconds per stage

**Resume.** Each stage keeps ``.stages/<stage>.json``: a fingerprint of its settings (its
arguments and the config sections in ``STAGE_KEYS``) and of its inputs (size and a hash of
the head and tail of each file; for ingest the markdown files' count, size and newest
mtime), and whether it completed. On a rerun a stage is
  - skipped when it completed with the same fingerprints and its outputs exist;
  - resumed (outputs kept) when it was interrupted with the same fingerprints, or when only
    the inputs of an append-only stage changed (ingest, and score after ingest added
    documents): ingest, score and correct pick up where they stopped;
  - rebuilt (outputs deleted first) when its settings changed, when the inputs of any other
    stage changed, or with ``--force``.
So a changed input (more documents, a new calibration, another fixer setting) reruns
exactly the stages downstream of it. correct's raw responses are kept per LLM config
(``corrected.responses.<hash>.jsonl``) and reused while the inputs are the same.

**LLM server** for correct (``serve``):
  auto      use the server(s) at ``hw.infer.urls`` if they answer, else start
            ``scripts/serve_vllm.sh`` for the stage and stop it afterwards (on one GPU the
            scorer in verify needs the memory back)
  external  the server(s) must already answer
  echo      ``ocrfix.llm.echo_server`` on the same ports (no model: smoke runs)

Correct needs flags from a calibration (``scorer.calibration``) unless ``correct_all``; without
one it is skipped with a note, and verify and eval follow.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ocrfix.io import list_shards

STAGES = ("ingest", "score", "fix", "rescore", "correct", "verify", "eval")
SERVE_MODES = ("auto", "external", "echo")
REPO = Path(__file__).resolve().parents[2]
# config keys whose values are part of each stage's fingerprint (a change reruns the stage)
STAGE_KEYS = {
    "ingest": ["corpus", "seed"],
    # not scorer.calibration: score rows carry spike_frac@<tau> for every candidate, and
    # correct applies the calibration to rows without `flag`, so a new calibration
    # doesn't need a rescore
    "score": ["scorer.tau", "scorer.tau_candidates", "scorer.boundary", "scorer.passthrough",
              "paths.scorer_model", "paths.tokenizer", "hw.scorer.max_seq_len", "hw.dtype",
              "hw.attn_impl"],
    "fix": ["fix"],
    "correct": ["llm", "hw.infer.max_model_len", "scorer.calibration"],
    "verify": ["verify", "corpus.lang", "scorer", "paths.scorer_model", "hw.dtype"],
    "eval": ["eval", "errors"],
}  # fmt: skip
STAGE_KEYS["rescore"] = STAGE_KEYS["score"]
# Score without a calibration: no stored `flag`, so correct always applies the current
# calibration (a stored flag would win over a newer one).
NO_FLAGS = ("scorer.calibration=null",)
# Stages whose existing output stays valid when only their input grew: ingest appends
# documents and never rewrites one, and score skips paragraph ids it already has, which is
# right for the append-only corpus (not for fixed.jsonl, which fix rewrites: rescore).
APPEND_SAFE = {"ingest", "score"}


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineConfig:
    input_dir: Path | None
    work_dir: Path
    stages: list[str] = field(default_factory=lambda: list(STAGES))
    sample_frac: float | None = None
    gold_eval: Path | None = None
    correct_all: bool = False
    serve: str = "auto"
    serve_timeout: float = 1800.0
    sets: list[str] = field(default_factory=list)
    input_records: Path | None = None  # paragraph Records instead of markdown: no ingest


def load_pipeline(path: Path) -> PipelineConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise PipelineError(f"{path}: expected a mapping")
    known = {"input_dir", "input_records", "work_dir", "stages", "sample_frac", "gold_eval",
             "correct_all", "serve", "serve_timeout", "set"}  # fmt: skip
    unknown = set(raw) - known
    if unknown:
        raise PipelineError(f"{path}: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    if not raw.get("work_dir"):
        raise PipelineError(f"{path}: work_dir is required")
    if bool(raw.get("input_dir")) == bool(raw.get("input_records")):
        raise PipelineError(f"{path}: set exactly one of input_dir, input_records")
    stages = list(raw.get("stages") or STAGES)
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise PipelineError(f"{path}: unknown stages {bad} (known: {list(STAGES)})")
    serve = raw.get("serve") or "auto"
    if serve not in SERVE_MODES:
        raise PipelineError(f"{path}: serve must be one of {SERVE_MODES}, got {serve!r}")
    frac = raw.get("sample_frac")
    if frac is not None and not 0 < float(frac) <= 1:
        raise PipelineError(f"{path}: sample_frac must be in (0, 1]")
    return PipelineConfig(
        input_dir=Path(raw["input_dir"]) if raw.get("input_dir") else None,
        input_records=Path(raw["input_records"]) if raw.get("input_records") else None,
        work_dir=Path(raw["work_dir"]),
        stages=[s for s in STAGES if s in stages],
        sample_frac=None if frac is None else float(frac),
        gold_eval=Path(raw["gold_eval"]) if raw.get("gold_eval") else None,
        correct_all=bool(raw.get("correct_all", False)),
        serve=serve,
        serve_timeout=float(raw.get("serve_timeout") or 1800),
        sets=[str(s) for s in raw.get("set") or []],
    )


# ---------------------------------------------------------------- fingerprints

_EDGE = 1 << 20  # bytes hashed at the head and the tail of each file


def file_fingerprint(path: Path) -> list[Any]:
    out: list[Any] = []
    for shard in list_shards(path):
        size = shard.stat().st_size
        h = hashlib.sha256()
        with open(shard, "rb") as f:
            h.update(f.read(_EDGE))
            if size > _EDGE:
                f.seek(max(_EDGE, size - _EDGE))
                h.update(f.read())
        out.append([shard.name, size, h.hexdigest()[:16]])
    return out


def dir_fingerprint(path: Path, pattern: str) -> list[Any]:
    n = total = newest = 0
    for p in path.glob(pattern):
        if p.is_file():
            st = p.stat()
            n, total, newest = n + 1, total + st.st_size, max(newest, st.st_mtime_ns)
    return [n, total, newest]


# ---------------------------------------------------------------- LLM server


def healthy(urls: list[str], timeout: float = 2.0) -> bool:
    import httpx

    try:
        return all(
            httpx.get(f"{u.rstrip('/')}/health", timeout=timeout).status_code == 200 for u in urls
        )
    except httpx.HTTPError:
        return False


def busy_ports(urls: list[str]) -> list[str]:
    """URLs whose host:port accepts a connection (something is listening there)."""
    import socket
    from urllib.parse import urlparse

    out = []
    for u in urls:
        p = urlparse(u)
        try:
            with socket.create_connection((p.hostname or "127.0.0.1", p.port or 80), timeout=1):
                out.append(u)
        except OSError:
            pass
    return out


def _ports(urls: list[str]) -> list[int]:
    from urllib.parse import urlparse

    ports = []
    for u in urls:
        p = urlparse(u)
        if p.hostname not in ("127.0.0.1", "localhost"):
            raise PipelineError(f"echo server only binds 127.0.0.1, not {u}")
        ports.append(p.port or 80)
    return ports


@contextmanager
def llm_server(
    mode: str, urls: list[str], argv: list[str], log: Path, timeout: float
) -> Iterator[str]:
    """Yields how the server was obtained: running | started | echo."""
    if healthy(urls) and mode != "echo":
        yield "running"
        return
    if mode == "external":
        raise PipelineError(
            f"no LLM server answers at {urls}; start scripts/serve_vllm.sh or use serve: auto"
        )
    busy = busy_ports(urls)
    if busy:
        raise PipelineError(
            f"{', '.join(busy)}: the port is in use by another process that is not an LLM "
            "server (no /health); free it or move the engines with --set "
            "hw.infer.base_port=... --set 'hw.infer.urls=[...]'"
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO
        )
    try:
        t0 = time.monotonic()
        while not healthy(urls):
            if proc.poll() is not None:
                raise PipelineError(f"LLM server exited with {proc.returncode}; see {log}")
            if time.monotonic() - t0 > timeout:
                raise PipelineError(f"LLM server not ready after {timeout:.0f}s; see {log}")
            time.sleep(0.2 if mode == "echo" else 5)
        yield "echo" if mode == "echo" else "started"
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


# ---------------------------------------------------------------- pipeline


@dataclass
class StagePlan:
    argv: list[str]
    inputs: list[Path]
    outputs: list[Path]
    skip: str | None = None  # reason the stage can't run now


@dataclass
class StageResult:
    stage: str
    status: str  # ran | cached | skipped | failed
    seconds: float = 0.0
    note: str = ""


class Pipeline:
    def __init__(
        self,
        pc: PipelineConfig,
        *,
        profile: str | None,
        sets: list[str] | None = None,
        limit: int | None = None,
        only: list[str] | None = None,
        force: list[str] | None = None,
        echo: Any = print,
    ) -> None:
        from ocrfix.config import load_config

        self.pc = pc
        self.profile = profile or os.environ.get("OCRFIX_PROFILE") or "rtx5090"
        self.sets = [*pc.sets, *(sets or [])]
        self.cfg = load_config(self.profile, self.sets)
        self.limit = limit
        self.only = only
        self.force = set(force or [])
        self.echo = echo
        w = pc.work_dir
        self.p = {
            "corpus": pc.input_records or w / "corpus.jsonl",
            "scores": w / "scores.jsonl",
            "fixed": w / "fixed.jsonl",
            "ops": w / "fixed.ops.jsonl",
            "scores_fixed": w / "scores_fixed.jsonl",
            "corrected": w / "corrected.jsonl",
            "verified": w / "verified.jsonl",
            "eval": w / "eval",
        }

    # -- commands ------------------------------------------------------------

    def _ocrfix(self, *args: str | Path, extra_sets: tuple[str, ...] = ()) -> list[str]:
        argv = [sys.executable, "-m", "ocrfix.cli", *map(str, args), "--profile", self.profile]
        for s in [*self.sets, *extra_sets]:
            argv += ["--set", s]
        return argv

    def responses_path(self) -> Path:
        """Responses are cached by input text only, so they go to a file per model/LLM
        setting: switching llm.model or llm.adapter never reuses another model's answers."""
        llm = OmegaConf.to_container(self.cfg.llm, resolve=True)
        h = hashlib.sha256(json.dumps(llm, sort_keys=True).encode()).hexdigest()[:10]
        return self.pc.work_dir / f"corrected.responses.{h}.jsonl"

    def calibration(self) -> Path | None:
        c = self.cfg.scorer.get("calibration")
        return Path(c) if c else None

    def plan(self, stage: str) -> StagePlan:
        p, pc = self.p, self.pc
        if stage == "ingest":
            if pc.input_dir is None:
                return StagePlan(self._ocrfix("ingest", "-", "-"), [], [], "input_records is set")
            args: list[str | Path] = ["ingest", pc.input_dir, p["corpus"]]
            if pc.sample_frac is not None:
                args += ["--sample-frac", str(pc.sample_frac)]
            if self.limit is not None:
                args += ["--limit", str(self.limit)]
            argv = self._ocrfix(*args)
            skip = None if pc.input_dir.is_dir() else f"input_dir {pc.input_dir} not found"
            return StagePlan(argv, [], [p["corpus"]], skip)
        if stage == "score":
            return StagePlan(
                self._ocrfix("score", p["corpus"], p["scores"], extra_sets=NO_FLAGS),
                [p["corpus"]],
                [p["scores"]],
            )
        if stage == "fix":
            return StagePlan(
                self._ocrfix("fix", p["corpus"], p["fixed"], "--ops", p["ops"]),
                [p["corpus"]],
                [p["fixed"], p["ops"]],
            )
        if stage == "rescore":
            return StagePlan(
                self._ocrfix("score", p["fixed"], p["scores_fixed"], extra_sets=NO_FLAGS),
                [p["fixed"]],
                [p["scores_fixed"]],
            )
        if stage == "correct":
            args = [
                "correct", p["fixed"], p["scores_fixed"], p["corrected"],
                "--responses", self.responses_path(),
            ]  # fmt: skip
            cal = self.calibration()
            inputs = [p["fixed"], p["scores_fixed"]]
            skip = None
            if pc.correct_all:
                args.append("--all")
            argv = self._ocrfix(*args)
            if not pc.correct_all and (cal is None or not cal.is_file()):
                skip = (
                    f"no calibration ({cal}): run `ocrfix calibrate` first, or set "
                    "correct_all: true"
                )
            elif not pc.correct_all and cal is not None:
                inputs.append(cal)
            return StagePlan(argv, inputs, [p["corrected"]], skip)
        if stage == "verify":
            return StagePlan(
                self._ocrfix("verify", p["corrected"], p["scores_fixed"], p["verified"]),
                [p["corrected"], p["scores_fixed"]],
                [p["verified"]],
            )
        if stage == "eval":
            gold = pc.gold_eval
            systems = [f"fixed={p['fixed']}"]
            inputs = [p["fixed"]]
            if p["verified"].exists():
                systems.append(f"verified={p['verified']}")
                inputs.append(p["verified"])
            skip = None
            if gold is None:
                skip = "gold_eval not set"
            elif not gold.is_file():
                skip = f"gold_eval {gold} not found"
            else:
                inputs.insert(0, gold)
            return StagePlan(
                self._ocrfix("eval", gold or "-", *systems, p["eval"]), inputs, [p["eval"]], skip
            )
        raise PipelineError(f"unknown stage {stage}")

    # -- resume --------------------------------------------------------------

    def _marker(self, stage: str) -> Path:
        return self.pc.work_dir / ".stages" / f"{stage}.json"

    def fingerprint(self, stage: str, plan: StagePlan) -> tuple[str, str]:
        """(settings, inputs). Settings = the stage's own arguments (not the interpreter,
        --profile or --set: those reach it through the STAGE_KEYS config sections, so an
        unrelated override doesn't rerun it) and those config sections."""
        core = plan.argv[1 : plan.argv.index("--profile")]
        limit = None
        if stage == "ingest" and "--limit" in core:  # more documents = growth, not a rebuild
            i = core.index("--limit")
            core, limit = core[:i] + core[i + 2 :], core[i + 1]
        settings = {
            "argv": core,
            "config": {
                k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
                for k in STAGE_KEYS[stage]
                for v in [OmegaConf.select(self.cfg, k)]
            },
        }
        inputs: dict[str, Any] = {str(i): file_fingerprint(i) for i in plan.inputs}
        if stage == "ingest" and self.pc.input_dir is not None:
            inputs["input_dir"] = dir_fingerprint(self.pc.input_dir, self.cfg.corpus.input_glob)
            inputs["limit"] = limit

        def h(x: Any) -> str:
            return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

        return h(settings), h(inputs)

    def decide(self, stage: str, fp: tuple[str, str], plan: StagePlan) -> str:
        """cached | resume (keep outputs: fresh, interrupted, or append-only inputs grew) |
        clean (outputs are stale: delete them first)."""
        m = self._marker(stage)
        if stage in self.force or "all" in self.force:
            return "clean"
        if not m.is_file():
            return "resume"
        prev = json.loads(m.read_text())
        same_settings = prev.get("settings") == fp[0]
        same_inputs = prev.get("inputs") == fp[1]
        if prev.get("complete") and same_settings and same_inputs:
            return "cached" if all(o.exists() for o in plan.outputs) else "clean"
        if not same_settings:
            return "clean"
        if same_inputs or stage in APPEND_SAFE:
            return "resume"
        return "clean"

    def _clean(self, stage: str, plan: StagePlan) -> None:
        for o in plan.outputs:
            for path in (o, o.with_name(o.name + ".parts"), o.with_name(o.name + ".tmp")):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    def _write_marker(self, stage: str, fp: tuple[str, str], plan: StagePlan, **extra: Any) -> None:
        marker = self._marker(stage)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"stage": stage, "settings": fp[0], "inputs": fp[1], "argv": plan.argv[1:]} | extra,
                indent=2,
            )
            + "\n"
        )

    # -- run -----------------------------------------------------------------

    def _exec(self, stage: str, argv: list[str]) -> None:
        log = self.pc.work_dir / "logs" / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(argv) + "\n")
            fh.flush()
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=Path.cwd()
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                self.echo(f"  [{stage}] {line.rstrip()}")
            rc = proc.wait()
        if rc != 0:
            raise PipelineError(f"stage {stage} failed (exit {rc}); see {log}")

    def _server_argv(self) -> list[str]:
        urls = list(self.cfg.hw.infer.urls)
        if self.pc.serve == "echo":
            argv = [sys.executable, "-m", "ocrfix.llm.echo_server"]
            for port in _ports(urls):
                argv += ["--port", str(port)]
            return argv
        argv = ["bash", str(REPO / "scripts" / "serve_vllm.sh"), "--profile", self.profile]
        for s in self.sets:
            argv += ["--set", s]
        return argv

    def stages(self) -> list[str]:
        if self.only:
            bad = [s for s in self.only if s not in STAGES]
            if bad:
                raise PipelineError(f"unknown stages {bad} (known: {list(STAGES)})")
            return [s for s in STAGES if s in self.only]
        return self.pc.stages

    def dry_run(self) -> list[str]:
        return [" ".join(self.plan(s).argv) for s in self.stages()]

    def run(self) -> list[StageResult]:
        self.pc.work_dir.mkdir(parents=True, exist_ok=True)
        results: list[StageResult] = []
        try:
            for stage in self.stages():
                results.append(self._run_stage(stage))
        finally:
            self._write_summary(results)
        return results

    def _run_stage(self, stage: str) -> StageResult:
        plan = self.plan(stage)
        missing = [str(i) for i in plan.inputs if not i.exists()]
        if plan.skip or missing:
            note = plan.skip or f"missing input {', '.join(missing)}"
            self.echo(f"{stage}: skipped ({note})")
            return StageResult(stage, "skipped", note=note)
        fp = self.fingerprint(stage, plan)
        action = self.decide(stage, fp, plan)
        if action == "cached":
            self.echo(f"{stage}: up to date")
            return StageResult(stage, "cached")
        if action == "clean":
            self._clean(stage, plan)
        self.echo(
            f"{stage}: running" + (" (outputs were stale: rebuilt)" if action == "clean" else "")
        )
        self._write_marker(stage, fp, plan, complete=False)
        t0 = time.monotonic()
        note = ""
        try:
            if stage == "correct":
                with llm_server(
                    self.pc.serve,
                    list(self.cfg.hw.infer.urls),
                    self._server_argv(),
                    self.pc.work_dir / "logs" / "llm_server.log",
                    self.pc.serve_timeout,
                ) as how:
                    note = f"server {how}"
                    self._exec(stage, plan.argv)
            else:
                self._exec(stage, plan.argv)
        except PipelineError as e:
            self.echo(f"{stage}: FAILED: {e}")
            raise
        secs = time.monotonic() - t0
        self._write_marker(stage, fp, plan, complete=True, seconds=round(secs, 1))
        self.echo(f"{stage}: done in {secs:.1f}s")
        return StageResult(stage, "ran", secs, note)

    def _write_summary(self, results: list[StageResult]) -> None:
        path = self.pc.work_dir / "pipeline.json"
        prev = json.loads(path.read_text()) if path.is_file() else {}
        stages = prev.get("stages", {})
        for r in results:
            if r.status == "ran":
                stages[r.stage] = {"seconds": round(r.seconds, 1), "note": r.note}
        path.write_text(
            json.dumps(
                {
                    "profile": self.profile,
                    "sets": self.sets,
                    "last_run": [r.__dict__ for r in results],
                    "stages": stages,
                },
                indent=2,
            )
            + "\n"
        )
