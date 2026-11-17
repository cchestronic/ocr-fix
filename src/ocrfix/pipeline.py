"""`ocrfix pipeline <stage>`: run a whole track with default settings, no typing per step.

    ./ocrfix pipeline prepare      # A (CPU): markup, gold chunks, errors table, chunk, corrupt
    ./ocrfix pipeline calibrate    # B (GPU): noise thresholds, scores, calibrate + gold check
    ./ocrfix pipeline train        # B (GPU): SFT data, train, merge, [DPO], eval
    ./ocrfix pipeline corpus       # C (GPU): chunk, score x2, noise-chunk limit, correct, ...
    ./ocrfix pipeline all          # the four in a row (skips the checks between them)

Each stage ends where a person should look before going on (README section 5). Every step is
an ordinary `ocrfix` command, run as its own process with the same ``--profile`` / ``--set``
(the step's own settings come last), so a stage does exactly what typing the README's
commands does. Folders: ``pipeline.data_dir`` (data/real), ``pipeline.calibration_dir``,
``pipeline.runs_dir``.

- **Resumable:** a step whose output already exists is skipped (``--redo`` runs it again);
  the steps that resume by themselves (markup, chunk, score, correct) always run and skip
  their finished files.
- **Servers:** with ``score.backend`` / ``correct.backend`` = vllm, the stage starts the
  engines a step needs (base; base + the trained model at ``pipeline.both_memory``), waits for
  ``/health``, and stops them before training. Engines that already answer are used as they
  are and never stopped. It starts engines only on **this machine** (urls at 127.0.0.1,
  localhost or ``hw.infer.host``); engines on other servers (``score_urls`` /
  ``correct_urls`` pointing elsewhere, e.g. several H200 servers) must already run: if one
  doesn't answer, the stage stops and says what to start where. With the transformers
  backends (``auto`` on a profile without engines, e.g. cpu) no server is started.
- **Logs:** every step's output is shown and kept in ``<data_dir>/pipeline_logs/``.
- ``--dry-run`` prints the steps (and which are done) without running anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES = ("prepare", "calibrate", "train", "corpus")

# what a step needs from the model servers
CPU = "cpu"  # nothing (servers may keep running)
BASE = "base"  # the base model
BOTH = "both"  # base + the trained model (correct.model)
FREE = "free"  # the GPUs to itself: no servers (training)


@dataclass
class Step:
    name: str
    args: list[str]
    needs: str = CPU
    done: Path | None = None  # skip when this exists
    sets: list[str] = field(default_factory=list)  # step-specific --set, after the user's
    tuned: str | None = None  # correct.model for this step (BOTH / correct)

    def command(self, profile: str | None, user_sets: list[str]) -> list[str]:
        cmd = [sys.executable, "-X", "utf8", "-m", "ocrfix", *self.args]
        if profile:
            cmd += ["--profile", profile]
        sets = [*user_sets, *self.sets]
        if self.tuned is not None:
            sets.append(f"correct.model={self.tuned}")
        for s in sets:
            cmd += ["--set", s]
        return cmd


def plan(stage: str, cfg) -> list[Step]:
    """The steps of one stage, with the folder layout of the README."""
    p = cfg.pipeline
    D, C, R = Path(p.data_dir), Path(p.calibration_dir), Path(p.runs_dir)
    cal = [f"score.calibration={C / 'calibration.yaml'}"]
    raw = ["score.calibration=null", "score.noise.remove=false"]
    E = [str(D / "errors" / "error_table.json"), str(D / "errors" / "confusion_map.json")]
    steps = int(p.max_steps) if p.max_steps is not None else None
    max_steps = ["--max-steps", str(steps)] if steps is not None else []
    dpo = bool(p.dpo)
    sft_model, dpo_model = str(R / "sft_merged"), str(R / "dpo_merged")
    tuned = str(p.model) if p.model else (dpo_model if dpo else sft_model)

    if stage == "prepare":
        out = []
        for src, dst in (
            ("corpus", "corpus_fixed"),
            ("gold/ocr", "gold_fixed/ocr"),
            ("gold/truth", "gold_fixed/truth"),
            ("clean", "clean_fixed"),
        ):
            out.append(Step(f"markup {src}", ["markup", str(D / src), str(D / dst)]))
        return [
            *out,
            Step(
                "gold chunks",
                ["gold", "chunks", str(D / "gold_fixed"), str(D / "gold_chunks")],
                done=D / "gold_chunks" / "stats.json",
            ),
            Step(
                "errors table",
                ["errors", "table", str(D / "gold_chunks" / "sft"), str(D / "errors")],
                done=D / "errors" / "error_table.json",
            ),
            Step("chunk clean", ["chunk", str(D / "clean_fixed"), str(D / "chunks_clean")]),
            Step(
                "corrupt (calibration, gold rates)",
                [
                    "corrupt",
                    str(D / "chunks_clean"),
                    *E,
                    str(D / "calib" / "corrupt" / "table.jsonl"),
                ],
                done=D / "calib" / "corrupt" / "table.jsonl",
            ),
            Step(
                "corrupt (calibration, 3%)",
                [
                    "corrupt",
                    str(D / "chunks_clean"),
                    *E,
                    str(D / "calib" / "corrupt" / "r0.03.jsonl"),
                    "--rate",
                    "0.03",
                ],
                done=D / "calib" / "corrupt" / "r0.03.jsonl",
            ),
            Step(
                "corrupt (training)",
                ["corrupt", str(D / "chunks_clean"), *E, str(D / "train" / "corrupt_table.jsonl")],
                sets=["seed=2"],
                done=D / "train" / "corrupt_table.jsonl",
            ),
        ]
    if stage == "calibrate":
        noise = ["calibrate-noise", str(D / "gold_chunks"), str(C)]
        if not p.write_noise:
            noise.append("--no-write")
        return [
            Step("calibrate-noise", noise, BASE, C / "noise_report.json"),
            Step(
                "score clean",
                ["score", str(D / "chunks_clean"), str(D / "calib" / "clean_scored")],
                BASE,
                sets=raw,
            ),
            Step(
                "score damaged",
                ["score", str(D / "calib" / "corrupt"), str(D / "calib" / "corrupt_scored")],
                BASE,
                sets=raw,
            ),
            Step(
                "score gold (sft)",
                ["score", str(D / "gold_chunks" / "sft"), str(D / "calib" / "gold_scored")],
                BASE,
                sets=raw,
            ),
            Step(
                "calibrate (+ gold check)",
                [
                    "calibrate",
                    str(D / "calib" / "clean_scored"),
                    str(D / "calib" / "corrupt_scored"),
                    str(C),
                    "--gold-scores",
                    str(D / "calib" / "gold_scored"),
                ],
                CPU,
                C / "calibration.yaml",
            ),
        ]
    if stage == "train":
        out = [
            Step(
                "data sft",
                [
                    "data",
                    "sft",
                    str(D / "sft"),
                    "--synthetic",
                    str(D / "train" / "corrupt_table.jsonl"),
                    "--gold",
                    str(D / "gold_chunks" / "sft"),
                    "--identity",
                    str(D / "chunks_clean"),
                ],
                CPU,
                D / "sft" / "train.jsonl",
            ),
            Step(
                "train sft",
                [
                    "train",
                    "sft",
                    str(D / "sft" / "train.jsonl"),
                    str(R / "sft"),
                    "--eval",
                    str(D / "sft" / "val.jsonl"),
                    *max_steps,
                ],
                FREE,
                R / "sft" / "ocrfix_train.json",
            ),
            Step(
                "merge sft",
                ["train", "merge", str(R / "sft"), sft_model],
                FREE,
                Path(sft_model) / "config.json",
            ),
        ]
        if dpo:
            out += [
                Step(
                    "corrupt (dpo)",
                    [
                        "corrupt",
                        str(D / "chunks_clean"),
                        *E,
                        str(D / "train" / "corrupt_dpo.jsonl"),
                    ],
                    sets=["seed=7"],
                    done=D / "train" / "corrupt_dpo.jsonl",
                ),
                Step(
                    "data dpo",
                    [
                        "data",
                        "dpo",
                        str(D / "dpo" / "pairs.jsonl"),
                        "--synthetic",
                        str(D / "train" / "corrupt_dpo.jsonl"),
                        "--gold",
                        str(D / "gold_chunks" / "dpo"),
                        "--exclude",
                        str(D / "sft" / "train.jsonl"),
                    ],
                    BOTH,
                    D / "dpo" / "pairs.jsonl",
                    sets=cal,
                    tuned=sft_model,
                ),
                Step(
                    "train dpo",
                    [
                        "train",
                        "dpo",
                        str(D / "dpo" / "pairs.jsonl"),
                        str(R / "sft"),
                        str(R / "dpo"),
                        *max_steps,
                    ],
                    FREE,
                    R / "dpo" / "ocrfix_train.json",
                ),
                Step(
                    "merge dpo",
                    ["train", "merge", str(R / "dpo"), dpo_model],
                    FREE,
                    Path(dpo_model) / "config.json",
                ),
            ]
        ev = D / "eval"
        out += [
            Step(
                "score gold eval",
                ["score", str(D / "gold_chunks" / "eval"), str(ev / "scored")],
                BASE,
                sets=cal,
            ),
            Step(
                "correct eval (base)",
                ["correct", str(ev / "scored"), str(ev / "base")],
                BASE,
                sets=[*cal, "correct.model=null"],
            ),
            Step(
                "correct eval (sft)",
                ["correct", str(ev / "scored"), str(ev / "sft")],
                BOTH,
                sets=cal,
                tuned=sft_model,
            ),
        ]
        systems = [f"base={ev / 'base'}", f"sft={ev / 'sft'}"]
        if dpo:
            out.append(
                Step(
                    "correct eval (dpo)",
                    ["correct", str(ev / "scored"), str(ev / "dpo")],
                    BOTH,
                    sets=cal,
                    tuned=dpo_model,
                )
            )
            systems.append(f"dpo={ev / 'dpo'}")
        out.append(
            Step(
                "eval", ["eval", str(D / "gold_fixed" / "truth"), *systems, str(ev / "report")], CPU
            )
        )
        return out
    if stage == "corpus":
        return [
            Step("chunk corpus", ["chunk", str(D / "corpus_fixed"), str(D / "chunks")]),
            Step(
                "score pass 1 (scores only)",
                ["score", str(D / "chunks"), str(D / "scored_raw")],
                BASE,
                sets=[*cal, "score.noise.remove=false", "score.noise.whole_chunks=false"],
            ),
            Step(
                "calibrate-noise-chunk",
                ["calibrate-noise-chunk", str(D / "gold_chunks"), str(D / "scored_raw"), str(C)],
                BASE,
                C / "noise_chunk_report.json",
                sets=cal,
            ),
            Step(
                "score pass 2 (noise removed)",
                ["score", str(D / "chunks"), str(D / "scored")],
                BASE,
                sets=cal,
            ),
            Step(
                "correct",
                ["correct", str(D / "scored"), str(D / "refined")],
                BOTH,
                sets=cal,
                tuned=tuned,
            ),
            Step("summary", ["summary", str(D / "refined")], CPU),
            Step(
                "report",
                [
                    "report",
                    str(D / "refined"),
                    "--clean",
                    str(D / "calib" / "clean_scored"),
                    "--json",
                    str(D / "corpus_report.json"),
                ],
                CPU,
                sets=cal,
            ),
            Step(
                "bpc by stage (raw / denoised / corrected)",
                [
                    "bpc-hist",
                    f"raw={D / 'scored_raw'}",
                    f"denoised={D / 'scored'}",
                    f"corrected={D / 'refined'}",
                    "--out",
                    str(D / "bpc_stages.png"),
                    "--json",
                    str(D / "bpc_stages.json"),
                ],
                CPU,
            ),
            Step(
                "reconstruct",
                ["reconstruct", str(D / "refined"), str(D / "corrected"), "--source", "current"],
                CPU,
            ),
        ]
    raise ValueError(f"unknown stage {stage!r}: {', '.join(STAGES)} or all")


# --- model servers ---------------------------------------------------------------------------


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


class Servers:
    """The engines the steps need, started and stopped as `./ocrfix serve` processes."""

    def __init__(self, cfg, profile: str | None, user_sets: list[str], log_dir: Path) -> None:
        self.cfg, self.profile, self.user_sets, self.log_dir = cfg, profile, user_sets, log_dir
        p = cfg.pipeline
        self.use_vllm = {
            "score": str(cfg.score.backend) == "vllm",
            "correct": str(cfg.correct.backend) == "vllm",
        }
        self.both_memory = [float(x) for x in p.both_memory]
        self.timeout = float(p.ready_timeout)
        self.procs: dict[str, tuple[subprocess.Popen, Any, str | None, list[str]]] = {}

    def _urls(self, kind: str, tuned: str | None) -> list[str]:
        from ocrfix.config import load_config

        sets = [*self.user_sets, *([f"correct.model={tuned}"] if tuned else [])]
        cfg = load_config(self.profile, sets)
        return list(cfg.hw.infer.score_urls if kind == "base" else cfg.hw.infer.correct_urls)

    def _start(self, kind: str, tuned: str | None, memory: float | None) -> None:
        urls = self._urls(kind, tuned)
        if all(_healthy(u) for u in urls):
            print(f"[pipeline] {kind} engines already running at {urls[0]}: used as they are")
            self.procs[kind] = (None, None, tuned, urls)  # type: ignore[assignment]
            return
        local = {"127.0.0.1", "localhost", "::1", "0.0.0.0", str(self.cfg.hw.infer.host)}
        remote = sorted({_host(u) for u in urls if _host(u) not in local})
        if remote:  # engines on other servers: this machine can't start them
            down = [u for u in urls if not _healthy(u)]
            what = (
                "serve --correct --set correct.model=" + str(tuned) if kind == "tuned" else "serve"
            )
            raise RuntimeError(
                f"{kind} engines on other servers don't answer: {', '.join(down)}. Start them "
                f"there (`./ocrfix {what} --set hw.infer.host=<that server's address>`), check "
                f"with `curl <url>/health`, then rerun the stage (finished steps are skipped)."
            )
        cmd = [sys.executable, "-X", "utf8", "-m", "ocrfix", "serve"]
        if kind == "tuned":
            cmd.append("--correct")
        if memory is not None:
            cmd += ["--memory", str(memory)]
        if self.profile:
            cmd += ["--profile", self.profile]
        for s in [*self.user_sets, *([f"correct.model={tuned}"] if tuned else [])]:
            cmd += ["--set", s]
        log = (self.log_dir / f"serve-{kind}.log").open("a", encoding="utf-8")
        print(f"[pipeline] starting {kind} engines ({' '.join(cmd[5:])}); log {log.name}")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not all(_healthy(u) for u in urls):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{kind} engines exited (code {proc.returncode}); see {log.name}"
                )
            if time.time() - t0 > self.timeout:
                proc.terminate()
                raise RuntimeError(f"{kind} engines not ready after {self.timeout:.0f}s")
            time.sleep(5)
        print(f"[pipeline] {kind} engines ready ({time.time() - t0:.0f}s)")
        self.procs[kind] = (proc, log, tuned, urls)

    def _stop(self, kind: str) -> None:
        proc, log, _, _ = self.procs.pop(kind)
        if proc is None:
            return  # not ours
        print(f"[pipeline] stopping {kind} engines")
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    def need(self, step: Step) -> None:
        """Bring the servers to what ``step`` needs."""
        if step.needs == CPU:
            return
        if step.needs == FREE:
            local = {"127.0.0.1", "localhost", "::1", "0.0.0.0", str(self.cfg.hw.infer.host)}
            for kind in list(self.procs):
                if self.procs[kind][0] is None:
                    if any(_host(u) in local for u in self.procs[kind][3]):
                        print(
                            f"[pipeline] warning: {kind} engines you started on this machine "
                            "are still running; training may run out of GPU memory"
                        )
                    self.procs.pop(kind)
                else:
                    self._stop(kind)
            return
        want_base = self.use_vllm["score"] or (step.needs == BASE and self.use_vllm["correct"])
        want_tuned = step.needs == BOTH and self.use_vllm["correct"]
        split = want_base and want_tuned and float(self.cfg.hw.infer.gpu_memory_utilization) > 0.5
        base_mem = self.both_memory[0] if split else None
        if "tuned" in self.procs and (not want_tuned or self.procs["tuned"][2] != step.tuned):
            self._stop("tuned")
        if (
            "base" in self.procs
            and self.procs["base"][0] is not None
            and want_tuned
            and split
            and getattr(self, "_base_mem", None) != base_mem
        ):
            self._stop("base")  # restart at half memory, to make room for the trained model
        if want_base and "base" not in self.procs:
            self._start("base", None, base_mem)
            self._base_mem = base_mem
        if want_tuned and "tuned" not in self.procs:
            self._start("tuned", step.tuned, self.both_memory[1] if split else None)

    def stop_all(self) -> None:
        for kind in list(self.procs):
            self._stop(kind)


# --- running ---------------------------------------------------------------------------------


def run_step(step: Step, cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONUTF8": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            log.write(line)
            if "it/s]" not in line and "examples/s]" not in line:  # progress bars: log only
                print("    " + line.rstrip())
        return proc.wait()


def run_pipeline(
    stage: str,
    cfg,
    *,
    profile: str | None,
    user_sets: list[str],
    dry_run: bool = False,
    redo: bool = False,
    start: int = 1,
) -> int:
    stages = list(STAGES) if stage == "all" else [stage]
    D = Path(cfg.pipeline.data_dir)
    log_dir = D / "pipeline_logs"
    servers = Servers(cfg, profile, user_sets, log_dir)
    n = 0
    t_all = time.time()
    try:
        for st in stages:
            print(f"=== stage {st}")
            for step in plan(st, cfg):
                n += 1
                cmd = step.command(profile, user_sets)
                done = step.done is not None and step.done.exists() and not redo
                tag = "done, skipped" if done else ("skipped (--from)" if n < start else "")
                if dry_run:
                    print(f"{n:>3} [{step.needs:>4}] {step.name:<34} {tag}")
                    print("      ./ocrfix " + " ".join(cmd[5:]))
                    continue
                if done or n < start:
                    print(f"{n:>3} {step.name}: {tag}")
                    continue
                if not dry_run:
                    log_dir.mkdir(parents=True, exist_ok=True)
                servers.need(step)
                print(f"{n:>3} {step.name}")
                t0 = time.time()
                safe = (
                    step.name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                )
                rc = run_step(step, cmd, log_dir / f"{n:02d}-{safe}.log")
                print(f"    -> exit {rc} in {time.time() - t0:.0f}s")
                if rc != 0:
                    print(
                        f"[pipeline] stopped at step {n} ({step.name}); fix it, then rerun "
                        f"(finished steps are skipped) or add --from {n}"
                    )
                    return rc
    finally:
        if not dry_run:
            servers.stop_all()
    if not dry_run:
        print(
            f"[pipeline] {' + '.join(stages)}: {n} steps in {time.time() - t_all:.0f}s. "
            f"Logs: {log_dir}"
        )
        print(NEXT.get(stages[-1], ""))
    return 0


NEXT = {
    "prepare": "Check before `pipeline calibrate`: <data_dir>/corpus_fixed/markup_report.jsonl "
    "(kept_original entries), gold_chunks/stats.json (most chunks ok), errors/summary.md.",
    "calibrate": "Check before `pipeline train`: <calibration_dir>/summary.md (rules per "
    "language; 'Real gold OCR chunks': heavily damaged chunks should be flagged).",
    "train": "Check before `pipeline corpus`: <data_dir>/eval/report/report.md (the trained "
    "model's CER should be below base and ocr; few false fixes).",
    "corpus": "Done: corrected files in <data_dir>/corrected, quality in corpus_report.json "
    "and bpc_stages.png / .json (mean and histogram: raw, after noise removal, corrected); "
    "<calibration_dir>/noise_chunk_samples.md shows what whole-chunk noise removal dropped.",
}
