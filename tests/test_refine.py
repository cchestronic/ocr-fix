"""`ocrfix refine score|correct` on chunks, against an in-process fake vLLM (httpx
MockTransport) and, for the transformers backend, the tiny model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest

from ocrfix.calibrate import Calibration, Rule, save_calibration
from ocrfix.chunks.rebuild import run_rebuild
from ocrfix.chunks.run import run_chunk
from ocrfix.config import load_config
from ocrfix.io import iter_jsonl
from ocrfix.refine.prompt import parse_answer
from ocrfix.refine.report import corpus_report, format_report
from ocrfix.refine.run import run_correct, run_score, summarize, text_checks
from ocrfix.refine.score import parse_prompt_logprobs

LN2 = math.log(2)
DOCS = {
    "a/damaged.md": "The results in Tab1e 3 show that the model works well on every page. " * 3
    + "\n\nWe use $x^2$ here and the method converges quickly on long documents.\n",
    "b/clean.md": "The model is trained on clean documents and the loss decreases steadily. " * 3
    + "\n\nThe encoder keeps the layout and the order of the paragraphs intact.\n",
}


class FakeVLLM:
    """/v1/completions: 1 bit per token, 12 bits for a token containing '1' (prompt_logprobs);
    /v1/chat/completions: the user message with ``fix`` applied."""

    def __init__(self, tokenizer_dir: Path, fix=lambda t: t.replace("Tab1e", "Table")):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.fix = fix
        self.calls: list[tuple[str, str]] = []  # (host, path)
        self.fail_chat = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.url.host + f":{request.url.port}", request.url.path))
        if request.url.path == "/v1/completions":
            plp = [None]
            for t in body["prompt"][1:]:
                bits = 12.0 if "1" in self.tok.decode([t]) else 1.0
                plp.append({str(t): {"logprob": -bits * LN2, "rank": 1}})
            return httpx.Response(200, json={"choices": [{"text": "", "prompt_logprobs": plp}]})
        if self.fail_chat:
            return httpx.Response(400, json={"error": "bad request"})
        user = body["messages"][-1]["content"]
        msg = {"role": "assistant", "content": "<think>\n\n</think>\n\n" + self.fix(user)}
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}]})


@pytest.fixture
def setup(tmp_path, tiny_model_dir):
    root = tmp_path / "md"
    for rel, text in DOCS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    cal = tmp_path / "calibration.yaml"
    save_calibration(Calibration(tau=8.0, rules={"*": Rule(spike_frac=0.0)}), cal)

    def cfg(*extra):
        return load_config("rtx5090", [
            f"paths.base_model={tiny_model_dir}", "chunk.min_tokens=40", "chunk.max_tokens=80",
            "refine.scorer=vllm", f"scorer.calibration={cal}", "hw.dtype=float32",
            "refine.score_urls=[http://127.0.0.1:9001]",
            "refine.correct_urls=[http://127.0.0.1:9002]", "llm.retries=0", *extra,
        ])  # fmt: skip

    run_chunk(cfg(), root, tmp_path / "chunks", workers=0)
    return tmp_path, cfg, FakeVLLM(tiny_model_dir)


def rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def test_parse_prompt_logprobs():
    plp = [None, {"5": {"logprob": -LN2}}, {"7": {"logprob": -3 * LN2}, "9": {"logprob": -0.1}}]
    assert parse_prompt_logprobs(plp, [0, 5, 7]).tolist() == pytest.approx([1.0, 3.0])
    with pytest.raises(ValueError, match="no logprob"):
        parse_prompt_logprobs([None, {"6": {"logprob": -1}}], [0, 5])
    with pytest.raises(ValueError, match="prompt tokens"):
        parse_prompt_logprobs([None], [0, 5])


def test_parse_answer_drops_think():
    assert parse_answer("<think>\nhmm\n</think>\n\n  fixed text \n") == "fixed text"


def test_score_writes_scores_into_chunks(setup):
    tmp, cfg, fake = setup
    st = run_score(cfg(), tmp / "chunks", tmp / "scored", transport=httpx.MockTransport(fake))
    out = rows(tmp / "scored")
    assert st.chunks == len(out) == len(rows(tmp / "chunks")) and st.status["scored"] == len(out)
    damaged = [r for r in out if "Tab1e" in r["source_text"]]
    clean = [r for r in out if r["source_path"] == "b/clean.md"]
    assert damaged and all(r["flag"] and r["score"]["spike_frac"] > 0 for r in damaged)
    assert all(r["flag"] is False and r["score"]["spike_frac"] == 0 for r in clean)
    for r in out:
        s = r["score"]
        assert s["n_tokens"] > 0 and s["n_prose_tokens"] < s["n_tokens"]  # start + boundary skipped
    assert {h for h, p in fake.calls} == {"127.0.0.1:9001"}
    assert json.loads((tmp / "scored" / "score_stats.json").read_text())["chunks"] == len(out)
    # resume: nothing left to do
    again = run_score(cfg(), tmp / "chunks", tmp / "scored", transport=httpx.MockTransport(fake))
    assert again.shards_skipped == again.shards


def test_placeholder_and_its_next_tokens_are_not_scored(setup):
    _, cfg, _ = setup
    from ocrfix.refine.score import Encoder, tokenize_chunk

    enc = Encoder(str(cfg().paths.tokenizer))
    tp = tokenize_chunk("We use ⟨M1⟩ here and the method converges quickly.", enc, 2)
    text = tp.text
    ph_end = text.index("⟩") + 1
    after = [i for i in range(len(tp)) if tp.starts[i] >= ph_end and tp.prose[i]]
    assert tp.boundary[after[:2]].all() and not tp.boundary[after[2:]].any()
    assert not any(tp.prose[i] for i in range(len(tp)) if tp.starts[i] < ph_end <= tp.ends[i])
    assert tp.boundary[np.flatnonzero(tp.prose)[:2]].all()  # the chunk start too


def test_correct_rescore_verify_per_chunk(setup):
    tmp, cfg, fake = setup
    t = httpx.MockTransport(fake)
    run_score(cfg(), tmp / "chunks", tmp / "scored", transport=t)
    fake.calls.clear()
    st = run_correct(cfg(), tmp / "scored", tmp / "refined", transport=t)
    out = rows(tmp / "refined")
    accepted = [r for r in out if r["status"] == "accepted"]
    assert accepted and st.status["accepted"] == len(accepted)
    for r in accepted:
        assert "Table 3" in r["text"] and "Tab1e" in r["source_text"]
        h = r["history"][-1]
        assert h["accepted"] and h["candidate"] == r["text"]
        assert h["score_after"]["bpc"] < h["score_before"]["bpc"] and r["score"] == h["score_after"]
        assert r["flag"] is False
    assert all(r["status"] == "scored" and r["text"] is None
               for r in out if r["source_path"] == "b/clean.md")  # fmt: skip
    # correction went to 9002, the re-score to 9001
    assert ("127.0.0.1:9002", "/v1/chat/completions") in fake.calls
    assert ("127.0.0.1:9001", "/v1/completions") in fake.calls
    assert ("127.0.0.1:9002", "/v1/completions") not in fake.calls
    # the corrected markdown
    run_rebuild(tmp / "refined", tmp / "fixed_md", source="current")
    fixed = (tmp / "fixed_md" / "a/damaged.md").read_text()
    assert "Tab1e" not in fixed and fixed.count("Table 3") == 3 and "$x^2$" in fixed
    assert (tmp / "fixed_md" / "b/clean.md").read_text() == DOCS["b/clean.md"]
    assert not run_rebuild(tmp / "refined", None).mismatched  # originals still byte-exact
    s = summarize(tmp / "refined")
    assert s["status"]["accepted"] == len(accepted) and s["mean_bpc_gain_accepted"] > 0


@pytest.mark.parametrize(
    ("fix", "status", "check"),
    [
        (lambda t: t.replace("Tab1e", "Table") + " ⟨M9⟩", "rejected", "placeholders"),
        (lambda t: "Completely different text. " * 20, "rejected", "edit_ratio"),
        (lambda t: t.replace("Tab1e", "Tab1e $y$"), "rejected", "markers"),
        (lambda t: t, "unchanged", None),
    ],
)
def test_bad_corrections_are_rejected(setup, fix, status, check):
    tmp, cfg, fake = setup
    run_score(cfg(), tmp / "chunks", tmp / "scored", transport=httpx.MockTransport(fake))
    fake.fix = fix
    run_correct(cfg(), tmp / "scored", tmp / "refined", transport=httpx.MockTransport(fake))
    sent = [r for r in rows(tmp / "refined") if r["history"]]
    assert sent and all(r["status"] == status and r["text"] is None for r in sent)
    if check:
        assert any(c["check"] == check and not c["passed"]
                   for r in sent for c in r["history"][-1]["checks"])  # fmt: skip


def test_request_errors_and_selection(setup):
    tmp, cfg, fake = setup
    run_score(cfg(), tmp / "chunks", tmp / "scored", transport=httpx.MockTransport(fake))
    fake.fail_chat = True
    st = run_correct(cfg(), tmp / "scored", tmp / "err", transport=httpx.MockTransport(fake))
    assert st.status["error"] > 0 and st.failed["request"] == st.status["error"]
    fake.fail_chat = False
    st = run_correct(cfg(), tmp / "scored", tmp / "all", select_all=True,
                     transport=httpx.MockTransport(fake))  # fmt: skip
    clean = [r for r in rows(tmp / "all") if r["source_path"] == "b/clean.md"]
    assert clean and all(r["status"] == "unchanged" for r in clean)  # sent, nothing to fix
    no_cal = cfg("scorer.calibration=null")
    run_score(no_cal, tmp / "chunks", tmp / "nocal", transport=httpx.MockTransport(fake))
    with pytest.raises(ValueError, match="calibrate first"):
        run_correct(no_cal, tmp / "nocal", tmp / "x", transport=httpx.MockTransport(fake))


def test_text_checks_on_masked_chunks():
    m = {"⟨M1⟩": "$x$", "⟨T1⟩": "<table><tr><td>1</td></tr></table>"}
    src = "Tab1e has ⟨M1⟩ and ⟨T1⟩ in it."
    ok = text_checks(src, "Table has ⟨M1⟩ and ⟨T1⟩ in it.", m, VERIFY, LANG)
    assert all(c.passed for c in ok), ok
    swapped = text_checks(src, "Table has ⟨T1⟩ and ⟨M1⟩ in it.", m, VERIFY, LANG)
    assert [c.name for c in swapped] == ["placeholders"] and not swapped[0].passed


VERIFY = {"length_ratio": [0.8, 1.25], "max_edit_ratio": 0.15, "max_script_shift": 0.1,
          "gap_fill_max_syllables": 4}  # fmt: skip
LANG = {"dominant": 0.8, "mixed_min": 0.1, "hangul_weight": 2.0}


def test_transformers_backend_scores_chunks(setup, model_device):
    """The same command with the local model instead of a server (the tiny model)."""
    tmp, cfg, _ = setup
    # a random model finds all text surprising: noise removal would cut everything
    st = run_score(cfg("refine.scorer=transformers", "refine.noise.remove=false"),
                   tmp / "chunks", tmp / "tf")  # fmt: skip
    out = rows(tmp / "tf")
    assert st.status["scored"] == len(out)
    for r in out:
        s = r["score"]
        assert s["bpc"] > 0 and 0 <= s["spike_frac"] <= 1 and r["flag"] is not None


# --- calibration on chunks: corrupt -> score -> calibrate -> flags -> correct ----------------

SENTS = [
    "The model is trained on clean documents and the loss decreases steadily.",
    "The encoder keeps the layout and the order of the paragraphs intact.",
    "We measure the error rate on every page and report the mean over all runs.",
    "The method converges quickly on long documents with many tables.",
]


class VocabVLLM(FakeVLLM):
    """Scores 1 bit for a token seen in the clean chunks, 12 bits for any other token."""

    def __init__(self, tokenizer_dir: Path, known: set[int]):
        super().__init__(tokenizer_dir)
        self.known = known

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/completions":
            return super().__call__(request)
        body = json.loads(request.content)
        plp = [None] + [
            {str(t): {"logprob": -(1.0 if t in self.known else 12.0) * LN2}}
            for t in body["prompt"][1:]
        ]
        return httpx.Response(200, json={"choices": [{"text": "", "prompt_logprobs": plp}]})


@pytest.fixture
def calib(tmp_path, tiny_model_dir):
    import random

    from tests.test_corrupt import CMAP, TABLE

    r = random.Random(3)
    root = tmp_path / "clean_md"
    for i in range(40):
        p = root / f"d{i:02d}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n".join(" ".join(r.choice(SENTS) for _ in range(4)) for _ in range(4)))
    (tmp_path / "table.json").write_text(json.dumps(TABLE))
    (tmp_path / "cmap.json").write_text(json.dumps(CMAP))

    def cfg(*extra):
        return load_config("rtx5090", [
            f"paths.base_model={tiny_model_dir}", "chunk.min_tokens=40", "chunk.max_tokens=80",
            "refine.scorer=vllm", "scorer.calibration=null", "calibrate.min_rows=5",
            "refine.score_urls=[http://127.0.0.1:9001]",
            "refine.correct_urls=[http://127.0.0.1:9002]", "llm.retries=0", *extra,
        ])  # fmt: skip

    run_chunk(cfg(), root, tmp_path / "clean", workers=0)
    from ocrfix.refine.score import Encoder

    enc = Encoder(str(tiny_model_dir))
    known = {t for row in iter_jsonl(tmp_path / "clean") for t in enc(row["source_text"])[0]}
    return tmp_path, cfg, VocabVLLM(tiny_model_dir, known)


def test_refine_corrupt_damages_prose_only(calib):
    from ocrfix.refine.corrupt import run_refine_corrupt

    tmp, cfg, _ = calib
    out = tmp / "c.jsonl"
    n = run_refine_corrupt(cfg(), tmp / "clean", tmp / "table.json", tmp / "cmap.json", out,
                           rate=0.03, n_variants=2)  # fmt: skip
    rows_ = rows(out)
    assert n == len(rows_) == 2 * len(rows(tmp / "clean"))
    assert all(r["chunk_id"].endswith(("~r0.03~v0", "~r0.03~v1")) for r in rows_)
    assert sum(r["source_text"] != r["gold_text"] for r in rows_) > 0.9 * n
    assert all(r["placeholders"] == r["gold_placeholders"] for r in rows_)
    assert all(
        r["error_rate"] > 0 and r["applied_ops"]
        for r in rows_
        if r["source_text"] != r["gold_text"]
    )
    again = tmp / "c2.jsonl"
    run_refine_corrupt(cfg(), tmp / "clean", tmp / "table.json", tmp / "cmap.json", again,
                       rate=0.03, n_variants=2)  # fmt: skip
    assert again.read_bytes() == out.read_bytes()  # deterministic


def test_protected_runs_are_not_damaged(tiny_model_dir):
    from tests.test_corrupt import CMAP, TABLE

    from ocrfix.corrupt.run import build_models, corrupt_opts
    from ocrfix.refine.corrupt import corrupt_chunks

    cfg = load_config("rtx5090", [f"paths.base_model={tiny_model_dir}"])
    mapping = {"⟨P1⟩": "https://github.com/example/ocrfix", "⟨P2⟩": "[12]", "⟨M1⟩": "$x$"}
    text = "See ⟨P1⟩ and the results in ⟨P2⟩ with ⟨M1⟩ here, many more words follow it."
    row = {"chunk_id": "d-c00000", "doc_id": "d", "lang": "en", "source_text": text,
           "text": None, "placeholders": mapping}  # fmt: skip
    models = build_models(TABLE, CMAP, corrupt_opts(cfg))
    out = list(corrupt_chunks([row], models, seed=1, n_variants=30, rate=0.3,
                              protect=list(cfg.llm.protect)))  # fmt: skip
    assert all(r["placeholders"] == mapping for r in out)
    assert any(r["source_text"] != text for r in out)


def test_calibrate_on_chunks_then_correct_uses_the_flags(calib):
    from ocrfix.calibrate.rules import load_calibration
    from ocrfix.calibrate.run import run_calibrate
    from ocrfix.refine.corrupt import run_refine_corrupt

    tmp, cfg, fake = calib
    t = httpx.MockTransport(fake)
    for rate in ("0.01", "0.03", "0.1", None):
        name = f"r{rate}" if rate else "table"
        run_refine_corrupt(cfg(), tmp / "clean", tmp / "table.json", tmp / "cmap.json",
                           tmp / "corrupt" / f"{name}.jsonl", rate=rate and float(rate),
                           n_variants=1)  # fmt: skip
    run_score(cfg(), tmp / "clean", tmp / "clean_scored", transport=t)
    run_score(cfg(), tmp / "corrupt", tmp / "corrupt_scored", transport=t)
    # recall on the heavily damaged set: table rates leave many chunks with 1-2 errors
    cal, _ = run_calibrate(cfg("calibrate.recall_on=[r0.1]"), tmp / "clean_scored",
                           tmp / "corrupt_scored", tmp / "cal", plots=False)  # fmt: skip
    path = tmp / "cal" / "calibration.yaml"
    assert path.is_file() and cal.rules["*"].recall > 0.8 and cal.rules["*"].false_flag <= 0.05
    loaded = load_calibration(path)
    # score with the calibration: corrupted chunks flagged, clean ones not
    with_cal = cfg(f"scorer.calibration={path}")
    run_score(with_cal, tmp / "clean", tmp / "clean_flagged", transport=t)
    run_score(with_cal, tmp / "corrupt" / "r0.1.jsonl", tmp / "table_flagged", transport=t)
    clean_flags = [r["flag"] for r in rows(tmp / "clean_flagged")]
    bad_flags = [r["flag"] for r in rows(tmp / "table_flagged")]
    assert sum(clean_flags) <= 0.05 * len(clean_flags) + 1
    assert sum(bad_flags) >= 0.8 * len(bad_flags)
    assert loaded.tau == cal.tau
    # correct picks exactly the flagged chunks
    fake.fix = lambda s: s  # the fake corrector changes nothing -> "unchanged"
    run_correct(with_cal, tmp / "table_flagged", tmp / "refined", transport=t)
    out = rows(tmp / "refined")
    assert {r["chunk_id"] for r in out if r["status"] == "unchanged"} == {
        r["chunk_id"] for r in out if r["flag"] and not r["score"]["low_confidence"]
    }


# --- noise removal -----------------------------------------------------------------------------

NOISE = "ⅡⅢ▯ ¤ʘ ⁂ 丨⺈ ꞉ ∯ ⸿ ☊ ⅏ ⌬ ⍟ ⎈ ▩ ◬ ⊗⍰ ∵◊※"  # 23 non-space chars: over min_chars


def test_noise_spans_and_removal_units():
    from ocrfix.refine.noise import remove_spans

    t = "Good text here. ⅡⅢ▯ ¤ʘ ⁂ 丨⺈ more good text.\n\nNext ⟨M1⟩ para."
    a = t.index("Ⅱ")
    b = t.index("⺈") + 1
    assert remove_spans(t, [(a, b, 9.0)]) == "Good text here. more good text.\n\nNext ⟨M1⟩ para."
    assert remove_spans("x\n\nJUNK\n\ny", [(3, 7, 9.0)]) == "x\n\ny"
    assert remove_spans("JUNK rest", [(0, 4, 9.0)]) == "rest"


def _char_tp(parts):
    """One token per char; ``parts`` = [(text, bits per char)], bits None = placeholder."""
    import numpy as np

    from ocrfix.scorer.tokens import TokenizedPara

    text = "".join(t for t, _ in parts)
    bits = np.concatenate([np.full(len(t), 0.0 if b is None else b) for t, b in parts])
    prose = np.concatenate([np.full(len(t), b is not None) for t, b in parts])
    n = len(text)
    idx = np.arange(n, dtype=np.int32)
    tp = TokenizedPara(("", ""), ("",), 0, text, idx, idx, idx + 1, prose, np.zeros(n, bool))
    return tp, bits.astype(np.float32)


def test_noise_spans_word_hysteresis():
    from ocrfix.refine.noise import noise_spans

    core, ext = {"hangul": 8.0, "other": 5.0}, {"hangul": 6.0, "other": 3.5}
    junk = "ⅡⅢ▯¤ʘ ⁂丨⺈꞉∯ ⸿☊⅏⌬"

    def spans(parts):
        tp, b = _char_tp(parts)
        return [tp.text[a:e] for a, e, _ in noise_spans(tp, b, core, ext, min_chars=12)]

    # surprising Korean (5.5 bits/char: above the Latin core, below the Hangul extend) stays;
    # the warm Latin word next to the junk joins the run, the clean one does not
    assert spans([("검출기는 문자 오류율을 ", 5.5), ("tbe ", 4.0), (junk, 11.0),
                  (" clean words", 1.0)]) == ["tbe " + junk]  # fmt: skip
    # garbled Hangul is noise by the Hangul threshold
    garbled = "쵺혀쀇 묥밂폖 뷁똷쀍 퉈뺹푑"
    assert spans([("좋은 문장이다 ", 2.0), (garbled, 12.0), (" 끝.", 2.0)]) == [garbled]
    # a placeholder ends a run: neither side swallows it
    assert spans([(junk, 11.0), (" ", 11.0), ("⟨M1⟩", None), (" " + junk, 11.0)]) == [junk, junk]
    # shorter than min_chars: an OCR glitch, left to the corrector
    assert spans([("text ", 1.0), ("ⅡⅢ▯ ¤ʘ", 11.0), (" text", 1.0)]) == []


def test_score_removes_noise_keeps_ocr_errors_and_placeholders(calib):
    tmp, cfg, fake = calib
    rows_ = rows(tmp / "clean")
    base = rows_[0]
    sent = f"{SENTS[0]} {SENTS[1]} {SENTS[2]}"
    cut = len(SENTS[0]) + 1
    from ocrfix.refine.score import Encoder

    fake.known |= set(Encoder(str(cfg().paths.tokenizer))(sent)[0])  # normal text = known
    variants = {
        "inline": sent[:cut] + NOISE + " " + sent[cut:],
        "ocr": sent.replace("the", "tbe", 2).replace("model", "rnodel", 1),
        "whole": NOISE + "\n\n" + NOISE,
        "ph": sent[:cut] + "⟨M1⟩ " + NOISE + " ⟨M2⟩ " + sent[cut:],
    }
    maps = {"ph": {"⟨M1⟩": "$a$", "⟨M2⟩": "$b$"}}
    src = tmp / "noisy.jsonl"
    src.write_text("".join(
        json.dumps({**base, "chunk_id": f"x-{k}", "source_text": v, "text": None,
                    "placeholders": maps.get(k, {})}, ensure_ascii=False) + "\n"
        for k, v in variants.items()
    ))  # fmt: skip
    st = run_score(cfg(), src, tmp / "denoised", transport=httpx.MockTransport(fake))
    out = {r["chunk_id"].split("-")[1]: r for r in rows(tmp / "denoised")}
    assert out["inline"]["text"] == sent and out["inline"]["status"] == "scored"
    h = out["inline"]["history"][-1]
    assert h["stage"] == "denoise" and h["removed"][0]["text"] == NOISE
    assert h["score_before"]["bpc"] > out["inline"]["score"]["bpc"]
    assert out["ocr"]["text"] is None and not out["ocr"]["history"]  # OCR errors are kept
    assert out["whole"]["status"] == "noise" and out["whole"]["text"] == ""
    assert out["ph"]["text"] == sent[:cut] + "⟨M1⟩ ⟨M2⟩ " + sent[cut:]
    assert st.failed["noise_spans"] == 3  # the two blocks of "whole" are one span
    off = run_score(cfg("refine.noise.remove=false"), src, tmp / "kept",
                    transport=httpx.MockTransport(fake))  # fmt: skip
    assert all(r["text"] is None for r in rows(tmp / "kept")) and not off.failed
    # the corrector never sees a noise-only chunk
    fake.fix = lambda s: s
    run_correct(cfg(), tmp / "denoised", tmp / "c", select_all=True,
                transport=httpx.MockTransport(fake))  # fmt: skip
    assert {r["chunk_id"]: r["status"] for r in rows(tmp / "c")}["x-whole"] == "noise"


def test_report_stages_clean_gap_and_flags(tmp_path):
    def sc(bpc, n):
        return {"bpc": bpc, "n_chars": n, "spike_frac": 0.0, "p90": 1.0}

    a = {"lang": "en", "status": "accepted", "score": sc(0.5, 90), "history": [
        {"stage": "denoise", "score_before": sc(2.0, 100), "removed": [{"text": "@@@@ ####"}]},
        {"stage": "correct", "score_before": sc(1.0, 90), "accepted": True},
    ]}  # fmt: skip
    b = {"lang": "en", "status": "scored", "score": sc(0.5, 100)}
    tried = {"stage": "correct", "score_before": sc(1.5, 50), "accepted": False}
    c = {"lang": "ko", "status": "rejected", "score": sc(1.5, 50), "history": [tried]}
    for d, rows in (("out", [a, b, c]), ("clean", [{"lang": "en", "score": sc(0.25, 100)}])):
        (tmp_path / d).mkdir()
        (tmp_path / d / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cal = Calibration(tau=16.0, rules={"*": Rule(spike_frac=None, bpc=0.8)})
    rep = corpus_report(tmp_path / "out", cal, tmp_path / "clean")
    en = rep["langs"]["en"]
    assert en["raw"]["bpc"] == 1.25  # (2.0 x 100 + 0.5 x 100) / 200: weighted by chars
    assert en["denoised"]["bpc"] == round(140 / 190, 4)
    assert en["final"]["bpc"] == 0.5
    assert [en[s]["flagged"] for s in ("raw", "denoised", "final")] == [0.5, 0.5, 0]
    assert en["clean"]["bpc"] == 0.25 and en["final"]["excess"] == 0.25
    assert en["closed_by_denoise"] == round(1.25 - 140 / 190, 4)
    assert en["closed_by_correct"] == round(140 / 190 - 0.5, 4)
    assert (en["noise_chunks"], en["noise_chars"], en["sent"], en["accepted"]) == (1, 8, 1, 1)
    ko = rep["langs"]["ko"]
    assert ko["raw"]["bpc"] == ko["final"]["bpc"] == 1.5 and "clean" not in ko
    assert ko["final"]["flagged"] == 1 and (ko["sent"], ko["accepted"]) == (1, 0)
    assert rep["langs"]["all"]["chunks"] == 3
    text = format_report(rep)
    assert "closed: denoise" in text and "accepted 1 (100.0%)" in text
    assert "flagged raw" not in format_report(corpus_report(tmp_path / "out"))
