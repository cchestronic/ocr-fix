from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ocrfix.scorer.model import is_hybrid, load_model, score_sequences  # noqa: E402
from ocrfix.scorer.tokens import Tokenizer, token_masks  # noqa: E402

# --- masks ------------------------------------------------------------------------------


def test_token_masks():
    # masked text: "ab ⟨M1⟩ cd ef gh ij ⟨T1⟩ kl"; placeholders at [3, 7) and [20, 24)
    phs = [(3, 7), (20, 24)]
    tokens = [
        (0, 2), (2, 3),  # "ab", " "
        (3, 5), (5, 7),  # placeholder pieces
        (7, 10), (10, 13), (13, 16), (16, 19),  # 4 prose tokens after it
        (19, 21),  # straddles into the table placeholder
        (21, 24),  # inside it
        (24, 27),  # prose after the table
        (27, 27),  # empty offsets: not prose
    ]  # fmt: skip
    starts, ends = (np.array(x, dtype=np.int32) for x in zip(*tokens, strict=True))
    prose, boundary = token_masks(starts, ends, phs, boundary=3)
    assert prose.tolist() == [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
    assert boundary.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    assert (prose & ~boundary).tolist() == [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def rec(pid: str, text: str, order: int = 0, doc: str = "d", **extra):
    from ocrfix.corpus.tagger import tag_spans
    from ocrfix.schema import Record

    return Record(doc, pid, order, "en", text, tag_spans(text), "", extra)


def test_scored_text_is_masked_like_the_llm_input(tiny_model_dir):
    from ocrfix.llm.protect import mask

    text = "The energy $E = mc^2$ is in [12] and <table><tr><td>x</td></tr></table> after."
    r = rec("p0", text)
    with Tokenizer(str(tiny_model_dir), workers=0, protect=["citations"]) as tok:
        (tp,) = tok([r])
    assert tp.text == mask(text, r.spans, ["citations"])[0]
    assert tp.text == "The energy ⟨M1⟩ is in ⟨P1⟩ and ⟨T1⟩ after."
    assert "$" not in tp.text and "<table>" not in tp.text
    for i in range(len(tp)):
        piece = tp.text[tp.starts[i] : tp.ends[i]]
        if any(c in piece for c in "⟨⟩"):
            assert not tp.prose[i]
    after_m1 = [i for i in range(len(tp)) if tp.prose[i] and tp.starts[i] >= tp.text.index("is")]
    assert tp.boundary[after_m1[0]]


def test_literal_placeholder_is_scored_unmasked(tiny_model_dir):
    with Tokenizer(str(tiny_model_dir), workers=0) as tok:
        (tp,) = tok([rec("p0", "already ⟨M1⟩ here")])
    assert tp.mask_error and tp.text == "already ⟨M1⟩ here"


# --- windows ----------------------------------------------------------------------------


def fake_tp(pid: str, n: int, order: int, group=("d", None, None), text: str | None = None):
    from ocrfix.scorer.tokens import TokenizedPara

    text = text if text is not None else "w " * n
    starts = np.arange(n, dtype=np.int32) * 2
    return TokenizedPara(
        (group[0], pid), group, order, text, np.arange(n, dtype=np.int32) + 10, starts,
        starts + 2, np.ones(n, bool), np.zeros(n, bool),
    )  # fmt: skip


def test_plan_windows_packs_consecutive_paragraphs():
    from ocrfix.scorer.tokens import plan_windows

    tps = [fake_tp("p0", 200, 0), fake_tp("p1", 200, 1), fake_tp("p2", 200, 2),
           fake_tp("p3", 300, 3), fake_tp("p5", 10, 5),  # gap in order: new window
           fake_tp("q0", 10, 6, group=("e", None, None))]  # other doc: new window  # fmt: skip
    plans = plan_windows(tps, sep_len=1, target=512, max_tokens=768, boundary=3)
    # p0+p1+p2 = 602 >= target closes; p3 starts the next; p5 (order gap) and q0 alone
    assert [[i for i, _, _ in p] for p in plans] == [[0, 1, 2], [3], [4], [5]]
    assert all(sum(hi - lo for _, lo, hi in p) + len(p) - 1 <= 768 for p in plans)


def test_plan_windows_respects_max_and_variants():
    from ocrfix.scorer.tokens import plan_windows

    a = ("d", "r0.01", 0)
    b = ("d", "r0.01", 1)
    tps = [fake_tp("p0", 500, 0, a), fake_tp("p1", 300, 1, a),  # 500+1+300 > 768
           fake_tp("p0v1", 10, 0, b), fake_tp("p1v1", 10, 1, b)]  # fmt: skip
    plans = plan_windows(tps, sep_len=1, target=512, max_tokens=768, boundary=3)
    assert [[i for i, _, _ in p] for p in plans] == [[0], [1], [2, 3]]


def test_long_paragraph_split_at_sentence_end():
    from ocrfix.scorer.tokens import plan_windows

    # 2000 tokens of "w " with a sentence end at token 700
    text = list("w " * 2000)
    text[700 * 2 + 1] = "."
    tp = fake_tp("p0", 2000, 0, text="".join(text))
    plans = plan_windows([tp, fake_tp("p1", 10, 1)], 1, target=512, max_tokens=768, boundary=3)
    pieces = [(lo, hi) for p in plans for i, lo, hi in p if i == 0]
    assert pieces[0] == (0, 701)  # cut after the sentence end
    assert all(hi - lo <= 768 for lo, hi in pieces) and pieces[-1][1] == 2000
    assert [i for i, _, _ in plans[-1]] == [1]  # the next paragraph starts fresh
    assert tp.boundary[701:704].all() and not tp.boundary[704]  # split point not scored
    assert not tp.boundary[:701].any()


def test_window_bits_equal_prefix_scoring(tiny_model_dir, model_device):
    """A paragraph's bits in a window = scoring the window up to it (causal)."""
    from ocrfix.scorer.tokens import build_window, para_bits

    model = load_model(str(tiny_model_dir), "float32", "sdpa", torch.device(model_device))
    eos = model.config.eos_token_id
    tps = [fake_tp("p0", 30, 0), fake_tp("p1", 20, 1)]
    sep = np.array([5], dtype=np.int32)
    w = build_window(tps, [(0, 0, 30), (1, 0, 20)], sep)
    assert len(w.ids) == 51
    (full,) = score_sequences(model, [w.ids], mode="padded", max_tokens=4096, logits_chunk=64,
                              eos_id=eos)  # fmt: skip
    b0, b1 = para_bits(tps, [w], [full])
    (solo0,) = score_sequences(model, [tps[0].ids], mode="padded", max_tokens=4096,
                               logits_chunk=64, eos_id=eos)  # fmt: skip
    assert np.abs(b0 - solo0).max() < 1e-3  # first paragraph: no context either way
    assert np.abs(b1 - full[31:]).max() < 1e-6 and not np.isnan(b1).any()


# --- packed vs padded -------------------------------------------------------------------


def _seqs(vocab: int, lengths: list[int], seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(4, vocab, size=n).astype(np.int32) for n in lengths]


LENGTHS = [1, 5, 17, 3, 64, 40, 2, 33, 9, 120]


def _bits(model, seqs, mode, max_tokens=160, chunk=1024):
    return score_sequences(
        model, seqs, mode=mode, max_tokens=max_tokens, logits_chunk=chunk,
        eos_id=model.config.eos_token_id,
    )  # fmt: skip


def _max_diff(a, b) -> float:
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b, strict=True))


def test_packed_equals_padded(tiny_attn_model_dir):
    model = load_model(str(tiny_attn_model_dir), "float32", "sdpa", torch.device("cpu"))
    assert not is_hybrid(model.config)
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    packed = _bits(model, seqs, "packed")
    padded = _bits(model, seqs, "padded")
    solo = [_bits(model, [s], "padded")[0] for s in seqs]  # batch of one: no packing, no pads
    assert [len(b) for b in packed] == LENGTHS
    assert _max_diff(packed, padded) < 1e-3
    assert _max_diff(padded, solo) < 1e-3
    # chunked lm_head/cross-entropy gives the same numbers
    assert _max_diff(_bits(model, seqs, "padded", chunk=7), padded) < 1e-4
    assert all((b > 0).all() for b in padded)


def test_hybrid_padded_equals_solo(tiny_model_dir, model_device):
    """Right padding is exact for linear-attention (recurrent + conv) layers too."""
    model = load_model(str(tiny_model_dir), "float32", "sdpa", torch.device(model_device))
    assert is_hybrid(model.config)
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    padded = _bits(model, seqs, "padded")
    solo = [_bits(model, [s], "padded")[0] for s in seqs]
    assert _max_diff(padded, solo) < 1e-3


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not all(importlib.util.find_spec(m) for m in ("fla", "causal_conv1d")),
    reason="packing a hybrid model needs CUDA + the optional kernels (fla, causal_conv1d)",
)
def test_hybrid_packed_equals_padded(tiny_model_dir):
    model = load_model(str(tiny_model_dir), "float32", "sdpa", torch.device("cuda"))
    seqs = _seqs(model.config.vocab_size, LENGTHS)
    assert _max_diff(_bits(model, seqs, "packed"), _bits(model, seqs, "padded")) < 1e-3


# --- end to end -------------------------------------------------------------------------


def test_score_end_to_end(fixtures_dir, tiny_model_dir, tmp_path):
    from ocrfix.config import load_config
    from ocrfix.corpus import ingest
    from ocrfix.scorer.run import score

    paras = tmp_path / "corpus.jsonl"
    ingest(fixtures_dir / "sample_mineru", paras)
    n_paras = len(paras.read_text().splitlines())
    cfg = load_config(
        "rtx5090",
        [
            f"paths.base_model={tiny_model_dir}",
            "hw.dtype=float32",
            "hw.scorer.tokenize_workers=0",
            "hw.scorer.max_batch_tokens=256",
            "scorer.window.target_tokens=8",
            "scorer.window.max_tokens=12",
            "scorer.shard_size=4",
        ],
    )
    out = tmp_path / "scores.jsonl"
    dump = tmp_path / "tokens"
    assert score(cfg, paras, out, limit=5, dump_tokens=dump) == 5
    assert score(cfg, paras, out, dump_tokens=dump) == n_paras - 5
    assert score(cfg, paras, out) == 0  # resumed: nothing left
    assert not (tmp_path / "scores.jsonl.parts").exists()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == n_paras == len({(r["doc_id"], r["para_id"]) for r in rows})
    for r in rows:
        assert "spans" not in r and "text" not in r  # features only, no token arrays
        for k in ("n_prose_tokens", "bpc", "p99", "max_pos", "spike_frac", "low_confidence"):
            assert k in r
        assert not any(k.startswith("flat_") for k in r)
        assert "math_mean_bits" not in r and "truncated" not in r
        assert r["low_confidence"] == (r["n_prose_tokens"] < 10)
        if r["n_prose_tokens"]:
            assert r["p50"] <= r["p90"] <= r["p99"] <= r["max_bits"]
            assert 0 <= r["spike_frac"] <= 1
    math_only = [r for r in rows if r["n_prose_tokens"] == 0]
    assert math_only and all(r["bpc"] is None and r["low_confidence"] for r in math_only)

    arr = np.load(next(dump.rglob("p00000.npy")))
    assert set(arr.dtype.names) >= {"id", "bits", "prose", "boundary"}


def test_score_with_calibration_and_passthrough(fixtures_dir, tiny_model_dir, tmp_path):
    from ocrfix.calibrate import Calibration, Rule, save_calibration
    from ocrfix.config import load_config
    from ocrfix.corpus import ingest
    from ocrfix.scorer.run import score

    paras = tmp_path / "corpus.jsonl"
    ingest(fixtures_dir / "sample_mineru", paras)
    rows = [json.loads(line) for line in paras.read_text().splitlines()]
    tagged = tmp_path / "tagged.jsonl"
    tagged.write_text(
        "".join(
            json.dumps(r | {"corruption": "r0.01", "junk": 1}, ensure_ascii=False) + "\n"
            for r in rows
        )
    )
    cal_path = tmp_path / "calibration.yaml"
    save_calibration(Calibration(tau=12.0, rules={"*": Rule(spike_frac=0.0)}), cal_path)
    cfg = load_config(
        "rtx5090",
        [
            f"paths.base_model={tiny_model_dir}",
            "hw.dtype=float32",
            "hw.scorer.tokenize_workers=0",
            f"scorer.calibration={cal_path}",
            "scorer.tau_candidates=[8,12]",
        ],
    )
    out = tmp_path / "scores.jsonl"
    score(cfg, tagged, out)
    for r in (json.loads(line) for line in out.read_text().splitlines()):
        assert r["corruption"] == "r0.01" and "junk" not in r and r["lang"]
        assert "spike_frac@8" in r and "flat_spike_frac@8" not in r
        assert "spike_frac@12" not in r  # tau 12 is the calibrated tau -> plain spike_frac
        want = r["spike_frac"] is not None and r["spike_frac"] > 0 and not r["low_confidence"]
        assert r["flag"] == want


def test_score_windows_match_single_paragraph_resume(fixtures_dir, tiny_model_dir, tmp_path):
    """Scoring in one go = scoring with a limit and resuming (whole documents re-windowed)."""
    from ocrfix.config import load_config
    from ocrfix.corpus import ingest
    from ocrfix.scorer.run import score

    paras = tmp_path / "corpus.jsonl"
    ingest(fixtures_dir / "sample_mineru", paras)
    cfg = load_config("rtx5090", [
        f"paths.base_model={tiny_model_dir}", "hw.dtype=float32",
        "hw.scorer.tokenize_workers=0", "scorer.calibration=null",
    ])  # fmt: skip
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    score(cfg, paras, one)
    score(cfg, paras, two, limit=6)
    score(cfg, paras, two)

    def load(p):
        return {(r["doc_id"], r["para_id"]): r for r in map(json.loads, p.read_text().splitlines())}

    a, b = load(one), load(two)
    assert a.keys() == b.keys()
    for k in a:
        if a[k]["bpc"] is not None:
            assert abs(a[k]["bpc"] - b[k]["bpc"]) < 1e-3, k


def test_rescore_uses_window_context(tiny_model_dir, model_device):
    """verify's rescore of an unchanged text reproduces the score (same window context)."""
    from ocrfix.config import load_config
    from ocrfix.scorer.run import InProcessScorer

    cfg = load_config("rtx5090", [
        f"paths.base_model={tiny_model_dir}", "hw.dtype=float32",
        "hw.scorer.tokenize_workers=0", "scorer.calibration=null",
    ])  # fmt: skip
    recs = [rec(f"p{i}", t, order=i) for i, t in enumerate([
        "The first paragraph sets the context for the rest.",
        "A second one, with Tab1e and $x$ inside.",
        "A third paragraph closes the document here.",
    ])]  # fmt: skip
    with InProcessScorer(cfg, torch.device(model_device)) as sc:
        base = {(f["doc_id"], f["para_id"]): f for f in sc(recs)}
        same = sc.rescore(recs, {("d", "p1"): recs[1].text})
        fixed = sc.rescore(recs, {("d", "p1"): "A second one, with Table and $x$ inside."})
        alone = sc([recs[1]])[0]
    assert abs(same[("d", "p1")]["bpc"] - base[("d", "p1")]["bpc"]) < 1e-3
    assert abs(alone["bpc"] - base[("d", "p1")]["bpc"]) > 1e-4  # context matters
    assert fixed[("d", "p1")]["n_prose_tokens"] > 0
