"""`ocrfix calibrate-noise`: labels from gold, inserted junk, the search, writing score.yaml."""

from __future__ import annotations

import json
import math
import random
import shutil

import httpx
import numpy as np
import pytest
import yaml

from ocrfix.calibrate.noise import (
    _SYMBOLS,
    INSERTED,
    REAL,
    collect,
    insert_junk,
    junk,
    noise_labels,
    removed_words,
    run_calibrate_noise,
    write_score_yaml,
)
from ocrfix.config import config_dir, load_config
from ocrfix.stages.noise import noise_spans
from ocrfix.stages.score import tokenize_chunk

LN2 = math.log(2)


def test_labels_deleted_junk_only():
    ocr = "Results ▯▯▯ ※¶ are good. 방볍 here 쨈몤듡 end ⟨M1⟩."
    gold = "Results are good. 방법 here 〈?〉 end ⟨M1⟩."
    lab = noise_labels(ocr, gold)
    noisy = "".join(c for c, x in zip(ocr, lab, strict=True) if x == REAL)
    assert noisy.replace(" ", "") == "▯▯▯※¶"  # deleted junk
    # a corrected word and a word replaced by 〈?〉 are text
    for w in ("방볍", "쨈몤듡", "Results", "⟨M1⟩"):
        i = ocr.index(w)
        assert not lab[i : i + len(w)].any()


def test_insert_junk_keeps_the_text_and_labels_it():
    text = "First sentence here. Second one ⟨M1⟩ there.\nNext line."
    lab = np.zeros(len(text), dtype=np.int8)
    for seed in range(40):
        t, ll = insert_junk(text, lab, random.Random(seed), (1, 3), (5, 20))
        assert len(ll) == len(t)
        kept = "".join(c for c, x in zip(t, ll, strict=True) if x != INSERTED)
        assert " ".join(kept.split()) == " ".join(text.split())  # only junk (and a separator) added
        assert "⟨M1⟩" in t and t.count("⟨") == 1
        n = int((ll == INSERTED).sum())
        assert n >= 5
    for _ in range(200):
        j = junk(random.Random(_), 5, 30)
        assert 5 <= sum(not c.isspace() for c in j) <= 30
        assert not set(j) & set("$|`\\<>[]#*_⟨⟩〈〉")


def _tp(text: str):
    """One token per character (ids = code points), so bits map to chars directly."""
    return tokenize_chunk(
        text, lambda t: ([ord(c) for c in t], [(i, i + 1) for i in range(len(t))]), 2
    )


def test_fast_search_applies_exactly_the_score_rule():
    rng = random.Random(0)
    words = ["text", "결과는", "▯▯▯", "lIl1", "⟨M1⟩", "쨈몤듡", ".,;", "1.5", "model", "※¶§"]
    texts, labels, tps, bits = [], [], [], []
    for _ in range(60):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(5, 40)))
        tp = _tp(text)
        b = np.array([rng.choice([0.5, 2.0, 4.0, 6.0, 9.0, 13.0]) for _ in text], dtype=np.float32)
        texts.append(text)
        labels.append(np.zeros(len(text), dtype=np.int8))
        tps.append(tp)
        bits.append(b)
    w = collect(texts, labels, tps, bits)
    settings = [
        ((8.0, 5.0), (6.0, 3.5), 8),
        ((6.0, 3.5), (5.0, 2.0), 4),
        ((12.0, 8.0), (9.0, 5.5), 12),
    ]
    for (ch, co), (eh, eo), m in settings:
        core, ext = {"hangul": ch, "other": co}, {"hangul": eh, "other": eo}
        mask = removed_words(w, core, ext, [m])[0]
        fast = np.bincount(w.chunk[mask], weights=w.size[mask], minlength=len(texts))
        exact = [sum(sum(not c.isspace() for c in t[a:b]) for a, b, _ in
                     noise_spans(tp, bb, core, ext, min_chars=m))
                 for t, tp, bb in zip(texts, tps, bits, strict=True)]  # fmt: skip
        assert fast.tolist() == exact


def test_write_score_yaml(tmp_path):
    path = tmp_path / "score.yaml"
    shutil.copy(config_dir() / "stages" / "score.yaml", path)
    before = path.read_text()
    fit = {"model": "/models/Qwen3.5-9B", "chunks": 123, "text_removed": 0.00001, "recall": 0.7,
           "min_chars": 16, "core": {"hangul": 9.0, "other": 4.5},
           "extend": {"hangul": 7.0, "other": 3.0}}  # fmt: skip
    for _ in range(2):  # a second write replaces the first, it doesn't stack
        write_score_yaml(path, fit)
    after = path.read_text()
    d = yaml.safe_load(after)["score"]["noise"]
    assert d["core"] == fit["core"] and d["extend"] == fit["extend"] and d["min_chars"] == 16
    assert after.count("# fitted by `ocrfix calibrate-noise`") == 1 and "Qwen3.5-9B" in after
    assert "set by hand" not in after
    # every other line is untouched
    strip = lambda s: [x for x in s.splitlines()  # noqa: E731
                       if not x.strip().startswith(("min_chars", "core", "extend", "# fitted",
                                                    "# set by hand"))]  # fmt: skip
    assert strip(before) == strip(after)


class JunkScorer:
    """A 'vLLM' that finds junk symbols and loose jamo surprising, everything else not."""

    def __init__(self, tokenizer_dir) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        plp = [None]
        for t in body["prompt"][1:]:
            s = self.tok.decode([t])
            odd = any(c in _SYMBOLS or "ㄱ" <= c <= "ㅣ" or c in "lI1!:;" for c in s)
            plp.append({str(t): {"logprob": -(14.0 if odd else 0.5) * LN2}})
        return httpx.Response(200, json={"choices": [{"prompt_logprobs": plp}]})


def test_end_to_end(tmp_path, tiny_model_dir):
    gold = tmp_path / "gold_chunks" / "sft"
    gold.mkdir(parents=True)
    rows = []
    for i in range(30):
        truth = f"The model {i} reaches good results on every page. The loss decreases steadily."
        ocr = truth.replace("results on", "results ▯▯◘ ※¶§ ▯▯▯ ※※ on") if i % 3 == 0 else truth
        rows.append({"chunk_id": f"d{i}-c0", "doc_id": f"d{i}", "source_text": ocr,
                     "gold_text": truth, "gold_status": "ok", "split": "sft"})  # fmt: skip
    (gold / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = load_config("rtx5090", [f"model={tiny_model_dir}", "score.backend=vllm",
                                  "score.noise.remove=false", "calibrate.noise.min_noise_chars=50",
                                  "calibrate.noise.max_text_removed=0.01",
                                  "hw.infer.score_urls=[http://127.0.0.1:9001]"])  # fmt: skip
    target = tmp_path / "score.yaml"
    shutil.copy(config_dir() / "stages" / "score.yaml", target)
    fake = httpx.MockTransport(JunkScorer(tiny_model_dir))
    fit = run_calibrate_noise(
        cfg, tmp_path / "gold_chunks", tmp_path / "out", score_yaml=target, transport=fake
    )
    assert fit["use"] and fit["written"] == str(target)
    # the annotator's deletions are counted apart (ignored); the inserted junk is the noise
    assert fit["deleted_chars"] > 0 and fit["noise_chars"] > fit["deleted_chars"]
    assert fit["text_removed"] <= 0.01 and fit["recall"] > 0 and fit["min_chars"] >= 20
    assert isinstance(fit["text_removed_examples"], list)
    assert yaml.safe_load(target.read_text())["score"]["noise"]["min_chars"] == fit["min_chars"]
    report = json.loads((tmp_path / "out" / "noise_report.json").read_text())
    assert report["top"] and "previous" in report


def test_needs_gold(tmp_path):
    (tmp_path / "g" / "sft").mkdir(parents=True)
    with pytest.raises(ValueError, match="no `gold chunks` rows"):
        run_calibrate_noise(load_config("rtx5090"), tmp_path / "g", tmp_path / "o", score_yaml=None)


def test_chinese_text_is_never_noise():
    """No spaces: a Chinese sentence is one long word with many bits per char."""
    text = "本文提出了一种新的光学字符识别方法实验结果表明该方法有效 and ▯▯▯⌘※¶∴⊗■□▪▫▲△▼▽◆◇○●◎★ x."
    tp = _tp(text)
    bits = np.full(len(text), 12.0, dtype=np.float32)
    core, ext = {"hangul": 8.0, "other": 5.0}, {"hangul": 6.0, "other": 3.5}
    removed = "".join(text[a:b] for a, b, _ in noise_spans(tp, bits, core, ext, min_chars=20))
    assert "▯▯▯⌘※¶" in removed and not any("一" <= c <= "鿿" for c in removed)
