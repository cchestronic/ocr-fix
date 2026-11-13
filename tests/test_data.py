from __future__ import annotations

import json
import random
from collections import Counter

import httpx
import pytest
from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.config import load_config
from ocrfix.corpus import tag_spans
from ocrfix.corpus.placeholders import PLACEHOLDER
from ocrfix.data.common import (
    EMPTY_THINK,
    Pair,
    assistant_content,
    group_chunks,
    mask_pair,
    to_chunk,
)
from ocrfix.data.dpo import _undo_some, constructed
from ocrfix.data.run import format_stats, run_dpo, run_sft, stats
from ocrfix.data.sft import quotas
from ocrfix.llm.parse import parse_chunk
from ocrfix.llm.prompt import SYSTEM

PROTECT = ["images", "code", "urls", "emails", "dois", "citations"]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


# --- masking / format ------------------------------------------------------------------------


def test_mask_pair_masks_every_span_like_correct():
    from ocrfix.corpus import tag_spans
    from ocrfix.llm.protect import mask

    src = "제안한 방법이 기존 방법 $x_1$ 보다 좋다 [3]. 결과는 $$y$$ 이다"
    gold = "제안한 방법이 기존 방법 $x_1$ 보다 좋다 [3]. 결과는 $$y$$ 이다."
    m = mask_pair(Pair("d", "p0", 0, "ko", src, gold, "gold"), PROTECT)
    assert m.input == "제안한 방법이 기존 방법 ⟨M1⟩ 보다 좋다 ⟨P1⟩. 결과는 ⟨M2⟩ 이다"
    assert m.target == "제안한 방법이 기존 방법 ⟨M1⟩ 보다 좋다 ⟨P1⟩. 결과는 ⟨M2⟩ 이다."
    assert m.mapping == {"⟨M1⟩": "$x_1$", "⟨P1⟩": "[3]", "⟨M2⟩": "$$y$$"}
    assert m.input == mask(src, tag_spans(src), PROTECT)[0]  # exactly what correct sends
    assert mask_pair(Pair("d", "p0", 0, "en", "has ⟨M1⟩", "has ⟨M1⟩", "gold"), PROTECT) is None


def test_mask_pair_drops_targets_that_change_a_span():
    """Garbled Hangul as math: masked at inference, so it is never taught as fixable."""
    src = "제안한 방법 $\\mathrm{Ol}$ 기존 방법 $x_1$ 보다 좋다."
    gold = "제안한 방법이 기존 방법 $x_1$ 보다 좋다."
    assert mask_pair(Pair("d", "p0", 0, "ko", src, gold, "gold"), PROTECT) is None
    # spans must also come back in order
    swapped = mask_pair(Pair("d", "p0", 0, "en", "$a$ then $b$", "$b$ then $a$", "gold"), PROTECT)
    assert swapped is None


@pytest.mark.parametrize("think", [True, False])
def test_assistant_turn_parses_back_to_target(think):
    pairs = [
        Pair(
            "d",
            "p0",
            0,
            "en",
            "Tab1e 1 has $x$ and http://a.org/b.",
            "Table 1 has $x$ and http://a.org/b.",
            "gold",
        ),
        Pair("d", "p1", 1, "en", "Seco nd one.", "Second one.", "gold"),
    ]
    items = [mask_pair(p, PROTECT) for p in pairs]
    content = assistant_content([m.target for m in items], think)
    assert content.startswith(EMPTY_THINK) is think
    results = parse_chunk(to_chunk(items), content)
    assert [r.text for r in results] == [p.target for p in pairs]


def test_group_chunks():
    items = [
        mask_pair(Pair("d", f"p{i}", i, "en", "x" * 10, "x" * 10, "s"), PROTECT) for i in range(20)
    ]
    chunks = group_chunks(items, random.Random(0), [1, 2, 3], [1, 1, 1], max_chars=1000)
    assert [m for c in chunks for m in c] == items  # consecutive, in order, all used
    assert {len(c) for c in chunks} <= {1, 2, 3} and len({len(c) for c in chunks}) > 1
    small = group_chunks(items, random.Random(0), [3], [1], max_chars=45)
    assert {len(c) for c in small} == {2}  # 3 x 20 chars > 45


def test_quotas():
    pools = {
        "synthetic": {"ko": [0] * 100, "en": [0] * 100},
        "gold": {"ko": [0] * 10, "en": [0] * 30},
        "identity": {"en": [0] * 50},
    }
    q = quotas(pools, {"synthetic": 0.6, "gold": 0.2, "identity": 0.2, "verified": 0.5},
               {"ko": 0.5, "en": 0.5}, None)  # fmt: skip
    # gold ko (10 at share 0.2 x 0.5) limits the total to 100; identity has only en
    assert q == {
        "synthetic": {"ko": 30, "en": 30},
        "gold": {"ko": 10, "en": 10},
        "identity": {"en": 20},
    }
    assert quotas(pools, {"synthetic": 1.0}, None, 50) == {"synthetic": {"ko": 25, "en": 25}}


# --- sft from fixture files ------------------------------------------------------------------

EN = ["The rnodel is efﬁcient on $x$ data.", "Results in Tab1e 2 show gains.", "We use [4] here."]
KO = ["본 연구 에서는 $y$ 를 사용 하였다.", "실험 걸과는 표 2 에 있다.", "정확도가 높았다."]


def fixture_files(tmp_path):
    corrupt, ident, gold, verified = [], [], [], []
    for d in range(12):
        lang, texts = ("en", EN) if d % 2 else ("ko", KO)
        doc = f"doc{d:02d}"
        for i, t in enumerate(texts):
            clean = (
                t.replace("rnodel", "model")
                .replace("efﬁcient", "efficient")
                .replace("Tab1e", "Table")
            )
            clean = clean.replace("걸과", "결과").replace(" 에서는", "에서는")
            for v in range(2):
                corrupt.append({
                    "doc_id": doc, "para_id": f"p{i:05d}~table~v{v}", "src_para_id": f"p{i:05d}",
                    "corruption": "table", "variant": v, "order": i, "lang": lang, "text": t,
                    "source_text": t, "gold_text": clean, "spans": [s.to_dict() for s in tag_spans(t)],
                })  # fmt: skip
            ident.append({"doc_id": doc, "para_id": f"p{i:05d}", "order": i, "lang": lang, "text": clean,
                          "spans": [s.to_dict() for s in tag_spans(clean)], "source_path": ""})  # fmt: skip
            gold.append({"doc_id": doc, "para_id": f"p{i:05d}", "order": i, "lang": lang,
                         "source_raw": t, "gold_raw": clean, "stratum": f"{lang}:low"})  # fmt: skip
            gain = 0.2 if i != 2 else 0.01  # the third paragraph is below the stricter threshold
            verified.append({"doc_id": doc, "para_id": f"p{i:05d}", "order": i, "lang": lang,
                             "accepted": i != 1 or d % 3 != 0, "original": t, "corrected": clean,
                             "before": {"bpc": 2.0}, "after": {"bpc": 2.0 - gain}})  # fmt: skip
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_docs": [], "eval_docs": ["doc00", "doc01"]}))
    return {
        "synthetic": write_jsonl(tmp_path / "corrupt.jsonl", corrupt),
        "identity": write_jsonl(tmp_path / "clean.jsonl", ident),
        "gold": write_jsonl(tmp_path / "gold.jsonl", gold),
        "verified": write_jsonl(tmp_path / "verified.jsonl", verified),
        "split": split,
    }


def sft_cfg(f, *extra):
    return load_config(
        "rtx5090",
        [f"data.exclude_split={f['split']}", "data.val_frac=0.2", "data.chunk.weights=[1,1,1]",
         "data.sft.lang_balance={ko: 0.5, en: 0.5}", *extra],
    )  # fmt: skip


def test_data_sft(tmp_path):
    f = fixture_files(tmp_path)
    cfg = sft_cfg(f, "data.empty_think=true")
    out = tmp_path / "sft"
    st = run_sft(
        cfg,
        out,
        synthetic=f["synthetic"],
        verified=f["verified"],
        gold=f["gold"],
        identity=f["identity"],
    )
    train, val = read_jsonl(out / "train.jsonl"), read_jsonl(out / "val.jsonl")
    rows = train + val
    assert len(rows) == st.total > 0 and st.excluded_eval_docs > 0
    assert not {r["doc_id"] for r in rows} & {"doc00", "doc01"}  # gold eval docs excluded
    assert not {r["doc_id"] for r in train} & {r["doc_id"] for r in val}  # split by document
    assert len({r["id"] for r in rows}) == len(rows)

    by_src = Counter(r["source"] for r in rows)
    assert set(by_src) == {"synthetic", "verified", "gold", "identity"}
    mix = {"synthetic": 0.6, "verified": 0.15, "gold": 0.1, "identity": 0.15}
    for s, w in mix.items():
        assert by_src[s] / len(rows) == pytest.approx(w, abs=0.05), s
    for s in by_src:  # language balance within every source
        langs = Counter(r["lang"] for r in rows if r["source"] == s)
        assert abs(langs["ko"] - langs["en"]) <= 1, (s, langs)

    for r in rows:
        sys_msg, user, asst = r["messages"]
        assert sys_msg == {"role": "system", "content": SYSTEM} and user["role"] == "user"
        assert asst["role"] == "assistant" and asst["content"].startswith(EMPTY_THINK)
        assert 1 <= len(r["para_ids"]) <= 3
        assert user["content"].count("<<<") == asst["content"].count("<<<") == len(r["para_ids"])
        assert PLACEHOLDER.findall(user["content"]) == PLACEHOLDER.findall(asst["content"])
        if r["source"] == "identity":
            assert asst["content"].removeprefix(EMPTY_THINK) == user["content"]
        if r["source"] == "synthetic":
            assert "ﬁ" not in user["content"]  # the unicode fixer ran on the input
            assert len(set(r["para_ids"])) == len(r["para_ids"])
            assert len({p.split("~")[2] for p in r["para_ids"]}) == 1  # one variant per chunk
    assert all(r["repeat"] in (0, 1, 2) for r in rows if r["source"] == "gold")
    assert st.pool["verified"] and sum(st.pool["verified"].values()) < sum(st.pool["gold"].values())

    # deterministic
    st2 = run_sft(cfg, tmp_path / "sft2", synthetic=f["synthetic"], verified=f["verified"], gold=f["gold"],
                  identity=f["identity"])  # fmt: skip
    assert (tmp_path / "sft2" / "train.jsonl").read_bytes() == (out / "train.jsonl").read_bytes()
    assert st2.total == st.total


def test_data_sft_no_think_and_single_source(tmp_path):
    f = fixture_files(tmp_path)
    cfg = sft_cfg(f, "data.empty_think=false", "data.sft.lang_balance=null")
    st = run_sft(cfg, tmp_path / "o", identity=f["identity"])
    rows = read_jsonl(tmp_path / "o" / "train.jsonl") + read_jsonl(tmp_path / "o" / "val.jsonl")
    assert len(rows) == st.total and all(r["source"] == "identity" for r in rows)
    assert not any(r["messages"][2]["content"].startswith("<think>") for r in rows)


# --- dpo -------------------------------------------------------------------------------------


def test_constructed_negatives():
    src = "The rnodel in Tab1e 2 uses $x$ and 〈?〉 data here."
    chosen = "The model in Table 2 uses $x$ and 〈?〉 data here."
    m = mask_pair(Pair("d", "p0", 0, "en", src, chosen, "gold"), PROTECT)
    got = dict(
        constructed(
            m, ["unchanged", "half_fixed", "marker_dropped", "gap_filled"], random.Random(0)
        )
    )
    assert got["unchanged"] == m.input
    assert got["half_fixed"] in ("The rnodel in Table 2 uses ⟨M1⟩ and 〈?〉 data here.",
                                 "The model in Tab1e 2 uses ⟨M1⟩ and 〈?〉 data here.")  # fmt: skip
    assert got["marker_dropped"] == "The model in Table 2 uses and 〈?〉 data here."
    assert "〈?〉" not in got["gap_filled"] and got["gap_filled"].startswith(
        "The model in Table 2 uses ⟨M1⟩ and "
    )
    assert _undo_some("abc", "abd", random.Random(0)) is None  # one edit: nothing to half-undo


def sampling_server(hits: Counter):
    """n choices: [chosen, fluent rewrite, placeholder lost, no markers]."""

    def handler(request):
        body = json.loads(request.content)
        hits["requests"] += 1
        hits["n"] = body["n"]
        assert body["temperature"] == 0.8 and body["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        user = body["messages"][1]["content"]
        fixed = user.replace("rnodel", "model").replace("Tab1e", "Table").replace("걸과", "결과")
        rewrite = (
            fixed.replace("<<<1>>>\n", "<<<1>>>\nIn summary, honestly, ")
            + " Moreover it is well known."
        )
        lost = PLACEHOLDER.sub("", fixed)
        texts = [fixed, rewrite, lost, "garbage without markers and more words"]
        choices = [{"message": {"content": t}, "finish_reason": "stop"} for t in texts[: body["n"]]]
        return httpx.Response(200, json={"choices": choices})

    return httpx.MockTransport(handler)


def test_data_dpo(tmp_path):
    f = fixture_files(tmp_path)
    cfg = sft_cfg(f, "data.empty_think=true", "llm.retries=0", "data.dpo.max_pairs_per_input=4")
    hits: Counter = Counter()
    out = tmp_path / "dpo.jsonl"
    st = run_dpo(cfg, out, gold=f["gold"], verified=f["verified"], transport=sampling_server(hits))
    rows = read_jsonl(out)
    assert hits["n"] == 4 and st.sampled_requests == hits["requests"] == st.inputs
    assert not {r["doc_id"] for r in rows} & {"doc00", "doc01"}
    per_input = Counter((r["source"], r["doc_id"], r["para_id"]) for r in rows)
    assert max(per_input.values()) <= 4
    types = Counter(r["rejected_type"] for r in rows)
    assert types["sampled:edit_ratio"] > 0 or types["sampled:length_ratio"] > 0
    assert types["unchanged"] > 0 and types["marker_dropped"] > 0
    for r in rows:
        assert r["prompt"][0]["content"] == SYSTEM and len(r["prompt"]) == 2
        c, rj = r["chosen"][0]["content"], r["rejected"][0]["content"]
        assert c != rj and c.startswith(EMPTY_THINK) and rj.startswith(EMPTY_THINK)
        assert r["rejected_type"] != "sampled:" and r["lang"] in ("ko", "en")
    # the correct completion (identical to chosen) is never a rejected answer;
    # fluent failures come before constructed ones for the same input
    first = {}
    for r in rows:
        first.setdefault((r["source"], r["doc_id"], r["para_id"]), r["rejected_type"])
    with_sample = [t for t in first.values() if t.startswith("sampled:")]
    assert with_sample and all(
        t in ("sampled:edit_ratio", "sampled:length_ratio") for t in with_sample
    )

    # resumable: a rerun samples nothing new
    before = hits["requests"]
    run_dpo(cfg, tmp_path / "dpo2.jsonl", gold=f["gold"], verified=f["verified"],
            transport=sampling_server(hits), responses=tmp_path / "dpo.responses.jsonl")  # fmt: skip
    assert hits["requests"] == before


def test_data_dpo_constructed_only_and_stats(tmp_path):
    f = fixture_files(tmp_path)
    cfg = sft_cfg(f, "data.dpo.constructed=[unchanged, marker_dropped]")
    out = tmp_path / "dpo.jsonl"
    st = run_dpo(cfg, out, gold=f["gold"], n_samples=0)
    rows = read_jsonl(out)
    assert st.sampled_requests == 0 and {r["rejected_type"] for r in rows} == {
        "unchanged",
        "marker_dropped",
    }

    run_sft(cfg, tmp_path / "sft", gold=f["gold"], identity=f["identity"])
    s = stats([tmp_path / "sft" / "train.jsonl", out])
    assert s["examples"] == len(rows) + len(read_jsonl(tmp_path / "sft" / "train.jsonl"))
    assert s["rejected_type"]["unchanged"] == sum(r["rejected_type"] == "unchanged" for r in rows)
    assert set(s["source"]) == {"gold", "identity"} and set(s["lang"]) == {"ko", "en"}
    text = format_stats(s)
    assert "by source:" in text and "by rejected_type:" in text
    res = CliRunner().invoke(app, ["data", "stats", str(out), "--json"])
    assert res.exit_code == 0 and json.loads(res.output)["rejected_type"]["unchanged"] > 0


def test_data_cli(tmp_path):
    f = fixture_files(tmp_path)
    res = CliRunner().invoke(
        app,
        ["data", "sft", str(tmp_path / "sft"), "--gold", str(f["gold"]), "--identity", str(f["identity"]),
         "--set", f"data.exclude_split={f['split']}"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "examples" in res.output and (tmp_path / "sft" / "stats.json").exists()
    res = CliRunner().invoke(
        app,
        ["data", "dpo", str(tmp_path / "d.jsonl"), "--gold", str(f["gold"]), "--n-samples", "0"],
    )
    assert res.exit_code == 0, res.output and "pairs" in res.output
