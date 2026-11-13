"""`ocrfix chunk` / `reconstruct`: size limits, atoms never cut, byte-exact round trip."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ocrfix.chunks.rebuild import rebuild_doc, run_rebuild
from ocrfix.chunks.run import chunk_rows, run_chunk
from ocrfix.chunks.split import normalize, raw_slices, split_document
from ocrfix.cli import app
from ocrfix.config import load_config
from ocrfix.corpus.placeholders import PLACEHOLDER, restore

PROTECT = ["images", "code", "urls", "emails", "dois", "citations"]
_TOKEN = re.compile(r"\s*⟨|\s*\S{1,3}|\s+")


def encode(text: str):
    """A stand-in tokenizer: up to 3 chars per token, leading whitespace joins the token."""
    ms = list(_TOKEN.finditer(text))
    return [m.start() for m in ms], [m.end() for m in ms]


def split(raw: str, lo: int = 40, hi: int = 60):
    return split_document(raw, encode, min_tokens=lo, max_tokens=hi, protect=PROTECT)


def rows(raw: str, lo: int = 40, hi: int = 60):
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return chunk_rows(split(raw, lo, hi), "a/b.md", sha, lo)


def round_trip(raw: str, lo: int = 40, hi: int = 60) -> list[dict]:
    rs = rows(raw, lo, hi)
    assert rebuild_doc(rs) == raw.encode("utf-8")
    return rs


SENT = "The energy $E = mc^2$ is shown in Tab1e 1 and [12]. "
KO = "본 연구에서는 오류를 $x_i$ 교정한다. "


def test_sizes_and_preferred_cuts():
    paras = [(SENT * 6 + KO * 2).strip() for _ in range(12)]
    raw = "\n\n".join(paras) + "\n"
    rs = round_trip(raw, 100, 160)
    assert len(rs) > 2
    for r in rs[:-1]:
        assert 100 <= r["n_tokens"] <= 160 and not r["short"]
        assert r["suffix"].strip() == "" and r["suffix"]
    assert rs[-1]["suffix"] == "\n" and rs[-1]["n_tokens"] <= 160
    # whole paragraphs fit the window, so every cut is at a blank line
    assert all(r["suffix"] == "\n\n" for r in rs[:-1])


def test_cuts_fall_to_sentence_then_whitespace_then_token():
    rs = round_trip(SENT * 60)  # no blank lines
    assert all(r["suffix"] == " " and r["source_text"].endswith(".") for r in rs[:-1])
    rs = round_trip("word " * 400)  # no sentence ends
    assert all(r["suffix"] == " " for r in rs[:-1])
    rs = round_trip("가" * 2000)  # no whitespace at all
    assert len(rs) > 2 and all(r["suffix"] == "" for r in rs)
    assert all(r["n_tokens"] <= 60 for r in rs)


def test_masking_and_numbering_per_chunk():
    raw = "\n\n".join(
        [f"Text {i} has $a_{i}$ and <table><tr><td>{i}</td></tr></table> and http://x.org/{i} end."
         for i in range(40)]
    )  # fmt: skip
    rs = round_trip(raw)
    for r in rs:
        for kind in "MTP":
            found = re.findall(rf"⟨{kind}(\d+)⟩", r["source_text"])
            assert [int(x) for x in found] == list(range(1, len(found) + 1))
        assert "$" not in r["source_text"] and "<table>" not in r["source_text"]
        assert "http" not in r["source_text"]
        assert list(r["placeholders"]) == PLACEHOLDER.findall(r["source_text"])
        assert restore(r["source_text"], r["placeholders"]).count("$") % 2 == 0


def test_a_long_table_is_one_atom():
    table = "<table>" + "<tr><td>cell</td></tr>" * 5000 + "</table>"
    rs = round_trip(f"Before.\n\n{table}\n\nAfter it.")
    assert len(rs) == 1 and rs[0]["source_text"] == "Before.\n\n⟨T1⟩\n\nAfter it."
    assert rs[0]["short"]


def test_blank_lines_inside_math_are_not_cuts():
    math = "\\begin{align}\na &= b\n\n" * 1 + "c &= d\n\\end{align}"
    rs = round_trip((SENT * 3 + "\n\n" + math + "\n\n") * 20)
    assert all("⟨M" in r["source_text"] for r in rs)
    assert all("\\begin" not in r["source_text"] for r in rs)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n\n  \n",
        "\ufeff",
        "\ufeffHello.\r\nSecond line.\r\n\r\nThird.\r\n",
        "lone\rcr and\r\ncrlf\n",
        "already has ⟨M1⟩ and ⟨T2⟩ literally, with $x$ too.",
        unicodedata.normalize("NFD", "한글 문장입니다. " * 300),
        unicodedata.normalize("NFD", "한글" * 800) + "\n",
        "cafe\u0301 " * 300,
        "\ufeff" + unicodedata.normalize("NFD", "가각 $x$ 나\r\n\r\n") * 200,
        "tail spaces   \n\n\n   leading\t\n",
        "a" * 5000,
        "$$\nx\n$$",
        "e\u0301",
        "\u0301 starts with a combining mark. " * 100,
    ],
)
def test_round_trip_is_byte_exact(raw):
    rs = round_trip(raw)
    assert rs and all(r["n_chunks"] == len(rs) for r in rs)
    assert all(r["n_tokens"] <= 60 for r in rs)
    assert [r["chunk_index"] for r in rs] == list(range(len(rs)))


def test_literal_placeholder_survives_masking():
    rs = round_trip("already has ⟨M1⟩ here, and $x$ too.")
    assert rs[0]["source_text"] == "already has ⟨P1⟩ here, and ⟨M1⟩ too."
    assert rs[0]["placeholders"] == {"⟨P1⟩": "⟨M1⟩", "⟨M1⟩": "$x$"}


def test_raw_only_where_needed():
    clean = round_trip("Plain NFC text. " * 100)
    assert all(r["raw"] is None for r in clean)
    mixed = "Plain NFC text. " * 60 + "\n\n" + unicodedata.normalize("NFD", "한글 문장. ") * 60
    rs = round_trip(mixed)
    assert rs[0]["raw"] is None and rs[-1]["raw"] is not None
    assert all(normalize(r["raw"]) == r["raw"] or r["raw"] != "" for r in rs if r["raw"])


def test_raw_slices_inside_a_run():
    raw = unicodedata.normalize("NFD", "한글문장")
    norm = normalize(raw)
    a, b = raw_slices(raw, norm, [2])
    assert a + b == raw and normalize(a) == "한글" and normalize(b) == "문장"


def test_random_documents_round_trip():
    rng = random.Random(7)
    bits = [
        SENT, KO, "\n\n", "\n", "\r\n", "  ", "$$y^2$$", "`code`", "http://a.org/x_y",
        "<table><tr><td>$1$</td></tr></table>", "⟨P9⟩", "\u0301", "각", "\u1100\u1161", "...",
        "| a | b |\n|---|---|\n| 1 | 2 |\n", "\\begin{equation}\n\nz\n\\end{equation}", "[3, 5-7]",
    ]  # fmt: skip
    mid_short = total = 0
    for _ in range(150):
        raw = "".join(rng.choice(bits) for _ in range(rng.randint(0, 200)))
        rs = round_trip(raw, 20, 30)
        assert all(r["n_tokens"] <= 30 for r in rs)
        mid_short += sum(r["short"] and r["chunk_index"] < len(rs) - 1 for r in rs)
        total += len(rs)
    assert mid_short <= total * 0.05  # tokens that can't be cut evenly, not a pattern


def test_tail_is_balanced_when_it_can_be():
    # 224 tokens in 100..160: greedy would leave a short tail; both parts can be >= 100
    rs = round_trip(SENT * 14, 100, 160)
    assert [r["n_tokens"] for r in rs] == [112, 112]


def test_current_text_uses_corrections_and_keeps_untouched_bytes():
    raw = "\ufeffOne Tab1e $x$ here.\r\n\r\n" + "Second part goes on. " * 100
    rs = rows(raw)
    assert rs[0]["raw"] is not None
    assert rebuild_doc(rs, "current") == raw.encode()  # nothing corrected yet
    assert rs[-1]["text"] is None
    rs[-1]["text"] = rs[-1]["source_text"].replace("Second", "2nd")
    out = rebuild_doc(rs, "current").decode()
    assert out.startswith("\ufeffOne Tab1e $x$ here.\r\n") and "2nd part" in out


def test_rebuild_refuses_missing_chunks():
    rs = rows("Words go here. " * 200)
    with pytest.raises(ValueError, match="expected chunks"):
        rebuild_doc(rs[:-1])


# --- the whole command, real files ---------------------------------------------------------


def make_tree(root: Path) -> dict[str, bytes]:
    files = {
        "a/one.md": (SENT * 200).encode(),
        "a/b/c/two.md": ("\ufeff" + "줄 하나 $x$.\r\n\r\n" * 100).encode(),
        "a/b/decomposed.md": unicodedata.normalize("NFD", "한글 문장입니다. " * 200).encode(),
        "empty.md": b"",
        "한글폴더/세 번째.md": ("표 앞. <table><tr><td>1</td></tr></table>\n" * 50).encode(),
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "a/not_markdown.txt").write_text("ignored")
    (root / "bad.md").write_bytes(b"bad \xff\xfe bytes")
    return files


@pytest.fixture
def chunk_cfg(tiny_model_dir):
    def make(*extra):
        return load_config(
            "rtx5090",
            [f"paths.base_model={tiny_model_dir}", "chunk.min_tokens=64", "chunk.max_tokens=96",
             "chunk.shard_bytes=4000", *extra],
        )  # fmt: skip

    return make


@pytest.mark.parametrize("workers", [0, 2])
def test_run_chunk_end_to_end(tmp_path, chunk_cfg, workers):
    root = tmp_path / "in"
    files = make_tree(root)
    out = tmp_path / "out"
    st = run_chunk(chunk_cfg(), root, out, workers=workers)
    assert st.docs == 5 and st.parts >= 2 and st.max_tokens <= 96 and st.mid_short == 0
    assert [e["path"] for e in st.errors] == ["bad.md"]
    assert json.loads((out / "stats.json").read_text())["chunks"] == st.chunks

    st2 = run_rebuild(out, tmp_path / "back")
    assert st2.docs == 5 and not st2.mismatched and not st2.errors
    for rel, data in files.items():
        assert (tmp_path / "back" / rel).read_bytes() == data
    assert not (tmp_path / "back" / "bad.md").exists()

    # resume: nothing to do
    again = run_chunk(chunk_cfg(), root, out, workers=0)
    assert again.parts_skipped == again.parts and again.chunks == st.chunks


def test_resume_refuses_another_chunking_and_runs_missing_parts(tmp_path, chunk_cfg):
    root = tmp_path / "in"
    make_tree(root)
    out = tmp_path / "out"
    st = run_chunk(chunk_cfg(), root, out, workers=0)
    shard = sorted(out.glob("part-*.jsonl"))[0]
    before = shard.read_bytes()
    shard.unlink()
    again = run_chunk(chunk_cfg(), root, out, workers=0)
    assert again.parts_skipped == st.parts - 1 and shard.read_bytes() == before
    assert again.chunks == st.chunks
    with pytest.raises(ValueError, match="different chunking"):
        run_chunk(chunk_cfg("chunk.max_tokens=200"), root, out, workers=0)


def test_gzip_limit_and_sample(tmp_path, chunk_cfg):
    root = tmp_path / "in"
    make_tree(root)
    st = run_chunk(chunk_cfg("io.gzip=true"), root, tmp_path / "gz", workers=0, limit=2)
    assert st.docs + len(st.errors) == 2 and list((tmp_path / "gz").glob("part-*.jsonl.gz"))
    assert not run_rebuild(tmp_path / "gz", None).mismatched
    none = run_chunk(chunk_cfg(), root, tmp_path / "s", workers=0, sample_frac=1e-9)
    assert none.docs == 0


def test_cli_chunk_and_reconstruct(tmp_path, tiny_model_dir):
    root = tmp_path / "in"
    make_tree(root)
    sets = ["--set", f"paths.base_model={tiny_model_dir}", "--set", "chunk.min_tokens=64",
            "--set", "chunk.max_tokens=96"]  # fmt: skip
    r = CliRunner().invoke(app, ["chunk", str(root), str(tmp_path / "o"), "--workers", "0", *sets])
    assert r.exit_code == 0, r.output
    assert "5 docs" in r.output and "bad.md" in r.output
    r = CliRunner().invoke(app, ["reconstruct", str(tmp_path / "o")])
    assert r.exit_code == 0 and "0 differ" in r.output
    row = json.loads(next((tmp_path / "o").glob("part-*.jsonl")).read_text().splitlines()[0])
    assert {"chunk_id", "source_text", "text", "placeholders", "raw", "n_tokens", "lang",
            "score", "status", "prefix", "suffix", "doc_sha256"} <= set(row)  # fmt: skip


def test_every_profile_names_a_model():
    from ocrfix.config import available_profiles

    for name in available_profiles():
        assert load_config(name).paths.base_model
    assert load_config("h200x4", ["paths.base_model=/x"]).llm.model == "/x"
