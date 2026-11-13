from __future__ import annotations

import gzip
import shutil
import unicodedata

from typer.testing import CliRunner

from ocrfix.cli import app
from ocrfix.corpus import doc_id_for, ingest
from ocrfix.io import iter_records


def test_ingest(fixtures_dir, tmp_path):
    src = fixtures_dir / "sample_mineru"
    out = tmp_path / "corpus.jsonl"
    stats = ingest(src, out)
    assert (stats.docs_written, stats.docs_skipped) == (3, 0)

    recs = list(iter_records(out))
    assert len(recs) == stats.paragraphs
    by_doc: dict[str, list] = {}
    for r in recs:
        by_doc.setdefault(r.doc_id, []).append(r)
        assert r.text == unicodedata.normalize("NFC", r.text)
        assert r.text == r.text.strip()
        assert r.doc_id == doc_id_for(r.source_path)
    for paras in by_doc.values():
        assert [p.order for p in paras] == list(range(len(paras)))
        assert [p.para_id for p in paras] == [f"p{i:05d}" for i in range(len(paras))]

    en = by_doc[doc_id_for("paper_en/auto/paper_en.md")]
    # display math and the align* env (which contains a blank line) stay whole paragraphs
    assert [s.kind for s in en[4].spans] == ["math"]
    assert en[4].text.startswith("\\begin{align*}") and en[4].text.endswith("\\end{align*}")
    ko = by_doc[doc_id_for("paper_ko/auto/paper_ko.md")]
    assert ko[1].text.startswith("실험 결과는")  # NFD on disk, NFC in output
    assert {r.lang for r in recs} == {"en", "ko", "mixed", "other"}

    # idempotent: rerun skips everything but the last doc, which is redone identically
    before = out.read_bytes()
    stats = ingest(src, out)
    assert (stats.docs_written, stats.docs_skipped) == (1, 2)
    assert out.read_bytes() == before

    # resume after a crash that tore the last line
    out.write_bytes(before[: len(before) - 25])
    ingest(src, out)
    assert out.read_bytes() == before


def test_ingest_limit_and_gzip(fixtures_dir, tmp_path):
    src = fixtures_dir / "sample_mineru"
    out = tmp_path / "corpus.jsonl.gz"
    assert ingest(src, out, limit=2).docs_written == 2
    assert ingest(src, out).docs_written == 1
    plain = tmp_path / "plain.jsonl"
    ingest(src, plain)
    with gzip.open(out, "rb") as f:
        assert f.read() == plain.read_bytes()


def test_ingest_cli(fixtures_dir, tmp_path):
    src = tmp_path / "in"
    shutil.copytree(fixtures_dir / "sample_mineru", src)
    out = tmp_path / "corpus.jsonl"
    res = CliRunner().invoke(app, ["ingest", str(src), str(out), "--limit", "1"])
    assert res.exit_code == 0, res.output
    assert "1 docs" in res.output
    assert len({r.doc_id for r in iter_records(out)}) == 1
