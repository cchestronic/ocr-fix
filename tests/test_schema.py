from __future__ import annotations

import unicodedata

from ocrfix.io import ShardedWriter, iter_records, read_ids
from ocrfix.schema import Record


def test_schema_roundtrip(fixtures_dir, tmp_path):
    recs = list(iter_records(fixtures_dir / "records.jsonl"))
    assert len(recs) == 3
    assert recs[2].extra == {"ocr_score": 0.42}
    assert {s.kind for r in recs for s in r.spans} == {"prose", "math", "table"}

    out = tmp_path / "out"
    with ShardedWriter(out, shard_size=2, gzip_output=True) as w:
        for r in recs:
            w.write(r)
    assert sorted(p.name for p in out.iterdir()) == ["part-00000.jsonl.gz", "part-00001.jsonl.gz"]

    back = list(iter_records(out))
    assert back == recs
    assert [Record.from_dict(r.to_dict()) for r in back] == recs
    assert read_ids(out) == {r.key for r in recs}

    # Decomposed Hangul (NFD) is normalized to NFC on input.
    nfd = unicodedata.normalize("NFD", "실험")
    assert Record.from_dict({**recs[0].to_dict(), "text": nfd, "spans": []}).text == "실험"
