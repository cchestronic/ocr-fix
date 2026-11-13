from __future__ import annotations

import pytest

from ocrfix.chunks.split import atoms_of, render
from ocrfix.corpus.placeholders import PlaceholderError, restore
from ocrfix.corpus.tagger import tag_spans


def substitute(text: str) -> tuple[str, dict[str, str]]:
    """Mask math and tables the way the chunker does."""
    return render(text, atoms_of(text, tag_spans(text), []), 0, len(text))


TEXT = "Let $x$ and \\(y\\) hold.\n<table><tr><td>$1</td></tr></table>\n$$z$$ 끝 〈?〉"


def test_roundtrip():
    masked, mapping = substitute(TEXT)
    assert masked == "Let ⟨M1⟩ and ⟨M2⟩ hold.\n⟨T1⟩\n⟨M3⟩ 끝 〈?〉"
    assert list(mapping) == ["⟨M1⟩", "⟨M2⟩", "⟨T1⟩", "⟨M3⟩"]
    assert mapping["⟨T1⟩"] == "<table><tr><td>$1</td></tr></table>"
    assert restore(masked, mapping) == TEXT
    # prose edits survive restore
    fixed = masked.replace("hold", "holds")
    assert restore(fixed, mapping) == TEXT.replace("hold", "holds")


def test_no_spans():
    assert substitute("plain") == ("plain", {})
    assert restore("plain", {}) == "plain"


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ("Let ⟨M1⟩ and hold.\n⟨T1⟩\n⟨M3⟩", "missing placeholder"),
        ("Let ⟨M1⟩ ⟨M1⟩ and ⟨M2⟩ hold.\n⟨T1⟩\n⟨M3⟩", "duplicated placeholder"),
        ("Let ⟨M2⟩ and ⟨M1⟩ hold.\n⟨T1⟩\n⟨M3⟩", "out of order"),
        ("Let ⟨M1⟩ and ⟨M2⟩ ⟨P1⟩ hold.\n⟨T1⟩\n⟨M3⟩", "unknown placeholder"),
    ],
)
def test_restore_rejects(broken, message):
    _, mapping = substitute(TEXT)
    with pytest.raises(PlaceholderError, match=message):
        restore(broken, mapping)
