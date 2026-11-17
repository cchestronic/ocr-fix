#!/usr/bin/env python3
"""Generate a synthetic MinerU-like test corpus (Korean + English, math, tables). No network.

    python scripts/make_test_corpus.py data/test --ocr-mb 50 --clean-mb 20 [--seed 1]

Writes, deterministically for a seed:
  <out>/truth/   documents without errors (the answer key)
  <out>/ocr/     the same documents with OCR errors injected (same relative paths)
  <out>/clean/   other documents, error-free (for SFT data; never the test documents)
  <out>/ocr_errors.jsonl   per document: counts of each injected error type, the exact
                           noise strings inserted (random symbols / junk to be removed)

The text is assembled from sentence templates and a vocabulary, so it is fluent enough for a
language model to score but repetitive compared with real papers. Use it to test plumbing and
speed, not to judge model quality.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# --- vocabulary ------------------------------------------------------------------------------

EN = {
    "subj": [
        "The proposed method",
        "Our model",
        "This approach",
        "The baseline",
        "The encoder",
        "The detector",
        "The resulting system",
        "Each layer",
        "The optimizer",
        "The network",
    ],
    "verb": [
        "reduces",
        "improves",
        "increases",
        "stabilizes",
        "predicts",
        "estimates",
        "outperforms",
        "approximates",
        "captures",
        "preserves",
    ],
    "obj": [
        "the character error rate",
        "the training loss",
        "the recognition accuracy",
        "the spatial resolution",
        "the number of parameters",
        "the convergence speed",
        "the alignment between pages",
        "the noise in scanned documents",
        "the variance of the gradient",
        "the quality of the output",
    ],
    "cond": [
        "on all benchmarks",
        "in most settings",
        "under heavy noise",
        "for long documents",
        "with a small batch size",
        "after fine-tuning",
        "without extra data",
        "at inference time",
        "on the validation set",
        "in the low-resource setting",
    ],
    "conn": [
        "However,",
        "Moreover,",
        "In contrast,",
        "As a result,",
        "In addition,",
        "Therefore,",
        "Notably,",
        "Similarly,",
    ],
    "topic": [
        "Introduction",
        "Related Work",
        "Method",
        "Experiments",
        "Results",
        "Discussion",
        "Ablation Study",
        "Conclusion",
        "Implementation Details",
        "Limitations",
    ],
}
KO = {
    "subj": [
        "제안한 방법은",
        "본 모델은",
        "이 접근법은",
        "기존 방법은",
        "인코더는",
        "검출기는",
        "전체 시스템은",
        "각 계층은",
        "최적화 기법은",
        "신경망은",
    ],
    "obj": [
        "문자 오류율을",
        "학습 손실을",
        "인식 정확도를",
        "공간 해상도를",
        "매개변수 수를",
        "수렴 속도를",
        "페이지 간 정렬을",
        "스캔 문서의 잡음을",
        "기울기의 분산을",
        "출력 품질을",
    ],
    "verb": [
        "크게 줄였다.",
        "향상시켰다.",
        "안정적으로 유지하였다.",
        "정확하게 예측하였다.",
        "효과적으로 추정하였다.",
        "개선하였다.",
        "보존하였다.",
        "효율적으로 처리하였다.",
    ],
    "cond": [
        "모든 실험에서",
        "대부분의 설정에서",
        "잡음이 심한 경우에도",
        "긴 문서에 대해",
        "작은 배치 크기에서",
        "미세 조정 이후",
        "추가 데이터 없이",
        "추론 단계에서",
        "검증 데이터에서",
        "자원이 부족한 환경에서",
    ],
    "conn": ["그러나", "또한", "반면에", "그 결과", "이에 더하여", "따라서", "특히"],
    "topic": [
        "서론",
        "관련 연구",
        "제안 방법",
        "실험",
        "결과",
        "논의",
        "구성 요소 분석",
        "결론",
        "구현 세부 사항",
        "한계점",
    ],
}
MATH_INLINE = [
    "$x_i$",
    "$E = mc^2$",
    "$\\alpha = 0.5$",
    "$n = 50$",
    "$10^{-3}$",
    "$f(x)$",
    "$\\mathcal{L}_{ce}$",
    "$p(y \\mid x)$",
    "$\\theta^*$",
    "$O(n \\log n)$",
]
MATH_BLOCK = [
    "$$\n\\mathcal{L} = -\\sum_{t=1}^{T} \\log p(y_t \\mid y_{<t}, x)\n$$",
    "$$\n\\int_0^1 f(x)\\,dx = F(1) - F(0)\n$$",
    "\\begin{equation}\n\\hat{y} = \\arg\\max_{y} p(y \\mid x; \\theta)\n\\end{equation}",
    "\\begin{align*}\na &= b + c \\\\\nd &= e \\cdot f\n\\end{align*}",
]
CITES = ["[1]", "[3]", "[12]", "[4, 7]", "[2-5]"]
URLS = ["https://github.com/example/ocrfix", "https://doi.org/10.1145/3292500.3330701"]


def en_sentence(r: random.Random) -> str:
    s = " ".join(r.choice(EN[k]) for k in ("subj", "verb", "obj", "cond"))
    x = r.random()
    if x < 0.25:
        s += f", where {r.choice(MATH_INLINE)} denotes the input"
    elif x < 0.35:
        s += f" {r.choice(CITES)}"
    elif x < 0.40:
        s += f" (see Table {r.randint(1, 6)})"
    s += "."
    if r.random() < 0.25:
        s = f"{r.choice(EN['conn'])} {s[0].lower()}{s[1:]}"
    return s


def ko_sentence(r: random.Random) -> str:
    s = " ".join(r.choice(KO[k]) for k in ("cond", "subj", "obj", "verb"))
    x = r.random()
    if x < 0.25:
        s += f" 여기서 {r.choice(MATH_INLINE)} 는 입력을 나타낸다."
    elif x < 0.35:
        s = s[:-1] + f" {r.choice(CITES)}."
    elif x < 0.42:
        s = f"이 결과는 Transformer encoder 구조를 사용한 경우이다. {s}"
    if r.random() < 0.25:
        s = f"{r.choice(KO['conn'])} {s}"
    return s


def table(r: random.Random, lang: str) -> str:
    rows = r.randint(2, 6)
    if r.random() < 0.5:
        head = "<tr><th>model</th><th>CER</th><th>WER</th></tr>"
        body = "".join(
            f"<tr><td>m{i}</td><td>{r.uniform(1, 9):.1f}</td><td>{r.uniform(5, 20):.1f}</td></tr>"
            for i in range(rows)
        )
        return f"<table>{head}{body}</table>"
    lines = ["| 모델 | 정확도 |" if lang == "ko" else "| model | accuracy |", "|---|---:|"]
    lines += [f"| m{i} | {r.uniform(80, 99):.1f} |" for i in range(rows)]
    return "\n".join(lines)


def document(r: random.Random, target_bytes: int) -> str:
    lang = r.choice(["ko", "en", "en", "ko", "mixed"])
    out = [f"# {r.choice(EN['topic'] if lang == 'en' else KO['topic'])} {r.randint(1, 99)}"]
    size = 0
    while size < target_bytes:
        x = r.random()
        if x < 0.08:
            sec = r.choice(EN["topic"] if lang == "en" else KO["topic"])
            block = f"## {r.randint(1, 9)}. {sec}"
        elif x < 0.14:
            block = r.choice(MATH_BLOCK)
        elif x < 0.19:
            block = table(r, lang)
        elif x < 0.21:
            fig = r.randint(1, 40)
            block = f"![](images/fig_{fig}.jpg)\n\nFigure {fig % 9 + 1}: {en_sentence(r)}"
        elif x < 0.22:
            block = f"Code and data are available at {r.choice(URLS)}."
        else:
            n = r.randint(2, 7)
            pick = {"en": en_sentence, "ko": ko_sentence}.get(lang)
            sents = [(pick or r.choice([en_sentence, ko_sentence]))(r) for _ in range(n)]
            block = " ".join(sents)
        out.append(block)
        size += len(block.encode())
    return "\n\n".join(out) + "\n"


# --- OCR noise --------------------------------------------------------------------------------

CONFUSIONS = [
    ("rn", "m", "m"),
    ("l", "1", "l"),
    ("e", "c", "e"),
    ("O", "0", "O"),
    ("h", "b", "h"),
    ("i", "l", "i"),
    ("t", "f", "t"),
    ("cl", "d", "d"),
    ("fi", "ﬁ", "fi"),
]
KO_CONF = [("을", "올"), ("는", "늗"), ("다", "디"), ("였", "었"), ("방", "벙"), ("결", "곁")]
MASKED = re.compile(
    r"\$\$[\s\S]*?\$\$|\$[^$\n]*\$|\\begin\{[\s\S]*?\\end\{[^}]*\}|<table>[\s\S]*?</table>"
    r"|^\|[^\n]*$|!\[[^\]\n]*\]\([^)\n]*\)|https?://\S+|\[[0-9, -]+\]|^#[^\n]*$",
    re.M,
)


def corrupt(text: str, r: random.Random, rate: float) -> tuple[str, Counter]:
    """Inject OCR-like errors into prose only (math, tables, links, headings untouched)."""
    kinds: Counter = Counter()
    keep = [(m.start(), m.end()) for m in MASKED.finditer(text)]
    pieces, pos = [], 0
    for a, b in [*keep, (len(text), len(text))]:
        pieces.append((text[pos:a], True))
        pieces.append((text[a:b], False))
        pos = b
    out = []
    for piece, prose in pieces:
        if not prose or not piece:
            out.append(piece)
            continue
        words = re.split(r"(\s+)", piece)  # separators kept; a word never holds a newline
        for i, w in enumerate(words):
            if not w or w.isspace() or r.random() > rate:
                continue
            k = r.random()
            if k < 0.35:
                for a, b, _ in r.sample(CONFUSIONS, len(CONFUSIONS)):
                    if a in w:
                        words[i] = w.replace(a, b, 1)
                        kinds["char_confusion"] += 1
                        break
                else:
                    for a, b in KO_CONF:
                        if a in w:
                            words[i] = w.replace(a, b, 1)
                            kinds["char_confusion"] += 1
                            break
            elif k < 0.55 and i + 2 < len(words) and words[i + 1] == " ":
                words[i + 1] = ""  # missing space
                kinds["missing_space"] += 1
            elif k < 0.70 and len(w) > 3:
                j = r.randint(1, len(w) - 1)
                words[i] = w[:j] + " " + w[j:]  # extra space inside a word
                kinds["extra_space"] += 1
            elif k < 0.82 and len(w) > 5 and w.isascii() and w.isalpha():
                j = r.randint(2, len(w) - 2)
                words[i] = w[:j] + "-\n" + w[j:]  # hyphenated line break
                kinds["hyphen_wrap"] += 1
            elif k < 0.90 and len(w) > 2:
                j = r.randint(0, len(w) - 1)
                words[i] = w[:j] + w[j + 1 :]  # dropped char
                kinds["dropped_char"] += 1
            elif k < 0.96:
                words[i] = w + "\n"  # line break mid-sentence
                if i + 1 < len(words) and words[i + 1] == " ":
                    words[i + 1] = ""
                kinds["newline_in_sentence"] += 1
            else:
                words[i] = f"{r.randint(2, 300)} {w}"  # page number left in the text
                kinds["header_footer_insert"] += 1
        out.append("".join(words))
    return "".join(out), kinds


# --- random noise (OCR garbage to be removed, not corrected) ----------------------------------

SYMBOLS = "ⅡⅢ▯¤ʘ⁂丨⺈꞉∯⸿☊⅏⌬⍟⎈▩◬⅋◊⌘※¶§†‡∴∵≋≈⊕⊗⌂⍰"
LATIN_JUNK = [
    "lIl1",
    ":;.",
    ",lI1",
    "rnvv",
    "/|",
    "Il1l",
    ".,;:",
    "Il|1",
    "vvrn",
    ";:,.",
    "l1I",
    "\\",
]


def noise_run(r: random.Random) -> str:
    kind = r.choice(["symbols", "latin", "hangul"])
    n = r.randint(12, 40)
    if kind == "symbols":
        out = "".join(r.choice(SYMBOLS + " ") for _ in range(n))
    elif kind == "latin":
        out = " ".join(r.choice(LATIN_JUNK) for _ in range(n // 4 + 1))
    else:
        out = "".join(chr(r.randint(0xAC00, 0xD7A3)) + ("" if r.random() < 0.7 else " ")
                      for _ in range(n // 2))  # fmt: skip
    return out.strip() or "▯▯▯▯"


def add_noise(text: str, r: random.Random) -> tuple[str, list[str]]:
    """Insert noise runs after sentence ends in prose, and whole noise paragraphs."""
    keep = [(m.start(), m.end()) for m in MASKED.finditer(text)]
    spots = [m.end() for m in re.finditer(r"[.!?] (?=\S)", text)
             if not any(a <= m.start() < b for a, b in keep)]  # fmt: skip
    breaks = [m.end() for m in re.finditer(r"\n\n", text)]
    inserts: list[tuple[int, str]] = []
    for pos in r.sample(spots, min(len(spots), r.randint(1, 4))):
        inserts.append((pos, noise_run(r) + " "))
    for pos in r.sample(breaks, min(len(breaks), r.randint(0, 2))):
        inserts.append((pos, " ".join(noise_run(r) for _ in range(r.randint(2, 5))) + "\n\n"))
    out, last = [], 0
    for pos, s in sorted(inserts):
        out += [text[last:pos], s]
        last = pos
    out.append(text[last:])
    return "".join(out), [s.strip() for _, s in sorted(inserts)]


# --- main ------------------------------------------------------------------------------------


def write_tree(root: Path, r: random.Random, total_mb: float, prefix: str):
    total, i = 0, 0
    while total < total_mb * 1e6:
        size = int(min(max(r.lognormvariate(10.6, 0.9), 3_000), 1_500_000))
        doc = document(r, size)
        rel = Path(f"vol{i % 5}") / f"paper_{i % 17:02d}" / f"{prefix}_{i:05d}.md"
        yield rel, doc
        total += len(doc.encode())
        i += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ocr-mb", type=float, default=50)
    ap.add_argument("--clean-mb", type=float, default=20)
    ap.add_argument("--error-rate", type=float, default=0.03, help="share of words damaged")
    ap.add_argument("--noise-docs", type=float, default=0.3,
                    help="share of test documents that also get random noise")  # fmt: skip
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for sub in ("truth", "ocr", "clean"):
        if (a.out / sub).exists() and any((a.out / sub).iterdir()):
            raise SystemExit(f"{a.out / sub} is not empty; remove it or pick another folder")

    r = random.Random(a.seed)
    n = 0
    total = Counter()
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "ocr_errors.jsonl", "w", encoding="utf-8") as log:
        for rel, doc in write_tree(a.out, r, a.ocr_mb, "doc"):
            # the rate varies per document: some are clean, some heavily damaged
            rate = a.error_rate * r.choice([0, 0.5, 1, 1, 2, 4])
            noisy, kinds = corrupt(doc, r, rate)
            truth = doc
            noise: list[str] = []
            if r.random() < a.noise_docs:
                noisy, noise = add_noise(noisy, r)
                kinds["noise"] += len(noise)
            for sub, text in (("truth", truth), ("ocr", noisy)):
                p = a.out / sub / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            row = {"path": rel.as_posix(), "rate": rate, "errors": kinds, "noise": noise}
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += kinds
            n += 1
    m = 0
    for rel, doc in write_tree(a.out, random.Random(a.seed + 1000), a.clean_mb, "clean"):
        p = a.out / "clean" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
        m += 1
    print(f"{a.out}: {n} test documents (ocr/ + truth/), {m} clean documents (clean/)")
    print("injected errors:", dict(total.most_common()))


if __name__ == "__main__":
    main()
