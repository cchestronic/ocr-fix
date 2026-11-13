# Gold annotation guideline (paragraph tasks, `gold sample` / `gold import`)

The chunk pipeline uses whole `.md` pairs instead: see `annotation_guideline.md`.

You are shown one paragraph of OCR output (MinerU) from a PDF page, the paragraphs before and
after it, and an image of the page. You write the **gold text**: the text that is printed in
the PDF, with the OCR damage removed. **Nothing else.**

The gold text is used to train and evaluate an OCR correction model. If gold contains
improvements the PDF doesn't have (style, spelling rules, translations), the model learns to
make those changes too, and we can no longer measure whether it is faithful to the PDF.

## The one rule

> **Gold = what the PDF says, character for character, minus what OCR broke.**

Always check the page image. If you are unsure whether something is an OCR error or is
really printed in the PDF, the image decides. If the image doesn't decide either, mark the
spot with `〈?〉` (see below). Never guess.

## What you see

| field | meaning |
|-------|---------|
| `source_text` | the OCR paragraph. Math and tables are replaced by placeholders `⟨M1⟩ ⟨M2⟩ … ⟨T1⟩ …` |
| Protected spans | what each placeholder stands for (the raw LaTeX / HTML) |
| `context_prev`, `context_next` | the neighboring paragraphs, **read-only**. Use them to understand the text. Never copy text from them into the gold text |
| page image | the PDF page. This is the reference |

The gold text field starts as a copy of `source_text`. Edit only what is damaged. If nothing
is damaged, submit it unchanged. That is a valid and common answer.

## Fix

- **Misrecognized characters**: `rn`→`m`, `1`/`l`/`I`, `0`/`O`, `cl`→`d`, `5`/`S`, broken
  diacritics. In Korean: wrong syllables or jamo (`걸과`→`결과`, `졍확도`→`정확도`), decomposed
  or garbage jamo.
- **Spaces OCR added or dropped**, only where the PDF differs: `나타 내었으며`→`나타내었으며`
  when the PDF prints `나타내었으며`.
- **Line-break hyphenation**: `pro- posed`, `pro-\nposed` → `proposed`. Keep the hyphen when the
  word is really hyphenated (`well-known`, `state-of-the-art`).
- **Line breaks inside a paragraph**: replace them with a space (Korean: with whatever the
  PDF shows, usually a space).
- **Junk OCR inserted**: stray characters from page furniture, dots, `■`, repeated fragments.
- **Hangul that OCR read as math**: label the span `misclassified-text` and type the real
  text in place of the placeholder (example KO-2).

## Do not fix

- **Anything the PDF itself prints**: the author's typos, grammar, outdated or nonstandard
  spelling (`되었읍니다`), nonstandard spacing (`할수 있다`), British/American spelling,
  inconsistent terminology. If it's in the PDF, it stays.
- **Do not apply spelling or spacing rules** (맞춤법, 띄어쓰기). Fix spaces only where OCR differs
  from the PDF.
- **Do not translate, romanize, or add English** to Korean text, or the reverse. Don't add
  explanations, expand abbreviations, or complete cut-off sentences.
- **Do not touch the placeholders** `⟨M#⟩ ⟨T#⟩`: no renumbering, no moving, no deleting, no
  editing what they stand for. The only exception is `misclassified-text`.
- **Do not change markdown structure**: keep `#` headings, `**bold**`, list markers and
  footnote marks as they are. Don't add or remove any.
- **Do not change numbers, units or citation marks** (`[12]`, `(Kim et al., 2021)`) unless
  the image clearly shows that OCR got them wrong.
- **Do not fix text in the context paragraphs**. They aren't the task.

## 〈?〉: unrecoverable text

Use `〈?〉` (U+3008 ? U+3009; copy it from here) where OCR destroyed the text and the
page image doesn't let you read it with confidence either (blur, a stain, a cut-off edge).

- Replace the **smallest** unreadable stretch, one `〈?〉` per contiguous stretch, whatever its
  length: `3〈?〉0 ℃` for one unreadable digit. If a whole clause is gone, use one `〈?〉`.
- Never guess a plausible word. A wrong guess teaches the model to invent text.
- If the source already contains `〈?〉`, keep it.
- `unrecoverable_count` is the number of `〈?〉` in your gold text. It is filled
  automatically if you leave it empty.

## Span labels

Every placeholder in the source gets exactly one label, written one per line as `M1=ok`
(the task comes prefilled with every span labeled `ok`):

| label | when | gold text |
|-------|------|-----------|
| `ok` | the LaTeX/HTML is a correct rendering of the PDF | keep the placeholder |
| `broken` | it is math/table in the PDF, but OCR damaged it (wrong symbol, missing brace, wrong cell) | keep the placeholder. We don't repair math or tables in this round |
| `misclassified-text` | it is **not** math/table in the PDF: ordinary text (very often Hangul) that OCR turned into LaTeX, or prose rendered as a table | **replace** the placeholder with the PDF text |

## Structure labels

Paragraph segmentation comes from OCR and is sometimes wrong. Don't fix the segmentation
inside the gold text. Label it instead:

| label | when |
|-------|------|
| *(none)* | segmentation is right (the common case) |
| `merge-with-next` | this paragraph and the next one are one paragraph in the PDF (split by a page or column break, a figure, a footnote) |
| `split-at:<offset>` | this paragraph is really two (e.g. a heading glued to the body text). `<offset>` = character position in **your gold text** where the second paragraph starts (0-based; the first character is 0) |

The gold text always covers exactly this paragraph's text. With `merge-with-next`, don't pull
in the next paragraph's text. With `split-at`, don't insert a line break.

## Automatic checks on import

`ocrfix gold import` rejects an annotation (and reports it) if:

- the gold text is empty;
- its length is outside 0.8–1.25× the source length (sources under 20 characters are not
  checked);
- a placeholder was dropped, duplicated, renumbered or moved (other than a
  `misclassified-text` span, which must be gone), or a span has no label or an unknown one;
- in a paragraph tagged `ko`, the gold text contains a Latin word that is not in the source
  and isn't an OCR fix of a source word (`rnodel`→`model` is fine; adding `structure` is not);
- the structure label is malformed or its offset falls outside the gold text;
- `unrecoverable_count` doesn't match the number of `〈?〉`.

If a real case trips a check (e.g. the source is so garbled that the true text is much
longer), still annotate it correctly, explain in `notes`, and tell the reviewer.

## Worked examples

### KO-1: misrecognized syllables and an OCR space (fix)

```
source:  실험 걸과는 표 2에 나타 내었으며, 졍확도는 기존 방법보다 높았다.
gold:    실험 결과는 표 2에 나타내었으며, 정확도는 기존 방법보다 높았다.
```
`걸과`→`결과` and `졍확도`→`정확도` are syllable confusions. `나타 내었으며` is fixed because the
page image shows `나타내었으며` with no space.

### KO-2: Hangul read as math, `misclassified-text` (fix)

```
source:  제안한 방법 ⟨M1⟩ 기존 방법보다 우수함을 확인하였다.
spans:   ⟨M1⟩ = $\mathrm{Ol}$
gold:    제안한 방법이 기존 방법보다 우수함을 확인하였다.
labels:  M1=misclassified-text
```
The PDF prints `방법이`. OCR turned `이` into the formula `Ol` and split it off with a space.
The placeholder is replaced by the real text, and the space OCR added goes with it.

### KO-3: paragraph cut by a page break, `merge-with-next` (fix the text, label the structure)

```
source:  본 연구의 결과는 기존의 연구와 달리 데이터의 규모가 작은 경우에도
next:    안정적인 성능을 보인다는 점에서 의의가 있다.
gold:    본 연구의 결과는 기존의 연구와 달리 데이터의 규모가 작은 경우에도
structure: merge-with-next
```
The sentence continues on the next page. The gold text stays this paragraph only. Don't
append the next paragraph's text and don't add a period.

### KO-4: unreadable digit, `〈?〉`

```
source:  반응 온도는 3口0 ℃로 유지하였으며 반응 시간은 2시간이었다.
gold:    반응 온도는 3〈?〉0 ℃로 유지하였으며 반응 시간은 2시간이었다.
unrecoverable_count: 1
```
OCR produced `口` for a digit, and in the image the digit is under a stain. It could be 2, 5
or 8, so it is marked `〈?〉`. Don't write `300` just because it looks plausible.

### KO-5: the PDF's own spelling and spacing (**do not fix**)

```
source:  이 방법은 소규모 데이터에도 적용할수 있으며, attention mechanism 을 사용하였읍니다.
gold:    이 방법은 소규모 데이터에도 적용할수 있으며, attention mechanism 을 사용하였읍니다.
```
`적용할수` (standard: `적용할 수`), `mechanism 을` and `하였읍니다` (standard: `하였습니다`) are all
printed this way in the PDF. Gold is the source unchanged. Don't apply 맞춤법 or 띄어쓰기 rules.

### EN-1: character confusions (fix)

```
source:  Tab1e 3 surnmarizes the resu1ts on the C0CO dataset; the rnodel reaches 41.2 AP.
gold:    Table 3 summarizes the results on the COCO dataset; the model reaches 41.2 AP.
```
`1`/`l`, `rn`/`m` and `0`/`O` are classic OCR confusions. `41.2` matches the image, so it
stays.

### EN-2: line-break hyphenation (fix)

```
source:  The pro- posed method out-
         performs the well- known baseline on all bench- marks.
gold:    The proposed method outperforms the well-known baseline on all benchmarks.
```
Hyphens that only split a word at a line end are removed, along with the line break and the
space. `well-known` is a hyphenated word anyway, so its hyphen stays and only the space goes.

### EN-3: damaged math, `broken` (fix the prose, label the math)

```
source:  where ⟨M1⟩ dcnotes the loss and ⟨M2⟩ is the learning rate.
spans:   ⟨M1⟩ = $\mathcal{L}$      ⟨M2⟩ = $\boldsymbol{\eta$
gold:    where ⟨M1⟩ denotes the loss and ⟨M2⟩ is the learning rate.
labels:  M1=ok
         M2=broken
```
`dcnotes` is fixed. `⟨M2⟩` is missing a brace, but math isn't edited in the gold text: it's
labeled `broken` and the placeholder stays where it is.

### EN-4: heading glued to body text, `split-at` (label the structure)

```
source:  3. Experimental Setup We trained all models for 10 epochs on four GPUs.
gold:    3. Experimental Setup We trained all models for 10 epochs on four GPUs.
structure: split-at:22
```
In the PDF, `3. Experimental Setup` is a heading on its own line. `We` starts at character
22 of the gold text (`3. Experimental Setup ` is 22 characters). The gold text itself is
unchanged: no line break inserted, no `#` added.

### EN-5: the author's own errors (**do not fix**)

```
source:  The datas was analysed using a modifed version of the algorithm in [12].
gold:    The datas was analysed using a modifed version of the algorithm in [12].
```
The page image shows `datas was` and `modifed` exactly like this. They are the author's
errors, not OCR's. `analysed` is British spelling. Gold is the source unchanged.
