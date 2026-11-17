# Gold annotation guideline (`.md` pairs)

Gold is a set of MinerU documents, each with a hand-corrected copy. They are used to measure
how much OCR damage there is (error rates for the training-data generator and calibration),
to train the correction model, and to evaluate it. `ocrfix gold chunks` turns them into
training and evaluation data; you only edit markdown files.

## Files

```
gold/
  ocr/     the MinerU .md files, copied unchanged. Never edit these.
  truth/   the same files, corrected. Same relative path and name as in ocr/.
```

1. Copy a MinerU file into `ocr/` (any sub-folders you like).
2. Copy it again to the same path under `truth/`.
3. Edit the `truth/` copy, with the PDF open beside it.

- **Encoding:** UTF-8.
- **Long documents:** you may correct only part of a long document. Cut **both** files at the
  same blank line, so `ocr/` and `truth/` hold the same part.
- **Choosing documents:** take them from all over the corpus: different sources, years and
  layouts; Korean, English and mixed; with math and tables. Don't pick only the clean-looking
  ones.
- **Leftovers:** a document without a matching `truth/` file is skipped (and listed).

## The one rule

> **Truth = what the PDF says, character for character, minus what OCR broke.**

If you're not sure whether something is an OCR error or is really printed in the PDF, the PDF
decides. If you can't tell what a destroyed word is, leave the OCR text as it is (see below).

## Fix

- **Misrecognized characters:**
  - Latin: `rn`→`m`, `1`/`l`/`I`, `0`/`O`, `cl`→`d`;
  - Korean: wrong syllables or jamo (`걸과`→`결과`, `졍확도`→`정확도`).
- **Spaces OCR added or dropped**, only where the PDF differs: `나타 내었으며`→`나타내었으며`
  when the PDF prints `나타내었으며`.
- **Line breaks inside a sentence or paragraph:** join the lines with a space. In Korean, use
  whatever the PDF shows, usually a space.
- **Line-break hyphenation:** `pro-\nposed` and `pro- posed` → `proposed`. Keep the hyphen of a
  really hyphenated word (`well-known`).
- **Page furniture in the text:** delete page numbers, running headers and footers, and line
  numbers that landed inside the text.
- **Junk:** delete stray symbols, garbage strings and repeated fragments that aren't in the PDF.
  A whole junk paragraph goes too, with its blank line.

## Do not change

- **Anything the PDF itself prints:**
  - the author's typos and grammar;
  - nonstandard spelling (`되었읍니다`) or spacing (`할수 있다`);
  - British or American spelling.

  Don't apply 맞춤법 or 띄어쓰기 rules.
- **Math, tables, links, image lines and citations stay exactly as they are,** even when OCR
  got them wrong. That covers `$...$`, `$$...$$`, `\begin{...}`, `<table>...</table>`,
  `![](...)`, URLs and `[12]`. The model never edits them (they are masked), and a chunk whose
  math or table changed can't be used for training.
- **Blank lines (paragraph breaks) stay where they are,** even when a page break split one
  paragraph in two. Only a deleted junk paragraph takes its blank line with it.
- **Markdown structure stays as it is:** `#` headings, list markers, `**bold**`.
- **No translation, no added text, no completing cut-off sentences.**

## Destroyed words

A word OCR destroyed beyond recognition (`효과적으로` came out as `흐괴ㅈ젹ㅇ□`, `training` as
`STJBT8RBY`):

- **The context makes it clear:** write the most likely word.
- **Otherwise:** leave the OCR text as it is. Don't invent text; the model learns from this
  file, so a guess here teaches it to invent.
- **Fixable damage:** if one or two characters are wrong and the word is clear, fix it
  (`방볍`→`방법`, `Tab1e`→`Table`).

## Checks (`ocrfix gold chunks`)

Each chunk of a document gets a status:

| status | meaning | used for training? |
|---|---|---|
| `ok` | the truth lines up with the OCR chunk | yes |
| `placeholders_changed` | math, a table or a link differs between `ocr/` and `truth/` (or OCR glued a word to a URL) | no |
| `cut_in_edit` | an edit of yours crosses the chunk boundary and can't be split | no |
| `empty` | the truth has no text for the chunk (a deleted junk chunk) | no |

Evaluation compares whole documents, so it uses every document whatever the statuses.
`stats.json` lists the counts, and the documents without a `truth/` file.

## Examples

**KO, characters and an OCR space:**
```
ocr:    실험 걸과는 표 2에 나타 내었으며, 졍확도는 기존 방법보다 높았다.
truth:  실험 결과는 표 2에 나타내었으며, 정확도는 기존 방법보다 높았다.
```

**KO, destroyed word, context unclear:**
```
ocr:    제안한 방법은 실험에서 흐괴ㅈ젹ㅇ□ 동작하였다.
truth:  제안한 방법은 실험에서 흐괴ㅈ젹ㅇ□ 동작하였다.
```
The PDF shows `효과적으로`, but the context doesn't make the word clear, so the OCR text stays.

**KO, the PDF's own spelling (don't fix):**
```
ocr:    이 방법은 소규모 데이터에도 적용할수 있으며, 사용하였읍니다.
truth:  이 방법은 소규모 데이터에도 적용할수 있으며, 사용하였읍니다.
```

**EN, confusions, hyphenation, a broken line:**
```
ocr:    Tab1e 3 surnmarizes the resu1ts of the pro-
        posed rnodel on the C0CO dataset.
truth:  Table 3 summarizes the results of the proposed model on the COCO dataset.
```

**EN, page number and header in the text:**
```
ocr:    the loss decreases steadily 12 Journal of Document Analysis as the batch grows.
truth:  the loss decreases steadily as the batch grows.
```

**Math stays, even when it's wrong:**
```
ocr:    where $\mathcal{L$ dcnotes the loss.
truth:  where $\mathcal{L$ denotes the loss.
```
Only `dcnotes` is fixed. The broken brace inside `$...$` stays.
