"""Human gold annotation: sample paragraphs for labeling, import and split the labels.

``ocrfix gold sample <scores.jsonl> <paragraphs.jsonl> <out.jsonl>`` -> annotation JSONL
(+ Label Studio import JSON and labeling config); ``ocrfix gold import <annotated> <out_dir>``
-> ``gold_train.jsonl`` / ``gold_eval.jsonl``. See docs/annotation_guideline_paragraphs.md.
"""

from ocrfix.gold.importer import ImportReport, import_annotations, run_import, split_docs
from ocrfix.gold.sample import SampleResult, run_sample, sample, write_sample
from ocrfix.gold.validate import SPAN_LABELS, Violation, validate_row

__all__ = [
    "SPAN_LABELS",
    "ImportReport",
    "SampleResult",
    "Violation",
    "import_annotations",
    "run_import",
    "run_sample",
    "sample",
    "split_docs",
    "validate_row",
    "write_sample",
]
