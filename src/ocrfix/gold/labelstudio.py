"""Label Studio glue: import tasks, labeling config, and reading an export back as rows.

Each task is prefilled (as a prediction) with gold_text = source_text and every span labeled
``ok``, so annotators only edit what is damaged. Span labels are a text area with one
``M1=ok`` line per placeholder. Cancelled annotations are skipped on import.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

NO_CONTEXT = "(none)"
MODEL_VERSION = "ocrfix-prefill"

_IMAGE_PANE = re.compile(r"[ \t]*<View[^>]*>\s*<Header value=\"Page image\"/>.*?</View>\n", re.S)
_LABEL_LINE = re.compile(r"^\s*⟨?([MTP]\d+)⟩?\s*[=:]\s*(\S+)\s*$")


def placeholder_name(ph: str) -> str:
    """⟨M1⟩ -> M1 (span_labels are keyed by the bare name)."""
    return ph.strip("⟨⟩")


def parse_span_labels(value: dict[str, str] | str | None) -> dict[str, str]:
    """``{"M1": "ok"}`` or text lines ``M1=ok`` / ``⟨M1⟩: ok``. Raises ValueError."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {placeholder_name(str(k).strip()): str(v).strip() for k, v in value.items()}
    out: dict[str, str] = {}
    for line in str(value).splitlines():
        if not line.strip():
            continue
        m = _LABEL_LINE.match(line)
        if not m:
            raise ValueError(f"bad span label line {line!r} (expected e.g. M1=ok)")
        if m.group(1) in out:
            raise ValueError(f"span {m.group(1)} labeled twice")
        out[m.group(1)] = m.group(2)
    return out


def _region(name: str, kind: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"from_name": name, "to_name": "source", "type": kind, "value": value}


def tasks(rows: list[dict[str, Any]], image_url_prefix: str = "") -> list[dict[str, Any]]:
    out = []
    for r in rows:
        phs = list(r["placeholders"])
        listing = "\n".join(f"{ph} = {raw}" for ph, raw in r["placeholders"].items())
        data = {
            "doc_id": r["doc_id"],
            "para_id": r["para_id"],
            "order": r["order"],
            "stratum": r["stratum"],
            "lang": r["lang"],
            "source_text": r["source_text"],
            "placeholders": r["placeholders"],
            "placeholder_list": listing or NO_CONTEXT,
            "context_prev": r["context_prev"] or NO_CONTEXT,
            "context_next": r["context_next"] or NO_CONTEXT,
            "page_image": image_url_prefix + r["page_image_path"] if r["page_image_path"] else "",
        }
        result = [_region("gold_text", "textarea", {"text": [r["source_text"]]})]
        if phs:
            labels = "\n".join(f"{placeholder_name(ph)}=ok" for ph in phs)
            result.append(_region("span_labels", "textarea", {"text": [labels]}))
        result.append(_region("structure", "choices", {"choices": ["none"]}))
        prefill = {"model_version": MODEL_VERSION, "result": result}
        out.append({"data": data, "predictions": [prefill]})
    return out


def labeling_config(with_image: bool = True) -> str:
    """Label Studio labeling config (labeling_config.xml); the page image pane is optional."""
    xml = (Path(__file__).with_name("labeling_config.xml")).read_text(encoding="utf-8")
    if not with_image:
        xml = _IMAGE_PANE.sub("", xml, count=1)
    return xml


def _annotator(ann: dict[str, Any]) -> str:
    who = ann.get("completed_by")
    if isinstance(who, dict):
        return str(who.get("email") or who.get("id") or "")
    return "" if who is None else str(who)


def _values(result: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in result:
        v = item.get("value", {})
        name = item.get("from_name")
        if "text" in v:
            out[name] = "\n".join(v["text"]) if isinstance(v["text"], list) else v["text"]
        elif "choices" in v:
            out[name] = v["choices"][0] if v["choices"] else None
        elif "number" in v:
            out[name] = v["number"]
    return out


def read_export(path: Path) -> list[dict[str, Any]]:
    """A Label Studio JSON export -> one annotation row per (task, annotation)."""
    with open(path, encoding="utf-8") as f:
        exported = json.load(f)
    rows = []
    for task in exported:
        d = task["data"]
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            v = _values(ann.get("result", []))
            structure = v.get("structure") or ""
            if structure == "none":
                structure = ""
            elif structure == "split-at":
                off = v.get("split_offset")
                structure = f"split-at:{int(off) if off is not None else ''}"
            rows.append(
                {
                    "doc_id": d["doc_id"],
                    "para_id": d["para_id"],
                    "order": d.get("order", 0),
                    "stratum": d.get("stratum", ""),
                    "lang": d["lang"],
                    "source_text": d["source_text"],
                    "placeholders": d.get("placeholders"),
                    "gold_text": v.get("gold_text", ""),
                    "span_labels": v.get("span_labels", ""),
                    "structure_label": structure,
                    "unrecoverable_count": None,
                    "annotator_id": _annotator(ann),
                    "notes": v.get("notes", ""),
                }
            )
    return rows
