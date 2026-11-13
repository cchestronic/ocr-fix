"""Fixer interface, registry, ops log and span-safe text editing.

A fixer is a class registered with ``@register("name")``. It gets one document's
paragraphs (Records, in order) and returns the new paragraphs plus an ``Op`` for every
change. Adding a fixer = a new module in this package + a ``fix.<name>`` config block and an
entry in ``fix.order``; the registry imports every module here, so the CLI never changes.

Ops: ``edit`` (text changed, same para_id), ``delete`` (para_id gone), ``merge`` (several
para_ids -> one; the destination may reuse the first source id), ``split`` (one -> several).
``check_ops`` enforces that the ops explain every para_id that appeared, disappeared or
changed text.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from ocrfix.schema import Record, Span, nfc

OP_KINDS = ("merge", "split", "delete", "edit")
MAX_EVIDENCE = 10  # examples kept per op


class FixerError(RuntimeError):
    pass


@dataclass
class Op:
    doc_id: str
    op: str
    src_para_ids: list[str]
    dst_para_ids: list[str]
    rule: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in OP_KINDS:
            raise ValueError(f"op must be one of {OP_KINDS}, got {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "op": self.op,
            "src_para_ids": self.src_para_ids,
            "dst_para_ids": self.dst_para_ids,
            "rule": self.rule,
            "evidence": self.evidence,
        }


class Fixer(ABC):
    name: ClassVar[str] = ""

    def __init__(self, **opts: Any) -> None:
        unknown = set(opts) - set(self.defaults())
        if unknown:
            raise ValueError(f"fixer {self.name}: unknown option(s) {sorted(unknown)}")
        self.opts = self.defaults() | opts

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {}

    @abstractmethod
    def apply(self, paras: list[Record]) -> tuple[list[Record], list[Op]]: ...

    def edit_op(self, rec: Record, rule: str, evidence: dict[str, Any]) -> Op:
        return Op(rec.doc_id, "edit", [rec.para_id], [rec.para_id], f"{self.name}.{rule}", evidence)

    def delete_op(self, rec: Record, rule: str, evidence: dict[str, Any]) -> Op:
        return Op(rec.doc_id, "delete", [rec.para_id], [], f"{self.name}.{rule}", evidence)


FIXERS: dict[str, type[Fixer]] = {}


def register(name: str) -> Callable[[type[Fixer]], type[Fixer]]:
    def deco(cls: type[Fixer]) -> type[Fixer]:
        if name in FIXERS and FIXERS[name] is not cls:
            raise ValueError(f"fixer {name!r} registered twice")
        cls.name = name
        FIXERS[name] = cls
        return cls

    return deco


def discover() -> dict[str, type[Fixer]]:
    """Import every module of this package so their @register calls run."""
    import ocrfix.fixers as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name not in ("base", "run"):
            importlib.import_module(f"ocrfix.fixers.{m.name}")
    return FIXERS


def build_fixers(fix_cfg: dict[str, Any]) -> list[Fixer]:
    """Enabled fixers in ``fix.order``; options are the ``fix.<name>`` block minus ``enabled``."""
    registry = discover()
    out = []
    for name in fix_cfg.get("order", []):
        if name not in registry:
            raise ValueError(f"fix.order: unknown fixer {name!r}; known: {sorted(registry)}")
        block = dict(fix_cfg.get(name) or {})
        if not block.pop("enabled", False):
            continue
        out.append(registry[name](**block))
    return out


# --- span-safe editing -------------------------------------------------------------------


def rebuild(rec: Record, pieces: list[tuple[str, str]]) -> Record:
    """New Record from (kind, text) per span, in order. Empty prose is dropped, adjacent
    prose merged, and whitespace at the paragraph's edges stripped (as ingest does)."""
    merged: list[list[str]] = []
    for kind, text in pieces:
        if kind == "prose" and merged and merged[-1][0] == "prose":
            merged[-1][1] += text
        else:
            merged.append([kind, text])
    if merged and merged[0][0] == "prose":
        merged[0][1] = merged[0][1].lstrip()
    if merged and merged[-1][0] == "prose":
        merged[-1][1] = merged[-1][1].rstrip()
    spans, parts, pos = [], [], 0
    for kind, text in merged:
        text = nfc(text)
        if not text and kind == "prose":
            continue
        spans.append(Span(pos, pos + len(text), kind))  # type: ignore[arg-type]
        parts.append(text)
        pos += len(text)
    return replace(rec, text="".join(parts), spans=spans)


def map_spans(rec: Record, fn: Callable[[str, str], str]) -> Record:
    """Apply ``fn(kind, text) -> text`` to every span."""
    return rebuild(rec, [(s.kind, fn(s.kind, rec.text[s.start : s.end])) for s in rec.spans])


def prose_mask(rec: Record) -> list[bool]:
    mask = [False] * len(rec.text)
    for s in rec.spans:
        if s.kind == "prose":
            mask[s.start : s.end] = [True] * (s.end - s.start)
    return mask


def apply_edits(rec: Record, edits: list[tuple[int, int, str]]) -> Record:
    """Replace non-overlapping [start, end) ranges; each must lie inside one prose span."""
    by_span: dict[int, list[tuple[int, int, str]]] = {}
    prose = [i for i, s in enumerate(rec.spans) if s.kind == "prose"]
    for a, b, repl in sorted(edits):
        i = next((i for i in prose if rec.spans[i].start <= a and b <= rec.spans[i].end), None)
        if i is None:
            raise FixerError(f"{rec.doc_id}/{rec.para_id}: edit [{a}, {b}) outside prose")
        by_span.setdefault(i, []).append((a, b, repl))
    pieces = []
    for i, s in enumerate(rec.spans):
        out, pos = [], s.start
        for a, b, repl in by_span.get(i, []):
            if a < pos:
                raise FixerError(f"{rec.doc_id}/{rec.para_id}: overlapping edits")
            out += [rec.text[pos:a], repl]
            pos = b
        out.append(rec.text[pos : s.end])
        pieces.append((s.kind, "".join(out)))
    return rebuild(rec, pieces)


def lines(text: str) -> list[tuple[int, int]]:
    """(start, end) of every line, without its newline."""
    out, pos = [], 0
    for part in text.split("\n"):
        out.append((pos, pos + len(part)))
        pos += len(part) + 1
    return out


def check_ops(
    doc_id: str, before: list[Record], after: list[Record], ops: list[Op], who: str
) -> None:
    """Every para_id that disappeared, appeared or changed text is explained by an op."""
    b = {r.para_id: r.text for r in before}
    a = {r.para_id: r.text for r in after}
    if len(a) != len(after):
        raise FixerError(f"{who}: {doc_id}: duplicate para_id in output")
    src = {p for o in ops if o.op != "edit" for p in o.src_para_ids}
    dst = {p for o in ops if o.op in ("merge", "split") for p in o.dst_para_ids}
    edited = {p for o in ops if o.op == "edit" for p in o.dst_para_ids}
    problems = []
    if gone := sorted(b.keys() - a.keys() - src):
        problems.append(f"removed without op: {gone}")
    if new := sorted(a.keys() - b.keys() - dst):
        problems.append(f"new without op: {new}")
    if changed := sorted(k for k in a.keys() & b.keys() if a[k] != b[k] and k not in edited | dst):
        problems.append(f"text changed without op: {changed}")
    if problems:
        raise FixerError(f"{who}: {doc_id}: " + "; ".join(problems))


def sample(items: list[Any], n: int = MAX_EVIDENCE) -> list[Any]:
    return items[:n]
