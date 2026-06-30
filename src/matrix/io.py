"""
matrix_io — turn a Rule (Python object) into matrix rows: defined here, once.

Two outputs for the same rule:
  * to_row(rule, jurisdiction, document) -> a flat row for matrix.csv (the
    filterable table opened in Excel); columns = COLUMNS, defined below;
  * to_record(rule, document)            -> the full record for matrix.json
    (ALL fields, including the FULL source snippet — the one that lets us cite).

Formatting choices are made explicit here rather than buried in the code:
  * SNIPPET_CSV_MAX — in the CSV the source snippet is truncated to stay readable
    in a spreadsheet; the full version lives in matrix.json.
  * conditions keep their type and note number: "only_if#468: text" — we no
    longer drop the info the notes glossary (P6) worked hard to attach.
  * both `document` (the corpus reference) AND `document_ref` (the one read in
    the text) are kept: a text may cite a standard other than itself.
  * needs_review / confidence are columns: a doubtful case is VISIBLE in the
    table — "flagged, never dropped".

MAPPED / DROPPED: what goes into the CSV and what is deliberately left out (all
of it still lives in the JSON). A test (tests/test_matrix_io.py) checks that
every Rule field is in one of the two sets and that the README documents each
column — code, benchmark and docs can no longer contradict each other without
breaking a test.
"""
from __future__ import annotations

from schema import Rule

SNIPPET_CSV_MAX = 300  # max snippet length in the CSV (full text is in matrix.json)


def _conditions(r: Rule) -> str:
    """All conditions in a single cell, losing nothing: "only_if#468: text | except: text"."""
    return " | ".join(
        f"{c.type.value}#{c.ref}: {c.text}" if c.ref else f"{c.type.value}: {c.text}"
        for c in r.conditions
    )


# THE column definition: name -> how to fill it from a Rule. COLUMNS follows.
_FIELDS = {
    "jurisdiction":   lambda r, jur, doc: jur,
    "document":       lambda r, jur, doc: doc,
    "document_ref":   lambda r, jur, doc: r.provenance.document_ref,
    "theme":          lambda r, jur, doc: r.theme.value,
    "rule_type":      lambda r, jur, doc: r.rule_type.value,
    "subject":        lambda r, jur, doc: r.subject_label(),
    "category":       lambda r, jur, doc: r.product_category,
    "consumer_group": lambda r, jur, doc: r.consumer_group.value,
    "value_min":      lambda r, jur, doc: r.norm.value_min,
    "value_max":      lambda r, jur, doc: r.norm.value_max,
    "unit":           lambda r, jur, doc: r.norm.unit,
    "basis":          lambda r, jur, doc: r.norm.basis.value if r.norm.basis else None,
    "conditions":     lambda r, jur, doc: _conditions(r),
    "page":           lambda r, jur, doc: r.provenance.page,
    "needs_review":   lambda r, jur, doc: r.quality.needs_review,
    "confidence":     lambda r, jur, doc: r.quality.confidence,
    "source":         lambda r, jur, doc: r.provenance.source_snippet[:SNIPPET_CSV_MAX],
}
COLUMNS = list(_FIELDS)

# Rule fields that end up in the columns above…
MAPPED = {
    "jurisdiction", "theme", "rule_type", "consumer_group", "claim",
    "product_category",
    "substance.name", "group.members", "labeling.element",
    "norm.value_min", "norm.value_max", "norm.unit", "norm.basis",
    "conditions.type", "conditions.text", "conditions.ref",
    "provenance.document_ref", "provenance.page", "provenance.source_snippet",
    "quality.needs_review", "quality.confidence",
}
# …and fields deliberately absent from the CSV (all still kept, in full, in matrix.json).
DROPPED = {
    "provenance.legal_reference", "provenance.version", "provenance.category_code",
    "substance.substance_class", "substance.identifier", "substance.chemical_form",
    "labeling.kind",
}


def to_row(rule: Rule, jurisdiction: str, document: str) -> dict:
    """ONE rule -> ONE table row (matrix.csv)."""
    return {col: fn(rule, jurisdiction, document) for col, fn in _FIELDS.items()}


def to_record(rule: Rule, document: str) -> dict:
    """ONE rule -> its full record for matrix.json (full source snippet: citable)."""
    return {"document": document, **rule.model_dump(mode="json", exclude_none=True)}
