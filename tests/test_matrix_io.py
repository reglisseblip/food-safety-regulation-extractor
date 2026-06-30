"""matrix_io: the Rule -> row conversion, locked down by tests
(columns, output choices, consistency with the README)."""
import json
import pathlib
from typing import get_args

from pydantic import BaseModel

from matrix import io as matrix_io
from schema import Rule

ROOT = pathlib.Path(__file__).resolve().parents[1]

FULL = Rule.model_validate({
    "jurisdiction": "Codex", "theme": "food_additives", "rule_type": "MAX",
    "product_category": "Confectionery",
    "substance": {"name": "Ascorbyl esters", "substance_class": "additive"},
    "norm": {"value_max": 500, "unit": "mg/kg", "basis": "as_consumed"},
    "conditions": [
        {"type": "only_if", "ref": "468", "text": "sauf farine de blé"},
        {"type": "except", "text": "hors nourrissons"},
    ],
    "provenance": {"document_ref": "CXS 192-1995", "page": 91, "category_code": "05.0", "source_snippet": "x" * 400},
    "quality": {"needs_review": True, "confidence": 0.7},
})


def _leaves() -> set:
    """Every leaf field of Rule (one sub-model level deep), in dotted notation."""
    def submodel(ann):
        for t in (ann, *get_args(ann)):
            if isinstance(t, type) and issubclass(t, BaseModel):
                return t
        return None

    leaves = set()
    for name, f in Rule.model_fields.items():
        sub = submodel(f.annotation)
        if sub is None:
            leaves.add(name)
        else:
            leaves.update(f"{name}.{k}" for k in sub.model_fields)
    return leaves


def test_les_colonnes_sont_exactement_les_cles_de_to_row():
    row = matrix_io.to_row(FULL, "Codex", "CXS 192-1995")
    assert list(row) == matrix_io.COLUMNS


def test_chaque_champ_de_rule_est_mappe_ou_explicitement_abandonne():
    # Adding a field to Rule without deciding whether it goes in the CSV fails this
    # test — intentional: a field can no longer be forgotten by accident.
    assert _leaves() == matrix_io.MAPPED | matrix_io.DROPPED
    assert not (matrix_io.MAPPED & matrix_io.DROPPED)


def test_conditions_sans_perte_type_et_ref():
    row = matrix_io.to_row(FULL, "Codex", "CXS 192-1995")
    assert row["conditions"] == "only_if#468: sauf farine de blé | except: hors nourrissons"


def test_snippet_tronque_en_csv_verbatim_integral_en_json():
    row = matrix_io.to_row(FULL, "Codex", "CXS 192-1995")
    assert len(row["source"]) == matrix_io.SNIPPET_CSV_MAX          # deliberately shortened (spreadsheet readability)
    record = matrix_io.to_record(FULL, "CXS 192-1995")
    assert record["provenance"]["source_snippet"] == "x" * 400      # the full snippet (for citation) lives in the JSON


def test_document_corpus_et_document_ref_provenance_tous_deux_emis():
    row = matrix_io.to_row(FULL, "Codex", "CXS 192 (corpus)")
    assert row["document"] == "CXS 192 (corpus)"
    assert row["document_ref"] == "CXS 192-1995"    # the reference read from the text is no longer overwritten


def test_needs_review_et_confidence_sont_des_colonnes():
    # A doubtful case must be visible in the table (flagged, never dropped).
    row = matrix_io.to_row(FULL, "Codex", "CXS 192-1995")
    assert row["needs_review"] is True and row["confidence"] == 0.7


def test_record_json_serialisable_enums_compris():
    json.dumps(matrix_io.to_record(FULL, "CXS 192-1995"))  # mode='json': enums -> strings


def test_readme_documente_chaque_colonne():
    txt = (ROOT / "matrix" / "README.md").read_text(encoding="utf-8")
    for col in matrix_io.COLUMNS:
        assert f"`{col}`" in txt, f"colonne {col!r} absente de matrix/README.md"
