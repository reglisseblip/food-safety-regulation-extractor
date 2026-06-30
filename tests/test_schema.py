"""A rule's subject (substance / group / labeling / claim):
a single accessor, a common label everywhere, and "exactly one subject" enforced."""
import pytest
from pydantic import ValidationError

from schema import Rule

BASE = {"jurisdiction": "EU", "theme": "food_additives", "rule_type": "MAX",
        "provenance": {"document_ref": "Reg 1333/2008", "source_snippet": "…"}}


def test_accesseur_et_libelle_canonique_par_famille():
    cases = {
        "substance": ({"substance": {"name": "Sorbic acid", "substance_class": "additive"}},
                      "Sorbic acid"),
        "group": ({"group": {"members": ["E 471", "E 472a"]}},
                  "group:E 471+E 472a"),
        "claim": ({"rule_type": "PERMITTED", "claim": "sans gluten"},
                  "sans gluten"),
        "labeling": ({"rule_type": "PERMITTED", "theme": "labeling",
                      "labeling": {"kind": "mandatory_element", "element": "liste des ingrédients"}},
                     "liste des ingrédients"),
    }
    for kind, (extra, label) in cases.items():
        r = Rule.model_validate({**BASE, **extra})
        assert r.subject[0] == kind          # the chain of `if`s lives in Rule, in one place
        assert r.subject_label() == label    # same label for dedup, matrix, benchmark


def test_deux_sujets_refuses():
    # "ONE subject per rule" is no longer just documented: it is rejected at validation.
    # (The error message goes back to the LLM via Instructor, which retries, so the model
    # corrects itself on its own during extraction.)
    with pytest.raises(ValidationError, match="PLUSIEURS sujets"):
        Rule.model_validate({**BASE,
                             "substance": {"name": "Gluten", "substance_class": "allergen"},
                             "claim": "sans gluten"})


def test_zero_sujet_refuse():
    # every rule carries exactly one subject: no subject -> rejected at validation.
    with pytest.raises(ValidationError, match="sans sujet"):
        Rule.model_validate(BASE)


def test_null_du_llm_vaut_le_defaut():
    # The LLM often writes `null` where it has nothing to say (rule with no numeric value → norm=null,
    # no conditions → conditions=null). These `null`s must NOT fail validation: without this
    # tolerance, a single null rule lost the whole batch (the entire chunk).
    r = Rule.model_validate({
        **BASE,
        "substance": {"name": "Agar", "substance_class": "additive"},
        "norm": None, "conditions": None, "quality": None, "consumer_group": None,
    })
    assert r.norm.value_max is None            # norm=null -> default Norm(), no error
    assert r.conditions == []                  # conditions=null -> empty list
    assert r.consumer_group.value == "general" # consumer_group=null -> default
    assert r.quality.needs_review is False
