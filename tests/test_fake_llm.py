"""The fake LLM itself: routing, shortcuts, guardrails.
(Every other test relies on it, so we test it too.)"""
import pytest
from pydantic import BaseModel, ValidationError

from fake_llm import FakeLLM, UnscriptedCall, fake_llm
from payloads import wire_rule
from pipeline.structurer import RuleBatch


def U(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_kind_inconnu_echoue_a_l_enregistrement():
    with pytest.raises(KeyError):
        fake_llm().on("rulez", [])  # a typo fails immediately instead of silently passing a hollow test


def test_respond_et_raises_mutuellement_exclusifs():
    with pytest.raises(ValueError):
        FakeLLM().on("rules", [], raises=ValueError())
    with pytest.raises(ValueError):
        FakeLLM().on("rules")


def test_response_model_inconnu_est_bruyant():
    class Autre(BaseModel):
        x: int = 0

    fake = fake_llm()
    with pytest.raises(UnscriptedCall):
        fake(U("peu importe"), Autre)
    assert len(fake.errors) == 1  # logged even when raising


def test_kind_non_scripte_repond_vide():
    assert fake_llm()(U("x"), RuleBatch).rules == []


def test_raccourci_liste_et_validation_reelle():
    batch = fake_llm(rules=[wire_rule("alpha")])(U("x"), RuleBatch)
    assert [r.substance.name for r in batch.rules] == ["alpha"]


def test_rulebatch_tolere_une_regle_cassee_les_bonnes_restent():
    # RuleBatch drops a malformed rule (defense in depth) instead of losing the whole batch.
    fake = fake_llm(rules=[wire_rule("alpha"), {"nimporte": "quoi"}])
    batch = fake(U("x"), RuleBatch)
    assert [r.substance.name for r in batch.rules] == ["alpha"]


def test_payload_hors_schema_leve_pour_un_modele_strict():
    # The "FakeLLM validates against the real schema" guarantee holds: a STRICT model (note
    # glossary) rejects an off-schema payload, just as a real LLM response would.
    from pipeline.structurer import NoteGlossary
    fake = fake_llm(notes=[{"nimporte": "quoi"}])
    with pytest.raises(ValidationError):
        fake(U("x"), NoteGlossary)
    assert len(fake.errors) == 1


def test_routage_par_contenu_premier_match_gagne():
    fake = (fake_llm()
            .on("rules", [wire_rule("alpha")], when="AAA")
            .on("rules", [wire_rule("beta")]))  # catch-all, registered last
    assert fake(U("... AAA ..."), RuleBatch).rules[0].substance.name == "alpha"
    assert fake(U("autre chose"), RuleBatch).rules[0].substance.name == "beta"
