"""Extraction pipeline tested offline (with the fake LLM):
extraction, fault tolerance (P16), notes glossary, deduplication."""
import pytest

from pipeline import structurer as st
from fake_llm import fake_llm
from llm import LLMCallError, LLMConfigError
from payloads import CTX, wire_rule
from schema import Rule

MD_DEUX_SECTIONS = ("# A\n\nAAA additif alpha, 500 mg/kg.\n\n"
                    "# B\n\nBBB sortie de scan abîmée.")


# ------------------------------ extraction ------------------------------
def test_cas_nominal_une_ligne():
    fake = fake_llm(rules=[wire_rule("alpha")])
    rules = st.extract_rules("Alpha : 500 mg/kg dans les confiseries.", CTX, llm=fake)
    assert [r.substance.name for r in rules] == ["alpha"]
    fake.assert_called("rules", times=1)   # single chunk: the doc fits in the window
    assert fake.count("splits") == 0       # so no boundary call


def test_p16_un_morceau_casse_le_document_survit():
    fake = (fake_llm()
            .on("rules", [wire_rule("alpha")], when="AAA")
            .on("rules", raises=LLMCallError("morceau B meurt"), when="BBB"))

    rules = st.extract_rules(MD_DEUX_SECTIONS, CTX, llm=fake, max_chars=40)

    assert [r.substance.name for r in rules] == ["alpha"]   # the healthy chunk survives
    fake.assert_called("rules", times=2)                    # the broken one was attempted
    assert len(fake.errors) == 1
    assert "BBB" in fake.errors[0].user_text                # and the right one died


def test_multi_morceaux_route_par_contenu_sans_doublon():
    fake = (fake_llm()
            .on("rules", [wire_rule("alpha")], when="AAA")
            .on("rules", [wire_rule("beta")], when="BBB"))
    rules = st.extract_rules(MD_DEUX_SECTIONS, CTX, llm=fake, max_chars=40, do_dedupe=False)
    assert sorted(r.substance.name for r in rules) == ["alpha", "beta"]


def test_rulebatch_garde_les_bonnes_regles_ecarte_la_cassee():
    # Defense in depth: a malformed rule (empty subject and not DEFINITION) is dropped
    # while its valid neighbors are kept, so one bad rule no longer loses the whole chunk.
    lot = st.RuleBatch.model_validate({"rules": [
        wire_rule("alpha"),
        {"jurisdiction": "EU", "theme": "food_additives", "rule_type": "MAX",  # no subject -> invalid
         "provenance": {"document_ref": "X", "source_snippet": "y"}},
        wire_rule("beta"),
    ]})
    assert [r.substance.name for r in lot.rules] == ["alpha", "beta"]


# --------------------------- notes glossary --------------------------
def test_le_glossaire_atteint_le_prompt_d_extraction():
    fake = (fake_llm(rules=[wire_rule("alpha")])
            .on("notes", [{"id": "468", "text": "sauf farine de blé : 60 mg/kg"}]))
    st.extract_rules("Alpha 500 mg/kg, voir note 468.", CTX, llm=fake)
    fake.assert_called("rules", containing="Note 468")  # build_messages injected it


def test_fusion_glossaire_garde_le_texte_le_plus_long():
    fake = (fake_llm()
            .on("notes", [{"id": "468", "text": "court"}], when="AAA")
            .on("notes", [{"id": "468", "text": "nettement plus long que court"}], when="BBB"))
    notes = st.build_notes(["AAA premier morceau", "BBB second morceau"], fake)
    assert notes == {"468": "nettement plus long que court"}


def test_notes_un_morceau_casse_les_autres_survivent():
    fake = (fake_llm()
            .on("notes", raises=LLMCallError("boom"), when="AAA")
            .on("notes", [{"id": "R1", "text": "restriction friture"}], when="BBB"))
    notes = st.build_notes(["AAA", "BBB"], fake)
    assert notes == {"R1": "restriction friture"}
    assert len(fake.errors) == 1  # the failure happened and was tolerated


# ------------------------------ dedup agent ------------------------------
def _rule(name: str) -> Rule:
    return Rule.model_validate(wire_rule(name))


def test_dedup_indices_hors_bornes_ignores():
    rules = [_rule("Blorbic acid"), _rule("Blorbic acid"), _rule("Zentanyl")]
    fake = fake_llm(dedupe=[1, 99, -1])    # 99 and -1 are out of bounds, filtered by structurer
    kept = st.dedupe(rules, llm=fake)
    assert len(kept) == 2                  # one Blorbic removed; Zentanyl intact
    fake.assert_called("dedupe", times=1)  # only the block with >= 2 candidates is judged


def test_dedup_moins_de_deux_regles_aucun_appel():
    fake = fake_llm()
    assert len(st.dedupe([_rule("Blorbic acid")], llm=fake)) == 1
    assert fake.count("dedupe") == 0


def test_dedup_echoue_le_bloc_est_rendu_non_dedoublonne():
    # Deduplication is the last and cheapest step: if its call fails we keep the block as-is
    # (a few duplicates) rather than losing rules already extracted and paid for.
    rules = [_rule("Blorbic acid"), _rule("Blorbic acid")]
    fake = fake_llm().on("dedupe", raises=LLMCallError("boom"))
    kept = st.dedupe(rules, llm=fake)
    assert len(kept) == 2          # block kept as-is, nothing lost
    assert len(fake.errors) == 1   # and the failure did happen (the test doesn't pass for nothing)


# --------------------------- the two error families ---------------------------
def test_exception_non_llm_n_est_plus_avalee():
    # A real bug (an exception that is NOT an LLM call failure) must propagate,
    # not disguise itself as "1 chunk skipped" in the logs.
    fake = fake_llm().on("rules", raises=TimeoutError("bug, pas un échec LLM"))
    with pytest.raises(TimeoutError):
        st.extract_rules("doc court.", CTX, llm=fake)


def test_config_error_remonte_hors_extract_rules():
    # A bad configuration never degrades into an empty matrix: it kills the run.
    fake = fake_llm().on("rules", raises=LLMConfigError("clé absente"))
    with pytest.raises(LLMConfigError):
        st.extract_rules("doc court.", CTX, llm=fake)
