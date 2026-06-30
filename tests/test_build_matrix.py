"""build(): matrix construction, tested without any PDF or network.

Two inputs are swapped: llm= (the fake LLM) and to_markdown= (a stub parser --
Docling is never even loaded: build_matrix imports it only for a real run).
"""
import pytest

from matrix import build as bm
from fake_llm import fake_llm
from llm import LLMConfigError
from payloads import wire_rule

DOC_A = bm.DocContext("Codex", "DOC-A", "food_additives", "a.pdf")
DOC_B = bm.DocContext("EU", "DOC-B", "food_additives", "b.pdf")


def _md(path):
    return f"contenu {path.name}"


def test_le_corpus_couvre_tous_les_fichiers_petits_d_abord():
    # DOCS = one entry per corpus file, nothing missed, nothing invented.
    fichiers = {f"{d.name}/{p.name}" for d in bm.SRC.iterdir() if d.is_dir()
                for p in d.iterdir() if p.is_file()}
    assert {doc.file for doc in bm.DOCS} == fichiers
    # small first: the giants (GSFA, Thai #381/#444) bring up the rear
    tailles = [d.path.stat().st_size for d in bm.DOCS]
    assert tailles == sorted(tailles)
    # a single scanned document in the whole corpus (checked by hand, 2026-07-08)
    assert sum(d.ocr for d in bm.DOCS) == 1
    # Thai notifications carry a short readable reference, not their full title
    assert any(d.document_ref == "MOPH 174" for d in bm.DOCS)
    assert all(d.jurisdiction in {"Codex", "EU", "India", "Thailand"} for d in bm.DOCS)


def test_build_ecrit_les_deux_vues_et_renvoie_les_lignes(tmp_path):
    fake = fake_llm(rules=[wire_rule("alpha")])
    rows = bm.build([DOC_A], llm=fake, stem="t", out=tmp_path, to_markdown=_md)
    assert [r["subject"] for r in rows] == ["alpha"]
    assert rows[0]["jurisdiction"] == "Codex" and rows[0]["document"] == "DOC-A"
    assert (tmp_path / "t.csv").exists() and (tmp_path / "t.json").exists()


def test_un_document_casse_ne_tue_pas_la_matrice(tmp_path):
    def md(path):
        if path.name == "b.pdf":
            raise ValueError("PDF illisible")
        return "contenu sain"

    fake = fake_llm(rules=[wire_rule("alpha")])
    rows = bm.build([DOC_B, DOC_A], llm=fake, stem="t", out=tmp_path, to_markdown=md)
    assert len(rows) == 1 and rows[0]["jurisdiction"] == "Codex"  # B -> [] ; A survives


def test_ecriture_incrementale_apres_chaque_document(tmp_path):
    partial_on_disk = []

    def md(path):
        # observed WHEN parsing each doc: is the previous doc's partial already written?
        partial_on_disk.append((tmp_path / "t.csv").exists())
        return "contenu"

    bm.build([DOC_A, DOC_B], llm=fake_llm(rules=[wire_rule("alpha")]),
             stem="t", out=tmp_path, to_markdown=md)
    assert partial_on_disk == [False, True]  # an interrupted run keeps the partial


def test_config_error_tue_le_run_pas_un_doc_en_echec(tmp_path):
    fake = fake_llm().on("rules", raises=LLMConfigError("clé absente"))
    with pytest.raises(LLMConfigError):
        bm.build([DOC_A], llm=fake, stem="t", out=tmp_path, to_markdown=_md)
