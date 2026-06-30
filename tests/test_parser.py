"""parser: choosing the extraction path (Docling by default, pymupdf4llm in fast mode, OCR forces Docling).
Offline: both engines are replaced by doubles, so we test the routing, not Docling itself.
"""
import pytest

from pipeline import parser


@pytest.fixture
def moteurs(monkeypatch):
    """Replace both engines with markers: the return value tells which path was taken."""
    monkeypatch.setattr(parser, "_pymupdf_markdown", lambda p: "VOIE-RAPIDE")
    monkeypatch.setattr(parser, "_docling_markdown", lambda p, ocr: f"DOCLING(ocr={ocr})")


def test_defaut_docling(moteurs, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    assert parser.to_markdown(pdf) == "DOCLING(ocr=False)"


def test_fast_prend_la_voie_rapide_sur_pdf(moteurs, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    assert parser.to_markdown(pdf, fast=True) == "VOIE-RAPIDE"


def test_ocr_force_docling_meme_en_fast(moteurs, tmp_path):
    # a scan has no text: the fast path is useless, OCR wins
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    assert parser.to_markdown(pdf, fast=True, ocr=True) == "DOCLING(ocr=True)"


def test_fast_ignore_hors_pdf(moteurs, tmp_path):
    # the fast path only reads PDFs: a .docx goes through Docling even with fast=True
    docx = tmp_path / "doc.docx"
    docx.write_bytes(b"fake")
    assert parser.to_markdown(docx, fast=True) == "DOCLING(ocr=False)"


def test_fichier_absent_erreur_claire(moteurs, tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.to_markdown(tmp_path / "absent.pdf")
