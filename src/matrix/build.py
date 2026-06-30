"""
build_matrix — builds THE consolidated matrix over the real corpus: runs the pipeline on each
document, aggregates rules with their PROVENANCE (country + document), writes a CSV (filterable) + a JSON.

The ENGINE (corpus traversal, per-document isolation, writing after each doc, resume/watch) lives in
corpus.run_corpus; here we only keep the document list (DOCS) and the command-line entry point.

Coverage: the WHOLE corpus (47 documents, ~3,000 pages), document by document, as required by the
assignment. Documents whose official reference is known are pinned by hand (_REFS_CONNUES); the rest
is inferred from the folders (country = folder, reference = file name, shortened to "MOPH n" for Thai
notifications). SMALL documents go first — the three giants (GSFA 568 p., Thai #381 609 p., Thai #444
808 p.) come last: the matrix fills up quickly and an interrupted run keeps the essentials.
Checked on the files (2026-07-08): only ONE corpus document is scanned (Thai #281, 4 p.) — its entry
carries ocr=True; all the others have an extractable text layer.

Run:  uv run python src/build_matrix.py     (or "MOPH 174" for a single doc -> matrix.test.*)
From code: build(docs, llm, ...) -> delegates to corpus.run_corpus. The same function serves the
command line, the watch mode (reprocess only changed documents) and the tests.
"""
from __future__ import annotations

import logging
import pathlib
import sys
from dataclasses import dataclass

import logs
from llm import bind

log = logging.getLogger("matrix")
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "source" / "reglementations"
OUT = ROOT / "matrix"
# gpt-5-mini (OpenAI) produces the delivered matrix: validated by the benchmark (F1=1.000 on
# complex_en), and the delivered corpus is digital/English — its only measured weakness (degraded
# extraction) does not apply here. A Claude model is obtained via the "claude" profile (official API,
# ANTHROPIC_API_KEY) or "openrouter" (model anthropic/claude-sonnet-4.6).
PROVIDER = "openai"
MODEL = "gpt-5-mini"


@dataclass(frozen=True)
class DocContext:
    """ONE corpus document: who (country), what (reference, theme), where (file)."""
    jurisdiction: str
    document_ref: str
    theme: str
    file: str  # path relative to source/reglementations/
    ocr: bool = False  # scanned document: read via OCR (slower) — decided entry by entry

    @property
    def path(self) -> pathlib.Path:
        """Absolute path of the document. corpus.run_corpus works with .path, like ExtractionTarget."""
        return SRC / self.file

    def contexte(self) -> dict:
        """The dict expected by the structurer (see build_messages)."""
        return {"jurisdiction": self.jurisdiction, "document_ref": self.document_ref, "theme": self.theme}


# Documents whose official reference and theme are known: pinned HERE, by hand.
# Any file missing from this table gets an entry inferred from its name (see _docs_du_corpus).
_REFS_CONNUES = {
    "Codex/CXS 1-1985 labeling of prepacked foods .pdf": ("CXS 1-1985", "labeling"),
    "Codex/CXS_180e_FSMPs lab_claims.pdf": ("CXS 180-1991", "claims"),
    "Codex/CXS_192-1995.pdf": ("CXS 192-1995", "food_additives"),
    "EU/Reg. EU 828-2014_Gluten_en.pdf": ("Reg 828/2014", "gluten_claims"),
    "EU/Reg. EU 1169-2011.pdf": ("Reg 1169/2011", "food_information"),
    "EU/Regulation 1333-2008 consolidated version 2025.pdf": ("Reg 1333/2008", "food_additives"),
    "EU/EU flavouring legislation last updated Apr 2025.pdf": ("Reg 1334/2008", "flavourings"),
    "EU/EU enzymes regulation last updated 2012.pdf": ("Reg 1332/2008", "enzymes"),
    "EU/EU FSMP Delegated Regulation 2016 128.pdf": ("Reg 2016/128", "fsmp"),
    "EU/EU Reg. 609-2013_en.pdf": ("Reg 609/2013", "fsmp"),
    "India/GMP additives.pdf": ("GMP additives", "food_additives"),
    "India/India_Nutra Regulations.pdf": ("Nutraceutical Reg.", "nutraceuticals"),
}

# Checked on the files: the only SCANNED corpus document (no text layer -> OCR).
_DOCS_SCANNES = {"Thailand/ประกาศ สธ # 281 (พ.ศ.2547) เรื่อง วัตถุเจือปนอาหาร.pdf"}

# The Thai keywords that give a notification's theme (for inferred entries).
_THEMES_THAI = [
    ("วัตถุเจือปนอาหาร", "food_additives"),   # food additives
    ("สารพิษตกค้าง", "residues"),             # pesticide residues
    ("ยาสัตว์ตกค้าง", "residues"),            # veterinary residues
    ("ปนเปื้อน", "contaminants"),             # contaminants
    ("จุลินทรีย์", "microbiology"),           # micro-organisms
    ("ฉลาก", "labeling"),                     # labeling
]


def _fiche_deduite(pays: str, nom: str) -> tuple[str, str]:
    """(reference, theme) inferred from the file NAME, for documents without a _REFS_CONNUES entry.
    Thai notifications carry their number in the name ("# 174") -> "MOPH 174", far more readable in
    the matrix than the long Thai title."""
    import re
    m = re.search(r"#\s*(\d+)", nom)
    ref = f"MOPH {m.group(1)}" if m else pathlib.Path(nom).stem[:60]
    theme = next((t for mot, t in _THEMES_THAI if mot in nom), "other")
    return ref, theme


def _docs_du_corpus() -> list[DocContext]:
    """The 47 corpus documents, ONE entry per file, small ones first.
    Country = folder name; reference/theme = _REFS_CONNUES when known, otherwise inferred from the name.
    Sorted by file size: the giants (GSFA, Thai #381/#444) come last — the matrix fills up quickly and
    an interrupted run keeps the essentials."""
    docs = []
    for dossier in sorted(SRC.iterdir()):
        if not dossier.is_dir():
            continue
        for fichier in sorted(dossier.iterdir()):
            if not fichier.is_file():
                continue
            rel = f"{dossier.name}/{fichier.name}"
            ref, theme = _REFS_CONNUES.get(rel) or _fiche_deduite(dossier.name, fichier.name)
            docs.append(DocContext(dossier.name, ref, theme, rel, ocr=rel in _DOCS_SCANNES))
    return sorted(docs, key=lambda d: d.path.stat().st_size)


DOCS = _docs_du_corpus()


def _default_to_markdown(docs: list[DocContext]):
    """The default reader: Docling, with OCR turned on DOCUMENT BY DOCUMENT (only scanned ones pay for it).
    Lazy import: Docling is loaded only if we actually parse — importing this module stays instant for
    the watch mode and for the tests."""
    from pipeline import parser as doc_parser
    ocr_par_chemin = {str(d.path): d.ocr for d in docs}

    def lire(path):
        return doc_parser.to_markdown(path, ocr=ocr_par_chemin.get(str(path), False))

    return lire


def build(docs: list[DocContext], llm, stem: str = "matrix", out: pathlib.Path = OUT,
          to_markdown=None, checkpointer=None, max_parallel_docs: int = 1) -> list[dict]:
    """Builds the matrix by delegating to corpus.run_corpus. checkpointer=None -> ephemeral InMemorySaver
    (one run per call, same behavior as before); a SqliteSaver makes the run resumable/incremental across
    invocations (watch mode). Returns the CSV rows.
    The same function serves the command line, the watch mode and the tests."""
    from orchestration import corpus
    to_markdown = to_markdown or _default_to_markdown(docs)
    return corpus.run_corpus(docs, llm=llm, to_markdown=to_markdown, out=out, stem=stem,
                             checkpointer=checkpointer, max_parallel_docs=max_parallel_docs)


# Number of documents processed AT THE SAME TIME for the full run (47 docs): while a large document is
# in Docling (CPU), the LLM calls for the others make progress. The global semaphore in llm.py (6)
# stays THE cap on calls, whatever the number of documents in flight.
PARALLEL_DOCS = 3

if __name__ == "__main__":
    logs.setup()
    log.setLevel(logging.INFO)
    # "test on 1 case" mode: `python -m matrix.build "MOPH 174"` -> processes only that/those doc(s),
    # and writes to matrix.test.* so as NOT to overwrite the real consolidated matrix.
    query = sys.argv[1].strip().lower() if len(sys.argv) > 1 else None
    docs = [d for d in DOCS if query in d.document_ref.lower()] if query else DOCS
    if query and not docs:
        log.error("Aucun document dont la référence contient « %s ». Réfs : %s",
                  query, [d.document_ref for d in DOCS])
        raise SystemExit(1)
    stem = "matrix.test" if query else "matrix"
    log.info("Matrice : %d document(s)%s · %s @ %s …", len(docs), " (test 1 cas)" if query else "", MODEL, PROVIDER)

    if query:
        # TRIAL run: ephemeral checkpointer — it does not touch the real run's watch state,
        # and two processes don't fight over the same sqlite file.
        rows = build(docs, llm=bind(model=MODEL, provider=PROVIDER), stem=stem)
    else:
        # Durable checkpointer (SqliteSaver): an interrupted run resumes, an unchanged document is skipped (watch mode).
        from langgraph.checkpoint.sqlite import SqliteSaver
        OUT.mkdir(exist_ok=True)
        with SqliteSaver.from_conn_string(str(OUT / "checkpoints.sqlite")) as saver:
            rows = build(docs, llm=bind(model=MODEL, provider=PROVIDER), stem=stem,
                         checkpointer=saver, max_parallel_docs=PARALLEL_DOCS)

    by_jur, by_type = {}, {}
    for r in rows:
        by_jur[r["jurisdiction"]] = by_jur.get(r["jurisdiction"], 0) + 1
        by_type[r["rule_type"]] = by_type.get(r["rule_type"], 0) + 1
    log.info("[bold]MATRICE — %d règles[/] → matrix/%s.csv + .json", len(rows), stem)
    log.info("  par pays : %s", by_jur)
    log.info("  par type : %s", by_type)
