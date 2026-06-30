"""corpus.run_corpus: watch mode (skip by content hash), reprocessing a modified doc, parallelism.

Offline: FakeLLM + InMemorySaver, no network. (Per-doc isolation, incremental writes
and abort on config-error are already covered via build() in test_build_matrix.py.)
"""
from langgraph.checkpoint.memory import InMemorySaver

from orchestration import corpus
from cli.app import ExtractionTarget
from fake_llm import fake_llm
from payloads import wire_rule


def _md(path):
    return path.read_text(encoding="utf-8")


def _target(path, ref="DOC-A"):
    return ExtractionTarget(path, "Codex", ref, "food_additives")


def test_veille_saute_les_documents_inchanges(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("Alpha 500 mg/kg", encoding="utf-8")
    saver = InMemorySaver()

    fake1 = fake_llm(rules=[wire_rule("alpha")])
    rows1 = corpus.run_corpus([_target(doc)], llm=fake1, to_markdown=_md,
                              out=tmp_path / "out", stem="m", checkpointer=saver)
    assert [r["subject"] for r in rows1] == ["alpha"]
    assert fake1.count("rules") >= 1

    # 2nd run, SAME checkpointer, unchanged content -> skipped: zero LLM calls, rows reused from the checkpoint
    fake2 = fake_llm(rules=[wire_rule("alpha")])
    rows2 = corpus.run_corpus([_target(doc)], llm=fake2, to_markdown=_md,
                              out=tmp_path / "out", stem="m", checkpointer=saver)
    assert fake2.count("rules") == 0
    assert [r["subject"] for r in rows2] == ["alpha"]


def test_veille_retraite_un_document_modifie(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("Alpha 500 mg/kg", encoding="utf-8")
    saver = InMemorySaver()
    corpus.run_corpus([_target(doc)], llm=fake_llm(rules=[wire_rule("alpha")]),
                      to_markdown=_md, out=tmp_path / "out", stem="m", checkpointer=saver)

    doc.write_text("Beta 999 mg/kg", encoding="utf-8")  # changed content -> new hash -> reprocessed
    fake2 = fake_llm(rules=[wire_rule("beta")])
    rows2 = corpus.run_corpus([_target(doc)], llm=fake2, to_markdown=_md,
                              out=tmp_path / "out", stem="m", checkpointer=saver)
    assert fake2.count("rules") >= 1
    assert [r["subject"] for r in rows2] == ["beta"]


def test_on_progress_suit_l_avancement_sequentiel(tmp_path):
    # The progress bar (CLI/UI) relies on this callback: before each document (i done out of n,
    # with the name of the current doc), then a final (n, n, "") once everything is finished.
    docs = []
    for i in range(2):
        d = tmp_path / f"d{i}.md"
        d.write_text(f"contenu {i}", encoding="utf-8")
        docs.append(_target(d, ref=f"DOC-{i}"))

    vus = []
    corpus.run_corpus(docs, llm=fake_llm(rules=[wire_rule("x")]), to_markdown=_md,
                      out=tmp_path / "out", stem="m",
                      on_progress=lambda done, total, label: vus.append((done, total, label)))
    assert vus[0] == (0, 2, "DOC-0")
    assert vus[1] == (1, 2, "DOC-1")
    assert vus[-1] == (2, 2, "")


def test_documents_en_parallele_tous_traites(tmp_path):
    docs = []
    for i in range(6):
        d = tmp_path / f"d{i}.md"
        d.write_text(f"document {i} contenu unique", encoding="utf-8")
        docs.append(_target(d, ref=f"DOC-{i}"))

    fake = fake_llm(rules=[wire_rule("x")])
    rows = corpus.run_corpus(docs, llm=fake, to_markdown=_md, out=tmp_path / "out",
                             stem="m", max_parallel_docs=3)
    assert len(rows) == 6                                      # all 6 documents processed
    assert {r["document"] for r in rows} == {f"DOC-{i}" for i in range(6)}
