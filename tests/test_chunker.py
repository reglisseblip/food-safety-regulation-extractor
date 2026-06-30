"""The chunker (document splitting): grouping, deterministic fallback, thresholds.

It calls the LLM in the same format as the rest of the pipeline, so the fake LLM
works here too, unchanged.
"""
from pipeline.chunker import MAX_ATOMS_FOR_LLM, MIN_SECTIONS_FOR_SPLIT, chunk
from fake_llm import fake_llm
from llm import LLMCallError


def test_petit_doc_un_seul_morceau_sans_appel():
    fake = fake_llm()
    md = "petit document qui tient dans la fenêtre"
    assert chunk(md, llm=fake) == [md]
    assert fake.count("splits") == 0


def test_atomes_jamais_coupes_et_titres_en_tete_de_morceau():
    md = "# A\n\nAAA premier paragraphe assez long.\n\n# B\n\nBBB second paragraphe."
    fake = fake_llm()  # empty LLM boundaries -> only structural splits remain
    chunks = chunk(md, max_chars=40, llm=fake)
    fake.assert_called("splits", times=1)      # boundaries requested once
    assert len(chunks) >= 2
    # no atom is split: concatenation yields exactly the original atoms
    atoms = [a for c in chunks for a in c.split("\n\n")]
    assert atoms == [a.strip() for a in md.split("\n\n")]
    assert any(c.startswith("# B") for c in chunks)  # structural split on heading


def test_atome_surdimensionne_reste_entier():
    gros = "x" * 300
    chunks = chunk(f"a\n\n{gros}\n\nb", max_chars=100, llm=fake_llm())
    assert gros in chunks  # alone in its chunk, but never split


def test_trop_d_atomes_repli_deterministe_sans_appel_llm():
    md = "\n\n".join(f"para {i} " + "x" * 20 for i in range(MAX_ATOMS_FOR_LLM + 1))
    fake = fake_llm()
    chunks = chunk(md, max_chars=200, llm=fake)
    assert fake.count("splits") == 0  # past the threshold: heading+size splits, no LLM call
    assert len(chunks) > 1


def test_doc_dense_qui_tient_est_coupe_sur_sections_sans_llm():
    # A doc that FITS in the window but is rich in sections is split ON HEADINGS (to
    # extract in parallel), without ever calling the LLM. Splitting only on sections -> a rule
    # stays whole within a chunk (P10, no overlap -> no fabricated duplicate).
    section = lambda i: f"# Section {i}\n\n" + ("bla " * 200)  # ~800 characters per section
    md = "\n\n".join(section(i) for i in range(MIN_SECTIONS_FOR_SPLIT + 3))
    assert len(md) <= 16000  # it fits: without the "dense" rule this would be a single chunk
    fake = fake_llm()
    chunks = chunk(md, llm=fake)
    assert fake.count("splits") == 0                          # structural split -> zero LLM calls
    assert len(chunks) > 1                                    # split for parallelism
    assert all(c.lstrip().startswith("# Section") for c in chunks)  # each chunk starts on a heading


def test_doc_peu_structure_qui_tient_reste_un_morceau():
    # Few sections (< threshold): splitting gains nothing -> 1 chunk, prior behavior.
    md = "# Titre unique\n\n" + ("bla " * 500)
    assert chunk(md, llm=fake_llm()) == [md]


def test_frontieres_llm_utilisees_et_indices_hors_bornes_filtres():
    # The LLM proposes boundaries: valid ones split, wild ones (0, negative, too large)
    # are ignored without breaking. Atoms without headings -> only LLM boundaries can split.
    md = "\n\n".join(f"paragraphe {i} " + "x" * 30 for i in range(6))
    fake = fake_llm(splits=[3, 0, -1, 99])  # only boundary 3 is within bounds
    chunks = chunk(md, max_chars=120, llm=fake)
    fake.assert_called("splits", times=1)
    assert any(c.startswith("paragraphe 3") for c in chunks)  # the LLM split did happen


def test_gros_tableau_coupe_en_lots_avec_entete_recopie():
    # A large table (300 rows, no blank line) = 1 atom: without splitting it would go out in ONE
    # call -> under-enumeration / truncation. We split it into batches, each batch replays the
    # header, and no data row is lost.
    entete = "| INS | Additive |\n|-----|----------|"
    lignes = [f"| {200+i} | Additif numéro {i} |" for i in range(300)]
    md = entete + "\n" + "\n".join(lignes)
    chunks = chunk(md, llm=fake_llm())
    assert len(chunks) > 1                                        # split, not a single giant call
    assert all(c.lstrip().startswith("| INS | Additive |") for c in chunks)  # header in every batch
    # all 300 data rows are present, exactly once, spread across the batches
    presentes = sum(c.count("Additif numéro ") for c in chunks)
    assert presentes == 300
    # each batch bounded: no more than TABLE_MAX_ROWS data rows
    from pipeline.chunker import TABLE_MAX_ROWS
    assert all(c.count("Additif numéro ") <= TABLE_MAX_ROWS for c in chunks)


def test_petit_tableau_reste_entier():
    # A small table (below the threshold) is NOT re-split — no over-fragmentation.
    md = "| INS | Additive |\n|-----|----------|\n| 200 | Sorbic acid |\n| 300 | Ascorbic acid |"
    assert chunk(md, llm=fake_llm()) == [md]


def test_panne_du_llm_de_frontieres_replie_sur_titres(caplog):
    # The boundary call FAILS (LLMCallError) -> deterministic fallback: split on headings,
    # splitting still completes. The LLM is a quality bonus, never a dependency.
    md = "# A\n\n" + "aaa " * 40 + "\n\n# B\n\n" + "bbb " * 40
    fake = fake_llm().on("splits", raises=LLMCallError)
    chunks = chunk(md, max_chars=100, llm=fake)
    assert fake.errors                                   # the failure was indeed triggered…
    assert len(chunks) >= 2                              # …and splitting completed without it
    assert any(c.startswith("# B") for c in chunks)      # split on the heading (fallback)
