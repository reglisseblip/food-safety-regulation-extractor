"""entrypoint: target resolution (file / folder / --country), CLI guardrails,
CSV+JSON writing. Offline: dry-run and FakeLLM, no key required."""
import json
import pathlib

import pytest

from cli import app as entrypoint
from fake_llm import fake_llm
from payloads import wire_rule

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_collection_depuis_fichier_et_dossier(tmp_path):
    one = tmp_path / "one.md"
    one.write_text("alpha", encoding="utf-8")
    two = tmp_path / "sub" / "two.txt"
    two.parent.mkdir()
    two.write_text("beta", encoding="utf-8")
    ignored = tmp_path / "notes.bin"
    ignored.write_text("x", encoding="utf-8")

    single = entrypoint.TargetCollection.from_paths(
        [one], jurisdiction="EU", theme="claims", document_ref="DOC-1")
    assert len(single.targets) == 1
    assert single.targets[0].document_ref == "DOC-1"

    coll = entrypoint.TargetCollection.from_paths(
        [tmp_path], jurisdiction="EU", theme="claims", recursive=True)
    assert [t.document_ref for t in coll.targets] == ["one", "two"]
    assert all(t.contexte()["jurisdiction"] == "EU" for t in coll.targets)


def test_collection_repare_chemin_git_bash_collapsed(monkeypatch):
    # The repair searches from the CURRENT directory: we pin it to the repo root
    # so the test passes regardless of the launcher (pytest at root, IDE, tests/ folder).
    monkeypatch.chdir(ROOT)
    coll = entrypoint.TargetCollection.from_paths(
        ["fixturessample_gsfa.md"],
        jurisdiction="Codex",
        theme="food_additives",
    )

    assert len(coll.targets) == 1
    assert coll.targets[0].path.name == "sample_gsfa.md"


def test_document_ref_refuse_sur_collection(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")

    with pytest.raises(ValueError, match="document-ref"):
        entrypoint.TargetCollection.from_paths(
            [tmp_path], jurisdiction="EU", document_ref="DOC", recursive=True)


def test_run_targets_ecrit_csv_et_json(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("Alpha : 500 mg/kg.", encoding="utf-8")
    target = entrypoint.ExtractionTarget(src, "Codex", "DOC-A", "food_additives")

    rows = entrypoint.run_targets(
        [target],
        llm=fake_llm(rules=[wire_rule("alpha")]),
        out=tmp_path / "out",
        stem="m",
        to_markdown=lambda path: path.read_text(encoding="utf-8"),
    )

    assert rows[0]["jurisdiction"] == "Codex"
    assert rows[0]["document"] == "DOC-A"
    assert (tmp_path / "out" / "m.csv").exists()
    records = json.loads((tmp_path / "out" / "m.json").read_text(encoding="utf-8"))
    assert records[0]["document"] == "DOC-A"


def test_country_resout_le_dossier_et_la_juridiction(capsys, tmp_path, monkeypatch):
    # "The country becomes the argument": --country India -> processes source/reglementations/India/,
    # jurisdiction=India, without needing a path or --jurisdiction. Dry-run = no LLM.
    # (Fake country folder: the test does not depend on the PDFs actually present in the repo.)
    (tmp_path / "India").mkdir()
    (tmp_path / "India" / "doc.md").write_text("règle", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "REGLEMENTATIONS", tmp_path)

    code = entrypoint.main(["run", "--country", "India", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    lignes = [l for l in out.splitlines() if l.strip()]
    assert lignes and all(l.startswith("India |") for l in lignes)  # jurisdiction = country, >= 1 document


def test_country_dossier_absent_message_clair(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint, "REGLEMENTATIONS", tmp_path)
    code = entrypoint.main(["run", "--country", "Atlantis", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 2 and "introuvable" in out  # actionable message, not a traceback


def test_run_sans_country_ni_cibles_est_refuse(capsys):
    code = entrypoint.main(["run", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 2 and "--country" in out


def test_country_et_cibles_a_la_fois_est_refuse(capsys, tmp_path, monkeypatch):
    # Both modes at once: we refuse rather than guess (and silently ignore a target).
    (tmp_path / "India").mkdir()
    monkeypatch.setattr(entrypoint, "REGLEMENTATIONS", tmp_path)
    code = entrypoint.main(["run", "un_fichier.md", "--country", "India", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 2 and "exclusifs" in out


def test_main_run_complet_ecrit_et_annonce_le_csv(capsys, tmp_path, monkeypatch):
    # The FULL CLI path (no --dry-run): bind replaced by the FakeLLM -> offline.
    src = tmp_path / "doc.md"
    src.write_text("Alpha : 500 mg/kg.", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "bind", lambda **_kw: fake_llm(rules=[wire_rule("alpha")]))

    code = entrypoint.main(["run", str(src), "--jurisdiction", "Codex",
                            "--out", str(tmp_path / "out"), "--stem", "m"])
    out = capsys.readouterr().out
    assert code == 0
    assert "OK: 1 règle(s)" in out
    assert (tmp_path / "out" / "m.csv").exists()
    assert (tmp_path / "out" / "checkpoints.sqlite").exists()  # the CLI checkpoint store is durable


def test_run_request_cache_resolution_llm_et_ecriture(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("Alpha : 500 mg/kg.", encoding="utf-8")
    request = entrypoint.ExtractionRequest(
        paths=(src,),
        jurisdiction="Codex",
        theme="food_additives",
        document_ref="DOC-A",
        out=tmp_path / "out",
        stem="m",
    )

    result = entrypoint.run_request(
        request,
        llm=fake_llm(rules=[wire_rule("alpha")]),
        to_markdown=lambda path: path.read_text(encoding="utf-8"),
    )

    assert result.rule_count == 1
    assert result.targets[0].document_ref == "DOC-A"
    assert result.csv_path == tmp_path / "out" / "m.csv"
    assert result.json_path.exists()
