"""
entrypoint — the user entry point: file(s) -> matrix.
Stable API over the pipeline (ExtractionTarget/TargetCollection) plus the `main()` CLI (agentreg).
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

# Default model and provider live in llm.py (single source, no drifting copy).
from llm import MODEL as DEFAULT_MODEL
from llm import PROVIDER as DEFAULT_PROVIDER
from llm import LLMCallError, LLMConfigError, bind

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "matrix"
REGLEMENTATIONS = ROOT / "source" / "reglementations"  # one subfolder per country: the country is the argument
SUPPORTED_INPUTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
_SKIPPED_PATH_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class ExtractionRequest:
    """Main interface for CLI/UI adapters.
    Adapters build the request; this module resolves files, binds the LLM, extracts, writes.
    """

    paths: tuple[str | pathlib.Path, ...]
    jurisdiction: str
    theme: str = "other"
    document_ref: str | None = None
    recursive: bool = False
    ocr: bool = False
    fast: bool = False   # fast pymupdf4llm path (digital PDF) instead of Docling
    out: pathlib.Path = DEFAULT_OUT
    stem: str = "extraction"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class ExtractionResult:
    rows: list[dict]
    targets: tuple["ExtractionTarget", ...]
    csv_path: pathlib.Path
    json_path: pathlib.Path

    @property
    def rule_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class ExtractionTarget:
    """One document to extract, with its business context."""

    path: pathlib.Path
    jurisdiction: str
    document_ref: str
    theme: str = "other"

    def contexte(self) -> dict:
        return {
            "jurisdiction": self.jurisdiction,
            "document_ref": self.document_ref,
            "theme": self.theme,
        }


@dataclass(frozen=True)
class TargetCollection:
    """Homogeneous set of targets built from files or directories."""

    targets: tuple[ExtractionTarget, ...]

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | pathlib.Path],
        *,
        jurisdiction: str,
        theme: str = "other",
        document_ref: str | None = None,
        recursive: bool = False,
    ) -> "TargetCollection":
        files = _resolve_files(paths, recursive=recursive)
        if not files:
            raise ValueError("Aucun fichier compatible trouvé.")
        if document_ref and len(files) > 1:
            raise ValueError("--document-ref est réservé à une cible unique.")
        return cls(tuple(
            ExtractionTarget(
                path=file,
                jurisdiction=jurisdiction,
                document_ref=document_ref or file.stem,
                theme=theme,
            )
            for file in files
        ))


def _resolve_files(paths: Sequence[str | pathlib.Path], *, recursive: bool) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for raw in paths:
        path = _coerce_path(raw)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            out.extend(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS:
            out.append(path)
        elif not path.exists():
            raise FileNotFoundError(path)
    return sorted({p.resolve() for p in out})


def _coerce_path(raw: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw)
    if path.exists():
        return path
    if any(sep in str(raw) for sep in ("/", "\\")):
        return path
    repaired = _repair_collapsed_shell_path(str(raw))
    return repaired or path


def _repair_collapsed_shell_path(raw: str) -> pathlib.Path | None:
    """Recover paths such as `fixturessample.md` produced by unquoted backslashes in Git Bash."""
    cwd = pathlib.Path.cwd()
    matches: list[pathlib.Path] = []
    for candidate in cwd.rglob("*"):
        if any(part in _SKIPPED_PATH_PARTS for part in candidate.parts):
            continue
        try:
            rel = candidate.relative_to(cwd).as_posix()
        except ValueError:
            continue
        if rel.replace("/", "") == raw:
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    return None


def _default_to_markdown(path: pathlib.Path, *, ocr: bool = False, fast: bool = False) -> str:
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8")
    from pipeline import parser as doc_parser
    return doc_parser.to_markdown(path, ocr=ocr, fast=fast)


def run_targets(
    targets: Iterable[ExtractionTarget],
    *,
    llm: Callable,
    out: pathlib.Path = DEFAULT_OUT,
    stem: str = "extraction",
    to_markdown: Callable[[pathlib.Path], str] | None = None,
    ocr: bool = False,
    fast: bool = False,
    checkpointer=None,
    max_parallel_docs: int = 1,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Extract rules from the targets via corpus.run_corpus (shared engine: per-document isolation,
    write after each doc) and write `<stem>.csv` + `<stem>.json`.
    An interrupted run keeps a usable partial. `on_progress(done, total, label)` tracks progress.
    Cross-run resume/caching needs a DURABLE `checkpointer` (SqliteSaver) — the CLI passes one by
    default; without it, every call starts over (ephemeral InMemorySaver)."""
    from orchestration import corpus
    markdown = to_markdown or (lambda path: _default_to_markdown(path, ocr=ocr, fast=fast))
    return corpus.run_corpus(targets, llm=llm, to_markdown=markdown, out=out, stem=stem,
                             checkpointer=checkpointer, max_parallel_docs=max_parallel_docs,
                             on_progress=on_progress)


def preview_targets(request: ExtractionRequest) -> tuple[ExtractionTarget, ...]:
    """Resolve a request into targets without parsing or calling the LLM."""
    return TargetCollection.from_paths(
        request.paths,
        jurisdiction=request.jurisdiction,
        theme=request.theme,
        document_ref=request.document_ref,
        recursive=request.recursive,
    ).targets


def run_request(
    request: ExtractionRequest,
    *,
    llm: Callable | None = None,
    to_markdown: Callable[[pathlib.Path], str] | None = None,
    checkpointer=None,
) -> ExtractionResult:
    """Run a full request and return the result written to disk."""
    targets = preview_targets(request)
    rows = run_targets(
        targets,
        llm=llm or bind(model=request.model, provider=request.provider),
        out=request.out,
        stem=request.stem,
        to_markdown=to_markdown,
        ocr=request.ocr,
        fast=request.fast,
        checkpointer=checkpointer,
    )
    return ExtractionResult(
        rows=rows,
        targets=targets,
        csv_path=request.out / f"{request.stem}.csv",
        json_path=request.out / f"{request.stem}.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentreg", description="Extraire une matrice depuis un ou plusieurs fichiers.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Traiter un fichier, plusieurs fichiers ou un dossier.")
    run.add_argument("targets", nargs="*", help="Fichier(s) ou dossier à traiter (inutile avec --country).")
    run.add_argument("--country", help="Nom du pays : traite source/reglementations/<pays>/ (récursif), "
                                       "jurisdiction=<pays>. « Le pays devient l'argument. » Ex. --country China.")
    run.add_argument("--jurisdiction", help="Pays / juridiction, ex. EU, Codex, India. Requis sans --country.")
    run.add_argument("--theme", default="other", help="Thème transmis au prompt.")
    run.add_argument("--document-ref", help="Référence document, seulement pour une cible unique.")
    run.add_argument("--recursive", action="store_true", help="Parcourir les dossiers récursivement.")
    run.add_argument("--ocr", action="store_true", help="Activer l'OCR Docling pour les PDFs scannés.")
    run.add_argument("--fast", action="store_true",
                     help="Voie rapide pymupdf4llm (PDF numérique, ~10x plus vite) au lieu de Docling.")
    run.add_argument("--out", default=str(DEFAULT_OUT), help="Dossier de sortie.")
    run.add_argument("--stem", default=None, help="Préfixe CSV/JSON (défaut : 'extraction', ou le pays avec --country).")
    run.add_argument("--provider", default=DEFAULT_PROVIDER, help="Fournisseur LLM.")
    run.add_argument("--model", default=DEFAULT_MODEL, help="Modèle chez le fournisseur.")
    run.add_argument("--dry-run", action="store_true", help="Lister les cibles sans appeler le LLM.")

    ui = sub.add_parser("ui", help="Lancer l'interface Textual.")
    ui.add_argument("target", nargs="?", help="Fichier ou dossier prérempli dans l'interface.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ui":
        from cli.ui import run_ui
        run_ui(initial_target=args.target)
        return 0

    # "The country becomes the argument": --country resolves its own folder and jurisdiction.
    if args.country:
        if args.targets:  # both at once: we don't guess which one the user meant
            print("--country et des cibles sont exclusifs : donne l'un OU l'autre. (agentreg run --help)")
            return 2
        folder = REGLEMENTATIONS / args.country
        if not folder.is_dir():
            print(f"Dossier pays introuvable : {folder}\n"
                  f"Dépose les PDF de « {args.country} » dans ce dossier, puis relance la même commande.")
            return 2
        paths = (str(folder),)
        jurisdiction = args.jurisdiction or args.country
        recursive = True
    else:
        if not args.targets or not args.jurisdiction:
            print("Précise soit --country <pays>, soit des cibles + --jurisdiction. (agentreg run --help)")
            return 2
        paths = tuple(args.targets)
        jurisdiction = args.jurisdiction
        recursive = args.recursive

    request = ExtractionRequest(
        paths=paths,
        jurisdiction=jurisdiction,
        theme=args.theme,
        document_ref=args.document_ref,
        recursive=recursive,
        ocr=args.ocr,
        fast=args.fast,
        out=pathlib.Path(args.out),
        stem=args.stem or (args.country.lower() if args.country else "extraction"),
        provider=args.provider,
        model=args.model,
    )
    try:
        if args.dry_run:
            for target in preview_targets(request):
                print(f"{target.jurisdiction} | {target.theme} | {target.document_ref} | {target.path}")
            return 0

        import logs
        logs.setup()  # without this, a multi-minute run would be silent (the pipeline's log.info calls would be lost)

        # Durable checkpointer = caching: rerunning the same command skips unchanged documents,
        # and an interrupted run resumes where it left off (see corpus.py — thread_id = content hash).
        from langgraph.checkpoint.sqlite import SqliteSaver
        request.out.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(request.out / "checkpoints.sqlite")) as saver:
            result = run_request(request, checkpointer=saver)
    except (ValueError, FileNotFoundError) as e:  # empty folder, invalid path…: clear message, no traceback
        print(f"Rien à traiter : {e}")
        return 2
    except LLMConfigError as e:  # missing/invalid key, unknown provider: actionable message, no traceback
        print(f"Configuration LLM à corriger : {e}")
        return 2
    except LLMCallError as e:    # network/provider down despite retries: same
        print(f"Appels LLM en échec : {e}")
        return 2
    print(f"OK: {result.rule_count} règle(s) -> {result.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
