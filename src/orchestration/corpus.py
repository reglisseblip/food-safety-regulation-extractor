"""
corpus — the engine that builds the matrix over a SET of documents (the corpus, a folder of rules).

Each document runs through the LangGraph doc_workflow (parse -> extract), keyed by a
thread_id = hash of its CONTENT: an unchanged document is SKIPPED (watch mode, without even
reparsing), and an interrupted run RESUMES where it stopped. matrix_io stays the ONLY place that
turns a Rule into a CSV row / JSON record.

Two error families (P16): a bad config (LLMConfigError) stops the WHOLE corpus; a document that
breaks any other way yields 0 rows but does not stop the others.

Sequential by default: we write to disk after each document -> an interrupted run keeps the partial
work already done. Inter-document parallelism turns on with max_parallel_docs > 1 (the global
semaphore in llm.py stays THE real cap on LLM calls, regardless of how many documents are in
flight); in that case writing happens on the main thread, as each document arrives.

This engine serves both build_matrix.build() (reference corpus) AND entrypoint.run_targets() (a
folder): it accepts any target exposing .path, .contexte(), .jurisdiction and .document_ref.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from matrix import io as matrix_io
from orchestration.doc_workflow import build_doc_workflow
from llm import LLMConfigError
from schema import Rule

log = logging.getLogger("matrix")


def _thread_id(path: pathlib.Path) -> str:
    """A document's identity = hash of its CONTENT: unchanged -> same id -> skipped (watch mode);
    modified -> new id -> reprocessed (never a resume on stale state). If the file is unreadable
    (e.g. the fake targets in tests), we fall back to the path -> still a stable id."""
    p = pathlib.Path(path)
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        digest = hashlib.sha256(str(p).encode("utf-8")).hexdigest()
    return "doc:" + digest[:16]


def _write(rows: list[dict], records: list[dict], stem: str, out: pathlib.Path) -> None:
    """CSV = the flat, filterable table (columns: matrix_io.COLUMNS); JSON = the full records."""
    (out / f"{stem}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=matrix_io.COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _rows_records(target, serialized_rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rehydrate the rules (JSON -> Rule) then convert them via matrix_io (the only place)."""
    rules = [Rule.model_validate(r) for r in serialized_rules]
    rows = [matrix_io.to_row(r, target.jurisdiction, target.document_ref) for r in rules]
    records = [matrix_io.to_record(r, target.document_ref) for r in rules]
    return rows, records


def run_corpus(targets, *, llm, to_markdown, out: pathlib.Path, stem: str = "matrix",
               checkpointer=None, resume: bool = True, max_parallel_docs: int = 1,
               on_progress=None) -> list[dict]:
    """Build the matrix over the corpus and return the CSV rows (as build() did before).
    - llm(messages, response_model) -> validated object and to_markdown(path) -> markdown: injected into the workflow;
    - checkpointer=None -> ephemeral InMemorySaver (one run per call); SqliteSaver -> resumable/watch mode;
    - max_parallel_docs > 1 -> documents in parallel (the global semaphore stays the real cap);
    - on_progress(done, total, label) -> called as documents complete (for a progress bar)."""
    from langgraph.checkpoint.memory import InMemorySaver

    targets = list(targets)
    checkpointer = checkpointer or InMemorySaver()
    wf = build_doc_workflow(checkpointer)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def _process(target) -> list[dict]:
        """-> the document's serialized rules; may raise LLMConfigError (fatal, propagates)."""
        config = {"configurable": {"thread_id": _thread_id(target.path),
                                   "llm": llm, "to_markdown": to_markdown}}
        if resume:
            state = wf.get_state(config)
            if state.next == () and isinstance(state.values, list):  # already done -> reuse (no parse, no LLM)
                log.info("[dim]%s déjà fait (veille) -> réutilisé[/]", target.document_ref)
                return state.values
        return wf.invoke({"path": str(target.path), "contexte": target.contexte()}, config=config)

    def _safe(target) -> tuple[list[dict], list[dict]]:
        """Per-document isolation: a bad config propagates (stop); any other failure -> [] (only that doc breaks)."""
        try:
            return _rows_records(target, _process(target))
        except LLMConfigError:  # bad config: this is not "a failed doc", we stop everything
            raise
        except Exception as e:
            log.error("[red]%s ÉCHEC[/] : %s", target.document_ref, type(e).__name__)
            return [], []

    rows: list[dict] = []
    records: list[dict] = []
    total = len(targets)
    if max_parallel_docs <= 1:
        for i, target in enumerate(targets):         # sequential: write after each document
            if on_progress:                          # before: show the document IN PROGRESS (i already done)
                on_progress(i, total, target.document_ref)
            r, rec = _safe(target)
            rows.extend(r)
            records.extend(rec)
            _write(rows, records, stem, out)
        if on_progress:
            on_progress(total, total, "")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=max_parallel_docs) as ex:
            futures = [ex.submit(_safe, t) for t in targets]
            try:
                for fut in as_completed(futures):    # write on arrival, on the main thread
                    r, rec = fut.result()            # a worker's LLMConfigError surfaces here -> stop
                    rows.extend(r)
                    records.extend(rec)
                    _write(rows, records, stem, out)
                    done += 1
                    if on_progress:                  # in parallel: on arrival (order is not guaranteed)
                        on_progress(done, total, "")
            except LLMConfigError:
                for f in futures:                    # "stop everything" means right now:
                    f.cancel()                       # documents still queued do not start
                raise
    return rows
