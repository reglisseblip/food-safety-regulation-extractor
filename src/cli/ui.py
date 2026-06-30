"""AgentReg Textual interface: two tabs (Extraction, Benchmark).

Design choice: a single accent color, well-spaced titled panels, and a detailed PROGRESS
area (live status + bar + ETA) — the element that anchors the screen. Work runs in a threaded
worker so the interface never freezes; each step shows up as it happens.
"""
from __future__ import annotations

import logging
import pathlib

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (Button, DataTable, Footer, Header, Input, ProgressBar,
                             RichLog, Static, Switch, TabbedContent, TabPane)

from cli.app import (DEFAULT_MODEL, DEFAULT_OUT, DEFAULT_PROVIDER,
                     ExtractionRequest, preview_targets, run_targets)

ROOT = pathlib.Path(__file__).resolve().parents[2]


class _JournalHandler(logging.Handler):
    """Reroutes pipeline warnings/errors to the interface journal.
    Without it, a failed document (log.error from corpus._safe) would be invisible: the bar
    would advance, then "Terminé" would show fewer rules with no explanation."""

    def __init__(self, app: "AgentRegApp") -> None:
        super().__init__(level=logging.WARNING)  # INFO stays silent (progress has its own bar)
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._app.call_from_thread(self._app._log, record.getMessage())
        except Exception:
            pass  # app is closing or not ready: never break the pipeline for a log line


class AgentRegApp(App[None]):
    TITLE = "AgentReg"
    SUB_TITLE = "Extraction de règles réglementaires"

    BINDINGS = [
        Binding("q", "quit", "Quitter"),
        Binding("ctrl+c", "quit", "Quitter", show=False),
    ]

    CSS = """
    Screen { background: $surface; }

    .panel {
        height: auto;
        margin: 1 2 0 2;
        padding: 1 2;
        border: round $panel;
        background: $boost;
        border-title-align: left;
        border-title-color: $accent;
    }
    .panel:focus-within { border: round $accent; }

    .field { height: auto; margin-bottom: 1; }
    .field > .label {
        width: 14;
        color: $text-muted;
        content-align: right middle;
        padding-right: 1;
    }
    Input { width: 1fr; }

    .actions { height: auto; align-horizontal: right; margin-top: 1; }
    #run, #run-bench { min-width: 18; }

    .status { height: auto; color: $text-muted; margin-bottom: 1; }
    ProgressBar { width: 1fr; }

    .journal, .results {
        height: 1fr;
        margin: 1 2 1 2;
        border: round $panel;
        border-title-align: left;
        border-title-color: $accent;
    }
    .journal { padding: 0 1; }
    """

    def __init__(self, initial_target: str | None = None) -> None:
        super().__init__()
        self.initial_target = initial_target or ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-extract"):
            with TabPane("Extraction", id="tab-extract"):
                with Vertical(id="extract-form", classes="panel"):
                    yield self._field("Cible", "target", self.initial_target)
                    yield self._field("Juridiction", "jurisdiction", "EU")
                    yield self._field("Thème", "theme", "other")
                    yield self._field("Fournisseur", "provider", DEFAULT_PROVIDER)
                    yield self._field("Modèle", "model", DEFAULT_MODEL)
                    yield self._field("Sortie", "out", str(DEFAULT_OUT))
                    yield self._field("Préfixe", "stem", "extraction")
                    with Horizontal(classes="field"):
                        yield Static("OCR", classes="label")
                        yield Switch(id="ocr", value=False, tooltip="PDF scanné : lecture par OCR (Docling), plus lent")
                    with Horizontal(classes="actions"):
                        yield Button("Extraire", id="run", variant="primary")
                with Vertical(id="extract-progress", classes="panel"):
                    yield Static("Prêt.", id="extract-status", classes="status")
                    yield ProgressBar(id="extract-bar", total=None, show_eta=True)
                yield RichLog(id="extract-log", classes="journal", wrap=True, markup=True)
            with TabPane("Benchmark", id="tab-bench"):
                with Vertical(id="bench-form", classes="panel"):
                    yield self._field("Fournisseur", "b-provider", "openai")
                    yield self._field("Modèle", "b-model", "gpt-5-mini")
                    yield self._field("Cas", "b-case", "")
                    with Horizontal(classes="actions"):
                        yield Button("Lancer le benchmark", id="run-bench", variant="primary")
                with Vertical(id="bench-progress", classes="panel"):
                    yield Static("Prêt. Cas vide = les quatre.", id="bench-status", classes="status")
                    yield ProgressBar(id="bench-bar", total=None, show_eta=True)
                yield DataTable(id="bench-table", classes="results")
        yield Footer()

    def on_mount(self) -> None:
        # Pipeline logs propagate to the root; without a handler Python would print them to
        # stderr and corrupt the Textual display. Route WARNING+ to the journal (a failed
        # document must be SEEN) and swallow the rest (progress already has its bar).
        root = logging.getLogger()
        root.addHandler(_JournalHandler(self))
        root.setLevel(logging.WARNING)
        try:
            self.theme = "flexoki"  # sober "paper" palette; otherwise keep the default theme
        except Exception:
            pass
        self.query_one("#extract-form").border_title = "Paramètres"
        self.query_one("#extract-progress").border_title = "Progression"
        self.query_one("#extract-log").border_title = "Journal"
        self.query_one("#bench-form").border_title = "Paramètres"
        self.query_one("#bench-progress").border_title = "Progression"
        table = self.query_one("#bench-table", DataTable)
        table.border_title = "Résultats"
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Cas", "F1", "Précision", "Rappel", "extrait/gold")

    def _field(self, label: str, widget_id: str, value: str) -> Horizontal:
        return Horizontal(Static(label, classes="label"), Input(value=value, id=widget_id), classes="field")

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self._extract()
        elif event.button.id == "run-bench":
            self._benchmark()

    # ------------------------------------------------------------------ UI updates (main thread)
    def _progress(self, which: str, done: int, total: int, label: str) -> None:
        self.query_one(f"#{which}-bar", ProgressBar).update(total=total or None, progress=done)
        text = f"{done}/{total}" + (f" · {label}" if label else "")
        self.query_one(f"#{which}-status", Static).update(text if done < total else f"{total}/{total} · terminé")

    def _log(self, message: str) -> None:
        self.query_one("#extract-log", RichLog).write(message)

    def _bench_reset(self) -> None:
        self.query_one("#bench-table", DataTable).clear()
        self.query_one("#bench-bar", ProgressBar).update(total=None, progress=0)

    def _bench_add(self, name: str, s: dict) -> None:
        f1 = s["f1"]
        colour = "green" if f1 >= 0.9 else "yellow" if f1 >= 0.7 else "red"
        self.query_one("#bench-table", DataTable).add_row(
            name, Text(f"{f1:.3f}", style=f"bold {colour}"),
            f"{s['precision']:.3f}", f"{s['recall']:.3f}", f"{s['extracted']}/{s['gold']}",
        )

    def _status(self, which: str, message: str) -> None:
        self.query_one(f"#{which}-status", Static).update(message)

    # ------------------------------------------------------------------ Extraction (worker)
    @work(thread=True, group="extract", exclusive=True)
    def _extract(self) -> None:
        target = self._value("target")
        if not target:
            self.call_from_thread(self._log, "[b red]Cible manquante.[/] Indiquez un fichier ou un dossier.")
            return
        try:
            request = ExtractionRequest(
                paths=(target,),
                jurisdiction=self._value("jurisdiction") or "EU",
                theme=self._value("theme") or "other",
                recursive=True,
                ocr=self.query_one("#ocr", Switch).value,
                out=pathlib.Path(self._value("out") or DEFAULT_OUT),
                stem=self._value("stem") or "extraction",
                provider=self._value("provider") or DEFAULT_PROVIDER,
                model=self._value("model") or DEFAULT_MODEL,
            )
            targets = preview_targets(request)  # resolve the files (may fail: invalid path)
            total = len(targets)
            self.call_from_thread(self._progress, "extract", 0, total, "démarrage…")
            self.call_from_thread(self._log, f"[b]Extraction[/] · {total} document(s) · {request.model} @ {request.provider}")

            from llm import bind
            llm = bind(model=request.model, provider=request.provider)  # validate the config right away

            def cb(done: int, tot: int, label: str) -> None:
                self.call_from_thread(self._progress, "extract", done, tot, label)
                if label:
                    self.call_from_thread(self._log, f"  · {label}")

            # Same watch mode as the CLI: unchanged documents skipped, interrupted run resumed (durable checkpoint).
            from langgraph.checkpoint.sqlite import SqliteSaver
            request.out.mkdir(parents=True, exist_ok=True)
            with SqliteSaver.from_conn_string(str(request.out / "checkpoints.sqlite")) as saver:
                rows = run_targets(targets, llm=llm, out=request.out, stem=request.stem,
                                   ocr=request.ocr, checkpointer=saver, on_progress=cb)
            csv = request.out / f"{request.stem}.csv"
            self.call_from_thread(self._log, f"[b green]Terminé.[/] {len(rows)} règle(s) → {csv}")
            self.call_from_thread(self.notify, f"Extraction terminée · {len(rows)} règle(s)")
        except Exception as exc:
            self.call_from_thread(self._log, f"[b red]Échec[/] {type(exc).__name__} : {exc}")
            self.call_from_thread(self._status, "extract", "Échec.")
            self.call_from_thread(self.notify, f"{type(exc).__name__} : {exc}", severity="error", timeout=8)

    # ------------------------------------------------------------------ Benchmark (worker)
    @work(thread=True, group="bench", exclusive=True)
    def _benchmark(self) -> None:
        import sys

        provider = self._value("b-provider") or "openai"
        model = self._value("b-model") or "gpt-5-mini"
        case_filter = self._value("b-case")
        self.call_from_thread(self._bench_reset)

        bench_dir = str(ROOT / "benchmark")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        try:
            import run as bench  # benchmark/run.py (CASES, run_case)
            from llm import bind

            llm = bind(model=model, provider=provider)  # validate the config right away (key, provider)
            cases = [c for c in bench.CASES if not case_filter or c["name"] == case_filter]
            if not cases:
                noms = ", ".join(c["name"] for c in bench.CASES)
                self.call_from_thread(self.notify, f"Cas inconnu « {case_filter} ». Cas : {noms}", severity="error", timeout=8)
                self.call_from_thread(self._status, "bench", "Cas inconnu.")
                return

            total = len(cases)
            self.call_from_thread(self._progress, "bench", 0, total, "démarrage…")
            for i, c in enumerate(cases):
                self.call_from_thread(self._progress, "bench", i, total, f"{c['name']} · parsing + extraction…")
                s = bench.run_case(c, llm=llm)
                self.call_from_thread(self._bench_add, c["name"], s)
                self.call_from_thread(self._progress, "bench", i + 1, total, c["name"])
            self.call_from_thread(self.notify, f"Benchmark terminé · {total} cas · {model}")
        except Exception as exc:
            self.call_from_thread(self._status, "bench", "Échec.")
            self.call_from_thread(self.notify, f"{type(exc).__name__} : {exc}", severity="error", timeout=8)


def run_ui(initial_target: str | None = None) -> None:
    AgentRegApp(initial_target=initial_target).run()
