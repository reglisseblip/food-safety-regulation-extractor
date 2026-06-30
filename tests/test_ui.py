"""agentreg_ui: smoke test — the app builds, opens (headless Textual pilot),
and its key widgets exist. Guards the `agentreg ui` demo against an import/CSS regression.
"""
from textual.widgets import DataTable, Input, Switch

from cli.ui import AgentRegApp


async def test_l_app_demarre_et_a_ses_widgets():
    app = AgentRegApp(initial_target="fixtures/sample_gsfa.md")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#target", Input).value == "fixtures/sample_gsfa.md"
        assert app.query_one("#run") is not None                  # Extract button
        assert app.query_one("#ocr", Switch).value is False       # OCR off by default
        table = app.query_one("#bench-table", DataTable)
        assert len(table.columns) == 5                            # Case, F1, Precision, Recall, extracted/gold
