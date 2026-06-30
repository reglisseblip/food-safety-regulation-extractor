"""
run — runs the extraction pipeline on each case, compares to the gold, prints the report.

  uv run python benchmark/run.py             # report per case (default model)
  uv run python benchmark/run.py --compare   # MODELS × CASES matrix

To compare two configs (e.g. with/without the notes glossary), pass kwargs to `run_case`
(use_notes=False, do_dedupe=False…): we FREEZE the matching+scorer and vary ONLY the extraction.
"""
from __future__ import annotations

import json
import logging
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

import metrics

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import logs  # noqa: E402
from matrix import io as matrix_io  # noqa: E402  (Rule -> row conversion: the same one used by the delivered matrix)
from pipeline import parser as doc_parser  # noqa: E402  (pipeline/parser.py: document -> markdown)
from pipeline import structurer as st  # noqa: E402
from llm import LLMConfigError, bind  # noqa: E402

log = logging.getLogger("benchmark")  # status + results (kept separate from pipeline noise)

# Synthetic only (invented substances = no memorization leak). All PDF (realistic, Docling first).
# TH/HI keep the source language (translate=False).
# Each case isolates ONE dimension (not a difficulty scale): noise / values / native numerals / degraded extraction.
CASES = [
    {"name": "noise_en", "input": "inputs/noise_en.pdf", "gold": "gold/noise_en.json",
     "ctx": {"jurisdiction": "Testland", "document_ref": "SYN-001", "theme": "food_additives"}},
    {"name": "complex_en", "input": "inputs/complex_en.pdf", "gold": "gold/complex_en.json",
     "ctx": {"jurisdiction": "Testland", "document_ref": "SYN-002", "theme": "food_additives"}},
    {"name": "numerals_thai", "input": "inputs/numerals_thai.pdf", "gold": "gold/numerals_thai.json",
     "ctx": {"jurisdiction": "Testland", "document_ref": "SYN-002", "theme": "food_additives"}, "translate": False},
    {"name": "degraded_hindi", "input": "inputs/degraded_hindi.pdf", "gold": "gold/degraded_hindi.json",
     "ctx": {"jurisdiction": "Testland", "document_ref": "SYN-002", "theme": "food_additives"}, "translate": False},
]


def _rule_to_dict(r) -> dict:
    """Go through matrix_io.to_row: the benchmark scores EXACTLY the conversion used for the
    real matrix (no second version that drifts). The compared fields stay the same."""
    row = matrix_io.to_row(r, jurisdiction="", document="")
    return {"substance": row["subject"], "category": row["category"], "rule_type": row["rule_type"],
            "value_min": row["value_min"], "value_max": row["value_max"], "unit": row["unit"]}


_MD: dict[str, str] = {}  # cache: run Docling ONCE per case (then all models reuse it)


def _markdown(case: dict) -> str:
    name = case["name"]
    if name not in _MD:
        src = HERE / case["input"]
        _MD[name] = doc_parser.to_markdown(src) if src.suffix.lower() == ".pdf" else src.read_text(encoding="utf-8")
    return _MD[name]


def run_case(case: dict, **extract_kwargs) -> dict:
    md = _markdown(case)
    gold = json.loads((HERE / case["gold"]).read_text(encoding="utf-8"))
    extract_kwargs.setdefault("translate", case.get("translate", True))
    extract_kwargs.setdefault("do_dedupe", False)  # synthetic docs have no duplicates -> dedup is useless (a duplicate would surface as spurious)
    rules = st.extract_rules(md, case["ctx"], **extract_kwargs)
    extracted = [_rule_to_dict(r) for r in rules]
    return metrics.score(gold, extracted)


# Approach comparison (A/B)
def _model_llm(model_id: str):
    """An llm(messages, response_model) bound to a different model (to compare models).
    The sweep goes through openrouter: it hosts ALL the models in the MODELS list
    (the project's default provider only knows its own)."""
    return bind(model=model_id, provider="openrouter")


# We FREEZE the matching + scorer; we vary ONLY the extraction MODEL (see README).
MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    "openai/gpt-5",
    "z-ai/glm-5.2",
    "mistralai/mistral-large",
    "meta-llama/llama-3.3-70b-instruct",
]
# Removed (break on our nested schema via OpenRouter, see probleme.md P12):
# google/gemini-2.5-flash, google/gemini-2.5-pro, qwen/qwen-2.5-72b-instruct


def compare() -> None:
    log.info("Docling : %d documents (une seule fois)…", len(CASES))
    for case in CASES:                      # Docling ONCE per case, BEFORE the parallel sweep
        _markdown(case)
    tasks = [(m, c) for m in MODELS for c in CASES]

    def _one(mc):
        model_id, case = mc
        try:
            return model_id, case["name"], f"{run_case(case, llm=_model_llm(model_id))['f1']}"
        except LLMConfigError:  # key/provider: the whole sweep would fail the same way -> stop immediately
            raise
        except Exception:
            return model_id, case["name"], "ÉCHEC"

    grid: dict[str, dict] = {m: {} for m in MODELS}
    with Progress(TextColumn("[cyan]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total} runs"), TimeElapsedColumn()) as prog, \
            ThreadPoolExecutor(max_workers=8) as ex:
        bar = prog.add_task("Benchmark (modèles × cas)", total=len(tasks))
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):  # 28 runs IN PARALLEL
            model_id, cname, f1 = fut.result()
            grid[model_id][cname] = f1
            prog.advance(bar)

    header = f"{'modèle':30}" + "".join(f"{c['name']:>18}" for c in CASES)
    rows = "\n".join(f"{m:30}" + "".join(f"{grid[m].get(c['name'], '-'):>18}" for c in CASES) for m in MODELS)
    log.info("[bold]MODÈLES × CAS — F1[/] (bruit · valeurs · chiffres · extraction)\n%s\n%s", header, rows)


if __name__ == "__main__":
    compare_mode = "--compare" in sys.argv
    logs.setup(logging.WARNING if compare_mode else logging.INFO)  # WARNING during the sweep -> clean progress bar
    log.setLevel(logging.INFO)  # our results still show even when everything else is at WARNING
    if compare_mode:
        compare()
        raise SystemExit(0)
    totals = {"gold": 0, "correct": 0, "extracted": 0}
    for case in CASES:
        s = run_case(case)
        log.info("[bold]%s[/] : gold=%d extrait=%d correct=%d | manquants=%d parasites=%d "
                 "(err_val=%d err_type=%d) | P=%.3f R=%.3f [bold]F1=%.3f[/]",
                 case["name"], s["gold"], s["extracted"], s["correct"], s["missing"], s["spurious"],
                 s["value_errors"], s["type_errors"], s["precision"], s["recall"], s["f1"])
        totals["gold"] += s["gold"]; totals["correct"] += s["correct"]; totals["extracted"] += s["extracted"]

    p = totals["correct"] / totals["extracted"] if totals["extracted"] else 0.0
    r = totals["correct"] / totals["gold"] if totals["gold"] else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    log.info("[bold]GLOBAL[/] : précision=%.3f rappel=%.3f F1=%.3f", p, r, f1)
