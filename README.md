<div align="center">

# Extract food-safety rules from regulatory texts into a filterable, citable matrix

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.7-1C3C3C?logo=langchain&logoColor=white)
![Docling](https://img.shields.io/badge/Docling-2.108.0-555555)
![Pydantic](https://img.shields.io/badge/Pydantic-2.13.4-E92063?logo=pydantic&logoColor=white)
![Textual](https://img.shields.io/badge/Textual-8.2.8-555555)

![Tests](https://img.shields.io/badge/Tests-88%20passed-brightgreen)
![Matrix](https://img.shields.io/badge/Matrix-1%2C775%20rules-blue)

Reads raw food-safety regulations (PDF, DOCX, Markdown) from Codex, the EU, India and Thailand, has an LLM turn them into structured rules (substance, food category, limit, unit, conditions), and consolidates everything into one CSV/JSON matrix. Every rule keeps the document, page and verbatim excerpt it came from, so it can be checked against the source.

</div>

<p align="center">
  <img src="assets/cli_ui.png" alt="Textual UI, Extraction tab" width="820">
</p>

The project delivers five outputs:

- a reference **schema** for a regulatory rule (`src/schema.py`);
- a document -> rules **pipeline** (PDF, DOCX or Markdown in, LLM-structured rules out), with monitoring and resumption;
- a consolidated CSV/JSON **matrix** with provenance (`matrix/`);
- a **benchmark** measuring precision, recall, F1, omissions and hallucinations (`benchmark/`);
- a **CLI and a Textual UI** to run an extraction on a file, a folder or a whole country.

## Prerequisites

- [uv](https://docs.astral.sh/uv/). It resolves and installs Python 3.11+ itself, nothing else to set up.
- An API key for one LLM provider. The default is OpenAI with `gpt-5-mini`, so `OPENAI_API_KEY` is enough to run every example below.

## Installation

```bash
git clone -c core.longpaths=true https://github.com/reglisseblip/food-safety-regulation-extractor.git
cd food-safety-regulation-extractor
uv sync
cp .env.example .env
```

Then fill in the API key in `.env`. The `core.longpaths` option matters on Windows only: some Thai file names in `source/` go past the 260-character path limit.

## Usage

### A whole country

The central requirement of the brief, "the country becomes the argument", is one command:

```bash
uv run agentreg run --country India
```

It processes `source/reglementations/India/` recursively, sets `jurisdiction=India` and writes `matrix/india.csv` and `matrix/india.json`. No other argument is needed.

List the files that would be processed, without calling the LLM:

```bash
uv run agentreg run --country India --dry-run
```

### A single file

```bash
uv run agentreg run fixtures/sample_gsfa.md --jurisdiction Codex --theme food_additives --stem sample
```

Outputs go to `matrix/` by default:

- `<stem>.csv`: flat view, one row per rule, filterable in a spreadsheet;
- `<stem>.json`: complete records with provenance.

| Option | Purpose |
| --- | --- |
| `--country <name>` | Derives the folder, the jurisdiction and the output name from the country. Mutually exclusive with positional targets. |
| `--jurisdiction` | Country or jurisdiction (e.g. EU, Codex, India). Required without `--country`. |
| `--theme` | Theme passed to the prompt (default: `other`). |
| `--document-ref` | Document reference, for a single target only. |
| `--recursive` | Traverse folders recursively. |
| `--ocr` | Enable Docling OCR for scanned PDFs (slower). |
| `--fast` | Fast PDF reading with pymupdf4llm, about 10x faster than Docling. |
| `--out` | Output folder (default: `matrix/`). |
| `--stem` | Prefix of the CSV/JSON files (default: `extraction`, or the country with `--country`). |
| `--provider` | LLM provider: `openai` (default), `claude`, `openrouter`, `ollama`. |
| `--model` | Model at that provider (default: `gpt-5-mini`). |
| `--dry-run` | List the targets without calling the LLM. |

Under Git Bash, write paths with `/` (`fixtures/sample_gsfa.md`). The CLI also tolerates some Windows paths with `\` that the shell mangles.

### Terminal UI

```bash
uv run agentreg ui
uv run agentreg ui fixtures/sample_gsfa.md   # with a pre-filled target
```

Two tabs: **Extraction** (file or folder, OCR toggle, live log) and **Benchmark** (run the synthetic cases and read the scores).

## Configuration

Each provider reads its own key from `.env` (see `.env.example`). Defaults live in one place, `src/llm.py`.

| Provider | `--provider` | Key in `.env` | Example `--model` |
| --- | --- | --- | --- |
| OpenAI (default) | `openai` | `OPENAI_API_KEY` | `gpt-5-mini` (default) |
| Anthropic API | `claude` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4.6` |
| Ollama Cloud (open-source models) | `ollama` | `OLLAMA_API_KEY` | `glm-5.2:cloud` |

```bash
uv run agentreg run --country India --provider claude --model claude-sonnet-4-6
```

The benchmark's `--compare` sweep goes through OpenRouter, which hosts every model in its list.

## Monitoring and resumption

Each document is identified by a fingerprint of its **content**, stored in `matrix/checkpoints.sqlite`:

- re-running the same command **skips** unchanged documents (no reading, no LLM call) and only reprocesses what changed;
- an **interrupted** run (outage, Ctrl+C) resumes where it stopped;
- deleting `matrix/checkpoints.sqlite` forces a full reprocessing.

A per-page fingerprint and scheduled re-runs are not implemented yet.

## Reproducing the deliverables

The consolidated matrix (`matrix/matrix.csv` and `matrix/matrix.json`), built from every document of the corpus, smallest first:

```bash
uv run python -m matrix.build
```

Test mode on the documents whose reference contains a given text, written to `matrix.test.*` so the real matrix is left untouched:

```bash
uv run python -m matrix.build "GMP additives"
```

The benchmark, for the default model, then as a models x cases table:

```bash
uv run python benchmark/run.py
uv run python benchmark/run.py --compare
```

Regenerate the benchmark's synthetic documents and gold answers (deterministic, no LLM):

```bash
uv run python benchmark/generate.py
```

The matrix committed in `matrix/` was produced with `gpt-5-mini` on OpenAI: **1,775 rules** from 35 of the 47 corpus documents (Thailand 1,051, India 388, EU 259, Codex 77). Its 17 columns are described in [`matrix/README.md`](matrix/README.md); the benchmark method and the scores of each model are in [`benchmark/README.md`](benchmark/README.md).

## Corpus

`source/reglementations/<country>/` holds the 47 raw documents: Codex 3, EU 7, India 2, Thailand 35. The brief of the assignment is `source/Food safety reglementation_ énoncé.docx`.

## Architecture

```text
document (PDF, DOCX, Markdown)
  -> parser       Docling by default, pymupdf4llm with --fast, OCR with --ocr
  -> chunker      adaptive splitting into chunks
  -> structurer   LLM extraction into Rule objects, notes glossary, deduplication
  -> matrix       17-column CSV + full JSON records with provenance

orchestrated by LangGraph, one checkpoint per document in matrix/checkpoints.sqlite
```

The code lives under `src/` (the import root): four top-level modules and four role-based subpackages.

| Path | Role |
| --- | --- |
| `src/schema.py` | Reference `Rule` model (Pydantic). |
| `src/llm.py` | Every LLM call goes through here; provider and model defaults. |
| `src/prompts.py` | Every text sent to the LLM, grouped for review. |
| `src/logs.py` | Shared colored logging (Rich). |
| `src/pipeline/parser.py` | Document -> Markdown (Docling, or pymupdf4llm with `--fast`). |
| `src/pipeline/chunker.py` | Adaptive splitting into chunks. |
| `src/pipeline/structurer.py` | LLM-structured extraction, notes glossary built chunk by chunk, deduplication. |
| `src/orchestration/doc_workflow.py` | Document-level engine: reading, splitting, extraction. |
| `src/orchestration/corpus.py` | Corpus-level engine: chains documents, carries monitoring and resumption. |
| `src/matrix/io.py` | Contract for the 17 CSV/JSON columns. |
| `src/matrix/build.py` | Builds `matrix/matrix.csv` and `matrix.json` (`python -m matrix.build`). |
| `src/cli/app.py` | The `agentreg` command (`run` and `ui` subcommands). |
| `src/cli/ui.py` | Textual interface. |
| `benchmark/` | Synthetic datasets, metrics and model comparison. |
| `matrix/` | Produced matrices, monitoring checkpoint, column documentation. |
| `fixtures/` | Small example documents to try the CLI. |
| `source/` | Raw regulatory corpus, by country under `source/reglementations/`. |

## Tests

```bash
uv run pytest -q
```

88 tests, all offline: they use a fake LLM, so no API key or network access is needed.

## Known limitations

- **Coverage of the committed matrix**: 12 of the 47 documents are not in it yet, the EU flavourings regulation (Reg 1334/2008) and 11 Thai notifications (MOPH 174, 281, 357, 363, 372, 376, 381, 384, 444, 450, 457), among them the largest ones. `python -m matrix.build` handles all 47; what is left is the **cost** of the run, not the code. The notes glossary is built chunk by chunk, so document size is no longer a blocker.
- **Scanned documents**: one document of the corpus has no text layer (Thai notification #281); the build reads it with OCR, which has not been run yet.
- **Monitoring**: the document level is implemented (see "Monitoring and resumption"); the per-page fingerprint and the re-run scheduler are not.

## References

**Libraries**

- [Docling](https://github.com/docling-project/docling): the document parser behind `src/pipeline/parser.py` (PDF/DOCX -> Markdown, with table-structure recovery and OCR).
- [LangGraph](https://langchain-ai.github.io/langgraph/): pipeline orchestration and durable checkpoints (`src/orchestration/`).
- [Pydantic](https://docs.pydantic.dev/): the `Rule` schema and structured LLM output (`src/schema.py`).
- [Textual](https://textual.textualize.io/): the terminal UI (`src/cli/ui.py`).

**Research**

- Brandon Smith and Anton Troynikov, [*Evaluating Chunking Strategies for Retrieval*](https://www.trychroma.com/research/evaluating-chunking) (Chroma, 2024). It motivates the adaptive, semantic chunking in `src/pipeline/chunker.py`: chunk boundaries measurably affect downstream quality, and cluster- or LLM-based semantic chunkers are competitive with well-tuned heuristics.
- Further evaluation-design sources (entity-matching metrics, embeddings vs. numbers) are listed in [`benchmark/README.md`](benchmark/README.md).
