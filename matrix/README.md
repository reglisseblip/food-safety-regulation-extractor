# matrix — the consolidated matrix

Produced by `uv run python src/build_matrix.py`: the pipeline (parser → chunker → structurer) run on the **real
corpus**, with all rules aggregated into **a single table**, together with their **provenance**.

## Files
- **`matrix.csv`** — **one row per rule**, filterable (opens in Excel/Sheets, UTF-8-BOM encoding).
  The `source` column is **truncated** here (constant `SNIPPET_CSV_MAX`, for spreadsheet readability).
- **`matrix.json`** — the **complete records** (`model_dump`, non-null fields): this is where the
  **full verbatim excerpt** lives (citability) along with every field absent from the CSV (identifiers, version…).
- **`checkpoints.sqlite`** — the **watch / resume** state (not versioned, created on the first run): every document
  already processed is remembered here by the fingerprint of its content; **deleting** it forces a **full re-run**.

The column contract has **a single source of truth**: `src/matrix_io.py` (`COLUMNS`).
This table is pinned by a golden test (`tests/test_matrix_io.py`) — it can no longer drift silently.

## Columns
| column | meaning |
|---|---|
| `jurisdiction` | country / source (Codex, EU, India…) |
| `document` | corpus ref. of the source text (e.g. `Reg 828/2014`) |
| `document_ref` | **extracted** document reference (provenance — may cite a third-party standard) |
| `theme` | normalized theme (food_additives, labeling, claims…) — filter axis |
| `rule_type` | family: `MAX` · `MIN` · `RANGE` · `PROHIBITED` · `PERMITTED` |
| `subject` | canonical subject label: substance / `group:…` / claim / labeling element |
| `category` | food category (label); the optional category code is in `matrix.json` → `provenance.category_code` |
| `consumer_group` | consumer group (general, infants…) — filter axis |
| `value_min` · `value_max` · `unit` · `basis` | the value (language-agnostic) |
| `conditions` | restrictions / notes, **lossless**: `type#ref: text` (e.g. `only_if#468: …`) |
| `page` | page of the source document |
| `needs_review` · `confidence` | quality (LLM output): a doubtful case is **flagged, never dropped** |
| `source` | excerpt of the source text (truncated here; **full verbatim in matrix.json**) |

→ filtering = sorting the CSV by any column (by country, by theme, by substance, by type, by value…).

## Coverage (transparent — no silent truncation)
A **digital and tractable** subset, multi-family, multi-country:
**Codex** (CXS 1-1985, CXS 180-1991) · **EU** (Reg 828/2014, Reg 1169/2011) · **India** (additives GMP, Nutraceuticals).

**Deliberately excluded:**
- **Codex CXS_192 (GSFA, 568 p.)** and **EU 1333-2008 consolidated** — this is **no longer a technical blocker**:
  the notes glossary is now built **piece by piece** (`build_notes` in `src/structurer.py`),
  so the size of the document is no longer a problem. It is a **cost** choice: a run of several
  hundred pages means that many LLM calls to pay for.
- **The 28 Thai documents** — mostly **scanned**; OCR exists (`--ocr`), but the batch has not
  been run.

→ The code is the same at scale; extending coverage = **launching the run** on these documents (the `--ocr` option
for the scanned ones), **without changing the code**.

## Delivered run
Matrix produced by **gpt-5-mini @ openai** (the project's default provider). The same run can be
redone with a Claude model via `--provider claude` (official API) or `--provider openrouter`,
without changing the code.

- **1,775 rules** across **35 documents** (of 47 in the corpus), 4 countries:
  Thailand 1,051 · India 388 · EU 259 · Codex 77.
- By type: MAX 1,064 · RANGE 260 · PERMITTED 231 · PROHIBITED 135 · MIN 85 — the **5 rule
  types** defined by the schema (`src/schema.py`).
- Quality: **0 rules without a source excerpt** (all citable); 426 flagged as "to review" (never dropped).
- Including large documents: Codex GSFA (CXS 192, 568 p.) and the EU additives regulation
  (Reg 1333/2008) — it is the batching of tables that makes their complete extraction possible.

> **Coverage 35/47.** The 3 largest documents (Thai #381 609 p., Thai #444 808 p., EU flavourings
> regulation 1334/2008) are deliberately left out: their Docling reading is very slow. A few short Thai
> amendments return **0 rules** — texts that refer to another regulation without any provision of their
> own. Thanks to the **watch** feature (`checkpoints.sqlite`), **re-running the same command** processes
> only the missing documents and completes the matrix — the ones already done are reused at no cost.