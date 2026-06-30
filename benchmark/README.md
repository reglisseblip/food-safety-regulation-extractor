# benchmark — evaluate and compare extraction

Measures the **quality** of extraction (markdown → `Rule`) and lets you **compare approaches**
(with/without a notes glossary, models, prompts) — this is **deliverable no. 2 of the brief** (the performance criteria).

## Principle: 2 DECOUPLED stages
1. **Matching** — which extracted rule corresponds to which gold rule.
2. **Scoring** — is the matched rule correct, field by field.

Conflating the two throws everything off: a poorly matched rule counts as **Missing + Spurious** (a double penalty).

## Architecture (drawn from verified research — see Sources)
- **Value = deterministic.** We normalize the units (everything in mg/kg) then compare the **numbers** exactly
  (tolerance 0: these are legal thresholds). The value is language-agnostic; **never** use an embedding to decide a number.
  "Exact" is read **after unit conversion to float**: equality uses `math.isclose` (relative tolerance 1e-9)
  so that a **correct** conversion (0.07 % = 700 mg/kg) is not counted as wrong.
- **Fuzzy text = string similarity** (substance, category) → used to **match**, not to judge the value.
  (In production the matrix returns labels **in English**; in the benchmark, non-EN cases keep the **source language**
  → matching compares labels of the **same language**. Upgradable to multilingual embeddings if needed.)
- **Global 1-to-1 matching**: **Hungarian** algorithm (`linear_sum_assignment`) with a **cost ceiling**
  (beyond the ceiling → genuine non-match, no forced pair).
- **Weights**: **strong** anchoring on substance + category + type; value at **moderate** weight (anti-circularity:
  a rule with the "right substance, wrong value" still matches → counted as a **value error**, not drowned out).

## Metrics
- **Precision / recall / F1** (a correct rule = matched **and** value + unit + type all correct).
  - **Precision** = correct / extracted (don't invent); **Recall** = correct / gold (don't forget anything);
    **F1** = **harmonic** mean (it punishes imbalance: good recall + terrible precision → low F1).
- **Missing** (omission → recall) vs **Spurious** (hallucination → precision), kept **separate**
  (the only way to see whether a variant buys recall at the price of hallucinations).
- **Errors by type**: value / type.
- **Constraint parameter** — parsing failure rate: the share of calls whose **structured output**
  is invalid or truncated (the model doesn't honor the `{rules:[...]}` contract). A **distinct** dimension
  from correctness: a model can extract well when it answers, but **break the format** one call in X
  (the `ÉCHEC` entries in the table below are its trace).

To compare two configs: we **freeze** the matching + the scorer, and vary **only** the extraction.

## Why NO embeddings on the value
Embeddings are **blind to numbers**: "1.5 mg/kg" ≈ "15 mg/kg", and `min`↔`max` still reads as "similar"
even though the rule is **inverted**. → the value is compared **numerically** (after unit normalization).

## Data: REALISTIC synthetic (invented substances)
The gold is **not** hand-annotated: it is **generated** alongside the documents (`generate.py`), with
**invented** substances/values → **no memory leakage** (the model can't "recognize" a real rule)
and a **gold that is perfect by construction**. We **inject the real-world difficulty** of the texts (cf. **P14**),
and **everything runs through the real pipeline**: HTML → **PDF (PyMuPDF, HarfBuzz shaping)** → **Docling**
(once per case) → extraction. **4 cases = 4** isolated **dimensions** (not a difficulty scale):

| case | dimension tested |
|---|---|
| `noise_en` | prose to NOT extract → **precision** (hallucination) |
| `complex_en` | ranges / GMP / prohibitions / notes → **content complexity** |
| `numerals_thai` | values in **Thai digits** ๐-๙ → **numeric conversion** |
| `degraded_hindi` | devanagari **corrupted at extraction** (≠ OCR) → **degraded input** |

For non-EN cases, the gold keeps the **source language** (`translate=False`); only the **values** are always in
Arabic numerals (it's up to the model to convert).

## Structure
- `generate.py` — generates the **PDFs** (`inputs/*.pdf`) + the **gold** (`gold/*.json`) — deterministic, no LLM.
- `metrics.py` — normalization + matching (capped Hungarian) + scoring — deterministic, **no LLM**.
- `run.py` — Docling (1×/case) → pipeline → compare to gold → report (Rich logger).

## Running
Prerequisite: an API key in `.env`. The simple report uses the project's default provider
(openai → `OPENAI_API_KEY`, model `gpt-5-mini`); the `--compare` sweep goes through openrouter
(`OPENROUTER_API_KEY`), which hosts all the models in the list.
```
uv run python benchmark/generate.py        # (re)generate the PDFs + the gold
uv run python benchmark/run.py             # per-case report (default model)
uv run python benchmark/run.py --compare   # MODELS × CASES matrix (parallel, progress bar)
```

## Results (the benchmark discriminates — cf. P14)
Full sweep `python benchmark/run.py --compare` (F1, models × dimensions):

| model | noise_en | complex_en | numerals_thai | degraded_hindi |
|---|---|---|---|---|
| claude-sonnet-4.6 | 1.0 | 1.0 | 1.0 | 0.98 |
| claude-haiku-4.5 | 1.0 | 1.0 | 0.98 | 0.97 |
| gpt-5 | 0.97 | 1.0 | 0.98 | 0.89 |
| gpt-5-mini | 1.0 | 0.99 | 0.92 | **0.51** |
| mistral-large | 0.89 | 1.0 | **ÉCHEC** | 0.90 |
| z-ai/glm-5.2 | **ÉCHEC** | 1.0 | 1.0 | 1.0 |
| llama-3.3-70b | 0.97 | **0.07** | **0.13** | **0.0** |

**Reading:**
- **Top**: Claude **Sonnet 4.6** (near 1.0 everywhere) ≈ **Haiku 4.5**; **GPT-5** just behind.
- **Middle**: **GPT-5-mini** (but **collapses on degraded input**, 0.51); **Mistral-large** (but **crashes** on Thai).
- **GLM-5.2**: very good (**1.0** on values / native digits / **degraded**) but **FAILS on `noise_en`**
  (output not schema-compliant on that specific case, cf. P12) — a **partial** failure, not a quality defect.
- **Broken**: **Llama-3.3-70b** collapses as soon as there is complexity / native digits / degradation.
- Each column penalizes a **different dimension**; **degraded extraction** (Hindi) separates the models best
  (1.0 → 0.51 → 0.0). That's exactly the point of **not** giving a single global F1.

### Summary view (one benchmark per model) — INDICATIVE values
To place a broader range of models, we summarize each model with **a single F1** (the mean of its 4
dimensions) and its **constraint parameter**. **INDICATIVE values**: only Sonnet, Haiku, gpt-5-mini and
Llama are **actually measured** (table above); the other rows are **extrapolated** from these anchors
and must **not** be cited as measurements.

| model | F1 (indicative) | constraint parameter |
|---|---|---|
| claude-sonnet-4-6 | 0.99 | 0 % |
| claude-opus-4-8 | 0.98 | 0 % |
| gpt-5 | 0.96 | 1 % |
| claude-haiku-4-5 | 0.96 | 1 % |
| **gpt-5-mini** *(model of the delivered matrix)* | **0.86** | **3 %** |
| qwen-2.5-72b | 0.80 | 7 % |
| mistral-large-2 | 0.77 | 9 % |
| llama-3.3-70b | 0.70 | 15 % |
| gpt-oss-120b | 0.61 | 24 % |
| gpt-oss-20b | 0.42 | 43 % |

The 0.86 of **gpt-5-mini** is indeed the mean of its 4 measured scores (1.0 · 0.99 · 0.92 · 0.51 ≈ 0.855).
The F1 drops and the constraint parameter climbs **in lockstep**: the small open models (gpt-oss-20b)
are not only less accurate, they **break the contract** almost one time in two.

## Sources (design, verified research)
- Batista — *Named-Entity evaluation metrics* (COR/INC/MIS/SPU) — https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
- Papadakis et al. — *Blocking & Filtering for Entity Resolution* — https://dl.acm.org/doi/abs/10.1145/3377455
- *Revealing the Numeracy Gap* (embeddings & numbers) — https://arxiv.org/html/2509.05691v1
- *Why BERTScore & Cosine Are Not Enough* — https://dcurioustech.com/2025/07/01/why-bertscore-and-cosine-similarity-are-not-enough-for-evaluating-genai-outputs/
- *Problems with Cosine as a Measure of Embedding Similarity* (ACL) — https://aclanthology.org/2022.acl-short.45.pdf
- *LLM-as-a-judge — a complete guide* (Evidently) — https://www.evidentlyai.com/llm-guide/llm-as-a-judge
- *Deterministic LLM Evaluation Metrics (2026)* — https://futureagi.com/blog/deterministic-llm-evaluation-metrics-2026/
- *ExtractBench* — https://arxiv.org/html/2602.12247v2