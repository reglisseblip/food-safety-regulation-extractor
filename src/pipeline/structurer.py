"""
structurer — component 2b: markdown -> Rule[] (structured extraction via LLM).

Pipeline (plain Python): split -> prompt -> LLM (structured output) -> validate -> accumulate.
Method: LangChain structured output (with_structured_output + Pydantic validation); retries
and the model/provider choice are centralized in src/llm.py (`llm.bind`, once per run).
The `llm` parameter can be swapped for a fake (same call signature everywhere, chunker included)
-> everything is testable without a key (tests/fake_llm.py).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from pydantic import BaseModel, Field, field_validator

from pipeline.chunker import chunk  # adaptive splitting (1 chunk if it fits, otherwise LLM meaning boundaries + fallback to headings/size)
from llm import MAX_WORKERS, LLMCallError, call  # centralized LLM access + the two error families
from prompts import DEDUPE_PROMPT, NOTES_PROMPT, SYSTEM_PROMPT  # grouped prompts
from schema import Rule  # schema.py lives in the same src/ folder

log = logging.getLogger(__name__)


_SKIP = object()  # internal sentinel: "this chunk was skipped"


def _parallel_map(fn, items: list, on_error="raise", label="tolérance"):
    """Apply fn to each item IN PARALLEL; return the results in order.
    This is where (and only here) we decide what to do with a failure (P16) — on_error:
      'raise'      -> any exception propagates (default);
      'skip'       -> an item whose fn raises LLMCallError is simply dropped from the result;
      f(item, exc) -> fallback: that item's result is replaced by this one.
    Only LLMCallError is forgiven — a bad config (LLMConfigError) or a real bug ALWAYS
    propagates: swallowing them would silently produce a wrong matrix."""
    def guarded(x):
        if on_error == "raise":
            return fn(x)
        try:
            return fn(x)
        except LLMCallError as e:
            log.warning("%s : 1 unité %s (%s)", label,
                        "dégradée (repli)" if callable(on_error) else "ignorée", e)
            return on_error(x, e) if callable(on_error) else _SKIP

    if len(items) <= 1:
        results = [guarded(x) for x in items]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items))) as ex:
            results = list(ex.map(guarded, items))
    return [r for r in results if r is not _SKIP]


# The LLM must return a BATCH of rules -> we wrap the list.
class RuleBatch(BaseModel):
    rules: list[Rule] = Field(default_factory=list)

    # Defense in depth: ONE malformed rule must not lose the WHOLE batch (and thus the whole
    # chunk). We validate rule by rule; those that fail are dropped and counted (never silently).
    # With the schema's `null` tolerance this is now rare — but if the model emits a truly broken
    # rule, we keep the good neighbors instead of throwing everything away.
    @field_validator("rules", mode="before")
    @classmethod
    def _garder_les_regles_valides(cls, v):
        if not isinstance(v, list):
            return v
        bonnes, rejetees = [], 0
        for item in v:
            try:
                bonnes.append(item if isinstance(item, Rule) else Rule.model_validate(item))
            except Exception:
                rejetees += 1
        if rejetees:
            log.warning("extraction : %d règle(s) malformée(s) écartée(s) sur %d (les bonnes gardées)",
                        rejetees, len(v))
        return bonnes


# ------------------------------------------------------------------ 1. splitting -> see src/chunker.py


# ------------------------------------------------------------------ 2. prompt -> see src/prompts.py
def build_messages(chunk_md: str, contexte: dict, notes: Optional[dict] = None, translate: bool = True) -> list[dict]:
    """Messages: system (role + instructions + examples) + one user message where EACH block is
    wrapped in an XML tag (context, notes glossary, extract) — the model sees unambiguously where
    each one starts and ends (OpenAI recommendation, see prompts.py).
    translate=False -> keep the labels IN THE SOURCE LANGUAGE (to evaluate extraction without translation)."""
    blocs = [
        f"<contexte>pays={contexte.get('jurisdiction')}, "
        f"document={contexte.get('document_ref')}, thème={contexte.get('theme')}.</contexte>"
    ]
    if not translate:
        blocs.append("<langue>EXCEPTION à la consigne 6 : garde TOUS les libellés (substance, catégorie, "
                     "périmètre, conditions) EXACTEMENT dans la langue d'origine, copiés tels quels ; "
                     "NE traduis PAS.</langue>")
    if notes:
        glossaire = "\n".join(f"- Note {k} : {v}" for k, v in notes.items())
        blocs.append("<glossaire_notes>\n" + glossaire + "\n</glossaire_notes>")
    blocs.append("<extrait>\n" + chunk_md + "\n</extrait>")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocs)},
    ]


# ------------------------------------------------------------------ 2b. notes glossary (P6)
class Note(BaseModel):
    id: str = Field(..., description="Identifiant EXACT de la note (ex. '468', 'XS89', 'XT100', 'Note 12').")
    text: str = Field(..., description="Texte de la note.")


class NoteGlossary(BaseModel):
    notes: list[Note] = Field(default_factory=list)


def _notes_from_chunk(chunk_md: str, llm: Callable) -> list[Note]:
    """Find the notes in a SINGLE chunk (generic, no regex). Failures are handled by _parallel_map."""
    return llm(
        [{"role": "system", "content": NOTES_PROMPT},
         {"role": "user", "content": "<extrait>\n" + chunk_md + "\n</extrait>"}],
        NoteGlossary,
    ).notes


def build_notes(chunks: list[str], llm: Callable) -> dict:
    """Glossary {id -> text} via MAP-REDUCE (see probleme.md P6/P11): notes PER chunk (in parallel) then UNION
    by id — never the whole document in a single call. Collision → keep the LONGEST text (note cut by a boundary)."""
    per_chunk = _parallel_map(lambda ch: _notes_from_chunk(ch, llm), chunks,
                              on_error="skip", label="notes")
    merged: dict[str, str] = {}
    for notes in per_chunk:
        for n in notes:
            k, t = n.id.strip(), n.text.strip()
            if k and (k not in merged or len(t) > len(merged[k])):
                merged[k] = t
    log.info("notes : %d note(s) au glossaire (map-reduce sur %d morceau(x))", len(merged), len(chunks))
    return merged


# ------------------------------------------------------------------ orchestration
def _summary(r: Rule) -> str:
    """Compact summary of a rule (so the dedup agent can compare without re-reading the whole JSON)."""
    subj = r.subject_label() or "?"
    cat = r.product_category or "—"
    n = r.norm
    val = f"{n.value_min if n.value_min is not None else ''}–{n.value_max if n.value_max is not None else ''} {n.unit or ''}".strip(" –")
    conds = "; ".join(c.text[:40] for c in r.conditions)
    return f"{r.rule_type.value} | {subj} | {cat} | {val}" + (f" | {conds}" if conds else "")


class _DedupDecision(BaseModel):
    remove: list[int] = Field(
        default_factory=list,
        description="Indices des règles à RETIRER car doublons d'une autre règle (gardée). Seulement de vrais doublons.",
    )


def _block_key(r: Rule) -> str:
    """LOOSE key to group dedup candidates. It only BRINGS candidates TOGETHER: the LLM decides afterwards."""
    base = r.subject_label() or r.rule_type.value
    words = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in base).split()
    return " ".join(words[:2]) if words else "?"


def _dedupe_block(rules: list[Rule], llm: Callable) -> list[Rule]:
    """Dedupe a SMALL group via the LLM (returns the rules to keep; mutates nothing)."""
    if len(rules) < 2:
        return rules
    listing = "\n".join(f"[{i}] {_summary(r)}" for i, r in enumerate(rules))
    decision = llm(
        [{"role": "system", "content": DEDUPE_PROMPT},
         {"role": "user", "content": "<regles>\n" + listing + "\n</regles>"}],
        _DedupDecision,
    )
    remove = {i for i in decision.remove if 0 <= i < len(rules)}
    return [r for i, r in enumerate(rules) if i not in remove]


def dedupe(rules: list[Rule], llm: Optional[Callable] = None) -> list[Rule]:
    """Dedup AT SCALE via *blocking*: group by subject (loose key), then the LLM agent dedupes
    WITHIN each group (in parallel, small calls). The key brings candidates together; the LLM decides (generic)."""
    if len(rules) < 2:
        return rules
    llm = llm or call
    blocks: dict[str, list[Rule]] = {}
    for r in rules:
        blocks.setdefault(_block_key(r), []).append(r)
    deduped = _parallel_map(lambda b: _dedupe_block(b, llm), list(blocks.values()),
                            # fallback if the call fails: keep the block as is — a few duplicates
                            # are better than losing rules already extracted
                            on_error=lambda block, exc: block,
                            label="dédoublonnage")
    out = [r for b in deduped for r in b]
    log.info("dédoublonnage : %d → %d règle(s) (%d groupe(s))", len(rules), len(out), len(blocks))
    return out


def _rules_from_chunk(chunk_md: str, contexte: dict, notes: dict, translate: bool, llm: Callable) -> list[Rule]:
    """Extract the rules from a SINGLE chunk. Failures are handled by _parallel_map (P16)."""
    return llm(build_messages(chunk_md, contexte, notes, translate), RuleBatch).rules


def extract_rules(markdown: str, contexte: dict, llm: Optional[Callable] = None,
                  do_dedupe: bool = True, use_notes: bool = True,
                  translate: bool = True, max_chars: int = 16000) -> list[Rule]:
    """markdown -> Rule[]: (1) split, (2) notes glossary MAP-REDUCE over those chunks (P6/P11),
    (3) PARALLEL extraction per chunk (TOLERANT), then dedup (blocking).
    Precision (not inventing a false rule) is carried by the PROMPT — instruction 2
    "provision ≠ explanation" — not by a separate filtering pass.
    max_chars is passed to the chunker — tests force multi-chunk on small documents.
    Failure handling: a failed call (LLMCallError) costs at worst its chunk (or leaves a
    block un-deduped); a bad config (LLMConfigError) stops everything."""
    llm = llm or call
    chunks = chunk(markdown, max_chars=max_chars, llm=llm)
    notes = build_notes(chunks, llm) if use_notes else {}
    per_chunk = _parallel_map(lambda ch: _rules_from_chunk(ch, contexte, notes, translate, llm), chunks,
                              on_error="skip", label="extraction")
    if chunks and not per_chunk:
        # ALL chunks failed (network outage, key revoked mid-run…): this document is not
        # "empty", it is FAILED. Without this, the watch would remember "0 rules" forever
        # and the document would never be reprocessed.
        raise LLMCallError("extraction : tous les morceaux ont échoué — document en échec, pas vide")
    rules = [r for batch in per_chunk for r in batch]
    log.info("extraction : %d règle(s) brute(s) sur %d morceau(x)", len(rules), len(chunks))
    return dedupe(rules, llm=llm) if do_dedupe else rules
