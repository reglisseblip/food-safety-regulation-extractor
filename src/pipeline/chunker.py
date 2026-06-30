"""
chunker — adaptive splitting for extraction (P10): a single chunk when the document fits in the
window, otherwise cuts at meaning boundaries (LLM) with a deterministic fallback (headings + size).
Unlike RAG chunking: the largest possible chunk, with NO overlap — an overlap would extract the same
rule twice (duplicates).
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from llm import LLMCallError, call
from prompts import CHUNK_BOUNDARIES

log = logging.getLogger(__name__)

MAX_ATOMS_FOR_LLM = 120  # beyond this: boundary prompt too large / output too fragile -> deterministic fallback (headings + size)

# Splitting a DENSE doc that still fits in the window: cut it on its sections to extract in parallel
# (one giant call -> several small calls + aggregation). Cut ONLY on headings (one rule = one chunk,
# never any overlap, see P10).
MIN_SECTIONS_FOR_SPLIT = 3   # minimum number of heading BOUNDARIES; below this, a doc that fits stays one chunk
SOFT_SPLIT_CHARS = 2500      # target chunk size; small neighbouring sections are merged

# A large TABLE (GMP list of 114 additives, GSFA annex of thousands of rows) is ONE atom (no blank
# line inside). Left untreated it goes out in a single call: the model does not enumerate everything
# (recall ~50% on a long list) or truncates its output -> SILENT loss of the richest chunk. Fix
# (research + audit agree): cut it into BATCHES of rows, COPYING the header into each batch (each
# sub-table stays valid and self-contained), and bound by NUMBER of rows (not characters — one row =
# one rule to output).
TABLE_MAX_ROWS = 50          # data rows per table batch (enough for one call, not too many for enumeration)


class _Splits(BaseModel):
    split_before: list[int] = Field(
        default_factory=list,
        description="Indices des atomes qui DÉBUTENT une nouvelle section/sujet (frontières de sens).",
    )


def _is_table_atom(a: str) -> bool:
    """An atom is a TABLE when most of its non-blank lines start with "|"."""
    lignes = [l for l in a.splitlines() if l.strip()]
    if len(lignes) < 2:
        return False
    pipes = sum(1 for l in lignes if l.lstrip().startswith("|"))
    return pipes >= max(2, int(0.6 * len(lignes)))


def _est_separateur(ligne: str) -> bool:
    """Markdown table separator line, e.g. "|---|:--:|" (only - : | and spaces, with at least one -)."""
    return "-" in ligne and bool(re.fullmatch(r"[\s:\-|]+", ligne.strip()))


def _split_table_atom(a: str, max_rows: int, max_chars: int) -> list[str]:
    """Cut a table into BATCHES of max_rows data rows, each batch prefixed by the header (title row +
    optional separator row) — each batch is a complete markdown table."""
    lignes = a.splitlines()
    entete, debut = [lignes[0]], 1
    if len(lignes) > 1 and _est_separateur(lignes[1]):
        entete.append(lignes[1])
        debut = 2
    donnees = [l for l in lignes[debut:] if l.strip()]
    if not donnees:
        return [a]
    hdr = "\n".join(entete)
    lots: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for row in donnees:
        if cur and (len(cur) >= max_rows or cur_len + len(row) > max_chars):
            lots.append(hdr + "\n" + "\n".join(cur))
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        lots.append(hdr + "\n" + "\n".join(cur))
    return lots


def _atomize(md: str) -> list[str]:
    """Atoms = paragraphs (separated by blank lines). The smallest unit we never cut — EXCEPT a large
    table, cut into batches of rows (header copied) to bound enumeration."""
    bruts = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
    atomes: list[str] = []
    for a in bruts:
        if _is_table_atom(a) and len(a) > SOFT_SPLIT_CHARS:
            atomes.extend(_split_table_atom(a, TABLE_MAX_ROWS, SOFT_SPLIT_CHARS))
        else:
            atomes.append(a)
    return atomes


def _pack(atoms: list[str], boundaries: set[int], max_chars: int) -> list[str]:
    """Group atoms: a new chunk at an LLM boundary, when max_chars is exceeded, or to ISOLATE a table
    batch (one batch = one chunk = one call with a bounded item count)."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for i, a in enumerate(atoms):
        tbl = _is_table_atom(a)
        prev_tbl = bool(cur) and _is_table_atom(cur[-1])
        if cur and (i in boundaries or cur_len + len(a) > max_chars or tbl or prev_tbl):
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(a)
        cur_len += len(a) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _pack_sections(atoms: list[str], boundaries: set[int], soft_max_chars: int) -> list[str]:
    """Group sections into chunks of about soft_max_chars, cutting ONLY on a heading (never in the
    middle of a section): a new chunk opens only at a section boundary AND when the current chunk is
    already large enough. Small neighbouring sections therefore merge, and a rule always stays whole
    in a single chunk."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for i, a in enumerate(atoms):
        if cur and i in boundaries and cur_len >= soft_max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(a)
        cur_len += len(a) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _llm_boundaries(atoms: list[str], llm: Callable) -> set[int]:
    """Ask the LLM where the meaning boundaries are (the indices of atoms that start a new section).
    The chunker builds its own messages: it calls the LLM in the same format as the rest of the pipeline."""
    numbered = "\n".join(f"[{i}] {a[:160]}" for i, a in enumerate(atoms))
    res = llm([{"role": "user",
                "content": CHUNK_BOUNDARIES + "\n<atomes>\n" + numbered + "\n</atomes>"}], _Splits)
    return {i for i in res.split_before if 0 < i < len(atoms)}


def _structural_boundaries(atoms: list[str]) -> set[int]:
    """DETERMINISTIC fallback: cut on markdown headings (#…) → keeps sections whole, without the LLM."""
    return {i for i, a in enumerate(atoms) if i > 0 and a.lstrip().startswith("#")}


def _safe_boundaries(atoms: list[str], llm: Callable) -> set[int]:
    """Meaning boundaries when possible, otherwise the DETERMINISTIC fallback (headings). The LLM is a
    quality bonus, never a correctness dependency: too many atoms OR a failed call → cut on headings + size."""
    if len(atoms) > MAX_ATOMS_FOR_LLM:
        log.warning("découpage : %d atomes > %d → coupes déterministes (titres + taille), pas d'appel LLM",
                    len(atoms), MAX_ATOMS_FOR_LLM)
        return _structural_boundaries(atoms)
    try:
        return _llm_boundaries(atoms, llm) | _structural_boundaries(atoms)
    except LLMCallError as e:  # only a failed call triggers plan B; a misconfiguration propagates
        log.warning("découpage : frontières LLM indisponibles (%s) → coupes déterministes", e)
        return _structural_boundaries(atoms)


def chunk(markdown: str, max_chars: int = 16000, llm: Optional[Callable] = None,
          soft_max_chars: int = SOFT_SPLIT_CHARS) -> list[str]:
    """Adaptive splitting:
      * doc that fits in the window AND is dense (many sections) -> cut ON THE SECTIONS (no LLM call)
        to extract in parallel; few sections -> one chunk;
      * large doc -> meaning cuts (LLM) with a deterministic fallback (headings + size)."""
    atoms = _atomize(markdown)
    if len(markdown) <= max_chars:
        structural = _structural_boundaries(atoms)
        if len(structural) >= MIN_SECTIONS_FOR_SPLIT:
            chunks = _pack_sections(atoms, structural, soft_max_chars)
            if len(chunks) > 1:  # otherwise (everything merges): it really does fit in one chunk
                log.info("découpage : %d morceau(x) sur sections (doc dense qui tient, %d frontières de titre, sans LLM)",
                         len(chunks), len(structural))
                return chunks
        # doc that fits but contains a LARGE table split into batches -> isolate the batches (recall + parallelism)
        if len(markdown) > soft_max_chars and sum(_is_table_atom(a) for a in atoms) >= 2:
            chunks = _pack(atoms, set(), max_chars)
            if len(chunks) > 1:
                log.info("découpage : %d morceau(x) sur lots de tableau (doc qui tient, sans LLM)", len(chunks))
                return chunks
        log.info("découpage : 1 morceau (le doc tient dans la fenêtre)")
        return [markdown]
    boundaries = _safe_boundaries(atoms, llm or call)
    chunks = _pack(atoms, boundaries, max_chars)
    log.info("découpage : %d morceau(x) (%d atomes)", len(chunks), len(atoms))
    return chunks