"""
metrics — benchmark matching and scoring (deterministic, NO LLM).

Two stages: (1) match each extracted rule to a gold rule, (2) score field by field.
VALUE is compared numerically after unit normalization (never via embeddings).
See benchmark/README.md for the design and sources.
"""
from __future__ import annotations

import difflib
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

# Concentrations -> factor to mg/kg (single base). %, ppm, mg/kg are comparable concentrations.
_CONC = {"mg/kg": 1.0, "ppm": 1.0, "mg/l": 1.0, "g/kg": 1000.0, "%": 10000.0, "g/100g": 10000.0}

# Matching-cost weights: STRONG anchor on substance/category/type, MODERATE on value (anti-circularity).
W_SUBJ, W_CAT, W_TYPE, W_VAL = 0.40, 0.25, 0.20, 0.15
CEILING = 0.60  # beyond this, no match (true Missing / Spurious)


def _norm(s) -> str:
    return " ".join((s or "").lower().split())


def _sim(a, b) -> float:
    """String similarity (0..1) — used for fuzzy text matching, never to judge the value."""
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def to_mgkg(value, unit):
    """Concentration -> mg/kg. Returns (value, comparable). A non-concentration unit (IU, mg/serving...) is not comparable."""
    if value is None or unit is None:
        return None, False
    u = _norm(unit).replace(" ", "")
    if u in _CONC:
        return value * _CONC[u], True
    return value, False


def _close(a, b, ua, ub, tol: float = 0.0) -> bool:
    """Whether one bound is correct after unit normalization. None on both sides -> correct.
    tol=0 keeps the "exact legal threshold" meaning, but in floating point: 0.07 % becomes
    700.0000000000001 mg/kg — without math.isclose, a CORRECT conversion would be scored wrong."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    av, ac = to_mgkg(a, ua)
    bv, bc = to_mgkg(b, ub)
    if ac and bc:
        if tol == 0.0:
            return math.isclose(av, bv, rel_tol=1e-9, abs_tol=1e-9)
        return abs(av - bv) <= tol * max(av, 1.0)
    return a == b and _norm(ua) == _norm(ub)


def value_correct(g: dict, e: dict, tol: float = 0.0) -> bool:
    """Value is correct = MIN AND MAX bounds correct (handles MAX, RANGE, GMP/forbidden). tol=0 -> exact (legal thresholds)."""
    return (_close(g.get("value_min"), e.get("value_min"), g.get("unit"), e.get("unit"), tol)
            and _close(g.get("value_max"), e.get("value_max"), g.get("unit"), e.get("unit"), tol))


def _cost(g: dict, e: dict) -> float:
    subj = 1 - _sim(g.get("substance"), e.get("substance"))
    cat = 1 - _sim(g.get("category"), e.get("category"))
    typ = 0.0 if g.get("rule_type") == e.get("rule_type") else 1.0
    gv, gc = to_mgkg(g.get("value_max"), g.get("unit"))
    ev, ec = to_mgkg(e.get("value_max"), e.get("unit"))
    if gc and ec and gv is not None and ev is not None:
        val = 0.0 if max(gv, ev) == 0 else min(1.0, abs(gv - ev) / max(gv, ev))
    else:
        val = 0.5  # neutral cost when the value is not comparable
    return W_SUBJ * subj + W_CAT * cat + W_TYPE * typ + W_VAL * val


def match(gold: list[dict], extracted: list[dict], ceiling: float = CEILING):
    """Global 1-to-1 matching (Hungarian) WITH a ceiling. Returns (pairs, missing_idx, spurious_idx)."""
    if not gold or not extracted:
        return [], list(range(len(gold))), list(range(len(extracted)))
    cost = np.array([[_cost(g, e) for e in extracted] for g in gold], dtype=float)
    rows, cols = linear_sum_assignment(cost)
    pairs, matched_g, matched_e = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= ceiling:  # ceiling: otherwise it is NOT a true pair
            pairs.append((int(r), int(c)))
            matched_g.add(int(r))
            matched_e.add(int(c))
    missing = [i for i in range(len(gold)) if i not in matched_g]
    spurious = [j for j in range(len(extracted)) if j not in matched_e]
    return pairs, missing, spurious


def score(gold: list[dict], extracted: list[dict]) -> dict:
    """Precision / recall / F1 + Missing vs Spurious + value/type errors."""
    pairs, missing, spurious = match(gold, extracted)
    correct = value_err = type_err = 0
    for r, c in pairs:
        g, e = gold[r], extracted[c]
        ok_type = g.get("rule_type") == e.get("rule_type")
        ok_val = value_correct(g, e)
        if ok_type and ok_val:
            correct += 1
        else:
            value_err += 0 if ok_val else 1
            type_err += 0 if ok_type else 1
    n_gold, n_ext = len(gold), len(extracted)
    precision = correct / n_ext if n_ext else 0.0
    recall = correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "gold": n_gold, "extracted": n_ext, "matched": len(pairs), "correct": correct,
        "missing": len(missing), "spurious": len(spurious),
        "value_errors": value_err, "type_errors": type_err,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
    }
