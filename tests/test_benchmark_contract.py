"""benchmark: the CONTRACT between the gold and the scorer can no longer drift silently.
Offline: we compare keys and files, no LLM call.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"


def test_gold_et_rule_to_dict_parlent_les_memes_cles():
    # If generate.py (the gold) and run._rule_to_dict (the extract) diverge on keys,
    # the matching would compare absent fields — and nobody would notice.
    import run as bench

    from payloads import wire_rule
    from schema import Rule

    extraite = bench._rule_to_dict(Rule.model_validate(wire_rule("alpha")))
    for case in bench.CASES:
        gold = json.loads((BENCH / case["gold"]).read_text(encoding="utf-8"))
        assert gold, f"gold vide : {case['gold']}"
        assert set(gold[0]) == set(extraite), f"clés gold ≠ clés extraites ({case['name']})"


def test_chaque_cas_a_son_pdf_et_son_gold():
    import run as bench

    for case in bench.CASES:
        assert (BENCH / case["input"]).exists(), f"PDF manquant : {case['input']}"
        assert (BENCH / case["gold"]).exists(), f"gold manquant : {case['gold']}"
