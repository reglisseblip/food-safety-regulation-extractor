"""Benchmark scoring, locked down: match weighting, cap, and unit conversion.
(If these scoring rules changed by accident, all past F1 scores would become incomparable.)"""
import metrics


def row(substance, category, rule_type="MAX", value_min=None, value_max=None, unit=None):
    return {"substance": substance, "category": category, "rule_type": rule_type,
            "value_min": value_min, "value_max": value_max, "unit": unit}


G1 = row("Blorbic acid", "Glimmer cakes", value_max=500, unit="mg/kg")
G2 = row("Zentanyl", "Frosted noodles", value_max=60, unit="mg/kg")


def test_listes_identiques_f1_parfait():
    s = metrics.score([G1, G2], [G1, G2])
    assert (s["f1"], s["correct"], s["missing"], s["spurious"]) == (1.0, 2, 0, 0)


def test_equivalence_d_unites_g_kg_vaut_1000x_mg_kg():
    ext = row("Blorbic acid", "Glimmer cakes", value_max=0.5, unit="g/kg")
    s = metrics.score([G1], [ext])
    assert s["f1"] == 1.0 and s["value_errors"] == 0


def test_paire_au_dessus_du_plafond_vrai_missing_et_spurious():
    s = metrics.score([G1], [row("Quexol", "Cured zalmon", rule_type="GMP")])
    assert (s["matched"], s["missing"], s["spurious"], s["f1"]) == (0, 1, 1, 0.0)


def test_valeur_fausse_appariee_mais_comptee_en_erreur():
    ext = row("Blorbic acid", "Glimmer cakes", value_max=400, unit="mg/kg")
    s = metrics.score([G1], [ext])
    assert s["matched"] == 1 and s["correct"] == 0 and s["value_errors"] == 1


def test_gmp_sans_valeur_des_deux_cotes_est_correct():
    g = row("Blorbic acid", "Glimmer cakes", rule_type="GMP")
    assert metrics.score([g], [g])["f1"] == 1.0
