"""payloads — reusable rule data for tests (validated by the real Pydantic
models at call time).

Made-up substances, as in the benchmark: the LLM cannot already "know" them.
"""

CTX = {"jurisdiction": "Codex", "document_ref": "CXS 192-1995", "theme": "food_additives"}


def wire_rule(name: str, value_max: float = 500.0, category: str | None = None) -> dict:
    """A minimal valid Rule in wire format (see the schema.py __main__ examples)."""
    r = {
        "jurisdiction": "Codex", "theme": "food_additives", "rule_type": "MAX",
        "substance": {"name": name, "substance_class": "additive"},
        "norm": {"value_max": value_max, "unit": "mg/kg"},
        "provenance": {"document_ref": "CXS 192-1995", "source_snippet": f"{name} {value_max} mg/kg"},
    }
    if category:
        r["product_category"] = category
    return r