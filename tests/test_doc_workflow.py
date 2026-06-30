"""doc_workflow: the LangGraph workflow for ONE document, offline (FakeLLM + InMemorySaver).

Checks that the LangGraph wrapper actually calls the pure functions, reads the seams from
config.configurable, and returns serializable (JSON) rules - with no network and no key.
"""
from langgraph.checkpoint.memory import InMemorySaver

from orchestration.doc_workflow import build_doc_workflow
from fake_llm import fake_llm
from payloads import wire_rule


def _md(path):
    return path.read_text(encoding="utf-8")


def test_workflow_produit_des_regles_serialisees(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("Alpha 500 mg/kg", encoding="utf-8")
    wf = build_doc_workflow(InMemorySaver())
    fake = fake_llm(rules=[wire_rule("alpha")])
    config = {"configurable": {"thread_id": "t1", "llm": fake, "to_markdown": _md}}

    rules = wf.invoke(
        {"path": str(doc), "contexte": {"jurisdiction": "Codex", "document_ref": "DOC-A", "theme": "food_additives"}},
        config=config,
    )

    assert isinstance(rules, list) and len(rules) == 1
    assert rules[0]["substance"]["name"] == "alpha"   # JSON-serializable (Rule.model_dump)
    fake.assert_called("rules")
    assert wf.get_state(config).next == ()            # workflow finished (checkpoint complete)
