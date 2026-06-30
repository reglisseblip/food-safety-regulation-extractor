"""
doc_workflow — the pipeline for ONE document, in LangGraph (durable execution + resume).

We WRAP (without rewriting them) the pure functions: first parser.to_markdown, then
structurer.extract_rules. Two `@task` under an `@entrypoint` with a checkpointer: an interrupted run
resumes without re-parsing or re-extracting what is already done, and an unchanged document can be
skipped (see corpus.py — the thread_id is the hash of the content).

The `llm` and `to_markdown` seams are callables (not serializable): we read them from
config.configurable AT RUNTIME (get_config), NEVER as a task argument — an argument would be
checkpointed and break serialization. All the P16 tolerance (a chunk that fails is ignored) stays
INSIDE extract_rules: LangGraph only orchestrates at the document level.
"""
from __future__ import annotations

import pathlib

from langgraph.config import get_config
from langgraph.func import entrypoint, task

from pipeline import structurer as st


def build_doc_workflow(checkpointer):
    """Compile the single-document workflow with the given checkpointer (InMemorySaver in tests,
    SqliteSaver for a real resumable run). The returned object exposes .invoke(job, config) and
    .get_state(config); `job` = {"path": <str>, "contexte": <dict>}, result = rules (JSON)."""

    @task
    def parse(path: str) -> str:
        """Document -> markdown. The str result is checkpointed, so a resume does not re-parse."""
        to_markdown = get_config()["configurable"]["to_markdown"]
        return to_markdown(pathlib.Path(path))

    @task
    def extract(md: str, contexte: dict) -> list[dict]:
        """markdown -> rules as serializable JSON (Rule.model_dump). P16 tolerance lives in extract_rules."""
        llm = get_config()["configurable"]["llm"]
        return [r.model_dump(mode="json") for r in st.extract_rules(md, contexte, llm=llm)]

    @entrypoint(checkpointer=checkpointer)
    def doc_workflow(job: dict) -> list[dict]:
        md = parse(job["path"]).result()
        return extract(md, job["contexte"]).result()

    return doc_workflow
