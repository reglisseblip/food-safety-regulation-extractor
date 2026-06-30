"""fake_llm — a FAKE LLM to test the whole pipeline with no key, no network, no cost.

It plugs in at the same place as the real one (the `llm=` parameter of the pipeline
functions) and follows the same call convention:
    llm(messages, response_model) -> validated object of the right type

Four kinds of responses flow through the pipeline: rules, notes, split boundaries and
deduplication. In tests we refer to them by a short name ("rules", "notes", "splits",
"dedupe"); the KINDS table maps them to the real classes — if a class is ever renamed,
there is only ONE line to fix here, zero tests.

Rules of the game:
  * Routing is by CONTENT: .on("rules", ..., when="MARKER") replies when the received text
    contains the marker (which the test placed in ITS own input markdown). Never by call
    order: the pipeline runs in threads, order is not guaranteed.
  * Responses (dicts) are validated by the REAL Pydantic models — you cannot make a test
    pass with data that does not match the schema.
  * Unscripted kind -> neutral empty response (a bare fake_llm() runs extract_rules end to
    end); wholly unknown kind -> UnscriptedCall, deliberately noisy.
  * PITFALL: the pipeline TOLERATES LLMCallError (P16) — it drops the chunk and keeps going.
    To simulate a failure, script raises=LLMCallError(...); any other exception propagates.
    Always ALSO check that the call actually happened (assert_called / errors): otherwise a
    mis-scripted test can pass green without having tested anything.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel

# Real class name -> short name used in tests. The only place that knows the classes.
KINDS = {"RuleBatch": "rules", "NoteGlossary": "notes", "_Splits": "splits",
         "_DedupDecision": "dedupe"}
# Neutral empty response per kind (each response kind has exactly ONE list field).
EMPTY = {"rules": {"rules": []}, "notes": {"notes": []},
         "splits": {"split_before": []}, "dedupe": {"remove": []}}


class UnscriptedCall(AssertionError):
    """The pipeline asked for a response kind the fake doesn't know — a new LLM call somewhere?
    (In that case: extend KINDS.)"""


@dataclass
class Call:
    """A call received by the fake — recorded EVEN if the call raised an error."""
    kind: str
    model: str
    user_text: str
    messages: list = field(default_factory=list, repr=False)
    result: Optional[BaseModel] = None
    error: Optional[BaseException] = None


@dataclass
class _Handler:
    match: Callable[[str], bool]
    respond: Any
    raises: Union[BaseException, type, None]


class FakeLLM:
    """Scriptable fake LLM: the response depends ONLY on the requested kind and the received text."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[_Handler]] = {}
        self._lock = threading.Lock()
        self._calls: list[Call] = []

    # ------------------------------ scripting ------------------------------
    def on(self, kind: str, respond: Any = None, *,
           when: Union[str, Callable[[str], bool], None] = None,
           raises: Union[BaseException, type, None] = None) -> "FakeLLM":
        """Script a behavior. Chainable; the first matching `when` wins.
        respond: dict (validated on call), list (shortcut: wrapped in the model's single
        list field), instance, or callable(Call); raises: exception to raise."""
        if kind not in EMPTY:
            raise KeyError(f"kind inconnu {kind!r} ; connus : {sorted(EMPTY)}")
        if (respond is None) == (raises is None):
            raise ValueError("exactement un de respond= / raises=")
        if when is None:
            match: Callable[[str], bool] = lambda _t: True
        elif isinstance(when, str):
            match = lambda t, s=when: s in t
        else:
            match = when
        self._handlers.setdefault(kind, []).append(_Handler(match, respond, raises))
        return self

    # --------------------- the call, like the real LLM ---------------------
    def __call__(self, messages: list[dict], response_model):
        model = response_model.__name__
        kind = KINDS.get(model)
        user_text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        rec = Call(kind=kind or "?", model=model, user_text=user_text, messages=messages)
        with self._lock:
            self._calls.append(rec)
        try:
            if kind is None:
                raise UnscriptedCall(f"type de réponse non routé : {model!r} — compléter KINDS ; "
                                     f"texte : {user_text[:120]!r}")
            rec.result = self._resolve(rec, response_model)
            return rec.result
        except BaseException as e:
            rec.error = e
            raise

    def _resolve(self, rec: Call, response_model):
        for h in self._handlers.get(rec.kind, ()):
            if h.match(rec.user_text):
                if h.raises is not None:
                    raise h.raises("échec scripté FakeLLM") if isinstance(h.raises, type) else h.raises
                payload = h.respond(rec) if callable(h.respond) else h.respond
                return _materialise(payload, response_model)
        return response_model.model_validate(EMPTY[rec.kind])

    # ------------------- ways to check what happened -------------------
    @property
    def calls(self) -> list[Call]:
        with self._lock:
            return list(self._calls)

    @property
    def errors(self) -> list[Call]:
        """Calls that raised an error — useful because the pipeline swallows tolerated failures (P16)."""
        return [c for c in self.calls if c.error is not None]

    def calls_for(self, kind: str) -> list[Call]:
        return [c for c in self.calls if c.kind == kind]

    def count(self, kind: str) -> int:
        return len(self.calls_for(kind))

    def assert_called(self, kind: str, times: Optional[int] = None,
                      containing: Optional[str] = None) -> None:
        """Check that the expected call ACTUALLY happened — without this, a test can pass
        green without having tested anything, since the pipeline tolerates failures (P16)."""
        got = self.calls_for(kind)
        if containing is not None:
            got = [c for c in got if containing in c.user_text]
        detail = f"{kind!r}" + (f" contenant {containing!r}" if containing else "")
        if times is None:
            assert got, f"aucun appel {detail}"
        else:
            assert len(got) == times, f"{detail} : attendu {times} appel(s), obtenu {len(got)}"


def _materialise(payload: Any, response_model):
    if isinstance(payload, BaseModel):
        return payload
    if isinstance(payload, list):  # shortcut: a list is wrapped in the model's single field
        payload = {next(iter(response_model.model_fields)): payload}
    return response_model.model_validate(payload)


def fake_llm(*, rules: Any = None, notes: Any = None, splits: Any = None,
             dedupe: Any = None) -> FakeLLM:
    """The simple one-liner: fake_llm(rules=[...]) runs extract_rules end to end.
    WARNING: a "flat" default replies the SAME thing to EVERY chunk — if the test forces
    multiple chunks (max_chars), route by content with .on(when=...) to avoid duplicates."""
    f = FakeLLM()
    for kind, payload in (("rules", rules), ("notes", notes), ("splits", splits),
                          ("dedupe", dedupe)):
        if payload is not None:
            f.on(kind, payload)
    return f
