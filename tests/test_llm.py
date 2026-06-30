"""The llm module: provider profiles (P16), bind, typed errors, 429 robustness, output cap — offline.

We replace `_model` (the chat-model factory) or `init_chat_model` with a fake, to check what the
module promises without ever touching the network or needing a key.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel

import llm


class _Out(BaseModel):
    ok: bool = True


class _FakeStructured:
    """What with_structured_output returns: a runnable whose .invoke does whatever we script."""
    def __init__(self, invoke):
        self._invoke = invoke

    def invoke(self, messages):
        return self._invoke(messages)


class _FakeModel:
    """A fake chat model: with_structured_output records the method and returns the scripted runnable."""
    def __init__(self, invoke, seen=None):
        self._invoke = invoke
        self.seen = seen if seen is not None else {}

    def with_structured_output(self, schema, method=None, **kw):
        self.seen["schema"] = schema
        self.seen["method"] = method
        return _FakeStructured(self._invoke)


def _patch_model(monkeypatch, invoke, seen=None):
    monkeypatch.setattr(llm, "_model", lambda provider, model: _FakeModel(invoke, seen))


def test_fiche_ollama_garde_le_fix_p16():
    # Guard: Ollama's P16 setting (JSON mode against tool_calls=None + large max_tokens) lives in
    # the PROVIDERS profile. If someone drops it while refactoring, this test breaks.
    p = llm.PROVIDERS["ollama"]
    assert p.method == "json_mode"
    assert p.extra == {"max_tokens": 48000}


def test_les_quatre_fiches_fournisseurs():
    # Guard: the project's four providers, each with its key and its SDK.
    # "claude" = the OFFICIAL Anthropic API (anthropic SDK, no URL to override).
    assert set(llm.PROVIDERS) == {"openai", "openrouter", "ollama", "claude"}
    cles = {nom: fiche.key_name for nom, fiche in llm.PROVIDERS.items()}
    assert cles == {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                    "ollama": "OLLAMA_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    assert llm.PROVIDERS["claude"].model_provider == "anthropic"
    assert llm.PROVIDERS["claude"].base_url is None
    assert all(f.model_provider == "openai" for n, f in llm.PROVIDERS.items() if n != "claude")


def test_fiche_claude_construit_avec_le_sdk_officiel(monkeypatch):
    # bind(provider="claude") must go through the official Anthropic SDK: model_provider="anthropic"
    # and NO base_url (the SDK targets api.anthropic.com on its own).
    seen = {}

    def fake_init(**kw):
        seen.update(kw)
        return _FakeModel(lambda messages: _Out(), seen)

    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    monkeypatch.setattr(llm, "_MODELS", {})
    monkeypatch.setattr(llm, "load_env", lambda name: "sk-test")

    out = llm.bind(model="claude-sonnet-4-6", provider="claude")([{"role": "user", "content": "x"}], _Out)
    assert out.ok
    assert seen["model"] == "claude-sonnet-4-6"
    assert seen["model_provider"] == "anthropic"
    assert "base_url" not in seen


def test_fournisseur_inconnu_echoue_au_bind():
    with pytest.raises(llm.LLMConfigError, match="Fournisseur inconnu"):
        llm.bind(provider="nope")


def test_cle_absente_echoue_au_bind(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda name: "")
    monkeypatch.setattr(llm, "_MODELS", {})  # don't inherit an already-built model
    with pytest.raises(llm.LLMConfigError, match="manquante"):
        llm.bind(provider="openrouter")  # fails AT STARTUP, not an empty matrix at the end of the run


def test_bind_construit_le_modele_avec_la_fiche(monkeypatch):
    # bind must build the chat model with the right model, URL, options AND output method from the
    # profile (P16: Ollama = json_mode + max_tokens 48000).
    seen = {}

    def fake_init(**kw):
        seen.update(kw)
        return _FakeModel(lambda messages: _Out(), seen)

    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    monkeypatch.setattr(llm, "_MODELS", {})
    monkeypatch.setattr(llm, "load_env", lambda name: "sk-test")

    out = llm.bind(model="glm-5.2:cloud", provider="ollama")([{"role": "user", "content": "x"}], _Out)
    assert out.ok
    assert seen["model"] == "glm-5.2:cloud"
    assert seen["base_url"] == llm.PROVIDERS["ollama"].base_url
    assert seen["max_tokens"] == 48000     # profile options (P16)
    assert seen["method"] == "json_mode"   # profile's output method (P16)


def test_mode_json_injecte_le_schema_dans_le_prompt(monkeypatch):
    # In JSON mode the API does not pass the schema to the model (unlike tool-calling): llm.call
    # must add it to the prompt — without it, the model invents its own shape and validation
    # rejects everything (seen with the Claude gateway, same family as P16).
    seen = {}

    def capture(messages):
        seen["messages"] = messages
        return _Out()

    _patch_model(monkeypatch, capture)
    llm.call([{"role": "user", "content": "x"}], _Out, model="m", provider="ollama")
    assert seen["messages"][-1]["role"] == "system"
    assert '"ok"' in seen["messages"][-1]["content"]  # _Out's schema is indeed passed through


def test_tool_calling_n_injecte_pas_le_schema(monkeypatch):
    # In tool-calling the schema goes through the API: the prompt must NOT be modified.
    seen = {}

    def capture(messages):
        seen["messages"] = messages
        return _Out()

    _patch_model(monkeypatch, capture)
    llm.call([{"role": "user", "content": "x"}], _Out, model="m", provider="openai")
    assert len(seen["messages"]) == 1


def test_echec_transitoire_epuise_devient_llmcallerror(monkeypatch):
    # An unrecognized error (neither fatal nor transient) surfaces as LLMCallError: the rest of the
    # pipeline never has to know LangChain's or OpenAI's internal exceptions.
    def boom(messages):
        raise RuntimeError("boom transport")

    _patch_model(monkeypatch, boom)
    with pytest.raises(llm.LLMCallError, match="boom transport"):
        llm.bind()([{"role": "user", "content": "x"}], _Out)


def test_429_est_retente_puis_reussit(monkeypatch):
    # A transient 429 is retried (no real wait in tests) then succeeds -> tolerated.
    calls = [0]

    class Rate(Exception):
        status_code = 429

    def flaky(messages):
        calls[0] += 1
        if calls[0] == 1:
            raise Rate("slow down")
        return _Out()

    _patch_model(monkeypatch, flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # no real wait
    out = llm.bind()([{"role": "user", "content": "x"}], _Out)
    assert out.ok and calls[0] == 2


def test_retry_after_negatif_ne_plante_pas(monkeypatch):
    # A server may return a negative Retry-After: we must NEITHER crash (sleep < 0),
    # NOR let a raw ValueError escape (it would break P16: only LLMCallError is tolerated).
    class _Resp:
        headers = {"retry-after": "-5"}

    class Rate(Exception):
        status_code = 429
        response = _Resp()

    calls = [0]

    def flaky(messages):
        calls[0] += 1
        if calls[0] == 1:
            raise Rate("slow down")
        return _Out()

    _patch_model(monkeypatch, flaky)
    out = llm.bind()([{"role": "user", "content": "x"}], _Out)  # retries (sleep clamped to 0) then succeeds
    assert out.ok and calls[0] == 2


def test_solde_negatif_402_est_fatal(monkeypatch):
    # 402 (OpenRouter negative balance) = bad config: we stop EVERYTHING, never tolerated per-chunk.
    class Billing(Exception):
        status_code = 402

    def broke(messages):
        raise Billing("negative balance")

    _patch_model(monkeypatch, broke)
    with pytest.raises(llm.LLMConfigError):
        llm.bind()([{"role": "user", "content": "x"}], _Out)


def test_budget_de_concurrence_applique(monkeypatch):
    # 18 calls on a pool 3x wider than the budget: the semaphore must hold the cap.
    lock, cur, peak = threading.Lock(), [0], [0]

    def busy(messages):
        with lock:
            cur[0] += 1
            peak[0] = max(peak[0], cur[0])
        time.sleep(0.03)
        with lock:
            cur[0] -= 1
        return _Out()

    _patch_model(monkeypatch, busy)
    bound = llm.bind()
    with ThreadPoolExecutor(max_workers=llm.MAX_WORKERS * 3) as ex:
        list(ex.map(lambda _: bound([{"role": "user", "content": "x"}], _Out), range(18)))
    assert peak[0] <= llm.MAX_WORKERS
