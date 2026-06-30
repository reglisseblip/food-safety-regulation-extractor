"""
llm — every LLM call goes through here, whatever the provider.

The whole pipeline calls the LLM the same way:
    llm(messages, response_model) -> validated Pydantic object
`bind(model, provider)` pins the model and provider ONCE for the whole run and returns a
ready-to-use function; `call(...)` does the same on a per-call basis.
(To test without a key or network, plug in a fake LLM instead: tests/fake_llm.py.)

Engine: **LangChain** — `init_chat_model(...).with_structured_output(response_model)`. Four
provider cards: openai, openrouter and ollama speak the same dialect (OpenAI client, only the
URL/key differ); `claude` goes through the OFFICIAL Anthropic SDK (the card names which SDK via
`model_provider`). The `llm(messages, response_model)` SIGNATURE never changes: the pipeline and
the tests don't see the engine swap.

Each provider has its card in PROVIDERS (URL, key name, output method, options). Adding a provider
= adding a card. Example: Ollama Cloud returns badly in tool-calling mode (see P16), so its card
forces `method="json_mode"` — with the setting IN the card, a future refactor can no longer lose it
by accident.

MAX_WORKERS limits the number of calls launched AT THE SAME TIME (beyond it, the provider returns
429s). The limit is enforced here by a semaphore: it holds even when several parallel loops stack
up (chunks x documents x benchmark). It is THE single ceiling.

Two families of errors (see P16):
  * LLMConfigError — bad configuration: STOP everything. Unknown provider, missing key, invalid
    key (401), negative balance (402), exhausted quota. Continuing would silently produce an empty
    or truncated matrix.
  * LLMCallError   — ONE call failed transiently (network, 429/5xx after retries, non-conforming
    output): the pipeline can skip that chunk and carry on.

Robustness: a 429/5xx/timeout is retried (server Retry-After, else exponential backoff WITH jitter)
before giving up with LLMCallError; the wait happens OUTSIDE the semaphore (we don't hold a slot
during backoff).
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import threading
import time
from dataclasses import dataclass, field

# to recognize the error families of the underlying SDKs (re-raised by LangChain)
import anthropic
import openai
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException

log = logging.getLogger(__name__)
ROOT = pathlib.Path(__file__).resolve().parents[1]


class LLMConfigError(RuntimeError):
    """Bad configuration (unknown provider, missing/invalid key, balance/quota) — stop the run."""


class LLMCallError(RuntimeError):
    """An LLM call failed transiently — skippable chunk by chunk."""


@dataclass(frozen=True)
class Provider:
    """A provider's card: everything that differs from one to another, in one place."""
    base_url: str | None                   # None = the SDK's official URL (nothing to override)
    key_name: str                          # key name in AgentReg/.env
    method: str = "function_calling"       # how LangChain retrieves the structured JSON
    model_provider: str = "openai"         # which SDK LangChain uses to talk to this provider
    extra: dict = field(default_factory=dict)  # options passed to the chat model (e.g. max_tokens)


# gpt-5-mini shares ONE output ceiling (128k) between its REASONING tokens (invisible) and the JSON.
# On a long enumeration, the reasoning can starve the JSON -> truncated output -> lost recall. We
# reserve a large margin (max_tokens; langchain-openai maps it to max_completion_tokens for
# reasoning models, see langchain#31024): the JSON has room, truncation becomes unlikely. It is a
# CEILING, not a target — no effect on small documents.
_OPENAI_BUDGET = {"max_tokens": 32000}

PROVIDERS = {
    "openai":     Provider("https://api.openai.com/v1",    "OPENAI_API_KEY", extra=_OPENAI_BUDGET),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # P16 — Ollama Cloud often returns the JSON in the text (tool_calls=None): tool-calling breaks.
    # JSON mode reads the JSON from the content -> reliable. And "thinking" models need a large
    # output budget (their reasoning consumes tokens before the answer).
    "ollama":     Provider("https://ollama.com/v1", "OLLAMA_API_KEY",
                           method="json_mode", extra={"max_tokens": 48000}),
    # OFFICIAL Anthropic API (api.anthropic.com), via its official SDK (langchain-anthropic) —
    # this is not an OpenAI-compatible endpoint: the card carries which SDK to use.
    # Models: claude-sonnet-4-6, claude-sonnet-5, claude-haiku-4-5... (never a date suffix).
    "claude":     Provider(None, "ANTHROPIC_API_KEY", model_provider="anthropic"),
}

PROVIDER = "openai"       # default provider
MODEL = "gpt-5-mini"      # default model (validated by the benchmark: F1=1.000 on complex_en)
MAX_WORKERS = 6                                # max concurrent LLM calls (beyond it: 429 errors)

_INFLIGHT = threading.BoundedSemaphore(MAX_WORKERS)   # the semaphore that enforces this limit
_MODELS: dict = {}                                    # one chat model per (provider, model), built once
_MODELS_LOCK = threading.Lock()

_TRANSIENT_STATUS = {429, 500, 502, 503, 529}         # retryable codes (rate limiting / server overload)


def load_env(name: str) -> str:
    """Read a variable from AgentReg/.env (else the environment).
    EXACT name match (not a mere prefix: OPENAI_API_KEY_BACKUP must not answer for
    OPENAI_API_KEY); surrounding quotes on the value are stripped."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip("'\"")
    return os.environ.get(name, "")


def _model(provider: str, model: str):
    """The LangChain chat model for (provider, model): checks the config (known provider, key
    present) and builds it only once. An invalid config raises LLMConfigError."""
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Fournisseur inconnu : {provider!r}. Connus : {list(PROVIDERS)}.")
    key = (provider, model)
    with _MODELS_LOCK:
        if key not in _MODELS:
            p = PROVIDERS[provider]
            api_key = load_env(p.key_name)
            if not api_key:
                raise LLMConfigError(f"{p.key_name} manquante (AgentReg/.env).")
            try:
                kwargs = dict(p.extra)
                if p.base_url:  # None = let the SDK target its official URL
                    kwargs["base_url"] = p.base_url
                _MODELS[key] = init_chat_model(
                    model=model, model_provider=p.model_provider,
                    api_key=api_key, **kwargs,
                )
            except Exception as e:  # invalid provider/params: this is config, not a failed call
                raise LLMConfigError(f"config LLM invalide ({model} @ {provider}) : {e}") from e
        return _MODELS[key]


def _is_truncation(exc: Exception) -> bool:
    """Truncated output: the model hit the token ceiling (finish_reason='length'). The OpenAI SDK
    raises LengthFinishReasonError. Do NOT retry as-is: the same input truncates again. The real fix
    is upstream (smaller chunks: table batching in the chunker + output margin in the card). Here we
    just recognize it so as not to mistake it for a transient fluke."""
    Length = getattr(openai, "LengthFinishReasonError", None)
    if Length is not None and isinstance(exc, Length):
        return True
    return "finish_reason" in str(exc).lower() and "length" in str(exc).lower()


def _is_fatal(exc: Exception) -> bool:
    """CONFIG error (stop everything): invalid key (401), negative balance (402), exhausted quota.
    Tolerating them would silently build an empty/truncated matrix. Each SDK (openai, anthropic)
    raises ITS own classes — the status_code test catches both."""
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return True
    if getattr(exc, "status_code", None) in (401, 402):
        return True
    return "insufficient_quota" in str(getattr(exc, "code", "") or "")


def _is_transient(exc: Exception) -> bool:
    """RETRYABLE error: rate limiting (429), overload/5xx, timeout/network, or non-conforming output."""
    if isinstance(exc, OutputParserException):
        return True
    if getattr(exc, "status_code", None) in _TRANSIENT_STATUS:
        return True
    return isinstance(exc, (openai.RateLimitError, openai.APITimeoutError,
                            openai.APIConnectionError, openai.InternalServerError,
                            anthropic.RateLimitError, anthropic.APITimeoutError,
                            anthropic.APIConnectionError, anthropic.InternalServerError))


def _retry_after(exc: Exception) -> float | None:
    """The delay imposed by the server (Retry-After header), if present — honored BEFORE our backoff."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        v = headers.get("retry-after")
        if v:
            try:
                return max(0.0, float(v))  # a server may send a negative value: never a sleep < 0
            except (TypeError, ValueError):
                return None  # unhandled HTTP-date format -> fall back to the backoff
    return None


def _with_schema_note(messages: list[dict], response_model, provider: str) -> list[dict]:
    """In JSON mode the API does NOT pass the schema to the model (unlike tool-calling): without
    instruction, the model invents its own shape and validation rejects everything (seen on the
    Claude gateway — same family as P16). So we add the expected schema to the prompt, as the old
    engine did. The other modes are unchanged."""
    if PROVIDERS[provider].method != "json_mode":
        return messages
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    note = ("Réponds UNIQUEMENT par un objet JSON valide et conforme à ce schéma "
            "(mêmes champs, AUCUN champ en plus, pas de texte autour) :\n" + schema)
    return list(messages) + [{"role": "system", "content": note}]


def call(messages: list[dict], response_model, model: str = MODEL,
         provider: str = PROVIDER, max_retries: int = 3):
    """A structured-output LLM call -> validated `response_model` object. Prefer `bind(...)`.
    A 429/5xx/timeout is retried (Retry-After, else jittered backoff) OUTSIDE the semaphore; a bad
    config raises LLMConfigError; an exhausted transient failure raises LLMCallError (tolerated
    upstream)."""
    structured = _model(provider, model).with_structured_output(
        response_model, method=PROVIDERS[provider].method
    )
    messages = _with_schema_note(messages, response_model, provider)
    delay, last = 1.0, None
    for attempt in range(max_retries + 1):
        with _INFLIGHT:  # wait our turn; the semaphore is THE global ceiling
            try:
                return structured.invoke(messages)
            except Exception as e:  # the only place that talks to the network -> the only place that sorts the failure
                if _is_fatal(e):
                    raise LLMConfigError(f"config LLM ({model} @ {provider}) : {type(e).__name__}: {e}") from e
                if _is_truncation(e):  # truncated: no point retrying as-is -> clean failure, clear message
                    raise LLMCallError(f"sortie TRONQUÉE ({model} @ {provider}) : morceau trop dense pour le "
                                       f"budget de sortie — à redécouper plus finement.") from e
                if not _is_transient(e) or attempt >= max_retries:
                    raise LLMCallError(f"échec d'appel ({model} @ {provider}) : {type(e).__name__}: {e}") from e
                last = e
        # semaphore RELEASED: wait before retrying (we don't hold a slot during the wait)
        wait = _retry_after(last)
        if wait is None:
            cap = min(delay, 30.0)
            wait = cap + random.uniform(0, cap)  # backoff + FULL jitter (avoids synchronized retries)
            delay *= 2
        log.warning("appel %s @ %s : tentative %d/%d après %s (attente %.1fs)",
                    model, provider, attempt + 1, max_retries, type(last).__name__, wait)
        time.sleep(max(0.0, wait))  # guard: never a negative sleep (P16: don't raise outside a typed error)
    # never reached in practice (the last attempt already raises in the loop) — safety net
    raise LLMCallError(f"échec d'appel ({model} @ {provider}) : épuisé après {max_retries} tentative(s)")


def bind(model: str = MODEL, provider: str = PROVIDER, max_retries: int = 3):
    """Pin (model, provider, retries) once and return the llm(messages, response_model) function the
    whole pipeline uses. Checks the config RIGHT AWAY: better to crash at launch than to get an
    empty matrix an hour later."""
    _model(provider, model)
    return lambda messages, response_model: call(
        messages, response_model, model=model, provider=provider, max_retries=max_retries
    )
