"""Thin wrapper around LiteLLM, shared by the agentic-collection modules.

The one place that knows how we talk to the LLM: model per tier, the
configured key passed explicitly (see _provider_kwargs), a request timeout,
retry of transient failures, and JSON-object extraction. extractor.py, the
planner, the assessor and the brief all go through it.
"""
import json
import re
import sys
import time
from typing import Optional

import litellm

from config import settings


class EmptyCompletion(RuntimeError):
    """The provider answered but produced no text. Raised (not returned as
    "") so retry, the fast->reasoning fallback, and each caller's own
    degraded path (e.g. the coverage-only brief) all engage."""


def model_for(tier: str = "reasoning") -> str:
    """Resolve the model id for a tier. 'fast' uses llm_provider_fast when set
    (e.g. a local Ollama model), otherwise everything falls back to
    llm_provider."""
    if tier == "fast" and settings.llm_provider_fast:
        return settings.llm_provider_fast
    return settings.llm_provider


_COHERE_PREFIXES = ("cohere/", "cohere_chat/")


def _provider_kwargs(model: str) -> dict:
    """litellm kwargs that depend on the provider: a local Ollama needs an
    api_base and no key; hosted providers get the configured key. When no key
    is set we pass none at all, so LiteLLM falls back to the vendor's own env
    var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).

    LLM_API_KEY is the provider-agnostic key and goes to whichever hosted
    model is configured. The legacy COHERE_API_KEY is a Cohere credential and
    only ever goes to a Cohere model — sending it to another vendor would
    leak it there and fail anyway."""
    if model.startswith(("ollama/", "ollama_chat/")):
        return {"api_base": settings.ollama_api_base}
    key = (settings.llm_api_key or "").strip()
    if not key and model.startswith(_COHERE_PREFIXES):
        key = (settings.cohere_api_key or "").strip()
    return {"api_key": key} if key else {}


RETRY_ATTEMPTS = 2
RETRY_DELAY_S = 1.0

# Errors a retry of the same model cannot fix (bad key, malformed request,
# unknown model, no access). Matched by class name so this needs no import
# from litellm (whose exception classes carry these names).
_NON_RETRYABLE = ("AuthenticationError", "BadRequestError",
                  "NotFoundError", "PermissionDeniedError")


def _complete(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        # Without a timeout a hung provider holds a mission worker (and its
        # job slot) forever.
        timeout=settings.llm_timeout_s,
        **_provider_kwargs(model),
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise EmptyCompletion(f"{model} returned an empty completion")
    return content


def _complete_retrying(model: str, system: str, user: str,
                       temperature: float, max_tokens: int) -> str:
    """Providers transiently refuse to generate (Cohere's
    NO_VALID_RESPONSE_GENERATED, rate limits, dropped connections, an empty
    completion). Retrying once usually succeeds and is far cheaper than
    losing a mission that has already paid for its crawling. Permanent
    errors (_NON_RETRYABLE) are raised at once."""
    last = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return _complete(model, system, user, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 - re-raised below if final
            last = e
            if type(e).__name__ in _NON_RETRYABLE:
                raise
            if attempt + 1 < RETRY_ATTEMPTS:
                print(f"[LLM] {model} failed ({type(e).__name__}); retrying",
                      file=sys.stderr, flush=True)
                time.sleep(RETRY_DELAY_S * (attempt + 1))
    raise last


def chat_ex(system: str, user: str, temperature: float = 0.0,
            max_tokens: int = 2000, tier: str = "reasoning",
            allow_fallback: bool = True) -> tuple[str, str]:
    """Return (assistant text, model that actually answered). If the 'fast'
    tier fails (e.g. local Ollama is down), fall back once to the reasoning
    model so brief/extraction still succeed instead of silently degrading.
    Raises if that also fails."""
    model = model_for(tier)
    try:
        return _complete_retrying(model, system, user, temperature, max_tokens), model
    except Exception as e:  # noqa: BLE001
        reasoning = model_for("reasoning")
        if allow_fallback and tier == "fast" and model != reasoning:
            print(f"[LLM] fast tier ({model}) failed ({type(e).__name__}); "
                  f"falling back to {reasoning}", file=sys.stderr, flush=True)
            return _complete_retrying(reasoning, system, user, temperature, max_tokens), reasoning
        raise


def chat(system: str, user: str, temperature: float = 0.0,
         max_tokens: int = 2000, tier: str = "reasoning",
         allow_fallback: bool = True) -> str:
    """chat_ex, discarding the model name."""
    text, _model = chat_ex(system, user, temperature=temperature,
                           max_tokens=max_tokens, tier=tier,
                           allow_fallback=allow_fallback)
    return text


_NOT_JSON = object()


def _loads(candidate: str):
    """json.loads, or _NOT_JSON when the candidate is not valid JSON."""
    try:
        return json.loads(candidate)
    except (ValueError, RecursionError):
        return _NOT_JSON


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_candidates(text: str) -> list[str]:
    """The strings model output most plausibly meant as its JSON answer, in
    order: the whole (stripped) text, then each ``` fence body."""
    text = (text or "").strip()
    if not text:
        return []
    return [text] + [m.group(1).strip() for m in _FENCE_RE.finditer(text)]


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a JSON *object* from model output. Returns a dict
    or None — never a list or scalar, because every caller goes on to call
    .get() on the result.

    Tries, in order: the whole text; each ```json fence; then a scan that
    decodes from every `{` and takes the first complete object (so prose
    with stray braces before the object, or a list before it, still works —
    a greedy first-`{`-to-last-`}` grab would not). When the whole text or a
    fence is valid JSON but not an object (e.g. an array of objects), that is
    the answer and it is not one: None, rather than an element fished out of
    it."""
    candidates = json_candidates(text)
    if not candidates:
        return None
    for candidate in candidates:
        value = _loads(candidate)
        if value is not _NOT_JSON:
            return value if isinstance(value, dict) else None
    text = candidates[0]
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def chat_json(system: str, user: str, temperature: float = 0.0,
              max_tokens: int = 2000, tier: str = "reasoning"):
    """Return (parsed_dict_or_None, raw_text)."""
    raw = chat(system, user, temperature=temperature, max_tokens=max_tokens, tier=tier)
    parsed = _extract_json(raw)
    if parsed is None:
        print(f"[LLM] could not parse JSON from response: {raw[:200]!r}",
              file=sys.stderr, flush=True)
    return parsed, raw
