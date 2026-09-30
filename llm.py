"""Thin wrapper around LiteLLM, shared by the agentic-collection modules.

The one place that knows how we talk to the LLM: model per tier, the
configured key passed explicitly (see _provider_kwargs), a request timeout,
retry of transient failures, JSON-object extraction, and per-call telemetry
(one `llm_calls` row per provider call that returned). extractor.py, the
planner, the assessor and the brief all go through it.
"""
import asyncio
import json
import re
import sys
import threading
import time
from typing import Optional

import litellm

import storage
from config import settings
from models import LlmCall


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


def _token_count(usage, name: str) -> int:
    """One token field of a provider's usage block, or 0 when it is missing
    or not a number (providers differ, and some omit fields)."""
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage(response) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from response.usage, or (0, 0)
    when the provider reported none."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return _token_count(usage, "prompt_tokens"), _token_count(usage, "completion_tokens")


def _complete(model: str, system: str, user: str, temperature: float,
              max_tokens: int) -> tuple[str, tuple[int, int]]:
    """One provider call: (content, (prompt_tokens, completion_tokens))."""
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
    return content, _usage(response)


def _complete_retrying(model: str, system: str, user: str, temperature: float,
                       max_tokens: int) -> tuple[str, tuple[int, int], int]:
    """Providers transiently refuse to generate (Cohere's
    NO_VALID_RESPONSE_GENERATED, rate limits, dropped connections, an empty
    completion). Retrying once usually succeeds and is far cheaper than
    losing a mission that has already paid for its crawling. Permanent
    errors (_NON_RETRYABLE) are raised at once.

    Returns (content, usage, duration_ms) of the attempt that returned; the
    duration covers that one provider call, not earlier failed attempts or
    the backoff between them."""
    last = None
    for attempt in range(RETRY_ATTEMPTS):
        started = time.monotonic()
        try:
            content, usage = _complete(model, system, user, temperature, max_tokens)
            return content, usage, int((time.monotonic() - started) * 1000)
        except Exception as e:  # noqa: BLE001 - re-raised below if final
            last = e
            if type(e).__name__ in _NON_RETRYABLE:
                raise
            if attempt + 1 < RETRY_ATTEMPTS:
                print(f"[LLM] {model} failed ({type(e).__name__}); retrying",
                      file=sys.stderr, flush=True)
                time.sleep(RETRY_DELAY_S * (attempt + 1))
    raise last


def _run_insert(call: LlmCall) -> None:
    """Run storage.insert_llm_call to completion from synchronous code.
    The callers of chat_ex are normally off the event loop
    (asyncio.to_thread), so asyncio.run in the calling thread is the path;
    a caller that is itself on a running loop gets a short-lived thread
    instead, since asyncio.run cannot nest."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(storage.insert_llm_call(call))
        return
    failure: list[BaseException] = []

    def worker() -> None:
        try:
            asyncio.run(storage.insert_llm_call(call))
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            failure.append(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if failure:
        raise failure[0]


def _record(purpose: str, tier: str, model: str, mission_id: Optional[str],
            usage: tuple[int, int], duration_ms: int) -> None:
    """Record one provider call that returned. Telemetry must never cost the
    caller its answer: any failure is logged to stderr and swallowed."""
    try:
        _run_insert(LlmCall(
            purpose=purpose, tier=tier, model=model, mission_id=mission_id,
            prompt_tokens=usage[0], completion_tokens=usage[1],
            duration_ms=duration_ms,
        ))
    except Exception as e:  # noqa: BLE001 - never fail the call itself
        print(f"[LLM] could not record {purpose} call telemetry: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)


def chat_ex(system: str, user: str, temperature: float = 0.0,
            max_tokens: int = 2000, tier: str = "reasoning",
            allow_fallback: bool = True, purpose: str = "chat",
            mission_id: Optional[str] = None) -> tuple[str, str]:
    """Return (assistant text, model that actually answered). If the 'fast'
    tier fails (e.g. local Ollama is down), fall back once to the reasoning
    model so brief/extraction still succeed instead of silently degrading.
    Raises if that also fails.

    Every provider call that returns is recorded as one LlmCall (`purpose`,
    the tier requested, the model that answered, its tokens and duration,
    `mission_id` or None); attempts that raised are not."""
    model = model_for(tier)
    try:
        text, usage, duration_ms = _complete_retrying(
            model, system, user, temperature, max_tokens)
    except Exception as e:  # noqa: BLE001
        reasoning = model_for("reasoning")
        if not (allow_fallback and tier == "fast" and model != reasoning):
            raise
        print(f"[LLM] fast tier ({model}) failed ({type(e).__name__}); "
              f"falling back to {reasoning}", file=sys.stderr, flush=True)
        model = reasoning
        text, usage, duration_ms = _complete_retrying(
            model, system, user, temperature, max_tokens)
    _record(purpose, tier, model, mission_id, usage, duration_ms)
    return text, model


def chat(system: str, user: str, temperature: float = 0.0,
         max_tokens: int = 2000, tier: str = "reasoning",
         allow_fallback: bool = True, purpose: str = "chat",
         mission_id: Optional[str] = None) -> str:
    """chat_ex, discarding the model name."""
    text, _model = chat_ex(system, user, temperature=temperature,
                           max_tokens=max_tokens, tier=tier,
                           allow_fallback=allow_fallback,
                           purpose=purpose, mission_id=mission_id)
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
              max_tokens: int = 2000, tier: str = "reasoning",
              purpose: str = "chat", mission_id: Optional[str] = None):
    """Return (parsed_dict_or_None, raw_text)."""
    raw = chat(system, user, temperature=temperature, max_tokens=max_tokens,
               tier=tier, purpose=purpose, mission_id=mission_id)
    parsed = _extract_json(raw)
    if parsed is None:
        print(f"[LLM] could not parse JSON from response: {raw[:200]!r}",
              file=sys.stderr, flush=True)
    return parsed, raw
