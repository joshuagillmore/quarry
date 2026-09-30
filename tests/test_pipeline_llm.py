"""llm.py: every call carries a timeout, an empty completion is a failure
(so retry and fallback engage), permanent errors are not retried, and the
legacy Cohere key is only ever sent to Cohere."""
import litellm
import pytest

import llm


class _Recorder:
    def __init__(self, content_for):
        self.calls = []
        self._content_for = content_for

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return litellm._Resp(self._content_for(kwargs["model"]))


def _models(monkeypatch, reasoning="cohere/command-a-03-2025", fast=""):
    monkeypatch.setattr(llm.settings, "llm_provider", reasoning)
    monkeypatch.setattr(llm.settings, "llm_provider_fast", fast)


def test_timeout_reaches_litellm(monkeypatch):
    _models(monkeypatch)
    monkeypatch.setattr(llm.settings, "llm_timeout_s", 37)
    rec = _Recorder(lambda model: "hello")
    monkeypatch.setattr(llm.litellm, "completion", rec)
    assert llm.chat("s", "u") == "hello"
    assert rec.calls[0]["timeout"] == 37


@pytest.mark.parametrize("empty", ["", "   \n", None])
def test_empty_completion_is_retried_then_raises(monkeypatch, empty):
    _models(monkeypatch)
    rec = _Recorder(lambda model: empty)
    monkeypatch.setattr(llm.litellm, "completion", rec)
    with pytest.raises(llm.EmptyCompletion):
        llm.chat("s", "u")
    assert len(rec.calls) == llm.RETRY_ATTEMPTS
    assert issubclass(llm.EmptyCompletion, RuntimeError)


def test_empty_fast_completion_falls_back_to_reasoning(monkeypatch):
    _models(monkeypatch, fast="ollama_chat/qwen2.5:14b")
    rec = _Recorder(lambda model: "" if model.startswith("ollama") else "real answer")
    monkeypatch.setattr(llm.litellm, "completion", rec)
    text, model = llm.chat_ex("s", "u", tier="fast")
    assert (text, model) == ("real answer", "cohere/command-a-03-2025")


@pytest.mark.parametrize("name", ["AuthenticationError", "BadRequestError",
                                  "NotFoundError", "PermissionDeniedError"])
def test_permanent_errors_are_not_retried(monkeypatch, name):
    _models(monkeypatch)
    permanent = type(name, (Exception,), {})
    calls = []

    def fails(model, system, user, temperature, max_tokens):
        calls.append(model)
        raise permanent("no")

    monkeypatch.setattr(llm, "_complete", fails)
    with pytest.raises(permanent):
        llm.chat("s", "u")
    assert len(calls) == 1


def test_permanent_fast_error_still_falls_back(monkeypatch):
    """Not retrying the same model is different from giving up: a fast model
    that is not pulled (NotFoundError) should still hand over to reasoning."""
    _models(monkeypatch, fast="ollama_chat/missing")
    not_found = type("NotFoundError", (Exception,), {})
    calls = []

    def fake(model, system, user, temperature, max_tokens):
        calls.append(model)
        if model.startswith("ollama"):
            raise not_found("model not found")
        return "ok"

    monkeypatch.setattr(llm, "_complete", fake)
    assert llm.chat("s", "u", tier="fast") == "ok"
    assert calls == ["ollama_chat/missing", "cohere/command-a-03-2025"]


# ---------- key per vendor ----------

def test_legacy_cohere_key_only_goes_to_cohere(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "")
    monkeypatch.setattr(llm.settings, "cohere_api_key", "legacy-cohere")
    assert llm._provider_kwargs("cohere/command-a-03-2025") == {"api_key": "legacy-cohere"}
    for model in ("openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"):
        assert llm._provider_kwargs(model) == {}, \
            "a Cohere key must never be sent to another vendor"


def test_llm_api_key_goes_to_any_hosted_vendor(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "k-1")
    monkeypatch.setattr(llm.settings, "cohere_api_key", "legacy-cohere")
    for model in ("cohere/command-a-03-2025", "openai/gpt-4o-mini"):
        assert llm._provider_kwargs(model) == {"api_key": "k-1"}
