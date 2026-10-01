import pytest

import llm

# llm._complete returns (content, (prompt_tokens, completion_tokens)).
_NO_USAGE = (0, 0)


def test_fast_falls_back_to_reasoning(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "cohere/command-a-03-2025")
    monkeypatch.setattr(llm.settings, "llm_provider_fast", "ollama_chat/qwen2.5:14b")

    def fake_complete(model, system, user, temperature, max_tokens):
        if model.startswith("ollama"):
            raise RuntimeError("Connection refused")
        return "from-reasoning", _NO_USAGE

    monkeypatch.setattr(llm, "_complete", fake_complete)
    # Fast tier fails -> transparently uses the reasoning model.
    assert llm.chat("s", "u", tier="fast") == "from-reasoning"


def test_reasoning_failure_raises(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "cohere/command-a-03-2025")
    monkeypatch.setattr(llm.settings, "llm_provider_fast", "ollama_chat/qwen2.5:14b")

    def boom(model, system, user, temperature, max_tokens):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_complete", boom)
    with pytest.raises(RuntimeError):
        llm.chat("s", "u", tier="reasoning")


def test_chat_ex_reports_model_that_answered(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "cohere/command-a-03-2025")
    monkeypatch.setattr(llm.settings, "llm_provider_fast", "ollama_chat/qwen2.5:14b")

    def fake_complete(model, system, user, temperature, max_tokens):
        if model.startswith("ollama"):
            raise RuntimeError("Connection refused")
        return "text", _NO_USAGE

    monkeypatch.setattr(llm, "_complete", fake_complete)
    text, model = llm.chat_ex("s", "u", tier="fast")
    assert text == "text"
    assert model == "cohere/command-a-03-2025"  # the fallback, not the intended fast model

    # And when the fast tier works, it reports the fast model.
    monkeypatch.setattr(llm, "_complete", lambda m, s, u, t, mt: ("ok", _NO_USAGE))
    _, model = llm.chat_ex("s", "u", tier="fast")
    assert model == "ollama_chat/qwen2.5:14b"


def test_no_fallback_when_fast_equals_reasoning(monkeypatch):
    """With no distinct fast model there is nothing to fall back *to*: the call
    may be retried, but only ever against the one configured model."""
    monkeypatch.setattr(llm.settings, "llm_provider", "cohere/command-a-03-2025")
    monkeypatch.setattr(llm.settings, "llm_provider_fast", "")
    calls = []

    def always_fails(model, system, user, temperature, max_tokens):
        calls.append(model)
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_complete", always_fails)
    monkeypatch.setattr(llm, "RETRY_DELAY_S", 0)
    with pytest.raises(RuntimeError):
        llm.chat("s", "u", tier="fast")
    assert set(calls) == {"cohere/command-a-03-2025"}, "must not try another model"
    assert len(calls) == llm.RETRY_ATTEMPTS, "retries only, no fallback duplication"


def test_transient_failure_is_retried(monkeypatch):
    """A provider that refuses once (Cohere's NO_VALID_RESPONSE_GENERATED)
    should not cost the caller its whole mission."""
    monkeypatch.setattr(llm.settings, "llm_provider", "cohere/command-a-03-2025")
    monkeypatch.setattr(llm.settings, "llm_provider_fast", "")
    monkeypatch.setattr(llm, "RETRY_DELAY_S", 0)
    calls = []

    def flaky(model, system, user, temperature, max_tokens):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("NO_VALID_RESPONSE_GENERATED")
        return "recovered", _NO_USAGE

    monkeypatch.setattr(llm, "_complete", flaky)
    assert llm.chat("s", "u") == "recovered"
    assert len(calls) == 2
