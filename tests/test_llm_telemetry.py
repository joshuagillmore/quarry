"""llm.py telemetry: every provider call that returns is recorded as one
LlmCall (purpose, tier requested, model that answered, tokens, duration);
attempts that raised are not; a recording failure never fails the call."""
import asyncio

import litellm
import pytest

import llm
import storage


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    async def fake_insert(call):
        calls.append(call)

    monkeypatch.setattr(storage, "insert_llm_call", fake_insert)
    return calls


def _models(monkeypatch, reasoning="cohere/command-a-03-2025", fast=""):
    monkeypatch.setattr(llm.settings, "llm_provider", reasoning)
    monkeypatch.setattr(llm.settings, "llm_provider_fast", fast)


# ---------- _complete reads usage ----------

def test_complete_returns_content_and_usage(monkeypatch):
    _models(monkeypatch)
    monkeypatch.setattr(llm.litellm, "completion",
                        lambda **k: litellm._Resp("hi", litellm._Usage(12, 3)))
    assert llm._complete("m", "s", "u", 0.0, 10) == ("hi", (12, 3))


def test_complete_without_usage_reports_zeros(monkeypatch):
    _models(monkeypatch)
    monkeypatch.setattr(llm.litellm, "completion", lambda **k: litellm._Resp("hi"))
    assert llm._complete("m", "s", "u", 0.0, 10) == ("hi", (0, 0))


def test_complete_tolerates_missing_token_fields(monkeypatch):
    _models(monkeypatch)
    monkeypatch.setattr(llm.litellm, "completion",
                        lambda **k: litellm._Resp("hi", litellm._Usage(None, "7")))
    assert llm._complete("m", "s", "u", 0.0, 10) == ("hi", (0, 7))


# ---------- chat_ex records ----------

def test_chat_ex_records_one_call(monkeypatch, recorded):
    _models(monkeypatch)
    monkeypatch.setattr(llm.litellm, "completion",
                        lambda **k: litellm._Resp("answer", litellm._Usage(100, 20)))
    text, model = llm.chat_ex("s", "u", purpose="assess", mission_id="m1")
    assert (text, model) == ("answer", "cohere/command-a-03-2025")
    [call] = recorded
    assert (call.purpose, call.tier, call.model, call.mission_id) == (
        "assess", "reasoning", "cohere/command-a-03-2025", "m1")
    assert (call.prompt_tokens, call.completion_tokens) == (100, 20)
    assert isinstance(call.duration_ms, int) and call.duration_ms >= 0


def test_chat_ex_defaults_to_purpose_chat_and_no_mission(monkeypatch, recorded):
    _models(monkeypatch)
    monkeypatch.setattr(llm, "_complete", lambda *a: ("ok", (1, 1)))
    llm.chat_ex("s", "u")
    [call] = recorded
    assert (call.purpose, call.mission_id) == ("chat", None)


def test_failed_retry_attempt_is_not_recorded(monkeypatch, recorded):
    _models(monkeypatch)
    attempts = []

    def flaky(model, system, user, temperature, max_tokens):
        attempts.append(model)
        if len(attempts) == 1:
            raise RuntimeError("NO_VALID_RESPONSE_GENERATED")
        return "recovered", (5, 2)

    monkeypatch.setattr(llm, "_complete", flaky)
    assert llm.chat("s", "u", purpose="plan", mission_id="m1") == "recovered"
    assert len(attempts) == 2
    [call] = recorded
    assert (call.purpose, call.prompt_tokens, call.completion_tokens) == ("plan", 5, 2)


def test_fallback_call_is_recorded_with_its_own_model(monkeypatch, recorded):
    _models(monkeypatch, fast="ollama_chat/qwen2.5:14b")

    def fake(model, system, user, temperature, max_tokens):
        if model.startswith("ollama"):
            raise RuntimeError("Connection refused")
        return "from-reasoning", (40, 9)

    monkeypatch.setattr(llm, "_complete", fake)
    text, model = llm.chat_ex("s", "u", tier="fast", purpose="brief", mission_id="m1")
    assert model == "cohere/command-a-03-2025"
    [call] = recorded
    assert call.model == "cohere/command-a-03-2025"
    assert call.tier == "fast", "the tier requested, not the one that answered"
    assert (call.purpose, call.prompt_tokens) == ("brief", 40)


def test_call_that_raises_records_nothing(monkeypatch, recorded):
    _models(monkeypatch)

    def down(*a):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_complete", down)
    with pytest.raises(RuntimeError):
        llm.chat_ex("s", "u")
    assert recorded == []


def test_recording_failure_is_logged_not_raised(monkeypatch, capsys):
    _models(monkeypatch)
    monkeypatch.setattr(llm, "_complete", lambda *a: ("fine", (1, 1)))

    async def broken(call):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(storage, "insert_llm_call", broken)
    assert llm.chat_ex("s", "u", purpose="assess") == ("fine", "cohere/command-a-03-2025")
    err = capsys.readouterr().err
    assert "database is locked" in err


def test_chat_and_chat_json_forward_purpose_and_mission(monkeypatch, recorded):
    _models(monkeypatch)
    monkeypatch.setattr(llm, "_complete", lambda *a: ('{"a": 1}', (3, 4)))
    llm.chat("s", "u", purpose="brief", mission_id="m1")
    parsed, _raw = llm.chat_json("s", "u", purpose="plan", mission_id="m2")
    assert parsed == {"a": 1}
    assert [(c.purpose, c.mission_id) for c in recorded] == [("brief", "m1"), ("plan", "m2")]


def test_recorded_on_an_event_loop_thread_too(monkeypatch, recorded):
    """chat_ex is synchronous and normally runs off the loop (to_thread);
    a caller already on a loop must still be recorded, not crash."""
    _models(monkeypatch)
    monkeypatch.setattr(llm, "_complete", lambda *a: ("ok", (2, 2)))

    async def on_loop():
        return llm.chat_ex("s", "u", purpose="assess", mission_id="m1")

    assert asyncio.run(on_loop())[0] == "ok"
    assert [c.purpose for c in recorded] == ["assess"]


def test_usage_lands_in_the_real_table(monkeypatch):
    _models(monkeypatch)
    asyncio.run(storage.init_db())
    monkeypatch.setattr(llm.litellm, "completion",
                        lambda **k: litellm._Resp("x", litellm._Usage(30, 7)))
    llm.chat("s", "u", purpose="assess", mission_id="m1")
    llm.chat("s", "u", purpose="assess", mission_id="m1")
    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert (usage["calls"], usage["prompt_tokens"], usage["completion_tokens"]) == (2, 60, 14)
    assert usage["by_purpose"]["assess"]["calls"] == 2
