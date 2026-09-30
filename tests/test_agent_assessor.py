import agent_assessor
from models import Requirement


def _req():
    return Requirement(id="r", mission_id="m", title="T", description="D")


def test_satisfied(monkeypatch):
    monkeypatch.setattr(agent_assessor, "chat_json", lambda s, u, **k: (
        {"satisfied": True, "confidence": "high", "missing": "", "next_queries": []}, "raw"))
    a = agent_assessor.assess_requirement(_req(), [])
    assert a.satisfied and a.confidence == "high"


def test_unsatisfied_with_next_queries(monkeypatch):
    monkeypatch.setattr(agent_assessor, "chat_json", lambda s, u, **k: (
        {"satisfied": False, "confidence": "low", "missing": "x", "next_queries": ["nq1", "nq2"]}, "raw"))
    a = agent_assessor.assess_requirement(_req(), [])
    assert not a.satisfied
    assert a.next_queries == ["nq1", "nq2"]


def test_provider_exception_does_not_propagate(monkeypatch):
    """A provider hiccup must cost the requirement an attempt, not the whole
    mission — the crawling already paid for is far more expensive."""
    def boom(*a, **k):
        raise RuntimeError("Cohere_chatException - NO_VALID_RESPONSE_GENERATED")

    monkeypatch.setattr(agent_assessor, "chat_json", boom)
    a = agent_assessor.assess_requirement(_req(), [])
    assert not a.satisfied
    assert a.next_queries == []
    assert "could not be completed" in a.missing


def test_parse_failure_is_not_satisfied(monkeypatch):
    monkeypatch.setattr(agent_assessor, "chat_json", lambda s, u, **k: (None, "raw"))
    a = agent_assessor.assess_requirement(_req(), [])
    assert not a.satisfied
    assert a.next_queries == []


import pytest  # noqa: E402


def _verdict(monkeypatch, **fields):
    base = {"satisfied": False, "confidence": "low", "missing": "", "next_queries": []}
    base.update(fields)
    monkeypatch.setattr(agent_assessor, "chat_json", lambda s, u, **k: (base, "raw"))
    return agent_assessor.assess_requirement(_req(), [])


@pytest.mark.parametrize("value", ["false", "False", "no", "", "maybe", 0, 1, None, [], {}])
def test_only_true_or_yes_counts_as_satisfied(monkeypatch, value):
    """bool("false") is True: a model answering with a string must not
    silently satisfy the requirement."""
    assert not _verdict(monkeypatch, satisfied=value).satisfied


@pytest.mark.parametrize("value", [True, "true", "TRUE", "yes", " Yes "])
def test_true_and_yes_are_satisfied(monkeypatch, value):
    assert _verdict(monkeypatch, satisfied=value).satisfied


def test_next_queries_string_is_not_split_into_characters(monkeypatch):
    a = _verdict(monkeypatch, next_queries="try this query")
    assert a.next_queries == []


def test_next_queries_non_strings_dropped(monkeypatch):
    a = _verdict(monkeypatch, next_queries=["ok", 3, None, {"q": 1}, "  ", "two"])
    assert a.next_queries == ["ok", "two"]
