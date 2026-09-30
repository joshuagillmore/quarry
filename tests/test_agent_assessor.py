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


def test_assessor_tags_its_call_with_purpose_and_mission(monkeypatch):
    seen = {}

    def fake(s, u, **k):
        seen.update(k)
        return {"satisfied": True, "confidence": "high"}, "raw"

    monkeypatch.setattr(agent_assessor, "chat_json", fake)
    agent_assessor.assess_requirement(_req(), [])
    assert (seen["purpose"], seen["mission_id"]) == ("assess", "m")


# ---------- search coverage note ----------

from models import Document  # noqa: E402
from prompt_templates import build_assess_prompt  # noqa: E402


def test_assess_prompt_carries_the_search_note():
    note = "Search coverage this pass: 3 queries, 11 results (brave)"
    assert note in build_assess_prompt("T", "D", "sources", search_note=note)
    assert "Search coverage" not in build_assess_prompt("T", "D", "sources")


def test_assessor_passes_the_search_note_to_the_prompt(monkeypatch):
    seen = {}

    def fake(s, u, **k):
        seen["prompt"] = u
        return {"satisfied": False, "confidence": "low"}, "raw"

    monkeypatch.setattr(agent_assessor, "chat_json", fake)
    agent_assessor.assess_requirement(
        _req(), [], search_note="Search coverage this pass: 2 queries, 0 results (none)")
    assert "Search coverage this pass: 2 queries, 0 results (none)" in seen["prompt"]


# ---------- cheaper assessment: skip the LLM when nothing is on topic ----------

def _termed_req():
    return Requirement(id="r", mission_id="m", title="Lithium battery degradation",
                       description="What mechanisms shorten battery lifetime?")


def _src(fit=None, md=None):
    return Document(id="d", url="http://x/1", domain="x", title="Some page",
                    search_query="q", crawled_at="t", content_fit=fit,
                    content_markdown=md, word_count=200)


def _no_llm(monkeypatch):
    def called(*a, **k):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(agent_assessor, "chat_json", called)


def test_key_terms_are_long_words_minus_stopwords():
    terms = agent_assessor.key_terms(_termed_req())
    assert {"lithium", "battery", "degradation", "mechanisms", "shorten",
            "lifetime"} <= terms
    assert "what" not in terms, "short words are not key terms"
    assert not agent_assessor.key_terms(Requirement(
        id="r", mission_id="m", title="Which of these", description="about their other"))


def test_off_topic_sources_skip_the_llm(monkeypatch, capsys):
    _no_llm(monkeypatch)
    a = agent_assessor.assess_requirement(
        _termed_req(), [_src(fit="A recipe for sourdough bread and pastry.")])
    assert a == agent_assessor.Assessment(
        False, "low", "no collected source mentions the requirement's key terms", [])
    assert "skipped" in capsys.readouterr().err


def test_no_sources_skip_the_llm(monkeypatch):
    _no_llm(monkeypatch)
    a = agent_assessor.assess_requirement(_termed_req(), [])
    assert not a.satisfied and a.next_queries == []
    assert a.missing == "no collected source mentions the requirement's key terms"


def test_a_matching_term_calls_the_llm(monkeypatch):
    calls = []

    def fake(s, u, **k):
        calls.append(u)
        return {"satisfied": True, "confidence": "high"}, "raw"

    monkeypatch.setattr(agent_assessor, "chat_json", fake)
    # Case-insensitive, and content_markdown counts when there is no fit text.
    a = agent_assessor.assess_requirement(
        _termed_req(), [_src(fit=None, md="Cell DEGRADATION rises with heat.")])
    assert a.satisfied and len(calls) == 1


def test_requirement_without_key_terms_always_calls_the_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_assessor, "chat_json", lambda s, u, **k: (
        calls.append(u) or ({"satisfied": False, "confidence": "low"}, "raw")))
    short = Requirement(id="r", mission_id="m", title="GDP 2024", description="Is it up?")
    agent_assessor.assess_requirement(short, [])
    agent_assessor.assess_requirement(short, [_src(fit="nothing relevant here")])
    assert len(calls) == 2
