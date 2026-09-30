import json

import pytest

import agent_planner
from models import Agent


def _agent():
    return Agent(id="a", name="n", expertise="x", persona_prompt="p", created_at="t")


def test_plan_parses_requirements(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (
        {"requirements": [
            {"title": "Evidence", "description": "D", "rationale": "R", "queries": ["q1", "q2"]},
        ]}, "raw"))
    reqs = agent_planner.build_collection_plan(_agent(), "m1", "question")
    assert len(reqs) == 1
    assert reqs[0].title == "Evidence"
    assert json.loads(reqs[0].next_queries_json) == ["q1", "q2"]


def test_plan_synthesizes_query_when_missing(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (
        {"requirements": [{"title": "Theories"}]}, "raw"))
    reqs = agent_planner.build_collection_plan(_agent(), "m1", "Big Bang")
    assert json.loads(reqs[0].next_queries_json) == ["Big Bang Theories"]


def test_plan_empty_raises(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: ({"requirements": []}, "raw"))
    with pytest.raises(ValueError):
        agent_planner.build_collection_plan(_agent(), "m1", "q")


def test_plan_parse_failure_raises(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (None, "garbage"))
    with pytest.raises(ValueError):
        agent_planner.build_collection_plan(_agent(), "m1", "q")


def test_plan_queries_string_is_not_split_into_characters(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (
        {"requirements": [{"title": "Theories", "queries": "one query"}]}, "raw"))
    reqs = agent_planner.build_collection_plan(_agent(), "m1", "Big Bang")
    assert json.loads(reqs[0].next_queries_json) == ["Big Bang Theories"]


def test_plan_skips_items_with_non_string_titles(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (
        {"requirements": [{"title": 5}, {"title": ["x"]}, {"title": "Real"}]}, "raw"))
    reqs = agent_planner.build_collection_plan(_agent(), "m1", "q")
    assert [r.title for r in reqs] == ["Real"]


def test_plan_non_string_text_fields_become_empty(monkeypatch):
    monkeypatch.setattr(agent_planner, "chat_json", lambda s, u, **k: (
        {"requirements": [{"title": "T", "description": ["x"], "rationale": 7,
                           "queries": ["good", 4, None]}]}, "raw"))
    r = agent_planner.build_collection_plan(_agent(), "m1", "q")[0]
    assert (r.description, r.rationale) == ("", "")
    assert json.loads(r.next_queries_json) == ["good"]


def test_planner_tags_its_call_with_purpose_and_mission(monkeypatch):
    seen = {}

    def fake(s, u, **k):
        seen.update(k)
        return {"requirements": [{"title": "T"}]}, "raw"

    monkeypatch.setattr(agent_planner, "chat_json", fake)
    agent_planner.build_collection_plan(_agent(), "m1", "q")
    assert (seen["purpose"], seen["mission_id"]) == ("plan", "m1")
