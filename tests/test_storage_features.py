"""Storage foundations for the features round: the llm_calls telemetry table,
the brief_warnings_json / search_stats_json columns, and the mission-pair
document query the compare view uses. Real temp SQLite per test (conftest)."""
import asyncio
import sqlite3

import pytest

import storage
from models import Agent, Document, LlmCall, Mission, Requirement


def _sql(query, args=()):
    db = sqlite3.connect(storage.DB_PATH)
    try:
        return db.execute(query, args).fetchall()
    finally:
        db.close()


def _columns(table):
    return {row[1] for row in _sql(f"PRAGMA table_info({table})")}


def _index_column_sets(table):
    out = set()
    for row in _sql(f"PRAGMA index_list({table})"):
        out.add(tuple(r[2] for r in _sql(f"PRAGMA index_info('{row[1]}')")))
    return out


def _call(mission_id, purpose, prompt, completion, **kw):
    return LlmCall(mission_id=mission_id, purpose=purpose, tier="reasoning",
                   model="cohere/command-a-03-2025", prompt_tokens=prompt,
                   completion_tokens=completion, duration_ms=10, **kw)


def _seed_missions(*mission_ids):
    async def run():
        await storage.init_db()
        await storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                         persona_prompt="p", created_at="t"))
        for mid in mission_ids:
            await storage.insert_mission(Mission(id=mid, agent_id="a1", question="Q",
                                                 status="done", created_at="t"))
            await storage.insert_requirement(Requirement(id=f"r-{mid}", mission_id=mid,
                                                         title="T"))
    asyncio.run(run())


# --- migration -----------------------------------------------------------------

# The schema as it stood before this round: every earlier migration applied,
# but no brief_warnings_json / search_stats_json and no llm_calls table.
_PRE_FEATURES_SCHEMA = (
    "CREATE TABLE missions (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, "
    "question TEXT NOT NULL, status TEXT NOT NULL, plan_json TEXT, budget_json TEXT, "
    "brief_markdown TEXT, brief_sources_json TEXT, job_id TEXT, parent_mission_id TEXT, "
    "error TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)",
    "CREATE TABLE requirements (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, "
    "title TEXT NOT NULL, description TEXT, rationale TEXT, "
    "status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER DEFAULT 0, "
    "next_queries_json TEXT, satisfied_doc_ids_json TEXT, assessment_missing TEXT, "
    "assessment_confidence TEXT, accepted_by_user INTEGER DEFAULT 0)",
    "INSERT INTO missions (id, agent_id, question, status, created_at) "
    "VALUES ('old', 'a1', 'Q', 'done', 't')",
    "INSERT INTO requirements (id, mission_id, title, description, rationale) "
    "VALUES ('old-r', 'old', 'T', '', '')",
)


def test_init_db_migrates_a_db_without_the_new_columns_or_llm_calls():
    db = sqlite3.connect(storage.DB_PATH)
    try:
        for stmt in _PRE_FEATURES_SCHEMA:
            db.execute(stmt)
        db.commit()
    finally:
        db.close()
    assert "llm_calls" not in {r[0] for r in _sql(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}

    asyncio.run(storage.init_db())
    asyncio.run(storage.init_db())  # idempotent: a second run is a no-op

    assert "brief_warnings_json" in _columns("missions")
    assert "search_stats_json" in _columns("requirements")
    assert _columns("llm_calls") == {
        "id", "mission_id", "purpose", "tier", "model", "prompt_tokens",
        "completion_tokens", "duration_ms", "created_at"}
    assert ("mission_id",) in _index_column_sets("llm_calls")

    old = asyncio.run(storage.get_mission("old"))
    assert old.status == "done" and old.brief_warnings_json is None
    reqs = asyncio.run(storage.get_requirements_for_mission("old"))
    assert [r.id for r in reqs] == ["old-r"] and reqs[0].search_stats_json is None


# --- new columns: insert / select / update allowlists ---------------------------

def test_brief_warnings_json_roundtrips_through_insert_and_update():
    _seed_missions("m1")
    warnings = '[{"kind": "uncited_paragraph", "detail": "para 2"}]'
    asyncio.run(storage.update_mission("m1", brief_warnings_json=warnings))
    assert asyncio.run(storage.get_mission("m1")).brief_warnings_json == warnings

    async def insert():
        await storage.insert_mission(Mission(id="m9", agent_id="a1", question="Q",
                                             created_at="t", brief_warnings_json="[]"))
        return await storage.get_mission("m9")
    assert asyncio.run(insert()).brief_warnings_json == "[]"
    listed = {m.id: m for m in asyncio.run(storage.list_missions())}
    assert listed["m1"].brief_warnings_json == warnings


def test_search_stats_json_roundtrips_through_insert_and_update():
    _seed_missions("m1")
    stats = '[{"pass": 1, "query": "q", "engine": "brave", "results": 4}]'
    asyncio.run(storage.update_requirement("r-m1", search_stats_json=stats))
    reqs = asyncio.run(storage.get_requirements_for_mission("m1"))
    assert reqs[0].search_stats_json == stats

    async def insert():
        await storage.insert_requirement(Requirement(
            id="r2", mission_id="m1", title="T2",
            search_stats_json='[{"pass": 1, "query": "z", "engine": null, "results": 0}]'))
        return await storage.get_requirements_for_mission("m1")
    by_id = {r.id: r for r in asyncio.run(insert())}
    assert by_id["r2"].search_stats_json.endswith('"results": 0}]')


def test_new_columns_are_allowlisted_but_unknown_keys_still_raise():
    _seed_missions("m1")
    assert "brief_warnings_json" in storage._MISSION_COLUMNS
    assert "search_stats_json" in storage._REQUIREMENT_COLUMNS
    with pytest.raises(ValueError):
        asyncio.run(storage.update_requirement("r-m1", search_stats_json="[]", bogus=1))
    with pytest.raises(ValueError):
        asyncio.run(storage.update_mission("m1", brief_warnings_json="[]", bogus=1))
    # The rejected updates changed nothing.
    assert asyncio.run(storage.get_requirements_for_mission("m1"))[0].search_stats_json is None
    assert asyncio.run(storage.get_mission("m1")).brief_warnings_json is None


# --- resume: stop_reason / resume_count ------------------------------------------

def test_init_db_adds_stop_reason_and_resume_count_to_an_older_missions_table():
    db = sqlite3.connect(storage.DB_PATH)
    try:
        for stmt in _PRE_FEATURES_SCHEMA:
            db.execute(stmt)
        db.commit()
    finally:
        db.close()
    asyncio.run(storage.init_db())
    asyncio.run(storage.init_db())  # idempotent

    assert {"stop_reason", "resume_count"} <= _columns("missions")
    old = asyncio.run(storage.get_mission("old"))
    # A mission finished before this change has no recorded reason and was
    # never resumed.
    assert old.stop_reason is None and old.resume_count == 0


def test_stop_reason_and_resume_count_roundtrip_through_insert_and_update():
    _seed_missions("m1")
    m = asyncio.run(storage.get_mission("m1"))
    assert (m.stop_reason, m.resume_count) == (None, 0)
    asyncio.run(storage.update_mission("m1", stop_reason="token_budget", resume_count=2))
    m = asyncio.run(storage.get_mission("m1"))
    assert (m.stop_reason, m.resume_count) == ("token_budget", 2)
    assert "stop_reason" in storage._MISSION_COLUMNS
    assert "resume_count" in storage._MISSION_COLUMNS

    async def insert():
        await storage.insert_mission(Mission(id="m9", agent_id="a1", question="Q",
                                             created_at="t", stop_reason="user_stop",
                                             resume_count=1))
        return await storage.get_mission("m9")
    m9 = asyncio.run(insert())
    assert (m9.stop_reason, m9.resume_count) == ("user_stop", 1)
    listed = {m.id: m for m in asyncio.run(storage.list_missions())}
    assert listed["m1"].stop_reason == "token_budget"


# --- llm_calls -------------------------------------------------------------------

def test_llm_call_defaults_fill_id_and_timestamp():
    a, b = _call("m1", "plan", 1, 2), _call("m1", "plan", 1, 2)
    assert a.id and b.id and a.id != b.id
    assert a.created_at and "T" in a.created_at
    with pytest.raises(TypeError):
        LlmCall("id", "m1", "plan", "reasoning", "model")  # keyword-only


def test_insert_llm_call_stores_every_field():
    asyncio.run(storage.init_db())
    call = LlmCall(id="c1", mission_id="m1", purpose="assess", tier="fast",
                   model="ollama_chat/qwen2.5:14b", prompt_tokens=120,
                   completion_tokens=30, duration_ms=842,
                   created_at="2026-09-30T07:00:00+00:00")
    assert asyncio.run(storage.insert_llm_call(call)) is None
    rows = _sql("SELECT id, mission_id, purpose, tier, model, prompt_tokens, "
                "completion_tokens, duration_ms, created_at FROM llm_calls")
    assert rows == [("c1", "m1", "assess", "fast", "ollama_chat/qwen2.5:14b",
                     120, 30, 842, "2026-09-30T07:00:00+00:00")]


def test_get_mission_llm_usage_aggregates_by_purpose():
    asyncio.run(storage.init_db())

    async def run():
        for call in (_call("m1", "plan", 100, 20),
                     _call("m1", "assess", 300, 40),
                     _call("m1", "assess", 200, 10),
                     _call("m2", "assess", 9999, 9999),   # another mission
                     _call(None, "extract", 7777, 7777)):  # one-shot, no mission
            await storage.insert_llm_call(call)
        return await storage.get_mission_llm_usage("m1")

    assert asyncio.run(run()) == {
        "calls": 3,
        "prompt_tokens": 600,
        "completion_tokens": 70,
        "by_purpose": {
            "plan": {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20},
            "assess": {"calls": 2, "prompt_tokens": 500, "completion_tokens": 50},
        },
    }


def test_get_mission_llm_usage_for_a_mission_without_calls_is_zeros():
    asyncio.run(storage.init_db())
    zero = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_purpose": {}}
    assert asyncio.run(storage.get_mission_llm_usage("nope")) == zero


def test_get_mission_llm_usage_counts_null_token_columns_as_zero():
    """A row written without token counts still counts as a call, and the
    sums stay integers rather than turning into None."""
    asyncio.run(storage.init_db())
    asyncio.run(storage.insert_llm_call(_call("m1", "brief", None, None)))
    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage == {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
                     "by_purpose": {"brief": {"calls": 1, "prompt_tokens": 0,
                                              "completion_tokens": 0}}}


def test_delete_mission_removes_its_llm_calls_only():
    _seed_missions("m1", "m2")

    async def run():
        await storage.insert_llm_call(_call("m1", "plan", 10, 1))
        await storage.insert_llm_call(_call("m1", "assess", 20, 2))
        await storage.insert_llm_call(_call("m2", "plan", 30, 3))
        await storage.delete_mission("m1")
        return (await storage.get_mission_llm_usage("m1"),
                await storage.get_mission_llm_usage("m2"))

    gone, kept = asyncio.run(run())
    assert gone["calls"] == 0
    assert kept["calls"] == 1 and kept["prompt_tokens"] == 30
    assert _sql("SELECT COUNT(*) FROM llm_calls WHERE mission_id = 'm1'")[0][0] == 0


# --- get_mission_pair_documents -----------------------------------------------------

def _doc(doc_id, url, crawled_at):
    return Document(id=doc_id, url=url, domain="x", title=doc_id, search_query="Q",
                    crawled_at=crawled_at, content_markdown=f"body {doc_id}")


def test_get_mission_pair_documents_returns_each_missions_docs_in_listing_order():
    _seed_missions("child", "parent")

    async def run():
        for doc_id, crawled in (("d1", "2026-01-01"), ("d2", "2026-01-02"),
                                ("d3", "2026-01-03"), ("d4", "2026-01-04")):
            await storage.upsert_document(_doc(doc_id, f"http://x/{doc_id}", crawled))
        # child: d1, d3, d4 (d4 linked under two requirements, listed once)
        for doc_id in ("d1", "d3", "d4"):
            await storage.link_mission_document("child", "r-child", doc_id)
        await storage.link_mission_document("child", "r-other", "d4")
        # parent: d2, d3 (d3 is shared with the child)
        for doc_id in ("d3", "d2"):
            await storage.link_mission_document("parent", "r-parent", doc_id)
        pair = await storage.get_mission_pair_documents("child", "parent")
        singles = (await storage.get_mission_documents("child"),
                   await storage.get_mission_documents("parent"))
        return pair, singles

    (child_docs, parent_docs), (child_single, parent_single) = asyncio.run(run())
    assert [d.id for d in child_docs] == ["d4", "d3", "d1"]   # crawled_at DESC
    assert [d.id for d in parent_docs] == ["d3", "d2"]
    assert [d.id for d in child_docs] == [d.id for d in child_single]
    assert [d.id for d in parent_docs] == [d.id for d in parent_single]


def test_get_mission_pair_documents_with_a_missing_parent_is_empty():
    _seed_missions("child")
    asyncio.run(storage.upsert_document(_doc("d1", "http://x/1", "2026-01-01")))
    asyncio.run(storage.link_mission_document("child", "r-child", "d1"))
    child_docs, parent_docs = asyncio.run(storage.get_mission_pair_documents("child", "gone"))
    assert [d.id for d in child_docs] == ["d1"] and parent_docs == []
