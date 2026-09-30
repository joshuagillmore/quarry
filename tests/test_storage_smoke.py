"""Storage against a real temp SQLite file (the autouse fixture in conftest.py
gives every test its own DB_PATH)."""
import asyncio
import sqlite3

import pytest

import storage
from models import Agent, Mission, Requirement, Document


def _doc(doc_id, url="http://x/1", query="Q", title="t", body="", fit=None, crawled_at="t"):
    return Document(id=doc_id, url=url, domain="x", title=title, search_query=query,
                    crawled_at=crawled_at, content_markdown=body, content_fit=fit)


def _sql(query, args=()):
    db = sqlite3.connect(storage.DB_PATH)
    try:
        return db.execute(query, args).fetchall()
    finally:
        db.close()


def _columns(table):
    return {row[1] for row in _sql(f"PRAGMA table_info({table})")}


def test_storage_roundtrip():
    async def run():
        await storage.init_db()

        # Agents
        await storage.insert_agent(Agent(
            id="a1", name="N", expertise="x", persona_prompt="p", created_at="t"))
        assert (await storage.get_agent("a1")).name == "N"
        assert len(await storage.list_agents()) == 1

        # Missions
        await storage.insert_mission(Mission(
            id="m1", agent_id="a1", question="Q", status="planning", created_at="t"))
        await storage.update_mission("m1", status="done", finished_at="t2")
        assert (await storage.get_mission("m1")).status == "done"
        assert len(await storage.list_missions()) == 1

        # Requirements
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="T", status="pending"))
        await storage.update_requirement("r1", status="satisfied", attempts=2)
        reqs = await storage.get_requirements_for_mission("m1")
        assert reqs[0].status == "satisfied" and reqs[0].attempts == 2

        # upsert_document reuses the id for the same (url, search_query)
        id1 = await storage.upsert_document(_doc("d1", title="t1", body="hello world"))
        id2 = await storage.upsert_document(_doc("dDIFFERENT", title="t2", body="updated body"))
        assert id1 == id2

        # Join table + lookups
        await storage.link_mission_document("m1", "r1", id1)
        md = await storage.get_mission_documents("m1")
        assert len(md) == 1 and md[0].title == "t2"  # content refreshed in place
        assert len(await storage.get_requirement_documents("m1", "r1")) == 1

        # Delta helpers
        assert await storage.get_prior_mission_urls("m1") == set()
        assert await storage.get_latest_finished_mission("a1", "Q", "m1") is None

        # FTS stayed consistent with the single upserted doc
        assert await storage.count_documents() == 1
        hits = await storage.search_documents_fts("updated")
        assert len(hits) == 1 and hits[0].url == "http://x/1"

    asyncio.run(run())


# --- documents: atomic upsert + FTS consistency --------------------------------

def test_upsert_replaces_the_fts_row_rather_than_adding_one():
    """Fails if upsert_document stops deleting the old FTS row: the stale row
    would still match the old content and double-count the document."""
    async def run():
        await storage.init_db()
        id1 = await storage.upsert_document(_doc("d1", body="hello zebrafish world"))
        id2 = await storage.upsert_document(_doc("d2", body="completely new content"))
        assert id1 == id2 == "d1"
        assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
        assert await storage.search_documents_fts("zebrafish") == []
        assert [d.id for d in await storage.search_documents_fts("completely")] == ["d1"]
    asyncio.run(run())


def test_insert_document_returns_the_authoritative_id():
    """A re-crawl of the same (url, search_query) keeps the original id, so
    extractions and mission links keyed on it are not orphaned."""
    async def run():
        await storage.init_db()
        first = await storage.insert_document(_doc("d1", body="old zebrafish text"))
        again = await storage.insert_document(_doc("d2", body="fresh text"))
        assert first == again == "d1"
        assert await storage.count_documents() == 1
        assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
        assert await storage.search_documents_fts("zebrafish") == []
        assert (await storage.get_document("d1")).content_markdown == "fresh text"
    asyncio.run(run())


def test_insert_document_raises_instead_of_swallowing_errors():
    async def run():
        await storage.init_db()
        await storage.insert_document(_doc("d1", url="http://x/1"))
        # Same primary key, different (url, search_query): not the upsert's
        # conflict target, so it must surface rather than print-and-continue.
        with pytest.raises(sqlite3.IntegrityError):
            await storage.insert_document(_doc("d1", url="http://x/other"))
    asyncio.run(run())


def test_concurrent_upserts_of_one_url_converge_on_one_row():
    """Select-then-insert races: two writers both see no row and the second
    INSERT hits the UNIQUE constraint. A single upsert statement cannot."""
    async def run():
        await storage.init_db()
        ids = await asyncio.gather(*[
            storage.upsert_document(_doc(f"d{i}", body=f"body {i}")) for i in range(8)
        ])
        assert len(set(ids)) == 1
        assert await storage.count_documents() == 1
        assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
    asyncio.run(run())


# --- documents: pagination + previews -----------------------------------------

def _seed_three():
    async def run():
        await storage.init_db()
        await storage.upsert_document(_doc("d1", url="http://x/1", query="A",
                                           body="0123456789", fit=None, crawled_at="2026-01-01"))
        await storage.upsert_document(_doc("d2", url="http://x/2", query="A",
                                           body="full markdown", fit="abcdefghij",
                                           crawled_at="2026-01-02"))
        await storage.upsert_document(_doc("d3", url="http://x/3", query="B",
                                           body="other query", crawled_at="2026-01-03"))
    asyncio.run(run())


def test_get_all_documents_defaults_are_unchanged():
    _seed_three()
    docs = asyncio.run(storage.get_all_documents())
    assert [d.id for d in docs] == ["d3", "d2", "d1"]
    assert docs[1].content_markdown == "full markdown"
    assert docs[1].content_fit == "abcdefghij"


def test_get_all_documents_pages():
    _seed_three()
    page1 = asyncio.run(storage.get_all_documents(limit=2))
    page2 = asyncio.run(storage.get_all_documents(limit=2, offset=2))
    assert [d.id for d in page1] == ["d3", "d2"]
    assert [d.id for d in page2] == ["d1"]
    assert asyncio.run(storage.get_all_documents(limit=2, offset=3)) == []
    assert [d.id for d in asyncio.run(storage.get_all_documents(offset=1))] == ["d2", "d1"]


def test_preview_chars_truncates_and_drops_full_markdown():
    _seed_three()
    docs = {d.id: d for d in asyncio.run(storage.get_all_documents(preview_chars=4))}
    assert docs["d2"].content_fit == "abcd"        # content_fit preferred
    assert docs["d1"].content_fit == "0123"        # falls back to content_markdown
    assert all(d.content_markdown is None for d in docs.values())
    assert docs["d2"].title == "t" and docs["d2"].url == "http://x/2"


def test_get_documents_by_search_pages_and_previews():
    _seed_three()
    assert [d.id for d in asyncio.run(storage.get_documents_by_search("A"))] == ["d2", "d1"]
    page = asyncio.run(storage.get_documents_by_search("A", limit=1, offset=1, preview_chars=3))
    assert [d.id for d in page] == ["d1"]
    assert page[0].content_fit == "012" and page[0].content_markdown is None


# --- missions ------------------------------------------------------------------

def _seed_agent_and_mission(status="planning", mission_id="m1", agent_id="a1"):
    async def run():
        await storage.init_db()
        if await storage.get_agent(agent_id) is None:
            await storage.insert_agent(Agent(id=agent_id, name="N", expertise="x",
                                             persona_prompt="p", created_at="t"))
        await storage.insert_mission(Mission(id=mission_id, agent_id=agent_id,
                                             question="Q", status=status, created_at="t"))
        await storage.insert_requirement(Requirement(id=f"r-{mission_id}",
                                                     mission_id=mission_id, title="T"))
    asyncio.run(run())


def test_claim_mission_status_is_compare_and_set():
    _seed_agent_and_mission(status="awaiting_approval")
    claim = storage.claim_mission_status
    assert asyncio.run(claim("m1", "awaiting_approval", "collecting")) is True
    assert asyncio.run(claim("m1", "awaiting_approval", "collecting")) is False
    assert asyncio.run(storage.get_mission("m1")).status == "collecting"
    assert asyncio.run(claim("nope", "awaiting_approval", "collecting")) is False


def test_agent_has_active_mission():
    _seed_agent_and_mission(status="done")
    assert asyncio.run(storage.agent_has_active_mission("a1")) is False
    _seed_agent_and_mission(status="awaiting_approval", mission_id="m2")
    assert asyncio.run(storage.agent_has_active_mission("a1")) is False
    for status in ("planning", "collecting", "synthesizing"):
        asyncio.run(storage.update_mission("m2", status=status))
        assert asyncio.run(storage.agent_has_active_mission("a1")) is True, status
    _seed_agent_and_mission(status="done", mission_id="m3", agent_id="a2")
    assert asyncio.run(storage.agent_has_active_mission("a2")) is False


def test_brief_sources_json_roundtrips():
    _seed_agent_and_mission()
    asyncio.run(storage.update_mission("m1", brief_sources_json='["d1", "d2"]'))
    assert asyncio.run(storage.get_mission("m1")).brief_sources_json == '["d1", "d2"]'

    async def insert():
        await storage.insert_mission(Mission(id="m9", agent_id="a1", question="Q",
                                             created_at="t", brief_sources_json='["d3"]'))
        return await storage.get_mission("m9")
    assert asyncio.run(insert()).brief_sources_json == '["d3"]'


@pytest.mark.parametrize("update_name, row_id", [
    ("update_mission", "m1"),
    ("update_requirement", "r-m1"),
    ("update_agent", "a1"),
])
def test_update_rejects_unknown_columns(update_name, row_id):
    _seed_agent_and_mission()
    update = getattr(storage, update_name)
    for bad in ({"bogus": 1}, {"status = 'done' --": 1}, {"id": "renamed"}):
        with pytest.raises(ValueError):
            asyncio.run(update(row_id, **bad))


def test_update_allowlists_match_the_schema():
    asyncio.run(storage.init_db())
    assert storage._AGENT_COLUMNS == _columns("agents") - {"id"}
    assert storage._MISSION_COLUMNS == _columns("missions") - {"id"}
    assert storage._REQUIREMENT_COLUMNS == _columns("requirements") - {"id"}


# --- init_db: paths, migrations, indexes ---------------------------------------

def test_bare_db_path_without_a_directory_initialises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage, "DB_PATH", "research.db")
    asyncio.run(storage.init_db())
    assert (tmp_path / "research.db").exists()


_OLD_SCHEMA = (
    "CREATE TABLE searches (id TEXT PRIMARY KEY, query TEXT NOT NULL, "
    "executed_at TEXT NOT NULL, result_count INTEGER DEFAULT 0)",
    "CREATE TABLE agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, expertise TEXT, "
    "persona_prompt TEXT, default_max_passes INTEGER DEFAULT 4, "
    "default_max_sources INTEGER DEFAULT 30, default_per_req_attempts INTEGER DEFAULT 3, "
    "schedule_cron TEXT, active INTEGER DEFAULT 1, created_at TEXT NOT NULL)",
    "CREATE TABLE missions (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, "
    "question TEXT NOT NULL, status TEXT NOT NULL, plan_json TEXT, budget_json TEXT, "
    "brief_markdown TEXT, job_id TEXT, parent_mission_id TEXT, error TEXT, "
    "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)",
    "CREATE TABLE requirements (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, "
    "title TEXT NOT NULL, description TEXT, rationale TEXT, "
    "status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER DEFAULT 0, "
    "next_queries_json TEXT, satisfied_doc_ids_json TEXT)",
    "INSERT INTO missions (id, agent_id, question, status, created_at) "
    "VALUES ('old', 'a1', 'Q', 'done', 't')",
)


def test_init_db_migrates_an_old_schema():
    """A DB created before these columns existed gains them, and a second
    init_db is a no-op rather than an error."""
    db = sqlite3.connect(storage.DB_PATH)
    try:
        for stmt in _OLD_SCHEMA:
            db.execute(stmt)
        db.commit()
    finally:
        db.close()

    asyncio.run(storage.init_db())
    asyncio.run(storage.init_db())

    assert "job_id" in _columns("searches")
    assert "schedule_question" in _columns("agents")
    assert "brief_sources_json" in _columns("missions")
    assert {"assessment_missing", "assessment_confidence",
            "accepted_by_user"} <= _columns("requirements")
    old = asyncio.run(storage.get_mission("old"))
    assert old.status == "done" and old.brief_sources_json is None


def _index_column_sets(table):
    out = set()
    for row in _sql(f"PRAGMA index_list({table})"):
        out.add(tuple(r[2] for r in _sql(f"PRAGMA index_info('{row[1]}')")))
    return out


def test_init_db_creates_lookup_indexes():
    asyncio.run(storage.init_db())
    assert ("search_query", "crawled_at") in _index_column_sets("documents")
    assert ("document_id",) in _index_column_sets("extractions")
    assert ("mission_id",) in _index_column_sets("requirements")
    assert ("document_id",) in _index_column_sets("mission_documents")
    assert ("agent_id", "status") in _index_column_sets("missions")
