"""The compare view: a run side by side with the run it follows
(`parent_mission_id`, set by the scheduler), with the sources each run
collected split into new, dropped and shared by URL."""
import asyncio
import re
from urllib.parse import urlsplit

import pytest

import storage
from models import Agent, Document, Mission


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def client():
    import app as app_mod
    app_mod.initialize()
    return app_mod.app.test_client()


def _path(resp):
    return urlsplit(resp.headers["Location"]).path


def _flashes(client):
    with client.session_transaction() as sess:
        return [msg for _cat, msg in sess.get("_flashes", [])]


SHARED = "https://shared.example/a"
DROPPED = "https://dropped.example/b"
NEW = "https://new.example/c"


def _seed_pair(parent=True, parent_id="p1",
               parent_brief="Old finding [1] with **bold**.",
               child_brief="New finding [1] and <script>alert(1)</script> more [2]."):
    """A parent run p1 (SHARED + DROPPED) and its child m2 (SHARED + NEW).
    With parent=False only m2 exists, still pointing at `parent_id`."""
    async def go():
        await storage.init_db()
        await storage.insert_agent(Agent(id="a1", name="Ada", expertise="orbital mechanics",
                                         persona_prompt="p", created_at="t"))
        if parent:
            await storage.insert_mission(Mission(
                id="p1", agent_id="a1", question="What changed on Monday?", status="done",
                brief_markdown=parent_brief, created_at="2026-01-01T07:00:00",
                finished_at="2026-01-01T07:05:00"))
        await storage.insert_mission(Mission(
            id="m2", agent_id="a1", question="What changed on Tuesday?", status="done",
            brief_markdown=child_brief, parent_mission_id=parent_id,
            created_at="2026-01-02T07:00:00", finished_at="2026-01-02T07:06:00"))

        async def doc(doc_id, url, query):
            return await storage.upsert_document(Document(
                id=doc_id, url=url, domain=urlsplit(url).hostname, title=f"Title {doc_id}",
                search_query=query, crawled_at="2026-01-01T07:01:00",
                content_markdown="An ordinary sentence about the topic. " * 20, word_count=120))

        shared = await doc("d-shared", SHARED, "q")
        await storage.link_mission_document("m2", "r", shared)
        await storage.link_mission_document("m2", "r", await doc("d-new", NEW, "q"))
        # The same URL stored under a second query must count once.
        await storage.link_mission_document("m2", "r", await doc("d-new-2", NEW, "q2"))
        if parent:
            await storage.link_mission_document("p1", "r", shared)
            await storage.link_mission_document("p1", "r", await doc("d-dropped", DROPPED, "q"))
    _run(go())


def _section(html, key):
    m = re.search(rf'<section class="cmp-list" data-cmp="{key}">(.*?)</section>', html, re.S)
    assert m, f"no {key} list"
    return m.group(1)


def test_compare_shows_both_briefs_and_the_source_delta(client):
    _seed_pair()
    r = client.get("/missions/m2/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)

    # Both questions and dates.
    assert "What changed on Monday?" in html and "What changed on Tuesday?" in html
    assert "2026-01-01 07:00" in html and "2026-01-02 07:00" in html

    # Both briefs, rendered through render_markdown (sanitized) and not
    # linkified: citation controls belong to the mission page's source rail.
    assert "<strong>bold</strong>" in html
    assert "<script>alert(1)</script>" not in html
    assert 'class="cite"' not in html
    assert "New finding [1]" in html and "Old finding [1]" in html

    new, dropped, shared = (_section(html, k) for k in ("new", "dropped", "shared"))
    assert NEW in new and DROPPED not in new and SHARED not in new
    assert new.count(NEW) == 1, "one URL is one source, whatever query stored it"
    assert DROPPED in dropped and NEW not in dropped and SHARED not in dropped
    assert SHARED in shared and NEW not in shared and DROPPED not in shared
    # Each entry opens the stored document.
    assert 'href="/document/d-dropped"' in dropped
    assert 'href="/document/d-shared"' in shared

    # Both runs link back to their mission pages.
    assert 'href="/missions/p1"' in html and 'href="/missions/m2"' in html


def test_compare_escapes_page_controlled_titles(client):
    _seed_pair()
    _run(storage.upsert_document(Document(
        id="d-new", url=NEW, domain="new.example", title="<img src=x onerror=alert(1)>",
        search_query="q", crawled_at="2026-01-01T07:01:00", content_markdown="x", word_count=1)))
    html = client.get("/missions/m2/compare").get_data(as_text=True)
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_compare_without_a_parent_flashes_back_to_the_mission(client):
    _seed_pair(parent=False, parent_id=None)
    r = client.get("/missions/m2/compare")
    assert r.status_code == 302 and _path(r) == "/missions/m2"
    assert any("no previous run" in m for m in _flashes(client))


def test_compare_with_a_deleted_parent_flashes_back_to_the_mission(client):
    _seed_pair()
    _run(storage.delete_mission("p1"))
    r = client.get("/missions/m2/compare")
    assert r.status_code == 302 and _path(r) == "/missions/m2"
    assert any("previous run was deleted" in m for m in _flashes(client))


def test_compare_unknown_mission(client):
    r = client.get("/missions/nope/compare")
    assert r.status_code == 302 and _path(r) == "/missions"
    assert any("Mission not found" in m for m in _flashes(client))


def test_mission_page_links_to_compare_only_when_the_parent_exists(client):
    _seed_pair()
    assert 'href="/missions/m2/compare"' in client.get("/missions/m2").get_data(as_text=True)
    # The parent itself has no parent: no link.
    assert "/compare" not in client.get("/missions/p1").get_data(as_text=True)
    _run(storage.delete_mission("p1"))
    assert "/compare" not in client.get("/missions/m2").get_data(as_text=True)


def test_compare_a_run_with_no_brief_yet(client):
    _seed_pair(child_brief=None)
    html = client.get("/missions/m2/compare").get_data(as_text=True)
    assert "No brief was written for this run" in html
    assert "Old finding [1]" in html
