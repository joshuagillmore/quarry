"""The corpus export actions: POST /missions/<id>/export and
POST /documents/export hand a document set to
vector_export.start_export_in_background (never run here), flash the
outcome and redirect back; GET /api/export/status exposes the last export
with its corpus check; the mission page renders the export state and keeps
it current through the poll."""
import asyncio
import re
from urllib.parse import urlsplit, parse_qs

import pytest

import config
import storage
import vector_export
from models import Agent, Document, DocumentExport, Mission

MID = "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"   # a mission id must be a UUID
OTHER = "0f8fad5b-d9cb-469f-a165-70867728950e"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def client():
    import app as app_mod
    app_mod.initialize()
    return app_mod.app.test_client()


@pytest.fixture()
def starter(monkeypatch):
    """Captures each start_export_in_background call as (doc_ids, kwargs)
    and answers `starter.result` (True = started, False = busy)."""
    calls = []

    def fake(doc_ids, **kw):
        calls.append((list(doc_ids), kw))
        return fake.result

    fake.result = True
    fake.calls = calls
    monkeypatch.setattr(vector_export, "start_export_in_background", fake)
    return fake


def _path(resp):
    return urlsplit(resp.headers["Location"]).path


def _query(resp):
    return parse_qs(urlsplit(resp.headers["Location"]).query)


def _flashes(client):
    with client.session_transaction() as sess:
        return [msg for _cat, msg in sess.get("_flashes", [])]


def _seed(n_docs=2, status="done", brief="See [1]."):
    """Mission MID with `n_docs` sources under search "alpha", plus one
    unrelated "beta" document."""
    async def go():
        await storage.init_db()
        await storage.insert_agent(Agent(id="a1", name="Ada", expertise="x",
                                         persona_prompt="p", created_at="t"))
        await storage.insert_mission(Mission(
            id=MID, agent_id="a1", question="Where is it?", status=status,
            brief_markdown=brief, created_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T01:00:00"))
        for i in range(1, n_docs + 1):
            doc_id = await storage.upsert_document(Document(
                id=f"doc{i:02d}", url=f"https://s{i}.example/", domain=f"s{i}.example",
                title=f"Source {i}", search_query="alpha",
                crawled_at=f"2026-01-01T00:{i:02d}:00",
                content_markdown="An ordinary sentence about the topic. " * 60,
                word_count=360))
            await storage.link_mission_document(MID, "r0", doc_id)
        await storage.upsert_document(Document(
            id="other", url="https://other.example/", domain="other.example",
            title="Other", search_query="beta", crawled_at="2026-01-01T00:30:00",
            content_markdown="Unrelated. " * 60, word_count=120))
    _run(go())


# --- POST /missions/<id>/export -------------------------------------------------

def test_mission_export_starts_the_background_export_with_the_mission_docs(client, starter):
    _seed()
    r = client.post(f"/missions/{MID}/export")
    assert r.status_code == 302 and _path(r) == f"/missions/{MID}"
    assert len(starter.calls) == 1
    ids, kw = starter.calls[0]
    assert sorted(ids) == ["doc01", "doc02"]
    assert kw == {"mission_id": MID}
    assert any(m == "Export started → 'quarry'" for m in _flashes(client))


def test_mission_export_flashes_the_configured_collection(client, starter, monkeypatch):
    _seed()
    monkeypatch.setattr(config.settings, "qdrant_export_collection", "fathom_osint")
    client.post(f"/missions/{MID}/export")
    assert any(m == "Export started → 'fathom_osint'" for m in _flashes(client))


def test_mission_export_busy(client, starter):
    _seed()
    starter.result = False
    r = client.post(f"/missions/{MID}/export")
    assert _path(r) == f"/missions/{MID}"
    assert any(m == "An export is already running" for m in _flashes(client))


def test_mission_export_with_no_sources_starts_nothing(client, starter):
    _seed(n_docs=0)
    r = client.post(f"/missions/{MID}/export")
    assert _path(r) == f"/missions/{MID}"
    assert starter.calls == []
    assert any("Nothing to export" in m for m in _flashes(client))


def test_mission_export_rejects_a_non_uuid_id(client, starter):
    _seed()
    r = client.post("/missions/not-a-uuid/export")
    assert r.status_code == 302 and _path(r) == "/missions"
    assert starter.calls == []
    assert any("Invalid mission id" in m for m in _flashes(client))


def test_mission_export_unknown_mission(client, starter):
    _seed()
    r = client.post(f"/missions/{OTHER}/export")
    assert r.status_code == 302 and _path(r) == "/missions"
    assert starter.calls == []
    assert any("Mission not found" in m for m in _flashes(client))


# --- POST /documents/export ------------------------------------------------------

def test_documents_export_by_search(client, starter):
    _seed()
    r = client.post("/documents/export", data={"search": "alpha"})
    assert r.status_code == 302 and _path(r) == "/documents"
    assert _query(r) == {"search": ["alpha"]}
    ids, kw = starter.calls[0]
    assert sorted(ids) == ["doc01", "doc02"]      # not the "beta" document
    assert kw == {"mission_id": None}
    assert any(m == "Export started → 'quarry'" for m in _flashes(client))


def test_documents_export_by_mission(client, starter):
    _seed()
    r = client.post("/documents/export", data={"mission": MID})
    assert _path(r) == "/documents" and _query(r) == {"mission": [MID]}
    ids, kw = starter.calls[0]
    assert sorted(ids) == ["doc01", "doc02"]
    assert kw == {"mission_id": MID}


def test_documents_export_mission_takes_precedence_over_search(client, starter):
    _seed()
    r = client.post("/documents/export", data={"mission": MID, "search": "beta"})
    assert _query(r) == {"mission": [MID]}
    assert sorted(starter.calls[0][0]) == ["doc01", "doc02"]


def test_documents_export_busy(client, starter):
    _seed()
    starter.result = False
    r = client.post("/documents/export", data={"search": "alpha"})
    assert _query(r) == {"search": ["alpha"]}
    assert any(m == "An export is already running" for m in _flashes(client))


def test_documents_export_bounds_the_search_query(client, starter):
    _seed()
    r = client.post("/documents/export", data={"search": "a" * 600})
    assert len(_query(r)["search"][0]) == 500
    # Nothing is stored under that query: nothing to export, nothing started.
    assert starter.calls == []
    assert any("Nothing to export" in m for m in _flashes(client))


def test_documents_export_rejects_bad_or_missing_targets(client, starter):
    _seed()
    r = client.post("/documents/export", data={})
    assert _path(r) == "/documents" and _query(r) == {}
    assert any("Choose a search or a mission" in m for m in _flashes(client))

    r = client.post("/documents/export", data={"mission": "m1"})
    assert _path(r) == "/documents"
    assert any("Invalid mission id" in m for m in _flashes(client))

    r = client.post("/documents/export", data={"mission": OTHER})
    assert _path(r) == "/documents"
    assert any("Mission not found" in m for m in _flashes(client))
    assert starter.calls == []


def test_export_posts_are_covered_by_the_csrf_guard(client, starter):
    _seed()
    for url, data in ((f"/missions/{MID}/export", {}),
                      ("/documents/export", {"search": "alpha"})):
        r = client.post(url, data=data, headers={"Origin": "http://evil.example"})
        assert r.status_code == 403, url
        r = client.post(url, data=data, headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403, url
    assert starter.calls == []


def test_export_routes_sit_behind_the_login_gate(client, starter, monkeypatch):
    _seed()
    monkeypatch.setattr(config.settings, "quarry_password", "hunter2-quarry")
    r = client.post(f"/missions/{MID}/export")
    assert r.status_code == 302 and _path(r) == "/login"
    r = client.post("/documents/export", data={"search": "alpha"})
    assert r.status_code == 302 and _path(r) == "/login"
    r = client.get("/api/export/status")
    assert r.status_code == 401 and r.get_json() == {"error": "authentication required"}
    assert starter.calls == []


# --- GET /api/export/status ------------------------------------------------------

def test_export_status_returns_the_exporter_state_with_its_check(client, monkeypatch):
    status = {
        "running": False, "collection": "quarry", "mission_id": MID, "doc_count": 2,
        "started_at": "2026-10-31T20:10:00+00:00", "finished_at": "2026-10-31T20:14:00+00:00",
        "error": None,
        "result": {"collection": "quarry", "exported": ["doc01"], "skipped": ["doc02"],
                   "failed": {}, "chunks": 5, "stats_path": None,
                   "manifest_path": "/data/export/manifest-quarry.json",
                   "check": {"collection": "quarry", "mission_id": MID, "k": 10,
                             "queries": [], "recall_at_k": 0.83, "mrr": 0.5,
                             "passed": True, "eval_dir": "/data/export/eval"},
                   "started_at": "2026-10-31T20:10:00+00:00",
                   "finished_at": "2026-10-31T20:14:00+00:00"},
    }
    monkeypatch.setattr(vector_export, "export_status", lambda: status)
    r = client.get("/api/export/status")
    assert r.status_code == 200 and r.mimetype == "application/json"
    assert r.get_json() == status
    assert r.get_json()["result"]["check"]["recall_at_k"] == 0.83


def test_export_status_idle_shape_for_real(client):
    body = client.get("/api/export/status").get_json()
    assert set(body) >= {"running", "collection", "mission_id", "doc_count",
                         "started_at", "finished_at", "result", "error"}
    assert body["running"] is False


# --- the mission page --------------------------------------------------------------

def _export_row(doc_id, exported_at, collection="quarry"):
    _run(storage.upsert_document_export(DocumentExport(
        document_id=doc_id, collection=collection, content_sha256="x" * 64,
        embed_model="cohere/embed-v5.0-pro", n_chunks=3, exported_at=exported_at)))


def _tele_acts(html):
    m = re.search(r'<div class="tele-cell tele-acts">(.*?)\n    </div>', html, re.S)
    assert m, "no done-state actions"
    return m.group(1)


def test_mission_page_shows_the_export_control_and_state(client):
    _seed()
    _export_row("doc01", "2026-10-31T20:14:07+00:00")
    html = client.get(f"/missions/{MID}").get_data(as_text=True)
    acts = _tele_acts(html)
    assert f'<form method="post" action="/missions/{MID}/export"' in acts
    assert "Export to corpus" in acts
    assert re.search(r'data-export-state>exported 1 of 2 → quarry · last 2026-10-31 20:14Z<', acts)
    # The brief download is "Brief .md": only one control says "Export".
    assert f'href="/missions/{MID}/brief.md"' in acts
    assert "Brief .md" in acts
    assert len(re.findall(r">\s*Export\s*<", acts)) == 0
    assert acts.count("Export to corpus") == 1


def test_mission_page_export_state_before_any_export(client):
    _seed()
    acts = _tele_acts(client.get(f"/missions/{MID}").get_data(as_text=True))
    assert re.search(r'data-export-state>exported 0 of 2 → quarry<', acts)


def test_mission_page_export_state_counts_only_this_collection(client, monkeypatch):
    _seed()
    _export_row("doc01", "2026-10-31T20:14:07+00:00", collection="elsewhere")
    _export_row("doc02", "2026-10-30T09:00:00+00:00")
    monkeypatch.setattr(config.settings, "qdrant_export_collection", "quarry")
    acts = _tele_acts(client.get(f"/missions/{MID}").get_data(as_text=True))
    assert "exported 1 of 2 → quarry · last 2026-10-30 09:00Z" in acts


def test_mission_page_shows_a_running_export(client, monkeypatch):
    _seed()
    monkeypatch.setattr(vector_export, "export_status", lambda: {
        "running": True, "collection": "quarry", "mission_id": MID, "doc_count": 2,
        "started_at": "t", "finished_at": None, "result": None, "error": None})
    acts = _tele_acts(client.get(f"/missions/{MID}").get_data(as_text=True))
    assert "exported 0 of 2 → quarry · export running" in acts
    # Another mission's export is not this one's.
    monkeypatch.setattr(vector_export, "export_status", lambda: {
        "running": True, "collection": "quarry", "mission_id": OTHER, "doc_count": 2,
        "started_at": "t", "finished_at": None, "result": None, "error": None})
    acts = _tele_acts(client.get(f"/missions/{MID}").get_data(as_text=True))
    assert "export running" not in acts


def test_api_mission_carries_the_export_state_for_the_poll(client):
    _seed()
    body = client.get(f"/api/mission/{MID}").get_json()
    assert body["export_state"] == {
        "exported": 0, "total": 2, "collection": "quarry", "latest": None,
        "running": False, "line": "exported 0 of 2 → quarry"}
    _export_row("doc01", "2026-10-31T20:14:07+00:00")
    _export_row("doc02", "2026-10-31T20:15:30+00:00")
    body = client.get(f"/api/mission/{MID}").get_json()
    assert body["export_state"] == {
        "exported": 2, "total": 2, "collection": "quarry",
        "latest": "2026-10-31T20:15:30+00:00", "running": False,
        "line": "exported 2 of 2 → quarry · last 2026-10-31 20:15Z"}
    # The page's poll handler sets the line from this field.
    html = client.get(f"/missions/{MID}").get_data(as_text=True)
    assert "s.export_state.line" in html and "[data-export-state]" in html


def test_export_state_line_escapes_a_hostile_collection_name(client, monkeypatch):
    # The Settings page validates the name, but settings.json is a file: the
    # template must autoescape whatever is there.
    _seed()
    monkeypatch.setattr(config.settings, "qdrant_export_collection", "<img src=x onerror=alert(1)>")
    html = client.get(f"/missions/{MID}").get_data(as_text=True)
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_token_rows_order_includes_rerank_and_embed():
    import app as app_mod
    assert app_mod._PURPOSE_ORDER == ("plan", "assess", "extract", "brief", "rerank", "embed")
    row = {"calls": 1, "prompt_tokens": 1, "completion_tokens": 0}
    usage = {"by_purpose": {p: dict(row) for p in
                            ("zzz", "embed", "brief", "rerank", "plan", "extract", "assess")}}
    assert [r["purpose"] for r in app_mod._token_rows(usage)] == \
        ["plan", "assess", "extract", "brief", "rerank", "embed", "zzz"]


# --- the poll never loads document bodies (review F4) --------------------------------

def test_api_mission_export_state_never_loads_document_bodies(client, monkeypatch):
    _seed()
    _export_row("doc01", "2026-10-31T20:14:07+00:00")
    import app as app_mod

    def bodies(*a, **k):
        raise AssertionError("api_mission loaded full document rows on a poll")

    monkeypatch.setattr(app_mod, "get_mission_documents", bodies)
    body = client.get(f"/api/mission/{MID}").get_json()
    assert body["export_state"]["exported"] == 1 and body["export_state"]["total"] == 2
    assert body["export_state"]["line"] == "exported 1 of 2 → quarry · last 2026-10-31 20:14Z"
