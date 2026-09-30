"""Route-level behaviour of the web layer: the login gate on every endpoint,
input bounds, the mission state machine's compare-and-set transitions, SSE,
crawl cancellation, Library pagination and form re-rendering.

Task-2 pipeline names (jobs.finish_job, brief.ordered_sources_for_mission,
scheduler.launch_scheduled_mission) are monkeypatched with raising=False where
a test depends on them, so this file is self-sufficient either way.
"""
import asyncio
import json
import re
import threading
from urllib.parse import urlsplit

import flask
import pytest

import auth
import config
import jobs
import storage
from models import Agent, Document, Mission, Requirement, SearchRecord

PW = "hunter2-quarry"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def app_mod():
    import app as app_mod
    return app_mod


@pytest.fixture()
def client(app_mod):
    # Initialise up front so the startup reconciliation (which fails any
    # mission left in an in-flight state) runs before a test seeds its rows.
    app_mod.initialize()
    return app_mod.app.test_client()


def _path(resp):
    return urlsplit(resp.headers["Location"]).path


def _flashes(client):
    with client.session_transaction() as sess:
        return [msg for _cat, msg in sess.get("_flashes", [])]


def _login(client):
    r = client.post("/login", data={"password": PW})
    assert r.status_code == 302


@pytest.fixture()
def finished(monkeypatch):
    """Stand-in for jobs.finish_job that records its calls."""
    calls = []

    def fake_finish(job_id, stage="done", error=None):
        calls.append((job_id, stage))
        jobs.update_job(job_id, stage=stage, done=True)

    monkeypatch.setattr(jobs, "finish_job", fake_finish, raising=False)
    return calls


@pytest.fixture()
def starts(app_mod, monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod, "start_collection", lambda mid: calls.append(mid))
    return calls


def _seed_agent(**kw):
    fields = dict(id="a1", name="Ada", expertise="orbital mechanics",
                  persona_prompt="p", created_at="2026-01-01T00:00:00")
    fields.update(kw)
    _run(storage.insert_agent(Agent(**fields)))


def _seed_mission(status="awaiting_approval", n_reqs=2, job_id=None,
                  mission_id="m1", with_agent=True, **req_kw):
    async def go():
        await storage.init_db()
        if with_agent:
            await storage.insert_agent(Agent(
                id="a1", name="Ada", expertise="orbital mechanics",
                persona_prompt="p", created_at="2026-01-01T00:00:00"))
        await storage.insert_mission(Mission(
            id=mission_id, agent_id="a1", question="Where is it?", status=status,
            job_id=job_id, budget_json=json.dumps({"max_sources": 7}),
            created_at="2026-01-01T00:00:00"))
        for i in range(n_reqs):
            # Requirement ids are global primary keys: prefix them for any
            # second mission a test seeds.
            rid = f"r{i}" if mission_id == "m1" else f"{mission_id}-r{i}"
            await storage.insert_requirement(Requirement(
                id=rid, mission_id=mission_id, title=f"Req {i}",
                next_queries_json=json.dumps([f"q{i}"]), **req_kw))
    _run(go())


def _mission(mid="m1"):
    return _run(storage.get_mission(mid))


def _reqs(mid="m1"):
    return {r.id: r for r in _run(storage.get_requirements_for_mission(mid))}


# --- the login gate -----------------------------------------------------

def test_every_endpoint_requires_login(client, app_mod, monkeypatch):
    monkeypatch.setattr(config.settings, "quarry_password", PW)
    checked = 0
    for rule in app_mod.app.url_map.iter_rules():
        if rule.endpoint in ("login", "static"):
            continue
        with app_mod.app.test_request_context():
            url = flask.url_for(rule.endpoint, **{a: "x" for a in rule.arguments})
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            r = client.open(url, method=method, follow_redirects=False)
            if rule.rule.startswith("/api/"):
                assert r.status_code == 401, (method, url, r.status_code)
                assert r.get_json() == {"error": "authentication required"}
            else:
                assert r.status_code == 302, (method, url, r.status_code)
                assert _path(r) == "/login", (method, url, r.headers["Location"])
            checked += 1
    assert checked >= 30, "the sweep should cover every route"


def test_settings_page_cannot_set_or_clear_the_password(client, monkeypatch):
    assert not auth.enabled()
    r = client.post("/settings", data={"quarry_password": "x", "llm_provider": "",
                                       "search_max_results": "5"})
    assert r.status_code == 302
    assert not auth.enabled()
    assert config.settings.quarry_password == ""

    monkeypatch.setattr(config.settings, "quarry_password", PW)
    _login(client)
    client.post("/settings", data={"quarry_password": "", "search_max_results": "5"})
    assert config.settings.quarry_password == PW


# --- input bounds -------------------------------------------------------

def test_search_inputs_are_bounded(client, app_mod, monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod, "create_job",
                        lambda q, n, e, p: calls.append((q, n, e, p)) or "job-x")
    monkeypatch.setattr(app_mod, "run_job_in_background", lambda jid: None)

    r = client.post("/search", data={"query": "a" * 600, "max_results": "999",
                                     "extract": "on", "extract_prompt": "p" * 6000})
    assert r.status_code == 302 and _path(r) == "/crawl/job-x"
    q, n, e, p = calls[-1]
    assert (len(q), n, e, len(p)) == (500, 20, True, 5000)

    for raw, want in (("0", 1), ("-5", 1), ("abc", 5), ("7", 7), ("21", 20)):
        client.post("/search", data={"query": "x", "max_results": raw})
        assert calls[-1][1] == want, raw


def test_full_text_query_is_bounded(client, app_mod, monkeypatch):
    seen = []

    async def fake_fts(q, search_filter=None):
        seen.append(q)
        return []

    monkeypatch.setattr(app_mod, "search_documents_fts", fake_fts)
    assert client.get("/documents?q=" + "z" * 300).status_code == 200
    assert seen == ["z" * 200]


def test_extract_prompt_is_bounded(client, monkeypatch):
    import extractor
    got = []
    monkeypatch.setattr(extractor, "extract_from_document",
                        lambda doc, prompt: got.append(prompt) or None)
    doc_id = _run(storage.upsert_document(Document(
        id="d1", url="https://x.example/1", domain="x.example", title="t",
        search_query="q", crawled_at="2026-01-01T00:00:00", content_markdown="body")))
    client.post(f"/extract/{doc_id}", data={"prompt": "p" * 6000})
    assert got and len(got[0]) == 5000


# --- initialize() -------------------------------------------------------

def test_initialize_is_idempotent_and_locked(app_mod, monkeypatch):
    calls = []

    async def counting_init():
        calls.append(1)

    async def no_stale():
        return 0

    monkeypatch.setattr(app_mod, "init_db", counting_init)
    monkeypatch.setattr(app_mod, "reconcile_interrupted_missions", no_stale)
    monkeypatch.setattr(app_mod, "start_scheduler", lambda: None)
    monkeypatch.setattr(app_mod.app, "_db_initialized", False, raising=False)

    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        app_mod.initialize()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    app_mod.initialize()
    assert calls == [1]
    assert app_mod.app._db_initialized is True


def test_first_request_runs_initialize(app_mod, monkeypatch):
    calls = []
    real = app_mod.initialize
    monkeypatch.setattr(app_mod, "initialize", lambda: (calls.append(1), real())[1])
    monkeypatch.setattr(app_mod.app, "_db_initialized", False, raising=False)
    assert app_mod.app.test_client().get("/").status_code == 200
    assert calls


# --- trusted hosts ------------------------------------------------------

def test_trusted_hosts_policy(app_mod, monkeypatch):
    base = ["localhost", "127.0.0.1", "[::1]"]
    monkeypatch.setattr(config.settings, "quarry_password", "")
    monkeypatch.setattr(config.settings, "quarry_trusted_hosts", "")
    assert app_mod._trusted_hosts() == base
    monkeypatch.setattr(config.settings, "quarry_trusted_hosts", " quarry.lan, box.ts.net ,")
    assert app_mod._trusted_hosts() == base + ["quarry.lan", "box.ts.net"]
    # Auth on: enforced only when the operator lists hosts.
    monkeypatch.setattr(config.settings, "quarry_password", PW)
    monkeypatch.setattr(config.settings, "quarry_trusted_hosts", "")
    assert app_mod._trusted_hosts() is None
    monkeypatch.setattr(config.settings, "quarry_trusted_hosts", "quarry.lan")
    assert app_mod._trusted_hosts() == base + ["quarry.lan"]


def test_default_config_enforces_loopback_hosts(app_mod):
    assert app_mod.app.config["TRUSTED_HOSTS"][:3] == ["localhost", "127.0.0.1", "[::1]"]


def test_untrusted_host_gets_400(client, app_mod, monkeypatch):
    monkeypatch.setitem(app_mod.app.config, "TRUSTED_HOSTS",
                        ["localhost", "127.0.0.1", "[::1]"])
    assert client.get("/", base_url="http://evil.example").status_code == 400
    assert client.get("/", base_url="http://localhost:5000").status_code == 200
    assert client.get("/", base_url="http://127.0.0.1:5000").status_code == 200
    # With auth on, a rebinding request is refused outright, not redirected
    # to the login page (and not a 500 from building that redirect).
    monkeypatch.setattr(config.settings, "quarry_password", PW)
    assert client.get("/", base_url="http://evil.example").status_code == 400
    assert client.get("/api/job/x", base_url="http://evil.example").status_code == 400
    r = client.post("/search", base_url="http://evil.example", data={"query": "x"})
    assert r.status_code == 400


# --- mission approval ---------------------------------------------------

def test_approve_twice_starts_one_collection(client, starts):
    _seed_mission()
    r1 = client.post("/missions/m1/approve", data={})
    assert r1.status_code == 302 and _path(r1) == "/missions/m1"
    r2 = client.post("/missions/m1/approve", data={})
    assert r2.status_code == 302 and _path(r2) == "/missions/m1"
    assert starts == ["m1"]
    assert any("not awaiting approval" in m for m in _flashes(client))
    m = _mission()
    assert m.status == "collecting"
    job = jobs.get_job(m.job_id)
    assert job is not None and not job.done
    assert job.crawl_total == 7          # max_sources from the mission budget


def test_concurrent_approvals_start_one_collection(client, app_mod, starts):
    _seed_mission()
    barrier = threading.Barrier(4)
    codes = []

    def worker():
        c = app_mod.app.test_client()
        barrier.wait()
        codes.append(c.post("/missions/m1/approve", data={}).status_code)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [302] * 4
    assert starts == ["m1"]
    assert sum(1 for j in jobs._store.values() if not j.done) == 1


def test_approve_with_every_row_dropped_changes_nothing(client, starts):
    _seed_mission()
    plan = [{"id": "r0", "title": "Req 0", "queries": ["q0"], "dropped": True},
            {"id": "r1", "title": "Req 1", "queries": ["q1"], "dropped": True}]
    r = client.post("/missions/m1/approve", data={"plan_json": json.dumps(plan)})
    assert r.status_code == 302 and _path(r) == "/missions/m1"
    assert set(_reqs()) == {"r0", "r1"}, "nothing may be deleted when nothing is approved"
    assert _mission().status == "awaiting_approval"
    assert starts == []
    assert any("at least one requirement" in m for m in _flashes(client))
    assert not jobs._store


def test_approve_drop_all_but_add_one_proceeds(client, starts):
    _seed_mission()
    plan = [{"id": "r0", "dropped": True}, {"id": "r1", "dropped": True},
            {"id": None, "title": "Brand new", "queries": [], "dropped": False}]
    client.post("/missions/m1/approve", data={"plan_json": json.dumps(plan)})
    reqs = list(_reqs().values())
    assert [r.title for r in reqs] == ["Brand new"]
    assert json.loads(reqs[0].next_queries_json) == ["Brand new"]
    assert starts == ["m1"]


def test_approve_explicit_empty_queries_fall_back_to_title(client, starts):
    _seed_mission()
    plan = [{"id": "r0", "title": "Renamed", "queries": [], "dropped": False}]
    client.post("/missions/m1/approve", data={"plan_json": json.dumps(plan)})
    reqs = _reqs()
    assert reqs["r0"].title == "Renamed"
    assert json.loads(reqs["r0"].next_queries_json) == ["Renamed"]
    assert json.loads(reqs["r1"].next_queries_json) == ["q1"]   # untouched


def test_approve_bounds_the_number_of_plan_rows(client, app_mod, starts):
    _seed_mission()
    plan = [{"id": None, "title": f"Added {i}", "queries": [], "dropped": False}
            for i in range(app_mod._MAX_PLAN_ROWS + 25)]
    client.post("/missions/m1/approve", data={"plan_json": json.dumps(plan)})
    assert len(_reqs()) == 2 + app_mod._MAX_PLAN_ROWS


def test_approve_ignores_malformed_plan_rows(client, starts):
    _seed_mission()
    plan = [["not", "a", "dict"], {"id": ["unhashable"], "title": 7, "queries": "q"},
            {"id": "r0", "title": "  Tidy  ", "queries": ["  a  ", "", 5]}]
    r = client.post("/missions/m1/approve", data={"plan_json": json.dumps(plan)})
    assert r.status_code == 302
    reqs = _reqs()
    assert set(reqs) == {"r0", "r1"}
    assert reqs["r0"].title == "Tidy"
    assert json.loads(reqs["r0"].next_queries_json) == ["a"]
    assert starts == ["m1"]


def test_approve_when_busy_reverts_to_the_gate(client, app_mod, starts, monkeypatch):
    _seed_mission()

    def full(*_a, **_k):
        raise jobs.JobLimitReached("6 jobs already running")

    monkeypatch.setattr(app_mod, "create_mission_job", full)
    r = client.post("/missions/m1/approve", data={})
    assert _path(r) == "/missions/m1"
    assert _mission().status == "awaiting_approval"
    assert starts == []
    assert any(m.startswith("Busy") for m in _flashes(client))


def test_approve_points_the_mission_at_a_fresh_job(client, starts):
    old = jobs.create_mission_job("Where is it?", 7)
    jobs.update_job(old, done=True, stage="awaiting_approval")
    _seed_mission(job_id=old)
    client.post("/missions/m1/approve", data={})
    new = _mission().job_id
    assert new and new != old
    assert jobs.get_job(new) is not None


def test_approve_unknown_mission(client, starts):
    r = client.post("/missions/nope/approve", data={})
    assert _path(r) == "/missions"
    assert starts == []


# --- re-tasking ---------------------------------------------------------

def test_retask_success(client, starts):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", status="unmet", attempts=3,
                                    assessment_missing="gap"))
    r = client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    assert _path(r) == "/missions/m1"
    req = _reqs()["r0"]
    assert (req.status, req.attempts) == ("pending", 0)
    assert json.loads(req.next_queries_json) == ["q0", "new angle"]
    m = _mission()
    assert m.status == "collecting"
    assert jobs.get_job(m.job_id) is not None
    assert starts == ["m1"]


def test_retask_when_busy_leaves_the_requirement_alone(client, app_mod, starts, monkeypatch):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", status="unmet", attempts=3))

    def full(*_a, **_k):
        raise jobs.JobLimitReached("6 jobs already running")

    monkeypatch.setattr(app_mod, "create_mission_job", full)
    client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    req = _reqs()["r0"]
    assert (req.status, req.attempts) == ("unmet", 3)
    assert json.loads(req.next_queries_json) == ["q0"]
    assert _mission().status == "done"
    assert starts == []
    assert any(m.startswith("Busy") for m in _flashes(client))


def test_retask_lost_race_releases_its_job(client, app_mod, starts, finished, monkeypatch):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", status="unmet", attempts=3))

    async def lost(*_a):
        return False

    monkeypatch.setattr(app_mod, "claim_mission_status", lost)
    client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    req = _reqs()["r0"]
    assert (req.status, req.attempts) == ("unmet", 3)
    assert starts == []
    assert len(finished) == 1 and finished[0][1] == "cancelled"
    assert not any(not j.done for j in jobs._store.values())


def test_retask_refused_while_running_or_at_the_gate(client, starts):
    for status in ("collecting", "awaiting_approval"):
        _seed_mission(status=status, mission_id=f"m-{status}", with_agent=status == "collecting")
        client.post(f"/missions/m-{status}/requirements/m-{status}-r0/retask",
                    data={"query": "x"})
        assert _mission(f"m-{status}").status == status
    assert starts == []
    assert not jobs._store


# --- delete -------------------------------------------------------------

def test_delete_at_the_gate_releases_a_live_job(client, finished):
    jid = jobs.create_mission_job("Where is it?", 7)
    _seed_mission(job_id=jid)
    r = client.post("/missions/m1/delete")
    assert _path(r) == "/missions"
    assert _mission() is None
    assert finished == [(jid, "cancelled")]


def test_delete_refused_while_running(client, finished):
    jid = jobs.create_mission_job("Where is it?", 7)
    _seed_mission(status="collecting", job_id=jid)
    r = client.post("/missions/m1/delete")
    assert _path(r) == "/missions/m1"
    assert _mission() is not None
    assert finished == []


def test_delete_finished_mission_leaves_done_job_alone(client, finished):
    jid = jobs.create_mission_job("Where is it?", 7)
    jobs.update_job(jid, done=True, stage="done")
    _seed_mission(status="done", job_id=jid)
    client.post("/missions/m1/delete")
    assert _mission() is None
    assert finished == []


# --- run the scheduled question now -------------------------------------

def test_run_scheduled_redirects_to_the_new_mission(client, monkeypatch):
    import scheduler
    _seed_agent(schedule_question="What changed overnight?", schedule_cron="0 7 * * *")
    monkeypatch.setattr(scheduler, "launch_scheduled_mission", lambda aid: "m-new",
                        raising=False)
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/missions/m-new"


def test_run_scheduled_when_busy(client, monkeypatch):
    import scheduler
    _seed_agent(schedule_question="What changed overnight?")

    def full(_aid):
        raise jobs.JobLimitReached("6 jobs already running")

    monkeypatch.setattr(scheduler, "launch_scheduled_mission", full, raising=False)
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents"
    assert any(m.startswith("Busy") for m in _flashes(client))


def test_run_scheduled_explains_why_nothing_started(client, monkeypatch):
    import scheduler
    _seed_agent(schedule_question="What changed overnight?")
    _run(storage.insert_mission(Mission(id="m-live", agent_id="a1", question="Q",
                                        status="collecting", created_at="t")))
    monkeypatch.setattr(scheduler, "launch_scheduled_mission", lambda aid: None,
                        raising=False)
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents"
    assert any("already" in m for m in _flashes(client))


def test_run_scheduled_needs_a_question(client, monkeypatch):
    import scheduler
    _seed_agent()
    called = []
    monkeypatch.setattr(scheduler, "launch_scheduled_mission",
                        lambda aid: called.append(aid), raising=False)
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents/a1/edit"
    assert called == []


# --- SSE ----------------------------------------------------------------

def _events(body):
    return [chunk for chunk in body.split("\n\n") if chunk.strip()]


def test_stream_unknown_job_is_204(client):
    r = client.get("/api/job/nope/stream")
    assert r.status_code == 204
    assert r.get_data() == b""


def test_stream_announces_a_job_that_disappears(client, app_mod, monkeypatch):
    jid = jobs.create_job("q", 5, False, "")

    class FakeTime:
        def sleep(self, _s):          # evicted between two polls
            with jobs._lock:
                jobs._store.pop(jid, None)

    monkeypatch.setattr(app_mod, "time", FakeTime())
    r = client.get(f"/api/job/{jid}/stream")
    assert r.status_code == 200
    events = _events(r.get_data(as_text=True))
    assert len(events) == 2
    assert events[0].startswith("data: ")
    assert events[1] == "event: gone\ndata: {}"


def test_stream_ignores_clock_only_changes(client, app_mod, monkeypatch):
    jid = jobs.create_job("q", 5, False, "")
    ticks = []

    class FakeTime:
        def sleep(self, _s):
            ticks.append(1)
            with jobs._lock:          # make `elapsed` differ on every poll
                jobs._store[jid].started_at -= 5
            if len(ticks) == 4:
                jobs.update_job(jid, done=True, stage="done")

    monkeypatch.setattr(app_mod, "time", FakeTime())
    body = client.get(f"/api/job/{jid}/stream").get_data(as_text=True)
    events = _events(body)
    assert len(ticks) == 4
    # One message for the initial state and one for the finish -- none for
    # the three polls where only the clock moved.
    assert len(events) == 2, events
    assert json.loads(events[-1][len("data: "):])["done"] is True


def test_api_job_passes_log_total_through(client):
    jid = jobs.create_job("q", 5, False, "")
    jobs.add_log(jid, "info", "hello")
    state = client.get(f"/api/job/{jid}").get_json()
    expected = jobs.job_state(jid)
    assert state.keys() == expected.keys()


def test_api_mission_trace_carries_log_total(client):
    jid = jobs.create_mission_job("Where is it?", 7)
    jobs.add_log(jid, "info", "one")
    jobs.add_log(jid, "info", "two")
    _seed_mission(status="awaiting_approval", job_id=jid)
    trace = client.get("/api/mission/m1").get_json()["trace"]
    assert trace["log_total"] == jobs.job_state(jid).get("log_total", 2)
    assert len(trace["log"]) == 2


# --- crawl cancel -------------------------------------------------------

def test_crawl_cancel_requests_a_stop(client):
    jid = jobs.create_job("q", 5, False, "")
    r = client.post(f"/crawl/{jid}/cancel")
    assert r.status_code == 302 and _path(r) == f"/crawl/{jid}"
    assert jobs.get_job(jid).cancel_requested is True


def test_crawl_cancel_of_a_finished_or_unknown_job(client):
    jid = jobs.create_job("q", 5, False, "")
    jobs.update_job(jid, done=True, stage="done")
    client.post(f"/crawl/{jid}/cancel")
    assert jobs.get_job(jid).cancel_requested is False
    assert any("not running" in m for m in _flashes(client))
    r = client.post("/crawl/nope/cancel")
    assert r.status_code == 302


def test_crawl_page_cancel_is_a_post_form(client):
    jid = jobs.create_job("q", 5, False, "")
    html = client.get(f"/crawl/{jid}").get_data(as_text=True)
    assert re.search(rf'<form[^>]*method="post"[^>]*action="/crawl/{jid}/cancel"', html)


def test_crawl_page_extract_label_is_json_encoded(client, monkeypatch):
    monkeypatch.setattr(config.settings, "llm_provider_fast", "ollama_chat/it's-a-model")
    jid = jobs.create_job("q", 5, True, "")
    html = client.get(f"/crawl/{jid}").get_data(as_text=True)
    assert "const EXTRACT_MODEL = " in html
    assert "'it's-a-model'" not in html


# --- Library ------------------------------------------------------------

def _seed_docs(n, query="alpha", start=0):
    async def go():
        for i in range(start, start + n):
            await storage.upsert_document(Document(
                id=f"d{i:04d}", url=f"https://site{i % 7}.example/p{i}",
                domain=f"site{i % 7}.example", title=f"Doc {i}", search_query=query,
                crawled_at=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                content_markdown="An ordinary sentence about the topic that reads like prose. " * 3,
                word_count=100))
    _run(go())


def _cards(html):
    return html.count('class="doc-card"')


def test_library_is_paginated(client, app_mod):
    _seed_docs(130)
    html = client.get("/documents").get_data(as_text=True)
    assert _cards(html) == app_mod.PAGE_SIZE == 60
    assert "showing 1–60 of 130" in html
    assert "page=2" in html

    html = client.get("/documents?page=3").get_data(as_text=True)
    assert _cards(html) == 10
    assert "showing 121–130 of 130" in html
    assert "page=2" in html

    # Out of range clamps to the last page; junk falls back to the first.
    assert _cards(client.get("/documents?page=99").get_data(as_text=True)) == 10
    assert _cards(client.get("/documents?page=abc").get_data(as_text=True)) == 60
    assert _cards(client.get("/documents?page=-4").get_data(as_text=True)) == 60


def test_library_pages_do_not_overlap(client):
    _seed_docs(130)
    seen = []
    for page in (1, 2, 3):
        html = client.get(f"/documents?page={page}").get_data(as_text=True)
        seen += re.findall(r'href="/document/(d\d{4})" class="doc-card"', html)
    assert len(seen) == 130 and len(set(seen)) == 130


def test_search_filter_is_paginated_with_a_total(client):
    _seed_docs(70, query="alpha")
    _seed_docs(5, query="beta", start=500)
    html = client.get("/documents?search=alpha").get_data(as_text=True)
    assert _cards(html) == 60
    assert "showing 1–60 of 70" in html
    assert "search=alpha" in html and "page=2" in html
    html = client.get("/documents?search=alpha&page=2").get_data(as_text=True)
    assert _cards(html) == 10


def test_small_library_has_no_pager(client):
    _seed_docs(3)
    html = client.get("/documents").get_data(as_text=True)
    assert _cards(html) == 3
    assert "showing" not in html


def test_full_text_query_does_not_seed_the_title_filter(client):
    _seed_docs(3)
    html = client.get("/documents?q=ordinary").get_data(as_text=True)
    tag = re.search(r'<input[^>]*id="libSearch"[^>]*>', html).group(0)
    assert "ordinary" not in tag
    assert _cards(html) == 3


def test_library_cards_use_the_preview_snippet(client):
    _seed_docs(1)
    html = client.get("/documents").get_data(as_text=True)
    assert "An ordinary sentence about the topic" in html


# --- history ------------------------------------------------------------

def test_history_rerun_uses_the_default_result_count(client, monkeypatch):
    monkeypatch.setattr(config.settings, "search_max_results", 5)
    _run(storage.insert_search(SearchRecord(
        id="s1", query="solar max", executed_at="2026-01-01T10:00:00", result_count=17)))
    html = client.get("/history").get_data(as_text=True)
    assert 'name="max_results" value="5"' in html
    assert 'name="max_results" value="17"' not in html


# --- agent form ---------------------------------------------------------

def test_agent_new_validation_rerenders_with_values(client):
    r = client.post("/agents/new", data={"name": "Keep Me", "expertise": "",
                                         "schedule_question": "Standing?",
                                         "max_sources": "42"})
    assert r.status_code == 400
    html = r.get_data(as_text=True)
    assert 'value="Keep Me"' in html
    assert 'value="Standing?"' in html
    assert 'value="42"' in html
    assert "Name and area of expertise are required." in html
    assert 'action="/agents/new"' in html
    assert _run(storage.list_agents()) == []


def test_agent_new_bad_cron_rerenders(client):
    r = client.post("/agents/new", data={"name": "N", "expertise": "E",
                                         "schedule_cron": "not a cron"})
    assert r.status_code == 400
    html = r.get_data(as_text=True)
    assert 'value="not a cron"' in html
    assert "a valid cron expression" in html
    assert _run(storage.list_agents()) == []


def test_agent_edit_validation_rerenders_with_values(client):
    _seed_agent()
    r = client.post("/agents/a1/edit", data={"name": "", "expertise": "new field"})
    assert r.status_code == 400
    html = r.get_data(as_text=True)
    assert 'value="new field"' in html
    assert 'action="/agents/a1/edit"' in html
    assert _run(storage.get_agent("a1")).expertise == "orbital mechanics"


# --- mission page -------------------------------------------------------

def _seed_briefed_mission(with_agent=True):
    _seed_mission(status="done", with_agent=with_agent)

    async def go():
        for i in (1, 2):
            doc_id = await storage.upsert_document(Document(
                id=f"doc{i}", url=f"https://s{i}.example/", domain=f"s{i}.example",
                title=f"Source {i}", search_query="Where is it?",
                crawled_at=f"2026-01-01T00:00:0{i}", content_markdown="body text " * 50,
                word_count=100))
            await storage.link_mission_document("m1", "r0", doc_id)
        await storage.update_mission("m1", brief_markdown="See [1] and [2].",
                                     finished_at="2026-01-01T01:00:00")
    _run(go())


def test_mission_page_numbers_sources_by_the_stored_order(client, monkeypatch):
    import brief
    _seed_briefed_mission()
    calls = []

    def fake_order(mission, docs):
        calls.append(mission.id)
        return sorted(docs, key=lambda d: d.id, reverse=True)

    monkeypatch.setattr(brief, "ordered_sources_for_mission", fake_order, raising=False)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert calls == ["m1"]
    rail = re.findall(r'data-src="(\d+)" href="/document/(doc\d)"', html)
    assert rail == [("1", "doc2"), ("2", "doc1")]
    assert 'class="cite" data-cite="1"' in html


def test_mission_page_rerun_needs_the_agent(client, monkeypatch):
    import brief
    monkeypatch.setattr(brief, "ordered_sources_for_mission",
                        lambda m, docs: list(docs), raising=False)
    _seed_briefed_mission(with_agent=False)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "/agents/a1/run" not in html
    _run(storage.insert_agent(Agent(id="a1", name="Ada", expertise="x",
                                    persona_prompt="p", created_at="t")))
    html = client.get("/missions/m1").get_data(as_text=True)
    assert 'action="/agents/a1/run"' in html


def test_gate_offers_discard(client, monkeypatch):
    import brief
    monkeypatch.setattr(brief, "ordered_sources_for_mission",
                        lambda m, docs: list(docs), raising=False)
    _seed_mission()
    html = client.get("/missions/m1").get_data(as_text=True)
    assert 'id="discardForm"' in html
    assert 'action="/missions/m1/delete"' in html
    assert 'form="discardForm"' in html


# --- templates use the log cursor ----------------------------------------

@pytest.mark.parametrize("name", ["crawl.html", "mission.html"])
def test_log_rendering_uses_the_monotonic_cursor(name):
    with open(f"templates/{name}", encoding="utf-8") as f:
        src = f.read()
    assert "log_total" in src
    assert "renderedLogs = " in src
