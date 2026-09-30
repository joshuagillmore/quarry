"""Route-level behaviour of the web layer: the login gate on every endpoint,
input bounds, the mission state machine's compare-and-set transitions, SSE,
crawl cancellation, Library pagination and form re-rendering.

The pipeline pieces the routes call (jobs.finish_job,
brief.ordered_sources_for_mission, scheduler.launch_scheduled_mission) run for
real; only the worker-thread launchers (start_collection, start_planning) are
replaced, so nothing outlives a test and writes into the next test's DB.
"""
import asyncio
import json
import re
import threading
import warnings
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


def _live_jobs():
    return [j for j in jobs._store.values() if not j.done]


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


def test_run_async_is_warning_free_with_or_without_a_running_loop(app_mod):
    async def answer():
        return 42

    async def from_inside_a_loop():
        return app_mod.run_async(answer())

    # A fresh policy: get_event_loop() warns only on the first loop-less call
    # per policy, and an earlier test in the process may have spent that.
    asyncio.set_event_loop_policy(None)
    results = []
    with warnings.catch_warnings():
        # The old get_event_loop() path warned here ("There is no current
        # event loop"); the suite output must stay warning-free.
        warnings.simplefilter("error")
        results.append(app_mod.run_async(answer()))          # no loop (main thread)
        results.append(asyncio.run(from_inside_a_loop()))     # a loop is running
        t = threading.Thread(target=lambda: results.append(app_mod.run_async(answer())))
        t.start()
        t.join()                                               # a request thread
    assert results == [42, 42, 42]


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


def test_approve_db_error_after_the_claim_reopens_the_gate(client, app_mod, starts, monkeypatch):
    # e.g. SQLite "database is locked" while collection workers write.
    import sqlite3
    _seed_mission()

    async def locked(_mission_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app_mod, "get_requirements_for_mission", locked)
    r = client.post("/missions/m1/approve", data={})
    assert r.status_code == 500
    assert _mission().status == "awaiting_approval"   # not stranded in `collecting`
    assert starts == []
    assert _live_jobs() == []


def test_approve_worker_start_failure_rolls_back(client, app_mod, monkeypatch):
    gate_job = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(gate_job, stage="awaiting_approval")
    _seed_mission(job_id=gate_job)

    def cannot_start(_mission_id):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(app_mod, "start_collection", cannot_start)
    r = client.post("/missions/m1/approve", data={})
    assert r.status_code == 500
    m = _mission()
    assert m.status == "awaiting_approval"
    assert m.job_id == gate_job                       # back on the planning trace
    assert _live_jobs() == []                         # the minted job was released
    [minted] = [j for j in jobs._store.values() if j.id != gate_job]
    assert minted.stage == "error"


def test_approve_deeply_nested_plan_json_is_not_a_500(client, starts):
    # json.loads raises RecursionError (not JSONDecodeError) on this, and it
    # is well under the form size limit.
    _seed_mission()
    r = client.post("/missions/m1/approve", data={"plan_json": "[" * 100000})
    assert r.status_code == 302 and _path(r) == "/missions/m1"
    # Unreadable edits are treated like JS-off: the plan is approved as drafted.
    assert _mission().status == "collecting"
    assert starts == ["m1"]
    assert set(_reqs()) == {"r0", "r1"}


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


def test_retask_lost_race_releases_its_job(client, app_mod, starts, monkeypatch):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", status="unmet", attempts=3))

    async def lost(*_a):
        return False

    monkeypatch.setattr(app_mod, "claim_mission_status", lost)
    client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    req = _reqs()["r0"]
    assert (req.status, req.attempts) == ("unmet", 3)
    assert starts == []
    # The job it minted first was finished for real, so it holds no slot.
    [job] = jobs._store.values()
    assert job.done and job.stage == "cancelled" and job.finished_at
    assert _live_jobs() == []


def test_retask_worker_start_failure_rolls_everything_back(client, app_mod, monkeypatch):
    old = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(old, stage="done")
    _seed_mission(status="error", job_id=old)
    _run(storage.update_mission("m1", error="earlier failure"))
    _run(storage.update_requirement("r0", status="unmet", attempts=3,
                                    assessment_missing="gap", assessment_confidence="low"))

    def cannot_start(_mission_id):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(app_mod, "start_collection", cannot_start)
    r = client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    assert r.status_code == 500
    m = _mission()
    assert (m.status, m.job_id, m.error) == ("error", old, "earlier failure")
    req = _reqs()["r0"]
    assert (req.status, req.attempts, req.assessment_missing, req.assessment_confidence) == \
        ("unmet", 3, "gap", "low")
    assert json.loads(req.next_queries_json) == ["q0"]
    assert _live_jobs() == []


def test_retask_db_error_on_the_claim_releases_its_job(client, app_mod, starts, monkeypatch):
    import sqlite3
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", status="unmet", attempts=3))

    async def locked(*_a):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app_mod, "claim_mission_status", locked)
    r = client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    assert r.status_code == 500
    assert _mission().status == "done"
    assert (_reqs()["r0"].status, _reqs()["r0"].attempts) == ("unmet", 3)
    assert starts == []
    assert _live_jobs() == []


def test_retask_refused_while_running_or_at_the_gate(client, starts):
    for status in ("collecting", "awaiting_approval"):
        _seed_mission(status=status, mission_id=f"m-{status}", with_agent=status == "collecting")
        client.post(f"/missions/m-{status}/requirements/m-{status}-r0/retask",
                    data={"query": "x"})
        assert _mission(f"m-{status}").status == status
    assert starts == []
    assert not jobs._store


# --- delete -------------------------------------------------------------

def test_delete_at_the_gate_releases_a_live_job(client):
    jid = jobs.create_mission_job("Where is it?", 7)
    _seed_mission(job_id=jid)
    r = client.post("/missions/m1/delete")
    assert _path(r) == "/missions"
    assert _mission() is None
    job = jobs.get_job(jid)
    assert job.done and job.stage == "cancelled" and job.finished_at
    assert _live_jobs() == []


def test_delete_refused_while_running(client):
    jid = jobs.create_mission_job("Where is it?", 7)
    _seed_mission(status="collecting", job_id=jid)
    r = client.post("/missions/m1/delete")
    assert _path(r) == "/missions/m1"
    assert _mission() is not None
    assert not jobs.get_job(jid).done


def test_delete_finished_mission_leaves_done_job_alone(client):
    jid = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(jid, stage="done")
    stamped = jobs.get_job(jid).finished_at
    _seed_mission(status="done", job_id=jid)
    client.post("/missions/m1/delete")
    assert _mission() is None
    job = jobs.get_job(jid)
    assert (job.stage, job.finished_at) == ("done", stamped)


# --- run the scheduled question now -------------------------------------

@pytest.fixture()
def plannings(monkeypatch):
    """The real scheduler.launch_scheduled_mission runs; only the planning
    worker thread it would start is replaced."""
    import agent_runner
    calls = []
    monkeypatch.setattr(agent_runner, "start_planning", lambda mid: calls.append(mid))
    return calls


def test_run_scheduled_redirects_to_the_new_mission(client, plannings):
    _seed_agent(schedule_question="What changed overnight?", schedule_cron="0 7 * * *")
    r = client.post("/agents/a1/run-scheduled")
    [mission] = _run(storage.list_missions())
    assert _path(r) == f"/missions/{mission.id}"
    assert mission.question == "What changed overnight?"
    assert json.loads(mission.budget_json)["auto_approve"] is True
    assert plannings == [mission.id]


def test_run_scheduled_when_busy(client, plannings):
    _seed_agent(schedule_question="What changed overnight?")
    for _ in range(jobs.MAX_ACTIVE_JOBS):
        jobs.create_job("filler", 5, False, "")
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents"
    assert any(m.startswith("Busy") for m in _flashes(client))
    assert _run(storage.list_missions()) == []
    assert plannings == []


def test_run_scheduled_explains_an_active_mission(client, plannings):
    _seed_agent(schedule_question="What changed overnight?")
    _run(storage.insert_mission(Mission(id="m-live", agent_id="a1", question="Q",
                                        status="collecting", created_at="t")))
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents"
    assert any("already has a mission running" in m for m in _flashes(client))
    assert plannings == []


def test_run_scheduled_explains_an_inactive_agent(client, plannings):
    _seed_agent(schedule_question="What changed overnight?", active=0)
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents"
    assert any("inactive" in m for m in _flashes(client))
    assert plannings == []


def test_run_scheduled_needs_a_question(client, plannings):
    _seed_agent()
    r = client.post("/agents/a1/run-scheduled")
    assert _path(r) == "/agents/a1/edit"
    assert _run(storage.list_missions()) == []
    assert plannings == []


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


def test_stream_sends_keepalives_while_idle(client, app_mod, monkeypatch):
    # With clock-only changes suppressed, a long extraction would otherwise
    # leave the stream silent long enough for a proxy idle timeout to cut it.
    jid = jobs.create_job("q", 5, False, "")
    every = app_mod._SSE_KEEPALIVE_POLLS
    ticks = []

    class FakeTime:
        def sleep(self, _s):
            ticks.append(1)
            if len(ticks) == 2 * every + 10:
                jobs.finish_job(jid)

    monkeypatch.setattr(app_mod, "time", FakeTime())
    events = _events(client.get(f"/api/job/{jid}/stream").get_data(as_text=True))
    assert every * 0.25 <= 15                          # about every 15 s
    assert events.count(": keepalive") == 2
    assert events[0].startswith("data: ") and events[-1].startswith("data: ")
    assert json.loads(events[-1][len("data: "):])["done"] is True


def test_api_job_passes_log_total_through(client):
    jid = jobs.create_job("q", 5, False, "")
    for i in range(250):
        jobs.add_log(jid, "info", f"line {i}")
    state = client.get(f"/api/job/{jid}").get_json()
    # `log` is the last-200 window; `log_total` keeps counting past it.
    assert state["log_total"] == 250
    assert len(state["log"]) == 200
    assert state["log"][-1]["msg"] == "line 249"
    assert state.keys() == jobs.job_state(jid).keys()


def test_api_mission_trace_carries_log_total(client):
    jid = jobs.create_mission_job("Where is it?", 7)
    for i in range(230):
        jobs.add_log(jid, "info", f"line {i}")
    _seed_mission(status="awaiting_approval", job_id=jid)
    trace = client.get("/api/mission/m1").get_json()["trace"]
    assert trace["log_total"] == 230
    assert len(trace["log"]) == 200
    assert trace["log"][0]["msg"] == "line 30"


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
    # Jinja's tojson escapes the quote, so the id cannot close a JS string.
    assert 'const EXTRACT_MODEL = "it\\u0027s-a-model";' in html
    # No fast tier: the label names the reasoning model the extractor falls
    # back to, not the sidebar's "—".
    monkeypatch.setattr(config.settings, "llm_provider_fast", "")
    monkeypatch.setattr(config.settings, "llm_provider", "cohere/command-a-03-2025")
    html = client.get(f"/crawl/{jid}").get_data(as_text=True)
    assert 'const EXTRACT_MODEL = "command-a-03-2025";' in html


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

def _seed_briefed_mission(with_agent=True, n_docs=2, stored_order=None,
                          brief_md="See [1] and [2]."):
    _seed_mission(status="done", with_agent=with_agent)

    async def go():
        for i in range(1, n_docs + 1):
            doc_id = await storage.upsert_document(Document(
                id=f"doc{i:02d}", url=f"https://s{i}.example/", domain=f"s{i}.example",
                title=f"Source {i}", search_query="Where is it?",
                crawled_at=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                content_markdown="An ordinary sentence about the topic. " * 60,
                word_count=360))
            await storage.link_mission_document("m1", "r0", doc_id)
        fields = dict(brief_markdown=brief_md, finished_at="2026-01-01T01:00:00")
        if stored_order is not None:
            fields["brief_sources_json"] = json.dumps(stored_order)
        await storage.update_mission("m1", **fields)
    _run(go())


def _rail(html):
    return re.findall(r'data-src="(\d+)" href="/document/(doc\d+)"', html)


def test_mission_page_numbers_sources_by_the_stored_order(client):
    _seed_briefed_mission(stored_order=["doc02", "doc01"])
    html = client.get("/missions/m1").get_data(as_text=True)
    assert _rail(html) == [("1", "doc02"), ("2", "doc01")]
    assert 'class="cite" data-cite="1"' in html
    assert 'class="cite" data-cite="2"' in html


def test_mission_rail_is_capped_like_the_brief(client, app_mod):
    import brief
    cited = ["doc25", "doc24", "doc23"]
    # "gone" was cited once but its document no longer exists: skipped, so
    # it neither shifts numbering nor widens the citation bound.
    _seed_briefed_mission(n_docs=25, stored_order=cited + ["gone"],
                          brief_md="See [1], [3], [4] and [21].")
    html = client.get("/missions/m1").get_data(as_text=True)
    rail = _rail(html)
    # Every uncited doc is appended after the stored order, but numbering
    # stops where a brief's citations can reach.
    assert len(rail) == brief.MAX_BRIEF_SOURCES
    assert [d for _n, d in rail[:3]] == cited
    assert [n for n, _d in rail] == [str(i) for i in range(1, brief.MAX_BRIEF_SOURCES + 1)]
    assert 'data-cite="1"' in html and 'data-cite="3"' in html
    # The brief cited 3 documents: [4] is rail entry 4, an uncited document a
    # later retask added, so it must stay plain text rather than link there.
    assert 'data-cite="4"' not in html and "[4]" in html
    assert 'data-cite="21"' not in html and "[21]" in html


def test_mission_page_without_a_stored_order_links_every_numbered_source(client):
    _seed_briefed_mission(n_docs=3, brief_md="See [1], [3] and [4].")
    html = client.get("/missions/m1").get_data(as_text=True)
    assert len(_rail(html)) == 3
    assert 'data-cite="1"' in html and 'data-cite="3"' in html
    assert 'data-cite="4"' not in html and "[4]" in html


def test_mission_page_rerun_needs_the_agent(client):
    _seed_briefed_mission(with_agent=False)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "/agents/a1/run" not in html
    _run(storage.insert_agent(Agent(id="a1", name="Ada", expertise="x",
                                    persona_prompt="p", created_at="t")))
    html = client.get("/missions/m1").get_data(as_text=True)
    assert 'action="/agents/a1/run"' in html


def test_gate_offers_discard(client):
    _seed_mission()
    html = client.get("/missions/m1").get_data(as_text=True)
    assert 'id="discardForm"' in html
    assert 'action="/missions/m1/delete"' in html
    assert 'form="discardForm"' in html


# --- templates use the log cursor ----------------------------------------

@pytest.mark.parametrize("name", ["crawl.html", "mission.html"])
def test_log_cursor_template_strings_present(name):
    """A source-text check of the template, not an execution of its JS: the
    log renderer must key its cursor off log_total."""
    with open(f"templates/{name}", encoding="utf-8") as f:
        src = f.read()
    assert "log_total" in src
    assert "renderedLogs = " in src
