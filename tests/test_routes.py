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
    """Records (mission_id, the mission's job_id in the DB at that moment,
    the job_id handed to the worker) for each start_collection call: the
    worker reads the mission row as soon as it starts, so the fresh job must
    already be on it, and the route passes that same job so the worker
    finishes it on every exit path."""
    calls = []
    monkeypatch.setattr(app_mod, "start_collection",
                        lambda mid, job_id=None: calls.append((mid, _mission(mid).job_id, job_id)))
    return calls


def _started(starts):
    return [mid for mid, _db_job_id, _passed_job_id in starts]


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
    swept = set()
    for rule in app_mod.app.url_map.iter_rules():
        if rule.endpoint in ("login", "static"):
            continue
        swept.add(rule.rule)
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
    assert checked >= 32, "the sweep should cover every route"
    assert "/missions/<mission_id>/compare" in swept
    assert "/missions/<mission_id>/resume" in swept


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
    assert _started(starts) == ["m1"]
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
    assert _started(starts) == ["m1"]
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
    assert _started(starts) == ["m1"]


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
    assert _started(starts) == ["m1"]


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

    def cannot_start(_mission_id, _job_id=None):
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
    assert _started(starts) == ["m1"]
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
    # The worker was started with the mission already pointing at the fresh,
    # live job, not the finished planning trace, and was handed that job.
    assert starts == [("m1", new, new)]
    job = jobs.get_job(new)
    assert job is not None and not job.done


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
    # The worker was started with the mission already pointing at the fresh,
    # live job, and was handed that job.
    assert m.job_id and starts == [("m1", m.job_id, m.job_id)]
    job = jobs.get_job(m.job_id)
    assert job is not None and not job.done


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

    def cannot_start(_mission_id, _job_id=None):
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


def _spend_tokens(mission_id, prompt, completion=0):
    from models import LlmCall
    _run(storage.insert_llm_call(LlmCall(
        purpose="assess", tier="reasoning", model="m", mission_id=mission_id,
        prompt_tokens=prompt, completion_tokens=completion)))


def _budget_mission(max_llm_tokens=None):
    _seed_mission(status="done")
    budget = {"max_sources": 7}
    if max_llm_tokens is not None:
        budget["max_llm_tokens"] = max_llm_tokens
    _run(storage.update_mission("m1", budget_json=json.dumps(budget)))
    _run(storage.update_requirement("r0", status="unmet", attempts=3,
                                    assessment_missing="not attempted: token budget reached"))


@pytest.mark.parametrize("own_budget, setting", [(1203, 0), (None, 1203)],
                         ids=["mission-budget", "setting-default"])
def test_retask_refused_once_the_token_budget_is_spent(client, starts, monkeypatch,
                                                       own_budget, setting):
    """A budget-stopped mission is over budget by definition: a retask could
    not collect and would still re-extract and re-write the brief. It is
    refused with nothing changed, not even a job created."""
    monkeypatch.setattr(config.settings, "max_llm_tokens", setting)
    _budget_mission(own_budget)
    _spend_tokens("m1", 1200, 3)   # exactly the budget: nothing is left to spend
    r = client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    assert _path(r) == "/missions/m1"
    req = _reqs()["r0"]
    assert (req.status, req.attempts) == ("unmet", 3)
    assert req.assessment_missing == "not attempted: token budget reached"
    assert json.loads(req.next_queries_json) == ["q0"]
    assert _mission().status == "done"
    assert starts == [] and not jobs._store
    assert _flashes(client) == [
        "This mission has used 1,203 of 1,203 tokens; raise the budget before re-tasking."]


@pytest.mark.parametrize("own_budget, used", [(100, 99), (0, 10 ** 6)],
                         ids=["under-budget", "unlimited"])
def test_retask_allowed_while_the_token_budget_has_room(client, starts, own_budget, used):
    _budget_mission(own_budget)
    _spend_tokens("m1", used)
    client.post("/missions/m1/requirements/r0/retask", data={"query": "new angle"})
    assert _reqs()["r0"].status == "pending"
    assert _mission().status == "collecting" and len(starts) == 1


# --- resume after a limit -----------------------------------------------

_STOPPED_BUDGET = {"max_sources": 7, "max_passes": 2, "per_req_attempts": 3,
                   "max_llm_tokens": 1000, "extract": True}


def _seed_stopped(stop_reason="token_budget", budget=None, status="done"):
    """A finished mission whose collection stopped on a limit: r0 satisfied,
    r1 never reached, r2 capped out after every attempt (3 of 3), r3 still
    pending."""
    _seed_mission(status=status, n_reqs=4)
    _run(storage.update_mission(
        "m1", stop_reason=stop_reason, brief_markdown="old brief",
        budget_json=json.dumps(_STOPPED_BUDGET if budget is None else budget)))
    _run(storage.update_requirement("r0", status="satisfied", attempts=1,
                                    assessment_missing="", assessment_confidence="high"))
    _run(storage.update_requirement("r1", status="unmet", attempts=0,
                                    assessment_missing="not attempted: token budget reached"))
    _run(storage.update_requirement("r2", status="unmet", attempts=3,
                                    assessment_missing="no primary source",
                                    assessment_confidence="low"))


def _snapshot():
    return _mission().model_dump(), {k: r.model_dump() for k, r in _reqs().items()}


def _budget():
    return json.loads(_mission().budget_json)


def test_resume_token_stopped_mission(client, starts):
    old = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(old, stage="done")
    _seed_stopped()
    _run(storage.update_mission("m1", job_id=old, error="stale"))
    _spend_tokens("m1", 1200)
    before = _reqs()

    r = client.post("/missions/m1/resume", data={"extra": "500"})
    assert _path(r) == "/missions/m1"

    reqs = _reqs()
    # Exactly the never-reached and still-pending requirements reopen, with
    # their assessment cleared and their attempts kept.
    for rid in ("r1", "r3"):
        assert reqs[rid].status == "pending"
        assert not reqs[rid].assessment_missing and not reqs[rid].assessment_confidence
        assert reqs[rid].attempts == before[rid].attempts
    # Satisfied and capped-out requirements are untouched.
    assert reqs["r0"] == before["r0"] and reqs["r2"] == before["r2"]

    m = _mission()
    assert m.status == "collecting"
    assert (m.resume_count, m.stop_reason, m.error) == (1, None, None)
    assert _budget() == {**_STOPPED_BUDGET, "max_llm_tokens": 1500}
    # A fresh live job, already on the row when the worker starts, and
    # handed to it.
    assert m.job_id and m.job_id != old
    assert starts == [("m1", m.job_id, m.job_id)]
    job = jobs.get_job(m.job_id)
    assert job is not None and not job.done
    assert _flashes(client) == [
        "Resuming — 2 requirements reopened, token budget raised to 1,500."]


@pytest.mark.parametrize("stop_reason, key, extra, want, phrase", [
    ("source_budget", "max_sources", "5", 12, "source budget raised to 12"),
    ("pass_budget", "max_passes", "3", 5, "pass budget raised to 5"),
])
def test_resume_raises_the_limit_that_stopped_it(client, starts, stop_reason, key,
                                                 extra, want, phrase):
    _seed_stopped(stop_reason)
    client.post("/missions/m1/resume", data={"extra": extra})
    assert _budget() == {**_STOPPED_BUDGET, key: want}
    assert _mission().status == "collecting" and len(starts) == 1
    assert _flashes(client) == [f"Resuming — 2 requirements reopened, {phrase}."]


def test_resume_job_is_sized_by_the_raised_source_budget(client, starts):
    _seed_stopped("source_budget")
    client.post("/missions/m1/resume", data={"extra": "5"})
    assert jobs.get_job(_mission().job_id).crawl_total == 12


def test_resume_after_a_user_stop_raises_nothing(client, starts):
    _seed_stopped("user_stop")
    client.post("/missions/m1/resume", data={"extra": "99"})
    assert _budget() == _STOPPED_BUDGET
    m = _mission()
    assert (m.status, m.resume_count) == ("collecting", 1)
    assert len(starts) == 1
    assert _flashes(client) == ["Resuming — 2 requirements reopened."]


def test_resume_leaves_an_unlimited_token_budget_unlimited(client, starts, monkeypatch):
    monkeypatch.setattr(config.settings, "max_llm_tokens", 0)
    _seed_stopped(budget={**_STOPPED_BUDGET, "max_llm_tokens": 0})
    client.post("/missions/m1/resume", data={"extra": "500"})
    assert _budget()["max_llm_tokens"] == 0
    assert _mission().status == "collecting" and len(starts) == 1
    assert _flashes(client) == ["Resuming — 2 requirements reopened."]


def test_resume_raises_the_setting_default_when_the_mission_has_no_own_budget(
        client, starts, monkeypatch):
    monkeypatch.setattr(config.settings, "max_llm_tokens", 9000)
    _seed_stopped(budget={k: v for k, v in _STOPPED_BUDGET.items() if k != "max_llm_tokens"})
    client.post("/missions/m1/resume", data={"extra": "1000"})
    assert _budget()["max_llm_tokens"] == 10000


def test_resume_counts_every_resume(client, starts):
    _seed_stopped()
    _run(storage.update_mission("m1", resume_count=2))
    client.post("/missions/m1/resume", data={"extra": "1"})
    assert _mission().resume_count == 3


@pytest.mark.parametrize("stop_reason, raw, want", [
    ("token_budget", "99999999", 1000 + 5_000_000),
    ("token_budget", "0", 1001),
    ("token_budget", "-40", 1001),
    ("token_budget", "lots", 2000),          # unreadable: the original limit again
    ("token_budget", "", 2000),
    ("source_budget", "500", 7 + 100),
    ("source_budget", "0", 8),
    ("pass_budget", "50", 2 + 10),
    ("pass_budget", "-1", 3),
], ids=["tokens-high", "tokens-zero", "tokens-negative", "tokens-junk", "tokens-blank",
        "sources-high", "sources-zero", "passes-high", "passes-negative"])
def test_resume_extra_is_clamped(client, starts, stop_reason, raw, want):
    _seed_stopped(stop_reason)
    client.post("/missions/m1/resume", data={"extra": raw})
    key = {"token_budget": "max_llm_tokens", "source_budget": "max_sources",
           "pass_budget": "max_passes"}[stop_reason]
    assert _budget()[key] == want


@pytest.mark.parametrize("status, stop_reason", [
    ("done", "complete"), ("done", None), ("error", "token_budget"),
    ("collecting", "token_budget"), ("awaiting_approval", None),
])
def test_resume_refused_for_a_mission_that_did_not_stop_on_a_limit(
        client, starts, status, stop_reason):
    _seed_stopped(stop_reason, status=status)
    before = _snapshot()
    r = client.post("/missions/m1/resume", data={"extra": "500"})
    assert _path(r) == "/missions/m1"
    assert _snapshot() == before
    assert starts == [] and not jobs._store
    assert _flashes(client) == ["This mission has nothing to resume."]


def test_resume_after_a_pass_budget_reopens_requirements_with_attempts_left(
        client, starts):
    """A pass-budget stop tries every requirement in its first pass, so none
    is "not attempted": the ones still open were marked unmet with their
    last gap while they had attempts left. Resume reopens exactly those,
    attempts kept, so each only gets its remaining tries."""
    _seed_stopped("pass_budget")
    _run(storage.update_requirement("r1", status="unmet", attempts=1,
                                    assessment_missing="gap after one pass",
                                    assessment_confidence="low"))
    _run(storage.update_requirement("r3", status="unmet", attempts=2,
                                    assessment_missing="gap after two passes",
                                    assessment_confidence="medium"))
    before = _reqs()
    assert client.get("/api/mission/m1").get_json()["resumable"] is True
    assert _resume_form(client.get("/missions/m1").get_data(as_text=True)) is not None

    client.post("/missions/m1/resume", data={"extra": "2"})

    reqs = _reqs()
    for rid, attempts in (("r1", 1), ("r3", 2)):
        assert (reqs[rid].status, reqs[rid].attempts) == ("pending", attempts)
        assert not reqs[rid].assessment_missing and not reqs[rid].assessment_confidence
    assert reqs["r0"] == before["r0"] and reqs["r2"] == before["r2"]
    assert _budget()["max_passes"] == 4
    assert _mission().status == "collecting" and len(starts) == 1
    assert _flashes(client) == [
        "Resuming — 2 requirements reopened, pass budget raised to 4."]


def test_resume_leaves_a_capped_requirement_alone(client, starts):
    _seed_stopped()
    _run(storage.update_requirement("r3", status="unmet", attempts=3,   # 3 of 3
                                    assessment_missing="still nothing",
                                    assessment_confidence="low"))
    before = _reqs()
    client.post("/missions/m1/resume", data={"extra": "500"})
    reqs = _reqs()
    assert reqs["r1"].status == "pending"
    assert reqs["r2"] == before["r2"] and reqs["r3"] == before["r3"]
    assert _flashes(client) == [
        "Resuming — 1 requirement reopened, token budget raised to 1,500."]


def test_resume_attempt_cap_falls_back_to_the_agent_default(client, starts):
    """A budget without per_req_attempts is capped at the agent's default,
    as the runner reads it."""
    _seed_stopped(budget={k: v for k, v in _STOPPED_BUDGET.items()
                          if k != "per_req_attempts"})
    _run(storage.update_agent("a1", default_per_req_attempts=2))
    _run(storage.update_requirement("r1", status="unmet", attempts=1,
                                    assessment_missing="gap"))     # 1 of 2: open
    _run(storage.update_requirement("r3", status="unmet", attempts=2,
                                    assessment_missing="gap"))     # 2 of 2: capped
    before = _reqs()
    assert client.get("/api/mission/m1").get_json()["resumable"] is True
    client.post("/missions/m1/resume", data={"extra": "500"})
    reqs = _reqs()
    assert (reqs["r1"].status, reqs["r1"].attempts) == ("pending", 1)
    assert reqs["r2"] == before["r2"] and reqs["r3"] == before["r3"]


def test_resume_attempt_cap_without_the_agent_is_three(client, starts):
    _seed_stopped(budget={k: v for k, v in _STOPPED_BUDGET.items()
                          if k != "per_req_attempts"})
    _run(storage.update_agent("a1", default_per_req_attempts=6))
    _run(storage.delete_agent("a1"))
    _run(storage.update_requirement("r1", status="unmet", attempts=2,
                                    assessment_missing="gap"))     # 2 of 3: open
    _run(storage.update_requirement("r3", status="unmet", attempts=3,
                                    assessment_missing="gap"))     # 3 of 3: capped
    before = _reqs()
    client.post("/missions/m1/resume", data={"extra": "500"})
    reqs = _reqs()
    assert reqs["r1"].status == "pending"
    assert reqs["r3"] == before["r3"]


def test_resume_refused_when_nothing_is_reopenable(client, starts):
    _seed_stopped()
    _run(storage.update_requirement("r1", status="satisfied"))
    _run(storage.update_requirement("r3", status="unmet", attempts=3,
                                    assessment_missing="gap"))
    before = _snapshot()
    client.post("/missions/m1/resume", data={"extra": "500"})
    assert _snapshot() == before
    assert starts == [] and not jobs._store
    assert _flashes(client) == [
        "Every requirement is satisfied or capped out — nothing to resume."]


def test_resume_unknown_mission(client, starts):
    r = client.post("/missions/nope/resume", data={"extra": "1"})
    assert _path(r) == "/missions"
    assert starts == []


def test_resume_when_busy_changes_nothing(client, app_mod, starts, monkeypatch):
    _seed_stopped()
    before = _snapshot()

    def full(*_a, **_k):
        raise jobs.JobLimitReached("6 jobs already running")

    monkeypatch.setattr(app_mod, "create_mission_job", full)
    r = client.post("/missions/m1/resume", data={"extra": "500"})
    assert _path(r) == "/missions/m1"
    assert _snapshot() == before
    assert starts == []
    assert any(m.startswith("Busy") for m in _flashes(client))


def test_resume_lost_claim_releases_its_job(client, app_mod, starts, monkeypatch):
    _seed_stopped()
    before = _snapshot()

    async def lost(*_a):
        return False

    monkeypatch.setattr(app_mod, "claim_mission_status", lost)
    client.post("/missions/m1/resume", data={"extra": "500"})
    assert _snapshot() == before
    assert starts == []
    [job] = jobs._store.values()
    assert job.done and job.stage == "cancelled" and job.finished_at
    assert _live_jobs() == []
    assert _flashes(client) == [
        "This mission changed state in the meantime — nothing was resumed."]


def test_resume_worker_start_failure_rolls_everything_back(client, app_mod, monkeypatch):
    old = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(old, stage="done")
    _seed_stopped()
    _run(storage.update_mission("m1", job_id=old, error="earlier note", resume_count=1))
    before = _snapshot()

    def cannot_start(_mission_id, _job_id=None):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(app_mod, "start_collection", cannot_start)
    r = client.post("/missions/m1/resume", data={"extra": "500"})
    assert r.status_code == 500
    assert _snapshot() == before
    assert _live_jobs() == []


def test_api_mission_reports_stop_reason_and_resumable(client):
    _seed_stopped()
    body = client.get("/api/mission/m1").get_json()
    assert (body["stop_reason"], body["resumable"]) == ("token_budget", True)
    _run(storage.update_mission("m1", stop_reason="complete"))
    body = client.get("/api/mission/m1").get_json()
    assert (body["stop_reason"], body["resumable"]) == ("complete", False)
    _run(storage.update_mission("m1", stop_reason=None))
    body = client.get("/api/mission/m1").get_json()
    assert (body["stop_reason"], body["resumable"]) == (None, False)


def test_api_mission_not_resumable_with_nothing_reopenable(client):
    _seed_stopped("pass_budget")
    for rid in ("r1", "r3"):
        _run(storage.update_requirement(rid, status="satisfied"))
    assert client.get("/api/mission/m1").get_json()["resumable"] is False


def _resume_form(html):
    m = re.search(r'<form[^>]*action="/missions/m1/resume".*?</form>', html, re.S)
    return m.group(0) if m else None


@pytest.mark.parametrize("stop_reason, spent, label, value, cap", [
    # A token stop has overshot its budget (the run stops at the next
    # checkpoint after crossing it): the prefill is the original budget plus
    # that overshoot, so the resumed run gets a full budget of headroom.
    ("token_budget", 1300,
     "Stopped: token budget reached · 1,300 of 1,000 used · 2 requirements still open",
     "1300", "5000000"),
    ("source_budget", 0, "Stopped: source budget reached · 2 requirements still open",
     "7", "100"),
    ("pass_budget", 0, "Stopped: pass budget reached · 2 requirements still open",
     "2", "10"),
])
def test_mission_page_offers_resume_with_the_limit_prefilled(client, stop_reason, spent,
                                                            label, value, cap):
    _seed_stopped(stop_reason)
    if spent:
        _spend_tokens("m1", spent)
    html = client.get("/missions/m1").get_data(as_text=True)
    form = _resume_form(html)
    assert form is not None
    assert label in html
    assert re.search(r'<input[^>]*name="extra"', form)
    assert f'value="{value}"' in form and f'max="{cap}"' in form and 'min="1"' in form
    assert re.search(r">\s*Resume\s*</button>", form)
    # Next to Re-run, in the done-state actions.
    assert html.index('action="/agents/a1/run"') < html.index('action="/missions/m1/resume"')


def test_mission_page_resume_after_a_user_stop_has_no_input(client):
    _seed_stopped("user_stop")
    _run(storage.update_requirement("r3", status="satisfied"))
    html = client.get("/missions/m1").get_data(as_text=True)
    form = _resume_form(html)
    assert form is not None and 'name="extra"' not in form
    assert "Stopped by you · 1 requirement still open" in html


def test_mission_page_token_prefill_uses_the_setting_default(client, monkeypatch):
    monkeypatch.setattr(config.settings, "max_llm_tokens", 9000)
    _seed_stopped(budget={k: v for k, v in _STOPPED_BUDGET.items() if k != "max_llm_tokens"})
    form = _resume_form(client.get("/missions/m1").get_data(as_text=True))
    assert 'value="9000"' in form


@pytest.mark.parametrize("stop_reason", ["complete", None])
def test_mission_page_has_no_resume_without_a_limit_stop(client, stop_reason):
    _seed_stopped(stop_reason)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "/missions/m1/resume" not in html
    assert "still open" not in html


def test_mission_page_has_no_resume_with_nothing_reopenable(client):
    _seed_stopped()
    for rid in ("r1", "r3"):
        _run(storage.update_requirement(rid, status="satisfied"))
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "/missions/m1/resume" not in html


# --- resume: a spent token budget must be raised too ----------------------

_SPENT_FLASH = ("This mission has used 1,000 of its 1,000-token budget; "
                "raise the token budget above 1,000 to resume.")


def test_resume_source_stop_over_its_token_budget_refused_without_extra_tokens(
        client, starts):
    """The resumed run would stop on the token budget before its first
    requirement (and still re-write the brief), so it is refused with
    nothing changed, not even a job created."""
    _seed_stopped("source_budget")
    _spend_tokens("m1", 1000)                  # exactly the budget: none left
    before = _snapshot()
    r = client.post("/missions/m1/resume", data={"extra": "5"})
    assert _path(r) == "/missions/m1"
    assert _snapshot() == before
    assert starts == [] and not jobs._store
    assert _flashes(client) == [_SPENT_FLASH]


@pytest.mark.parametrize("raw", ["", "lots"], ids=["blank", "junk"])
def test_resume_unreadable_extra_tokens_counts_as_not_given(client, starts, raw):
    _seed_stopped("source_budget")
    _spend_tokens("m1", 1000)
    before = _snapshot()
    client.post("/missions/m1/resume", data={"extra": "5", "extra_tokens": raw})
    assert _snapshot() == before and starts == []
    assert _flashes(client) == [_SPENT_FLASH]


def test_resume_source_stop_over_its_token_budget_resumes_with_extra_tokens(
        client, starts):
    _seed_stopped("source_budget")
    _spend_tokens("m1", 1000)
    client.post("/missions/m1/resume", data={"extra": "5", "extra_tokens": "1000"})
    assert _budget() == {**_STOPPED_BUDGET, "max_sources": 12, "max_llm_tokens": 2000}
    m = _mission()
    assert m.status == "collecting" and starts == [("m1", m.job_id, m.job_id)]
    assert _flashes(client) == [
        "Resuming — 2 requirements reopened, source budget raised to 12, "
        "token budget raised to 2,000."]


def test_resume_user_stop_over_its_token_budget_raises_only_the_tokens(client, starts):
    _seed_stopped("user_stop")
    _spend_tokens("m1", 1200)
    client.post("/missions/m1/resume", data={"extra": "9", "extra_tokens": "800"})
    assert _budget() == {**_STOPPED_BUDGET, "max_llm_tokens": 1800}
    assert _flashes(client) == [
        "Resuming — 2 requirements reopened, token budget raised to 1,800."]


def test_resume_refused_when_the_raise_still_leaves_the_tokens_spent(client, starts):
    _seed_stopped("pass_budget")
    _spend_tokens("m1", 1500)
    before = _snapshot()
    client.post("/missions/m1/resume", data={"extra": "1", "extra_tokens": "500"})
    assert _snapshot() == before and starts == []
    assert _flashes(client) == [
        "This mission has used 1,500 of its 1,000-token budget; "
        "raise the token budget above 1,500 to resume."]


def test_resume_token_stop_uses_extra_as_its_one_token_field(client, starts):
    """For a token-budget stop `extra` is the token raise; a stray
    extra_tokens is ignored, and a raise that is still not enough is
    refused."""
    _seed_stopped("token_budget")
    _spend_tokens("m1", 3000)
    before = _snapshot()
    client.post("/missions/m1/resume", data={"extra": "500", "extra_tokens": "9000"})
    assert _snapshot() == before and starts == []
    assert _flashes(client) == [
        "This mission has used 3,000 of its 1,000-token budget; "
        "raise the token budget above 3,000 to resume."]
    client.post("/missions/m1/resume", data={"extra": "2500", "extra_tokens": "9000"})
    assert _budget()["max_llm_tokens"] == 3500
    assert len(starts) == 1


@pytest.mark.parametrize("raw, want", [("99999999", 1000 + 5_000_000), ("0", 1001),
                                       ("-7", 1001)],
                         ids=["high", "zero", "negative"])
def test_resume_extra_tokens_is_clamped(client, starts, raw, want):
    _seed_stopped("source_budget")
    _spend_tokens("m1", 1000)
    client.post("/missions/m1/resume", data={"extra": "1", "extra_tokens": raw})
    assert _budget()["max_llm_tokens"] == want
    assert len(starts) == 1


def test_resume_extra_tokens_never_limits_an_unlimited_budget(client, starts):
    _seed_stopped("source_budget", budget={**_STOPPED_BUDGET, "max_llm_tokens": 0})
    _spend_tokens("m1", 10 ** 6)
    client.post("/missions/m1/resume", data={"extra": "1", "extra_tokens": "500"})
    assert _budget()["max_llm_tokens"] == 0
    assert len(starts) == 1


def _inputs(form):
    return re.findall(r'<input[^>]*name="(extra(?:_tokens)?)"[^>]*value="(\d+)"', form)


def test_mission_page_adds_a_token_field_when_the_tokens_are_spent(client):
    _seed_stopped("source_budget")
    html = client.get("/missions/m1").get_data(as_text=True)
    assert _inputs(_resume_form(html)) == [("extra", "7")]
    assert "tokens used" not in html
    _spend_tokens("m1", 1000)
    html = client.get("/missions/m1").get_data(as_text=True)
    form = _resume_form(html)
    assert _inputs(form) == [("extra", "7"), ("extra_tokens", "1000")]
    assert re.search(r'<input[^>]*name="extra_tokens"[^>]*required', form)
    assert 'max="5000000"' in form
    assert ("Stopped: source budget reached · 1,000 of 1,000 tokens used · "
            "2 requirements still open") in html


def test_mission_page_extra_tokens_prefill_covers_the_overshoot(client):
    _seed_stopped("pass_budget")
    _spend_tokens("m1", 1250)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert _inputs(_resume_form(html)) == [("extra", "2"), ("extra_tokens", "1250")]
    assert "1,250 of 1,000 tokens used" in html


def test_mission_page_token_stop_shows_one_token_field(client):
    _seed_stopped("token_budget")
    _spend_tokens("m1", 1200)
    form = _resume_form(client.get("/missions/m1").get_data(as_text=True))
    assert _inputs(form) == [("extra", "1200")]


def test_mission_page_token_prefill_is_clamped(client):
    _seed_stopped("token_budget", budget={**_STOPPED_BUDGET, "max_llm_tokens": 4_000_000})
    _spend_tokens("m1", 7_000_000)
    form = _resume_form(client.get("/missions/m1").get_data(as_text=True))
    assert _inputs(form) == [("extra", "5000000")]


def test_resume_blank_extra_after_a_token_stop_uses_the_overshoot_prefill(client, starts):
    _seed_stopped("token_budget")
    _spend_tokens("m1", 1300)
    client.post("/missions/m1/resume", data={"extra": ""})
    assert _budget()["max_llm_tokens"] == 1000 + 1300   # 1,000 of headroom past 1,300
    assert len(starts) == 1


def test_mission_page_user_stop_over_budget_shows_only_the_token_field(client):
    _seed_stopped("user_stop")
    _spend_tokens("m1", 1000)
    form = _resume_form(client.get("/missions/m1").get_data(as_text=True))
    assert _inputs(form) == [("extra_tokens", "1000")]


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


def test_delete_stranded_mission_whose_job_already_finished(client):
    # The worker died after its job ended but before the mission row left
    # `collecting`: the finished trace is still in memory, yet nothing is
    # running, so the delete must not wait for the trace to be evicted.
    jid = jobs.create_mission_job("Where is it?", 7)
    jobs.finish_job(jid, stage="error", error="worker crashed")
    stamped = jobs.get_job(jid).finished_at
    _seed_mission(status="collecting", job_id=jid)
    r = client.post("/missions/m1/delete")
    assert _path(r) == "/missions"
    assert _mission() is None
    job = jobs.get_job(jid)
    assert (job.stage, job.error, job.finished_at) == ("error", "worker crashed", stamped)


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
    monkeypatch.setattr(agent_runner, "start_planning",
                        lambda mid, job_id=None: calls.append((mid, job_id)))
    return calls


def test_run_scheduled_redirects_to_the_new_mission(client, plannings):
    _seed_agent(schedule_question="What changed overnight?", schedule_cron="0 7 * * *")
    r = client.post("/agents/a1/run-scheduled")
    [mission] = _run(storage.list_missions())
    assert _path(r) == f"/missions/{mission.id}"
    assert mission.question == "What changed overnight?"
    assert json.loads(mission.budget_json)["auto_approve"] is True
    # The planner is handed the job the launch created, the one on the row.
    assert mission.job_id and plannings == [(mission.id, mission.job_id)]


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


# --- mission stop -------------------------------------------------------

def test_mission_stop_says_it_lands_after_the_current_requirement(client):
    """The runner honours a stop before the next requirement, not at the
    next pass, and the page and flash say so."""
    jid = jobs.create_mission_job("Where is it?", 7)
    _seed_mission(status="collecting", job_id=jid)
    page = client.get("/missions/m1").get_data(as_text=True)
    assert "Stop after the current requirement" in page
    assert "stopping after the current requirement" in page   # the live meta line
    assert "after this pass" not in page
    r = client.post("/missions/m1/stop")
    assert _path(r) == "/missions/m1"
    assert jobs.is_cancelled(jid)
    assert _flashes(client) == [
        "Stopping after the current requirement — the brief will still be written."]


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


def _lib_input(html):
    tags = re.findall(r'<input[^>]*id="libSearch"[^>]*>', html)
    assert len(tags) == 1, "the Library has exactly one search input"
    return tags[0]


def test_full_text_query_seeds_the_one_input_without_refiltering(client):
    _seed_docs(3)
    html = client.get("/documents?q=ordinary").get_data(as_text=True)
    tag = _lib_input(html)
    # Typing filters this page; Enter submits the same input as `q`.
    assert 'name="q"' in tag and 'value="ordinary"' in tag
    assert re.search(r'<form[^>]*method="get"[^>]*action="/documents"', html)
    # The seeded value is the server's full-text query. The client filter
    # sees only titles/snippets, so re-applying it would hide matches whose
    # hit is in the body: the page tells the script which value to skip.
    assert 'const FTS_QUERY = "ordinary";' in html
    assert _cards(html) == 3


def test_zero_hit_full_text_search_keeps_the_input(client):
    # With no matches there are no cards, but the query must stay editable.
    _seed_docs(2)
    html = client.get("/documents?q=zzzznomatch").get_data(as_text=True)
    assert 'value="zzzznomatch"' in _lib_input(html)
    assert _cards(html) == 0
    assert "No pages match" in html
    assert "No documents yet" not in html


def test_library_without_a_query_leaves_the_input_empty(client):
    _seed_docs(2)
    html = client.get("/documents").get_data(as_text=True)
    assert 'value=""' in _lib_input(html)
    assert 'const FTS_QUERY = "";' in html


def test_library_query_inside_a_mission_filters_client_side(client):
    # The mission listing takes precedence over full-text search, so `q` is
    # not an FTS result set there: the input carries it and the client
    # filter applies it, and the page does not claim FTS "matches".
    _seed_briefed_mission()
    html = client.get("/documents?mission=m1&q=ordinary").get_data(as_text=True)
    assert 'value="ordinary"' in _lib_input(html)
    assert 'const FTS_QUERY = "";' in html
    assert "Sources from mission" in html
    assert "match" not in re.search(r'<p class="page-sub">(.*?)</p>', html, re.S).group(1)
    assert _cards(html) == 2


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


def _rail_numbers(html):
    """Every numbered rail slot, available or removed, in page order."""
    return [int(n) for n in re.findall(r'class="src(?: src-gone)?" data-src="(\d+)"', html)]


def test_mission_page_numbers_sources_by_the_stored_order(client):
    _seed_briefed_mission(stored_order=["doc02", "doc01"])
    html = client.get("/missions/m1").get_data(as_text=True)
    assert _rail(html) == [("1", "doc02"), ("2", "doc01")]
    assert 'class="cite" data-cite="1"' in html
    assert 'class="cite" data-cite="2"' in html


def test_mission_rail_is_capped_like_the_brief(client, app_mod):
    import brief
    cited = ["doc25", "doc24", "doc23"]
    # "gone" was cited once but its document no longer exists. It is last in
    # the stored order, so whether brief.ordered_sources_for_mission skips it
    # or keeps its slot as a removed source, it shifts no number and does not
    # widen the citation bound. (A gap mid-list: see the removed-slot tests.)
    _seed_briefed_mission(n_docs=25, stored_order=cited + ["gone"],
                          brief_md="See [1], [3], [4] and [21].")
    html = client.get("/missions/m1").get_data(as_text=True)
    rail = _rail(html)
    # Every uncited doc is appended after the stored order, but numbering
    # stops where a brief's citations can reach.
    assert _rail_numbers(html) == list(range(1, brief.MAX_BRIEF_SOURCES + 1))
    assert [d for _n, d in rail[:3]] == cited
    assert 'data-cite="1"' in html and 'data-cite="3"' in html
    # The brief cited 3 documents that still exist: [4] is the removed slot
    # or an uncited document a later retask added, so it stays plain text.
    assert 'data-cite="4"' not in html and "[4]" in html
    assert 'data-cite="21"' not in html and "[21]" in html


def test_mission_rail_keeps_a_removed_source_in_its_slot(client):
    # brief.ordered_sources_for_mission keeps a stored id whose document is
    # gone as a None slot, so later numbers do not shift.
    _seed_briefed_mission(n_docs=3, stored_order=["doc01", "gone", "doc02"],
                          brief_md="See [1], [2], [3] and [4].")
    html = client.get("/missions/m1").get_data(as_text=True)
    # Numbers never shift: doc02 stays [3], the number the brief gave it.
    assert _rail(html) == [("1", "doc01"), ("3", "doc02"), ("4", "doc03")]
    assert _rail_numbers(html) == [1, 2, 3, 4]
    assert re.search(r'<div class="src src-gone" data-src="2"', html)
    assert "source no longer available" in html
    # Every stored slot up to the last surviving one is citable ([2] lights
    # the removed row); [4] is an uncited document appended later.
    for n in (1, 2, 3):
        assert f'data-cite="{n}"' in html
    assert 'data-cite="4"' not in html and "[4]" in html
    # Requirement source lists carry the same numbers.
    assert re.search(r'href="/document/doc02"[^>]*>\s*<span class="req-src-n">\[3\]</span>', html)
    assert re.search(r'href="/document/doc03"[^>]*>\s*<span class="req-src-n">\[4\]</span>', html)


def test_mission_rail_trailing_removed_slot_does_not_widen_the_bound(client):
    _seed_briefed_mission(n_docs=2, stored_order=["gone", "doc01", "doc02", "gone-too"],
                          brief_md="See [1], [2], [3] and [4].")
    html = client.get("/missions/m1").get_data(as_text=True)
    assert _rail(html) == [("2", "doc01"), ("3", "doc02")]
    assert _rail_numbers(html) == [1, 2, 3, 4]
    for n in (1, 2, 3):
        assert f'data-cite="{n}"' in html
    assert 'data-cite="4"' not in html and "[4]" in html


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


# --- mission page: brief checks -------------------------------------------

def _brief_checks(html):
    m = re.search(r'<div class="brief-checks"[^>]*>(.*?)</ul>', html, re.S)
    return m.group(1) if m else None


def test_mission_page_shows_brief_checks_above_the_brief(client):
    _seed_briefed_mission()
    _run(storage.update_mission("m1", brief_warnings_json=json.dumps([
        {"kind": "uncited_paragraph", "detail": "<script>alert(1)</script> A long claim"},
        {"kind": "junk_citation", "detail": "[2] cites a block page"},
        {"kind": "requirement_unmentioned", "detail": "Req 1"},
        {"kind": "<b>novel</b>", "detail": "a kind this page has no label for"},
        "not a dict", ["nor", "this"], {"kind": 7, "detail": None},
    ])))
    html = client.get("/missions/m1").get_data(as_text=True)
    box = _brief_checks(html)
    assert box is not None
    assert html.index('class="brief-checks"') < html.index('id="briefBody"')
    assert "Brief checks" in box
    assert box.count('class="bcheck"') == 4
    for label in ("Uncited claim", "Cites an unusable source", "Requirement not addressed"):
        assert label in box
    # Everything the LLM or a page could influence is escaped.
    assert "&lt;script&gt;alert(1)&lt;/script&gt; A long claim" in box
    assert "<script>alert(1)" not in html
    assert "&lt;b&gt;novel&lt;/b&gt;" in box


@pytest.mark.parametrize("raw", [None, "[]", "{not json", '{"kind": "x"}', "[1, 2]"])
def test_mission_page_without_brief_checks(client, raw):
    _seed_briefed_mission()
    if raw is not None:
        _run(storage.update_mission("m1", brief_warnings_json=raw))
    r = client.get("/missions/m1")
    assert r.status_code == 200
    assert _brief_checks(r.get_data(as_text=True)) is None


# --- mission page: LLM token telemetry -----------------------------------

def _seed_llm_calls(mission_id="m1"):
    from models import LlmCall

    async def go():
        for purpose, prompt, completion in (("plan", 1000, 200),
                                            ("assess", 400, 100), ("assess", 400, 100)):
            await storage.insert_llm_call(LlmCall(
                purpose=purpose, tier="reasoning", model="cohere/command-a",
                mission_id=mission_id, prompt_tokens=prompt, completion_tokens=completion))
        # Another mission's usage never counts.
        await storage.insert_llm_call(LlmCall(
            purpose="plan", tier="reasoning", model="cohere/command-a",
            mission_id="other", prompt_tokens=9000, completion_tokens=9000))
    _run(go())


def test_mission_page_shows_llm_tokens(client):
    _seed_mission(status="done")
    _run(storage.update_mission("m1", budget_json=json.dumps(
        {"max_sources": 7, "max_llm_tokens": 50000})))
    _seed_llm_calls()
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "LLM tokens" in html
    assert "<span data-tele-tokens>2,200</span> <small>/ 50,000</small>" in html
    assert 'data-tele-tokens-bar style="width:4%"' in html
    # By purpose: compact on the cell, exact in its hover title.
    assert re.search(r'<small class="tele-sub">\s*plan 1.2k · assess 1k\s*</small>', html)
    assert "plan: 1 call, 1,000 prompt + 200 completion" in html
    assert "assess: 2 calls, 800 prompt + 200 completion" in html


def test_mission_page_token_cap_defaults_to_the_setting(client, monkeypatch):
    _seed_mission(status="done")          # its budget sets no max_llm_tokens
    monkeypatch.setattr(config.settings, "max_llm_tokens", 9000)
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "<span data-tele-tokens>0</span> <small>/ 9,000</small>" in html
    assert "none recorded" in html
    monkeypatch.setattr(config.settings, "max_llm_tokens", 0)   # unlimited
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "<span data-tele-tokens>0</span></div>" in html
    assert "<span data-tele-tokens-bar" not in html


@pytest.mark.parametrize("tokens, shown", [
    (999, "999"), (1_000, "1k"), (12_340, "12.3k"), (999_949, "999.9k"),
    (999_950, "1M"), (1_260_000, "1.3M"),
])
def test_token_breakdown_rounds_before_choosing_a_unit(client, tokens, shown):
    from models import LlmCall
    _seed_mission(status="done")
    _run(storage.insert_llm_call(LlmCall(
        purpose="plan", tier="reasoning", model="m", mission_id="m1",
        prompt_tokens=tokens, completion_tokens=0)))
    html = client.get("/missions/m1").get_data(as_text=True)
    assert re.search(rf'<small class="tele-sub">\s*plan {re.escape(shown)}\s*</small>', html)


def test_api_mission_reports_llm_tokens(client):
    _seed_mission(status="collecting")
    assert client.get("/api/mission/m1").get_json()["llm_tokens"] == 0
    _seed_llm_calls()
    assert client.get("/api/mission/m1").get_json()["llm_tokens"] == 2200


def test_mission_poll_updates_the_token_cell():
    with open("templates/mission.html", encoding="utf-8") as f:
        src = f.read()
    assert "'[data-tele-tokens]'" in src and "s.llm_tokens" in src


# --- mission page: search signals -----------------------------------------

def test_requirement_detail_shows_search_signals(client):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", search_stats_json=json.dumps([
        {"pass": 1, "query": "a", "engine": "brave", "results": 5},
        {"pass": 1, "query": "b", "engine": "brave", "results": 6},
        {"pass": 1, "query": "c", "engine": "bing", "results": 0},
        {"pass": 2, "query": "d", "engine": None, "results": 0},
        {"pass": 2, "query": "e", "engine": None, "results": 0},
        "junk", {"query": "no pass"}, {"pass": "x", "results": 3},
    ])))
    _run(storage.update_requirement("r1", search_stats_json="{not json"))
    html = client.get("/missions/m1").get_data(as_text=True)
    rows = re.findall(r'<li class="search-stat( zero)?">\s*(.*?)\s*</li>', html, re.S)
    assert rows == [
        ("", "pass 1 · 3 queries · 11 results · brave, bing"),
        (" zero", "pass 2 · 2 queries · 0 results · no engine answered"),
    ]


def test_search_signals_keep_a_retask_run_apart(client):
    # Rows are appended in order and every collection run numbers its passes
    # from 1, so a retask's pass 1 must not merge into the first run's.
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", search_stats_json=json.dumps([
        {"pass": 1, "query": "a", "engine": "brave", "results": 4},
        {"pass": 1, "query": "b", "engine": "brave", "results": 2},
        {"pass": 2, "query": "c", "engine": "bing", "results": 3},
        {"pass": 1, "query": "d", "engine": None, "results": 0},
        {"pass": 2, "query": "e", "engine": "brave", "results": 5},
    ])))
    html = client.get("/missions/m1").get_data(as_text=True)
    rows = re.findall(r'<li class="search-stat( zero)?">\s*(.*?)\s*</li>', html, re.S)
    assert rows == [
        ("", "pass 1 · 2 queries · 6 results · brave"),
        ("", "pass 2 · 1 query · 3 results · bing"),
        (" zero", "run 2 · pass 1 · 1 query · 0 results · no engine answered"),
        ("", "run 2 · pass 2 · 1 query · 5 results · brave"),
    ]


def test_search_signal_engine_names_are_escaped(client):
    _seed_mission(status="done")
    _run(storage.update_requirement("r0", search_stats_json=json.dumps([
        {"pass": 1, "query": "q", "engine": "<i>x</i>", "results": 1}])))
    html = client.get("/missions/m1").get_data(as_text=True)
    assert "pass 1 · 1 query · 1 result · &lt;i&gt;x&lt;/i&gt;" in html


# --- run form: token budget and job hand-off ------------------------------

@pytest.mark.parametrize("raw, want", [
    ("12345", 12345), ("9999999", 5_000_000), ("-3", 0), ("0", 0),
    ("abc", 777), ("", 777), (None, 777),
])
def test_agent_run_stores_a_token_budget(client, app_mod, monkeypatch, raw, want):
    monkeypatch.setattr(config.settings, "max_llm_tokens", 777)
    monkeypatch.setattr(app_mod, "start_planning", lambda mid, job_id=None: None)
    _seed_agent()
    form = {"question": "Where is it?"}
    if raw is not None:
        form["max_llm_tokens"] = raw
    client.post("/agents/a1/run", data=form)
    [m] = _run(storage.list_missions())
    assert json.loads(m.budget_json)["max_llm_tokens"] == want


def test_agent_run_hands_its_job_to_the_planner(client, app_mod, monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod, "start_planning",
                        lambda mid, job_id=None: calls.append((mid, job_id)))
    _seed_agent()
    r = client.post("/agents/a1/run", data={"question": "Where is it?"})
    [m] = _run(storage.list_missions())
    assert _path(r) == f"/missions/{m.id}"
    assert m.job_id and calls == [(m.id, m.job_id)]
    assert not jobs.get_job(m.job_id).done


@pytest.mark.parametrize("setting, shown", [
    (250000, "250,000"), (8_000_000, "8,000,000"), (-1, "unlimited"), (0, "unlimited"),
])
def test_run_form_offers_a_token_budget(client, monkeypatch, setting, shown):
    monkeypatch.setattr(config.settings, "max_llm_tokens", setting)
    _seed_agent()
    html = client.get("/").get_data(as_text=True)
    tag = re.search(r'<input[^>]*name="max_llm_tokens"[^>]*>', html).group(0)
    assert 'min="0"' in tag and 'max="5000000"' in tag
    # The field starts blank (blank means the setting) so its value is always
    # in range: the panel is hidden until Agentic Crawl is chosen, and an
    # invalid hidden field silently blocks every submit of the search form.
    value = re.search(r'\svalue="([^"]*)"', tag)
    assert value is None or value.group(1) == ""
    assert f'placeholder="{shown}"' in tag


# --- crawl page: skipped count --------------------------------------------

def _crawl_job(n_done, n_skipped, n_total):
    jid = jobs.create_job("q", n_total, False, "")
    urls = [f"https://s{i}.example/" for i in range(n_total)]
    jobs.add_urls(jid, [jobs.JobUrl(url=u) for u in urls])
    for u in urls[:n_done]:
        jobs.update_url(jid, u, status="done")
    for u in urls[n_done:n_done + n_skipped]:
        jobs.update_url(jid, u, status="skipped")
    # The crawler counts a skipped page as processed (crawl_done).
    jobs.update_job(jid, crawl_total=n_total, crawl_done=n_done + n_skipped)
    jobs.finish_job(jid, stage="cancelled" if n_skipped else "done")
    return jid


def _url_meta(html):
    return re.search(r'id="urlMeta">([^<]*)<', html).group(1).strip()


def test_crawl_page_shows_skipped_separately(client):
    jid = _crawl_job(n_done=6, n_skipped=2, n_total=8)
    assert jobs.job_state(jid)["skipped"] == 2
    html = client.get(f"/crawl/{jid}").get_data(as_text=True)
    assert _url_meta(html) == "6/8 crawled · 2 skipped"


def test_crawl_page_without_skips_says_nothing_about_them(client):
    jid = _crawl_job(n_done=3, n_skipped=0, n_total=3)
    html = client.get(f"/crawl/{jid}").get_data(as_text=True)
    assert _url_meta(html) == "3/3 crawled"


def test_crawl_page_live_update_uses_the_skipped_count():
    with open("templates/crawl.html", encoding="utf-8") as f:
        src = f.read()
    assert "s.skipped" in src and "' skipped'" in src


# --- agents page ----------------------------------------------------------

def test_agents_page_run_buttons_are_full_height_submits(client):
    _seed_agent(schedule_question="What changed overnight?", schedule_cron="0 7 * * *")
    html = client.get("/agents").get_data(as_text=True)
    ask = re.search(r'<form class="agent-ask" method="post" action="/agents/a1/run">(.*?)</form>',
                    html, re.S).group(1)
    # Enter in the question submits this form; the button is its submit.
    assert re.search(r'<input type="text" name="question"', ask)
    assert re.search(r'<button type="submit" class="btn btn-accent agent-go"', ask)
    sched = re.search(r'<form class="agent-sched" method="post" action="/agents/a1/run-scheduled">'
                      r'(.*?)</form>', html, re.S).group(1)
    assert re.search(r'<button type="submit" class="btn agent-go"', sched)
    # Room below the last card so the fixed tweaks button never covers them.
    assert 'class="page page-agents"' in html


# --- templates use the log cursor ----------------------------------------

@pytest.mark.parametrize("name", ["crawl.html", "mission.html"])
def test_log_cursor_template_strings_present(name):
    """A source-text check of the template, not an execution of its JS: the
    log renderer must key its cursor off log_total."""
    with open(f"templates/{name}", encoding="utf-8") as f:
        src = f.read()
    assert "log_total" in src
    assert "renderedLogs = " in src
