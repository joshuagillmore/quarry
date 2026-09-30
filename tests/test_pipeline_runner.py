"""agent_runner: the approval gate releases its job slot, a crashed worker
finishes its job, a retask neither re-crawls nor charges what the mission
already holds, unattempted requirements say why, and the delta is taken
against the mission's own parent."""
import asyncio
import json

import agent_runner
import jobs
import storage
from agent_assessor import Assessment
from models import Agent, Document, Mission, Requirement, SearchResult


def _init():
    async def go():
        await storage.init_db()
        await storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                         persona_prompt="p", created_at="t"))
    asyncio.run(go())


def _mission(mid, job_id=None, status="collecting", budget=None, parent=None,
             finished_at=None):
    return Mission(id=mid, agent_id="a1", question="Q", status=status,
                   job_id=job_id, budget_json=json.dumps(budget or {}),
                   parent_mission_id=parent, created_at="t",
                   finished_at=finished_at)


def _doc(url, crawled_at="2026-01-01T00:00:00", title=None, words=100):
    return Document(id="d-" + url, url=url, domain=url.split("/")[2],
                    title=title or ("Real " + url), search_query="Q",
                    crawled_at=crawled_at, content_markdown="w " * words,
                    word_count=words)


def _active_jobs() -> int:
    with jobs._lock:
        return sum(1 for j in jobs._store.values() if not j.done)


# ---------- planning gate ----------

def test_gate_leaves_no_active_job(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 10)
    asyncio.run(storage.insert_mission(_mission("m1", jid, status="planning")))
    monkeypatch.setattr(agent_runner, "build_collection_plan", lambda agent, mid, q: [
        Requirement(id="r1", mission_id=mid, title="T", next_queries_json='["q"]')])

    asyncio.run(agent_runner._run_planning("m1"))

    assert asyncio.run(storage.get_mission("m1")).status == "awaiting_approval"
    j = jobs.get_job(jid)
    assert j.done and j.stage == "awaiting_approval"
    assert _active_jobs() == 0
    with jobs._lock:
        jobs._admit_locked()  # a waiting mission holds no slot


def test_planning_failure_finishes_the_job(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 10)
    asyncio.run(storage.insert_mission(_mission("m1", jid, status="planning")))

    def no_plan(agent, mid, q):
        raise ValueError("planner returned no requirements")

    monkeypatch.setattr(agent_runner, "build_collection_plan", no_plan)
    asyncio.run(agent_runner._run_planning("m1"))
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"
    assert asyncio.run(storage.get_mission("m1")).status == "error"


# ---------- _thread crash handling ----------

class _Fatal(BaseException):
    pass


def _crash_case(coro_fn):
    _init()
    jid = jobs.create_mission_job("Q", 10)
    asyncio.run(storage.insert_mission(_mission("m1", jid)))
    agent_runner._thread(coro_fn, "m1")
    return jid, asyncio.run(storage.get_mission("m1"))


def test_crashed_thread_finishes_its_job():
    async def boom(mission_id):
        raise RuntimeError("kaput")

    jid, m = _crash_case(boom)
    assert m.status == "error" and "kaput" in (m.error or "")
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"
    assert _active_jobs() == 0


def test_base_exception_in_thread_still_finishes_its_job():
    async def fatal(mission_id):
        raise _Fatal("interpreter going down")

    jid, m = _crash_case(fatal)
    assert m.status == "error"
    assert jobs.get_job(jid).done


def test_thread_that_returns_early_finishes_its_job():
    async def quits(mission_id):
        return None

    jid, _m = _crash_case(quits)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"


def test_thread_leaves_a_properly_finished_job_alone():
    async def gate(mission_id):
        m = await storage.get_mission(mission_id)
        jobs.finish_job(m.job_id, stage="awaiting_approval")

    jid, _m = _crash_case(gate)
    assert jobs.get_job(jid).stage == "awaiting_approval"


# ---------- collection ----------

def _wire(monkeypatch, urls, satisfied=False, crawled=None, seen_attempted=None):
    monkeypatch.setattr(agent_runner, "web_search", lambda q, n=5: [
        SearchResult(url=u, title=u, snippet="") for u in urls])

    async def fake_crawl(results, question, job_id, attempted=None):
        if seen_attempted is not None:
            seen_attempted.append(set(attempted) if attempted is not None else None)
        if crawled is not None:
            crawled.extend(sr.url for sr in results)
        if attempted is not None:
            attempted.update(sr.url for sr in results)
        return [_doc(sr.url) for sr in results]

    monkeypatch.setattr(agent_runner, "crawl_urls_with_progress", fake_crawl)
    monkeypatch.setattr(agent_runner, "assess_requirement", lambda req, docs: Assessment(
        satisfied, "high" if satisfied else "low", "" if satisfied else "gap", []))
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls: "brief")


def test_retask_skips_held_urls_and_keeps_its_budget(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 1)
    budget = {"max_passes": 1, "max_sources": 1, "per_req_attempts": 1}

    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        await storage.insert_requirement(Requirement(
            id="r_old", mission_id="m1", title="Done already", status="satisfied"))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="Retasked", next_queries_json='["q"]'))
        held = await storage.upsert_document(_doc("http://held.example/1"))
        await storage.link_mission_document("m1", "r_old", held)
    asyncio.run(setup())

    crawled, seen = [], []
    _wire(monkeypatch, ["http://held.example/1", "http://new.example/1"],
          satisfied=True, crawled=crawled, seen_attempted=seen)
    asyncio.run(agent_runner._run_collection("m1"))

    # The held page is not crawled again, and does not use up the budget of 1.
    assert crawled == ["http://new.example/1"]
    assert seen and "http://held.example/1" in seen[0]
    assert jobs.get_job(jid).sources_used == 1
    # It resurfaced for the retasked requirement, so it is linked there too.
    r1_docs = asyncio.run(storage.get_requirement_documents("m1", "r1"))
    assert {d.url for d in r1_docs} == {"http://held.example/1", "http://new.example/1"}
    m = asyncio.run(storage.get_mission("m1"))
    assert m.status == "done"
    j = jobs.get_job(jid)
    assert j.done and j.stage == "done"


def test_unattempted_requirement_says_budget_exhausted(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 1)
    budget = {"max_passes": 2, "max_sources": 1, "per_req_attempts": 2}

    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        for rid in ("r1", "r2"):
            await storage.insert_requirement(Requirement(
                id=rid, mission_id="m1", title=rid, next_queries_json='["q"]'))
    asyncio.run(setup())
    _wire(monkeypatch, ["http://a.example/1", "http://a.example/2"])

    asyncio.run(agent_runner._run_collection("m1"))
    reqs = {r.id: r for r in asyncio.run(storage.get_requirements_for_mission("m1"))}
    assert reqs["r1"].status == "unmet" and reqs["r1"].assessment_missing == "gap"
    assert reqs["r2"].status == "unmet"
    assert reqs["r2"].attempts == 0
    assert reqs["r2"].assessment_missing == "not attempted: source budget exhausted"


def test_unattempted_requirement_says_stopped_by_user(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    budget = {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2}

    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="r1", next_queries_json='["q"]'))
    asyncio.run(setup())
    crawled = []
    _wire(monkeypatch, ["http://a.example/1"], crawled=crawled)
    assert jobs.request_cancel(jid)

    asyncio.run(agent_runner._run_collection("m1"))
    r1 = asyncio.run(storage.get_requirements_for_mission("m1"))[0]
    assert crawled == []
    assert r1.status == "unmet"
    assert r1.assessment_missing == "not attempted: stopped by user"
    # A stop still produces a brief from whatever was collected.
    assert asyncio.run(storage.get_mission("m1")).status == "done"


# ---------- delta ----------

def _delta_fixture():
    _init()

    async def setup():
        ids = {}
        for u in ("http://a.example/1", "http://b.example/1", "http://c.example/1"):
            ids[u] = await storage.upsert_document(_doc(u))
        await storage.insert_mission(_mission("P", status="done", finished_at="2026-01-01"))
        await storage.insert_mission(_mission("L", status="done", finished_at="2026-02-01"))
        await storage.insert_mission(_mission("C", parent="P"))
        await storage.insert_mission(_mission("C2"))
        links = {"P": ["http://a.example/1"],
                 "L": ["http://a.example/1", "http://b.example/1"],
                 "C": list(ids), "C2": list(ids)}
        for mid, urls in links.items():
            await storage.insert_requirement(Requirement(id="r" + mid, mission_id=mid, title="T"))
            for u in urls:
                await storage.link_mission_document(mid, "r" + mid, ids[u])
    asyncio.run(setup())


def _captured_new_urls(monkeypatch, mission_id):
    captured = {}

    def fake_brief(mission, reqs, docs, new_urls):
        captured["new"] = set(new_urls)
        return "brief"

    monkeypatch.setattr(agent_runner, "synthesize_brief", fake_brief)
    asyncio.run(agent_runner._synthesize(mission_id, None, None))
    return captured["new"]


def test_delta_is_against_the_parent_mission(monkeypatch):
    _delta_fixture()
    # L is the most recent finished run, but C descends from P.
    assert _captured_new_urls(monkeypatch, "C") == {
        "http://b.example/1", "http://c.example/1"}


def test_delta_falls_back_to_latest_run_without_a_parent(monkeypatch):
    _delta_fixture()
    assert _captured_new_urls(monkeypatch, "C2") == {"http://c.example/1"}


def test_runner_no_longer_imports_prior_mission_urls():
    assert not hasattr(agent_runner, "get_prior_mission_urls")
