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
    async def boom(mission_id, job_id=None):
        raise RuntimeError("kaput")

    jid, m = _crash_case(boom)
    assert m.status == "error" and "kaput" in (m.error or "")
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"
    assert _active_jobs() == 0


def test_base_exception_in_thread_still_finishes_its_job():
    async def fatal(mission_id, job_id=None):
        raise _Fatal("interpreter going down")

    jid, m = _crash_case(fatal)
    assert m.status == "error"
    assert jobs.get_job(jid).done


def test_thread_that_returns_early_finishes_its_job():
    async def quits(mission_id, job_id=None):
        return None

    jid, _m = _crash_case(quits)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"


def test_thread_leaves_a_properly_finished_job_alone():
    async def gate(mission_id, job_id=None):
        m = await storage.get_mission(mission_id)
        jobs.finish_job(m.job_id, stage="awaiting_approval")

    jid, _m = _crash_case(gate)
    assert jobs.get_job(jid).stage == "awaiting_approval"


# ---------- collection ----------

def _wire(monkeypatch, urls, satisfied=False, crawled=None, seen_attempted=None,
          redirects=None):
    """Stub search/crawl/assess/brief. `redirects` maps a requested URL to the
    URL its document is stored under; the fake honours the crawler contract
    for `attempted` and `aliases` exactly as crawler.py does."""
    redirects = redirects or {}
    monkeypatch.setattr(agent_runner, "web_search_ex", lambda q, n=5: (
        [SearchResult(url=u, title=u, snippet="") for u in urls], "stub-engine"))

    async def fake_crawl(results, question, job_id, attempted=None, aliases=None):
        if seen_attempted is not None:
            seen_attempted.append(set(attempted) if attempted is not None else None)
        if crawled is not None:
            crawled.extend(sr.url for sr in results)
        docs = []
        for sr in results:
            final = redirects.get(sr.url, sr.url)
            if attempted is not None:
                attempted.update({sr.url, final})
            if aliases is not None and final != sr.url:
                aliases[sr.url] = final
            docs.append(_doc(final))
        return docs

    monkeypatch.setattr(agent_runner, "crawl_urls_with_progress", fake_crawl)
    monkeypatch.setattr(agent_runner, "assess_requirement", lambda req, docs, **k: Assessment(
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


# ---------- redirects: one page, several names ----------

def test_redirected_url_resurfacing_is_linked_not_recrawled(monkeypatch):
    """Requirement 1 crawls A, stored under B. When requirement 2's search
    returns A again, A must neither be re-crawled nor left unlinked."""
    _init()
    jid = jobs.create_mission_job("Q", 5)
    budget = {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1}

    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        for rid in ("r1", "r2"):
            await storage.insert_requirement(Requirement(
                id=rid, mission_id="m1", title=rid, next_queries_json='["q"]'))
    asyncio.run(setup())

    crawled = []
    _wire(monkeypatch, ["http://a.example/start"], satisfied=True, crawled=crawled,
          redirects={"http://a.example/start": "http://b.example/final"})
    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == ["http://a.example/start"], "crawled once"
    for rid in ("r1", "r2"):
        docs = asyncio.run(storage.get_requirement_documents("m1", rid))
        assert [d.url for d in docs] == ["http://b.example/final"], rid
    assert jobs.get_job(jid).sources_used == 1


def test_retask_resolves_held_redirects_from_metadata(monkeypatch):
    """A retask knows a held page by every name it was crawled under: the
    requested URL (stored under its final URL) and the redirect target
    (stored under the requested URL, which is what crawl4ai reports)."""
    _init()
    jid = jobs.create_mission_job("Q", 1)
    budget = {"max_passes": 1, "max_sources": 1, "per_req_attempts": 1}

    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        await storage.insert_requirement(Requirement(
            id="r_old", mission_id="m1", title="old", status="satisfied"))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="retasked", next_queries_json='["q"]'))
        held = [
            _doc("http://b.example/final").model_copy(update={
                "metadata_json": json.dumps({"requested_url": "http://a.example/start"})}),
            _doc("http://c.example/asked").model_copy(update={
                "metadata_json": json.dumps({"redirected_url": "http://d.example/landed"})}),
        ]
        for doc in held:
            await storage.link_mission_document(
                "m1", "r_old", await storage.upsert_document(doc))
    asyncio.run(setup())

    crawled = []
    _wire(monkeypatch, ["http://a.example/start", "http://d.example/landed",
                        "http://new.example/1"], satisfied=True, crawled=crawled)
    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == ["http://new.example/1"]
    assert jobs.get_job(jid).sources_used == 1
    r1_urls = {d.url for d in asyncio.run(storage.get_requirement_documents("m1", "r1"))}
    assert r1_urls == {"http://b.example/final", "http://c.example/asked",
                       "http://new.example/1"}


# ---------- telemetry: extraction is charged to the mission ----------

def test_extract_sources_passes_the_mission_id(monkeypatch):
    _init()

    async def setup():
        await storage.insert_mission(_mission("m1"))
        await storage.insert_requirement(Requirement(id="r1", mission_id="m1", title="T"))
        doc_id = await storage.upsert_document(_doc("http://a.example/1"))
        await storage.link_mission_document("m1", "r1", doc_id)
    asyncio.run(setup())
    seen = []

    def fake_extract(doc, prompt, mission_id=None):
        seen.append((doc.url, mission_id))
        return None

    monkeypatch.setattr(agent_runner, "extract_from_document", fake_extract)
    asyncio.run(agent_runner._extract_sources("m1", "p", None))
    assert seen == [("http://a.example/1", "m1")]


# ---------- search signals ----------

def _one_requirement(jid, budget, queries='["q1", "q2"]', stats=None):
    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="r1", next_queries_json=queries,
            search_stats_json=stats))
    asyncio.run(setup())


def _stats():
    [r1] = asyncio.run(storage.get_requirements_for_mission("m1"))
    return json.loads(r1.search_stats_json or "[]")


def _engine_per_query(monkeypatch, answers):
    """answers: query -> (urls, engine)."""
    def fake(q, n=5):
        urls, engine = answers[q]
        return [SearchResult(url=u, title=u, snippet="") for u in urls], engine
    monkeypatch.setattr(agent_runner, "web_search_ex", fake)


def test_search_stats_accumulate_across_passes_including_empty_queries(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2})
    _wire(monkeypatch, [])
    _engine_per_query(monkeypatch, {"q1": (["http://a.example/1"], "brave"),
                                    "q2": ([], None)})

    asyncio.run(agent_runner._run_collection("m1"))

    assert _stats() == [
        {"pass": 1, "query": "q1", "engine": "brave", "results": 1},
        {"pass": 1, "query": "q2", "engine": None, "results": 0},
        {"pass": 2, "query": "q1", "engine": "brave", "results": 1},
        {"pass": 2, "query": "q2", "engine": None, "results": 0},
    ]


def test_search_stats_keep_only_the_last_40(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    old = [{"pass": 0, "query": f"old{i}", "engine": "bing", "results": 1}
           for i in range(39)]
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1},
                     stats=json.dumps(old))
    _wire(monkeypatch, [])
    _engine_per_query(monkeypatch, {"q1": ([], None), "q2": ([], None)})

    asyncio.run(agent_runner._run_collection("m1"))

    stats = _stats()
    assert len(stats) == 40
    assert stats[0]["query"] == "old1", "the oldest entry is dropped"
    assert [s["query"] for s in stats[-2:]] == ["q1", "q2"]


def test_assessor_gets_this_pass_search_coverage(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    _wire(monkeypatch, [])
    _engine_per_query(monkeypatch, {
        "q1": (["http://a.example/1", "http://a.example/2"], "brave"),
        "q2": (["http://b.example/1"], "bing")})
    notes = []
    monkeypatch.setattr(agent_runner, "assess_requirement", lambda req, docs, search_note="": (
        notes.append(search_note) or Assessment(False, "low", "gap", [])))

    asyncio.run(agent_runner._run_collection("m1"))

    assert notes == ["Search coverage this pass: 2 queries, 3 results (brave, bing)"]


def test_search_note_when_no_engine_answered():
    assert agent_runner._search_note([
        {"pass": 1, "query": "q", "engine": None, "results": 0}]) == (
        "Search coverage this pass: 1 query, 0 results (no engine answered)")


# ---------- stop inside a pass ----------

def _two_requirements(jid, budget):
    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        for rid in ("r1", "r2"):
            await storage.insert_requirement(Requirement(
                id=rid, mission_id="m1", title=rid, next_queries_json='["q"]'))
    asyncio.run(setup())


def test_stop_mid_pass_leaves_later_requirements_unattempted(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 3, "max_sources": 5, "per_req_attempts": 3})
    crawled = []
    _wire(monkeypatch, ["http://a.example/1"], crawled=crawled)
    real_crawl = agent_runner.crawl_urls_with_progress

    async def crawl_then_stop(results, question, job_id, attempted=None, aliases=None):
        jobs.request_cancel(jid)   # the user clicks Stop while r1 is crawling
        return await real_crawl(results, question, job_id, attempted, aliases)

    monkeypatch.setattr(agent_runner, "crawl_urls_with_progress", crawl_then_stop)
    asyncio.run(agent_runner._run_collection("m1"))

    reqs = {r.id: r for r in asyncio.run(storage.get_requirements_for_mission("m1"))}
    assert crawled == ["http://a.example/1"], "r2 never searched or crawled"
    assert reqs["r1"].attempts == 1 and reqs["r1"].assessment_missing == "gap"
    assert reqs["r2"].status == "unmet" and reqs["r2"].attempts == 0
    assert reqs["r2"].assessment_missing == "not attempted: stopped by user"
    assert asyncio.run(storage.get_mission("m1")).status == "done"
    assert "stop requested" in " ".join(entry.msg for entry in jobs.get_job(jid).log)


# ---------- token budget ----------

def _spend(mission_id, tokens, purpose="assess"):
    from models import LlmCall
    asyncio.run(storage.insert_llm_call(LlmCall(
        purpose=purpose, tier="reasoning", model="m", mission_id=mission_id,
        prompt_tokens=tokens, completion_tokens=0)))


def _log_text(jid):
    return " ".join(entry.msg for entry in jobs.get_job(jid).log)


def test_token_budget_stops_before_the_next_requirement(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2,
                            "max_llm_tokens": 100})
    crawled = []
    _wire(monkeypatch, ["http://a.example/1"], crawled=crawled)

    def costly_assess(req, docs, **k):
        _spend("m1", 150)          # this assessment blows the budget
        return Assessment(False, "low", "gap", [])

    monkeypatch.setattr(agent_runner, "assess_requirement", costly_assess)
    asyncio.run(agent_runner._run_collection("m1"))

    reqs = {r.id: r for r in asyncio.run(storage.get_requirements_for_mission("m1"))}
    assert reqs["r1"].attempts == 1
    assert reqs["r2"].attempts == 0 and reqs["r2"].status == "unmet"
    assert reqs["r2"].assessment_missing == "not attempted: token budget reached"
    assert "token budget reached" in _log_text(jid)
    assert asyncio.run(storage.get_mission("m1")).status == "done"


def _record_extraction(monkeypatch):
    calls = []

    async def fake_extract(mission_id, prompt, job_id):
        calls.append(mission_id)

    monkeypatch.setattr(agent_runner, "_extract_sources", fake_extract)
    return calls


def test_token_budget_stop_skips_extraction_but_still_writes_the_brief(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2,
                            "max_llm_tokens": 100, "extract": True, "extract_prompt": "p"})
    _spend("m1", 101)
    _wire(monkeypatch, ["http://a.example/1"])
    extracted = _record_extraction(monkeypatch)

    asyncio.run(agent_runner._run_collection("m1"))

    assert extracted == []
    assert "skipping extraction: token budget reached" in _log_text(jid)
    m = asyncio.run(storage.get_mission("m1"))
    assert m.status == "done" and m.brief_markdown == "brief"


def test_extraction_still_runs_when_collection_ends_otherwise(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1,
                            "max_llm_tokens": 10 ** 6, "extract": True})
    _wire(monkeypatch, ["http://a.example/1"])
    extracted = _record_extraction(monkeypatch)

    asyncio.run(agent_runner._run_collection("m1"))

    assert extracted == ["m1"]
    assert "skipping extraction" not in _log_text(jid)


def test_token_budget_checked_at_the_pass_boundary(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2,
                            "max_llm_tokens": 100})
    _spend("m1", 101)              # the plan alone used it up
    crawled = []
    _wire(monkeypatch, ["http://a.example/1"], crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == []
    for r in asyncio.run(storage.get_requirements_for_mission("m1")):
        assert r.assessment_missing == "not attempted: token budget reached"
    assert "token budget reached" in _log_text(jid)


def test_token_budget_defaults_to_the_setting(monkeypatch):
    _init()
    monkeypatch.setattr(agent_runner.settings, "max_llm_tokens", 100)
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 2, "max_sources": 5, "per_req_attempts": 2})
    _spend("m1", 500)
    crawled = []
    _wire(monkeypatch, ["http://a.example/1"], crawled=crawled)
    asyncio.run(agent_runner._run_collection("m1"))
    assert crawled == []


def test_zero_token_budget_is_unlimited(monkeypatch):
    _init()
    monkeypatch.setattr(agent_runner.settings, "max_llm_tokens", 100)
    jid = jobs.create_mission_job("Q", 5)
    _two_requirements(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1,
                            "max_llm_tokens": 0})   # the mission's own 0 wins
    _spend("m1", 10 ** 6)
    crawled = []
    _wire(monkeypatch, ["http://a.example/1", "http://a.example/2"], crawled=crawled)
    asyncio.run(agent_runner._run_collection("m1"))
    assert crawled, "an unlimited budget never stops collection"
    assert "token budget" not in _log_text(jid)


# ---------- the job a worker was given is the one it finishes ----------

def test_worker_whose_mission_was_deleted_still_releases_its_job():
    _init()
    jid = jobs.create_mission_job("Q", 10)   # the mission row never exists / is gone
    agent_runner._thread(agent_runner._run_collection, "gone", jid)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"
    assert _active_jobs() == 0


def test_worker_finishes_its_own_job_not_the_missions_newer_one():
    """A retask may point the mission at a new job (another worker's) while
    this worker is still exiting; only the job it was given is finished."""
    _init()
    mine = jobs.create_mission_job("Q", 10)
    newer = jobs.create_mission_job("Q", 10)
    asyncio.run(storage.insert_mission(_mission("m1", newer)))

    async def quits(mission_id, job_id=None):
        return None

    agent_runner._thread(quits, "m1", mine)
    assert jobs.get_job(mine).done
    assert not jobs.get_job(newer).done


def test_launchers_hand_the_job_id_to_the_worker(monkeypatch):
    import threading as _threading
    seen, ran = [], _threading.Event()

    def fake_thread(coro_fn, mission_id, job_id=None):
        seen.append((coro_fn.__name__, mission_id, job_id))
        if len(seen) == 2:
            ran.set()

    monkeypatch.setattr(agent_runner, "_thread", fake_thread)
    agent_runner.start_planning("m1", "j1")
    agent_runner.start_collection("m2", job_id="j2")
    assert ran.wait(5)
    assert sorted(seen) == [("_run_collection", "m2", "j2"), ("_run_planning", "m1", "j1")]


# ---------- brief checks are stored with the brief ----------

def test_synthesize_stores_brief_warnings_and_logs_them(monkeypatch):
    _init()
    jid = jobs.create_mission_job("Q", 5)

    async def setup():
        await storage.insert_mission(_mission("m1", jid))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title="Lithium battery degradation",
            status="satisfied"))
        await storage.insert_requirement(Requirement(
            id="r2", mission_id="m1", title="Sodium supply chains", status="unmet"))
        for doc in (_doc("http://good.example/1", words=500),
                    _doc("http://junk.example/1", title="Just a moment...", words=5)):
            await storage.link_mission_document("m1", "r1", await storage.upsert_document(doc))
    asyncio.run(setup())
    monkeypatch.setattr(agent_runner, "synthesize_brief", lambda mission, reqs, docs, new_urls: (
        "## Key Findings\n- Lithium battery cells fade quickly when stored hot and fully "
        "charged for long periods.\n- Cells fade [2].\n"))

    asyncio.run(agent_runner._synthesize("m1", None, jid))

    m = asyncio.run(storage.get_mission("m1"))
    kinds = sorted(w["kind"] for w in json.loads(m.brief_warnings_json))
    assert kinds == ["junk_citation", "requirement_unmentioned", "uncited_paragraph"]
    assert "brief checks: 3 warning(s)" in _log_text(jid)


def test_synthesize_stores_empty_warnings_for_a_clean_brief(monkeypatch):
    _init()

    async def setup():
        await storage.insert_mission(_mission("m1"))
        await storage.insert_requirement(Requirement(id="r1", mission_id="m1", title="T"))
    asyncio.run(setup())
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls: "Fine.")
    asyncio.run(agent_runner._synthesize("m1", None, None))
    assert json.loads(asyncio.run(storage.get_mission("m1")).brief_warnings_json) == []


def test_token_budget_parsing(monkeypatch):
    monkeypatch.setattr(agent_runner.settings, "max_llm_tokens", 700)
    assert agent_runner._token_budget({}) == 700
    assert agent_runner._token_budget({"max_llm_tokens": None}) == 700, "null means not set"
    assert agent_runner._token_budget({"max_llm_tokens": 0}) == 0
    assert agent_runner._token_budget({"max_llm_tokens": "250"}) == 250
    assert agent_runner._token_budget({"max_llm_tokens": "lots"}) == 700
    assert agent_runner._token_budget({"max_llm_tokens": -5}) == 0


# ---------- one job per worker: the coroutine uses the job it was given ----------

def test_thread_hands_the_given_job_to_the_coroutine():
    _init()
    seen = []

    async def coro(mission_id, job_id=None):
        seen.append(job_id)

    agent_runner._thread(coro, "m1", "j-given")
    agent_runner._thread(coro, "m1")
    assert seen == ["j-given", None]


def _run_given(monkeypatch, coro_fn, given, other):
    _wire(monkeypatch, ["http://a.example/1"], satisfied=True)
    asyncio.run(coro_fn("m1", given))
    return jobs.get_job(given), jobs.get_job(other)


def test_run_collection_uses_the_given_job(monkeypatch):
    _init()
    given, other = jobs.create_mission_job("Q", 5), jobs.create_mission_job("Q", 5)
    _one_requirement(other, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    g, o = _run_given(monkeypatch, agent_runner._run_collection, given, other)
    assert g.done and g.stage == "done" and g.log
    assert not o.done and o.log == []


def test_run_planning_uses_the_given_job(monkeypatch):
    _init()
    given, other = jobs.create_mission_job("Q", 5), jobs.create_mission_job("Q", 5)
    asyncio.run(storage.insert_mission(_mission("m1", other, status="planning")))
    monkeypatch.setattr(agent_runner, "build_collection_plan", lambda agent, mid, q: [
        Requirement(id="r1", mission_id=mid, title="T", next_queries_json='["q"]')])
    g, o = _run_given(monkeypatch, agent_runner._run_planning, given, other)
    assert g.done and g.stage == "awaiting_approval"
    assert not o.done and o.log == []


def test_run_collection_falls_back_to_the_rows_job(monkeypatch):
    _init()
    row_job = jobs.create_mission_job("Q", 5)
    _one_requirement(row_job, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    _wire(monkeypatch, ["http://a.example/1"], satisfied=True)
    asyncio.run(agent_runner._run_collection("m1"))
    assert jobs.get_job(row_job).stage == "done"
