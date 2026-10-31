"""WP7: the pipeline's ranking, embedding and export hooks.

Driven through the litellm stub (tests/stubs/litellm.py): its reranker
scores a string by the share of the query's tokens it contains, so a search
order is reversed by writing titles whose overlap with the requirement grows
down the list; its embedder is deterministic and records every call. The
search, crawl and assess steps are faked the way test_pipeline_runner does.
"""
import asyncio
import json

import litellm
import pytest

import agent_runner
import crawler
import embeddings
import extractor
import jobs
import search
import storage
import vector_export
from agent_assessor import Assessment
from brief import ordered_sources
from config import settings
from models import Agent, Document, LlmCall, Mission, Requirement, SearchResult

RERANK = "cohere/rerank-v5.0-pro"
PRO = "cohere/embed-v5.0-pro"
DIM = 8


class AuthenticationError(Exception):
    """Named like litellm's: not retried, so one failed attempt."""


@pytest.fixture(autouse=True)
def _models(monkeypatch):
    """Rerank and embeddings on (stubbed), export off, budget excludes
    embed/rerank rows: the defaults a user gets, at a small dimension."""
    litellm.reset_calls()
    monkeypatch.setattr(settings, "rerank_model", RERANK)
    monkeypatch.setattr(settings, "embed_model", PRO)
    monkeypatch.setattr(settings, "embed_query_model", "")
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "embed_tokens_count_toward_budget", False)
    monkeypatch.setattr(settings, "export_on_mission_done", False)
    asyncio.run(storage.init_db())
    asyncio.run(storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                           persona_prompt="p", created_at="t")))
    yield
    litellm.reset_calls()


def _mission(mid, job_id=None, status="collecting", budget=None, question="Q"):
    return Mission(id=mid, agent_id="a1", question=question, status=status,
                   job_id=job_id, budget_json=json.dumps(budget or {}), created_at="t")


def _doc(url, body="w " * 100, crawled_at="2026-01-01T00:00:00", title=None, words=100):
    return Document(id="d-" + url, url=url, domain=url.split("/")[2],
                    title=title or ("Real " + url), search_query="Q",
                    crawled_at=crawled_at, content_markdown=body, content_fit="",
                    word_count=words)


def _log_text(jid):
    return " ".join(entry.msg for entry in jobs.get_job(jid).log)


def _one_requirement(jid, budget, title="r1", description=""):
    async def setup():
        await storage.insert_mission(_mission("m1", jid, budget=budget))
        await storage.insert_requirement(Requirement(
            id="r1", mission_id="m1", title=title, description=description,
            next_queries_json='["q"]'))
    asyncio.run(setup())


def _wire(monkeypatch, results, crawled=None):
    """Fake search (fixed results), crawl (one doc per url, recorded in
    `crawled`), assess (always a gap) and brief."""
    monkeypatch.setattr(agent_runner, "web_search_ex", lambda q, n=5: (list(results), "stub"))

    async def fake_crawl(results, question, job_id, attempted=None, aliases=None):
        if crawled is not None:
            crawled.extend(sr.url for sr in results)
        if attempted is not None:
            attempted.update(sr.url for sr in results)
        return [_doc(sr.url) for sr in results]

    monkeypatch.setattr(agent_runner, "crawl_urls_with_progress", fake_crawl)
    monkeypatch.setattr(agent_runner, "assess_requirement",
                        lambda req, docs, **k: Assessment(False, "low", "gap", []))
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")


def _sr(url, title):
    return SearchResult(url=url, title=title, snippet="")


def _candidate_reranks():
    """The rerank calls made over search candidates. The brief's
    rank_for_brief reranks too, against the mission question ("Q")."""
    return [c for c in litellm.rerank_calls if c["query"] != "Q"]


# Search order: the engine's first result shares no token with the
# requirement "cats mats", the second one, the third both, so the overlap
# reranker reverses it exactly.
SEARCH = [_sr("http://s/dogs", "dogs chase cars"),
          _sr("http://s/mats", "rubber mats"),
          _sr("http://s/cats", "cats on mats")]
BUDGET = {"max_passes": 1, "max_sources": 2, "per_req_attempts": 1}


# ---------- _collect_one: candidates reranked before the cap ----------

def test_candidates_are_reranked_before_the_cap(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, BUDGET, title="cats mats")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == ["http://s/cats", "http://s/mats"], "top 2 by relevance, not search order"
    assert "reranked 3 candidates → crawling top 2" in _log_text(jid)
    [kw] = _candidate_reranks()
    assert kw["query"] == "cats mats"
    assert kw["documents"] == ["dogs chase cars", "rubber mats", "cats on mats"]
    assert asyncio.run(storage.get_mission("m1")).status == "done"


def test_rerank_query_is_title_and_description(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, BUDGET, title="cats", description="and mats")
    _wire(monkeypatch, SEARCH)
    asyncio.run(agent_runner._run_collection("m1"))
    assert [c["query"] for c in _candidate_reranks()] == ["cats. and mats"]


def test_search_order_kept_when_rerank_disabled(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "")
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, BUDGET, title="cats mats")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == ["http://s/dogs", "http://s/mats"]
    assert litellm.rerank_calls == []
    assert "reranked" not in _log_text(jid)


def test_search_order_kept_when_rerank_fails(monkeypatch):
    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", boom)
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, BUDGET, title="cats mats")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == ["http://s/dogs", "http://s/mats"]
    assert "reranked" not in _log_text(jid)
    assert asyncio.run(storage.get_mission("m1")).status == "done"


def test_no_rerank_when_every_candidate_fits_the_cap(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1},
                     title="cats mats")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)
    asyncio.run(agent_runner._run_collection("m1"))
    assert crawled == [sr.url for sr in SEARCH]
    assert _candidate_reranks() == []
    assert "reranked" not in _log_text(jid)


def test_unchanged_order_is_not_logged_as_reranked(monkeypatch):
    """The reranker agrees with the engine: nothing to tell the operator."""
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, BUDGET, title="cats mats")
    _wire(monkeypatch, list(reversed(SEARCH)))
    asyncio.run(agent_runner._run_collection("m1"))
    assert len(_candidate_reranks()) == 1
    assert "reranked" not in _log_text(jid)


# ---------- _collect_one: the stored pages are embedded ----------

def test_collected_pages_are_embedded_and_logged(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    _wire(monkeypatch, SEARCH)

    asyncio.run(agent_runner._run_collection("m1"))

    docs = asyncio.run(storage.get_mission_documents("m1"))
    rows = asyncio.run(storage.get_chunk_embeddings(PRO, [d.id for d in docs]))
    assert {did for _cid, did, _blob in rows} == {d.id for d in docs}
    assert all(len(blob) == DIM * 4 for _c, _d, blob in rows)
    assert f"embedded {len(rows)} chunks · {PRO}" in _log_text(jid)
    assert litellm.embedding_calls and all(
        c["model"] == PRO for c in litellm.embedding_calls)
    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage["by_purpose"]["embed"]["calls"] >= 1, "charged to the mission's telemetry"


def test_embedding_failure_costs_nothing(monkeypatch):
    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "embedding", boom)
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert len(crawled) == 3
    assert "embedded" not in _log_text(jid)
    docs = asyncio.run(storage.get_mission_documents("m1"))
    assert asyncio.run(storage.get_chunk_embeddings(PRO, [d.id for d in docs])) == []
    m = asyncio.run(storage.get_mission("m1"))
    assert m.status == "done" and m.brief_markdown == "brief"
    assert jobs.get_job(jid).stage == "done"


def test_embeddings_off_is_silent(monkeypatch):
    monkeypatch.setattr(settings, "embed_model", "")
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    _wire(monkeypatch, SEARCH)
    asyncio.run(agent_runner._run_collection("m1"))
    assert litellm.embedding_calls == []
    assert "embedded" not in _log_text(jid)


# ---------- token budget: embed/rerank rows do not count by default ----------

def _spend(mission_id, tokens, purpose):
    asyncio.run(storage.insert_llm_call(LlmCall(
        purpose=purpose, tier=purpose, model="m", mission_id=mission_id,
        prompt_tokens=tokens, completion_tokens=0)))


def test_budget_excludes_embed_and_rerank_rows_by_default(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1,
                           "max_llm_tokens": 100})
    _spend("m1", 150, "embed")
    _spend("m1", 150, "rerank")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled, "embedding spend is not LLM spend"
    assert "token budget" not in _log_text(jid)
    # The telemetry roll-up is unfiltered: the page still shows every row.
    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage["prompt_tokens"] >= 300


def test_budget_counts_embed_rows_when_the_setting_says_so(monkeypatch):
    monkeypatch.setattr(settings, "embed_tokens_count_toward_budget", True)
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1,
                           "max_llm_tokens": 100})
    _spend("m1", 150, "embed")
    crawled = []
    _wire(monkeypatch, SEARCH, crawled=crawled)

    asyncio.run(agent_runner._run_collection("m1"))

    assert crawled == []
    assert "token budget reached" in _log_text(jid)
    [r1] = asyncio.run(storage.get_requirements_for_mission("m1"))
    assert r1.assessment_missing == "not attempted: token budget reached"


def test_budget_exclusions_follow_the_setting(monkeypatch):
    assert agent_runner._budget_exclusions() == ("embed", "rerank")
    monkeypatch.setattr(settings, "embed_tokens_count_toward_budget", True)
    assert agent_runner._budget_exclusions() == ()


# ---------- _synthesize: one reranked order for prompt and rail ----------

QUESTION = "solar battery storage"


def _mission_with_sources(jid=None, question=QUESTION):
    """Two usable sources: the newest (first from storage, so first in
    ordered_sources) is off topic, the older one answers the question."""
    async def setup():
        await storage.insert_mission(_mission("m1", jid, question=question))
        await storage.insert_requirement(Requirement(id="r1", mission_id="m1", title="T"))
        for doc_id, body, when in [
                ("offtopic", "sourdough bread and pastry in the kitchen " * 20,
                 "2026-01-03"),
                ("ontopic", "solar battery storage in one paragraph " * 20,
                 "2026-01-01")]:
            real_id = await storage.upsert_document(Document(
                id=doc_id, url="http://x/" + doc_id, domain="x", title="Title " + doc_id,
                search_query="Q", crawled_at=when, content_markdown=body,
                content_fit="", word_count=140))
            await storage.link_mission_document("m1", "r1", real_id)
    asyncio.run(setup())


def test_synthesize_persists_the_reranked_order_and_passes_it_to_the_brief(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    captured = {}

    def fake_brief(mission, reqs, docs, new_urls, ordered=None):
        captured["docs"] = docs
        captured["ordered"] = ordered
        return "brief citing [1]"

    monkeypatch.setattr(agent_runner, "synthesize_brief", fake_brief)
    asyncio.run(agent_runner._synthesize("m1", None, jid))

    m = asyncio.run(storage.get_mission("m1"))
    assert m.status == "done"
    ids = json.loads(m.brief_sources_json)
    assert ids == ["ontopic", "offtopic"]
    assert ids == [d.id for d in captured["ordered"]], "the prompt was numbered with it"
    assert ids != [d.id for d in ordered_sources(captured["docs"])], "today's order differs"
    assert "brief sources reranked" in _log_text(jid)
    assert litellm.rerank_calls[-1]["query"] == QUESTION


def test_synthesize_keeps_todays_order_when_rerank_is_off(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "")
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    captured = {}

    def fake_brief(mission, reqs, docs, new_urls, ordered=None):
        captured["docs"], captured["ordered"] = docs, ordered
        return "brief"

    monkeypatch.setattr(agent_runner, "synthesize_brief", fake_brief)
    asyncio.run(agent_runner._synthesize("m1", None, jid))

    m = asyncio.run(storage.get_mission("m1"))
    ids = json.loads(m.brief_sources_json)
    assert ids == [d.id for d in ordered_sources(captured["docs"])] == ["offtopic", "ontopic"]
    assert ids == [d.id for d in captured["ordered"]]
    assert "brief sources reranked" not in _log_text(jid)
    assert litellm.rerank_calls == []


def test_brief_warnings_resolve_citations_against_the_reranked_order(monkeypatch):
    """[1] is the on-topic source once reranked; a junk citation check that
    re-derived today's order would judge the wrong document."""
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)

    async def add_junk():
        real_id = await storage.upsert_document(Document(
            id="junk", url="http://x/junk", domain="x", title="Just a moment...",
            search_query="Q", crawled_at="2026-01-02", content_markdown="w " * 5,
            content_fit="", word_count=5))
        await storage.link_mission_document("m1", "r1", real_id)
    asyncio.run(add_junk())
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "Solar [3].")
    asyncio.run(agent_runner._synthesize("m1", None, jid))

    m = asyncio.run(storage.get_mission("m1"))
    assert json.loads(m.brief_sources_json) == ["ontopic", "offtopic", "junk"]
    kinds = [w["kind"] for w in json.loads(m.brief_warnings_json)]
    assert "junk_citation" in kinds, "[3] is the junk page in the stored order"


# ---------- _synthesize: auto-export never fails the mission ----------

def _export_result(**kw):
    res = vector_export.ExportResult(collection="quarry")
    for k, v in kw.items():
        setattr(res, k, v)
    return res


def test_auto_export_is_off_by_default(monkeypatch):
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    calls = []

    async def fake_export(ids, **kw):
        calls.append(ids)
        return _export_result()

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")
    asyncio.run(agent_runner._synthesize("m1", None, jid))
    assert calls == []
    assert "export" not in _log_text(jid)


def test_auto_export_runs_after_done_and_logs_the_summary(monkeypatch):
    monkeypatch.setattr(settings, "export_on_mission_done", True)
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    seen = {}

    async def fake_export(ids, **kw):
        seen["ids"] = list(ids)
        seen["kw"] = kw
        seen["status"] = (await storage.get_mission("m1")).status
        seen["job_done"] = jobs.get_job(jid).done
        return _export_result(exported=list(ids), chunks=4)

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")
    asyncio.run(agent_runner._synthesize("m1", None, jid))

    assert sorted(seen["ids"]) == ["offtopic", "ontopic"]
    assert seen["kw"]["mission_id"] == "m1"
    assert seen["status"] == "done", "the mission is marked done before the export"
    assert seen["job_done"] is False, "and the job is finished after it"
    # Escaped like every other trace line (log strings are injected into the DOM).
    assert "exported 2 · skipped 0 · failed 0 · 4 chunks → &#x27;quarry&#x27;" in _log_text(jid)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "done"


def test_auto_export_failure_does_not_fail_the_mission(monkeypatch):
    monkeypatch.setattr(settings, "export_on_mission_done", True)
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)

    async def fake_export(ids, **kw):
        raise vector_export.ExportError("cannot reach Qdrant at <http://q:6333>")

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")
    asyncio.run(agent_runner._synthesize("m1", None, jid))

    m = asyncio.run(storage.get_mission("m1"))
    assert m.status == "done" and m.brief_markdown == "brief" and m.error is None
    log = _log_text(jid)
    assert "export failed: ExportError: cannot reach Qdrant at &lt;http://q:6333&gt;" in log
    j = jobs.get_job(jid)
    assert j.done and j.stage == "done" and j.error is None


def test_auto_export_runs_inside_a_full_collection(monkeypatch):
    """End to end: a collection run with export on finishes done with the
    export line in its trace, whether the export worked or not."""
    monkeypatch.setattr(settings, "export_on_mission_done", True)

    async def fake_export(ids, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    jid = jobs.create_mission_job("Q", 5)
    _one_requirement(jid, {"max_passes": 1, "max_sources": 5, "per_req_attempts": 1})
    _wire(monkeypatch, SEARCH)
    asyncio.run(agent_runner._run_collection("m1"))
    assert asyncio.run(storage.get_mission("m1")).status == "done"
    assert "export failed: RuntimeError: boom" in _log_text(jid)
    assert jobs.get_job(jid).stage == "done"


# ---------- jobs._run_job: the one-shot crawl embeds what it stored ----------

def _wire_job(monkeypatch, urls):
    monkeypatch.setattr(search, "web_search", lambda query, max_results=5: [
        SearchResult(url=u, title="t " + u, snippet="s") for u in urls])

    async def fake_crawl(results, query, job_id, attempted=None, skip_on_cancel=False):
        return [Document(id="new-" + sr.url, url=sr.url, domain="d.example",
                         title="T " + sr.url, search_query=query, crawled_at="t",
                         content_markdown="word " * 100, content_fit="",
                         word_count=100) for sr in results]

    monkeypatch.setattr(crawler, "crawl_urls_with_progress", fake_crawl)
    monkeypatch.setattr(extractor, "extract_from_document", lambda doc, prompt="": None)


def test_run_job_embeds_the_stored_documents(monkeypatch):
    _wire_job(monkeypatch, ["http://a/1", "http://a/2"])
    jid = jobs.create_job("q", 5, False, "")
    asyncio.run(jobs._run_job(jid))

    j = jobs.get_job(jid)
    assert j.done and j.stage == "done"
    rows = asyncio.run(storage.get_chunk_embeddings(PRO, j.document_ids))
    assert {did for _c, did, _b in rows} == set(j.document_ids)
    assert f"embedded <em>{len(rows)}</em> chunks · {PRO}" in _log_text(jid)
    assert all(c.get("mission_id") is None for c in litellm.embedding_calls)


def test_run_job_survives_an_embedding_failure(monkeypatch):
    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "embedding", boom)
    _wire_job(monkeypatch, ["http://a/1"])
    jid = jobs.create_job("q", 5, False, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert j.done and j.stage == "done" and j.error is None
    assert "embedded" not in _log_text(jid)
    assert len(j.document_ids) == 1


def test_run_job_embedding_call_goes_through_embed_pending_documents(monkeypatch):
    seen = []

    async def fake_embed(ids, mission_id=None):
        seen.append((list(ids), mission_id))
        return 3

    monkeypatch.setattr(embeddings, "embed_pending_documents", fake_embed)
    _wire_job(monkeypatch, ["http://a/1", "http://a/2"])
    jid = jobs.create_job("q", 5, False, "")
    asyncio.run(jobs._run_job(jid))
    assert seen == [(jobs.get_job(jid).document_ids, None)]
    assert "embedded <em>3</em> chunks" in _log_text(jid)


def test_auto_export_is_skipped_while_another_export_runs(monkeypatch):
    """The auto-export goes through vector_export's one-at-a-time lock, so a
    background export in flight means one skip line, not a second export."""
    monkeypatch.setattr(settings, "export_on_mission_done", True)
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    calls = []

    async def fake_export(ids, **kw):
        calls.append(list(ids))
        return _export_result(exported=list(ids))

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")
    assert vector_export._export_lock.acquire(blocking=False)
    try:
        asyncio.run(agent_runner._synthesize("m1", None, jid))
    finally:
        vector_export._export_lock.release()
    assert calls == []
    assert "export skipped: another export is running" in _log_text(jid)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "done"


def test_auto_export_sets_the_shared_export_status_while_it_runs(monkeypatch):
    monkeypatch.setattr(settings, "export_on_mission_done", True)
    jid = jobs.create_mission_job("Q", 5)
    _mission_with_sources(jid)
    seen = {}

    async def fake_export(ids, **kw):
        status = vector_export.export_status()
        seen["running"] = status["running"]
        seen["mission_id"] = status["mission_id"]
        seen["locked"] = vector_export._export_lock.locked()
        return _export_result(exported=list(ids))

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    monkeypatch.setattr(agent_runner, "synthesize_brief",
                        lambda mission, reqs, docs, new_urls, ordered=None: "brief")
    asyncio.run(agent_runner._synthesize("m1", None, jid))
    assert seen == {"running": True, "mission_id": "m1", "locked": True}
    status = vector_export.export_status()
    assert status["running"] is False and status["mission_id"] == "m1"
    assert status["result"]["exported"] and not vector_export._export_lock.locked()
    assert "exported 2 · skipped 0" in _log_text(jid)
