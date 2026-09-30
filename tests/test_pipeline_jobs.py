"""Job store lifecycle (finish_job, log_total, TTL) and the one-shot crawl
pipeline in jobs._run_job: authoritative ids, bounded concurrent extraction,
cooperative cancel, and a crashed worker never leaking its job slot."""
import asyncio
import threading
import time

import crawler
import extractor
import jobs
import search
import storage
from models import Document, ExtractedData, SearchResult


# ---------- finish_job / log_total / job_state ----------

def test_finish_job_sets_terminal_state_once():
    jid = jobs.create_job("q", 5, False, "")
    jobs.finish_job(jid, stage="error", error="boom")
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error" and j.error == "boom"
    first = j.finished_at
    assert first is not None

    time.sleep(0.01)
    jobs.finish_job(jid, stage="error", error="boom")   # idempotent
    j = jobs.get_job(jid)
    assert j.finished_at == first, "finished_at is stamped once"
    assert (j.done, j.stage, j.error) == (True, "error", "boom")


def test_finish_job_defaults_and_keeps_existing_error():
    jid = jobs.create_job("q", 5, False, "")
    jobs.update_job(jid, error="earlier")
    jobs.finish_job(jid)
    j = jobs.get_job(jid)
    assert (j.done, j.stage, j.error) == (True, "done", "earlier")


def test_finish_job_unknown_id_is_a_noop():
    jobs.finish_job("no-such-job")  # must not raise


def test_finish_if_running_only_finishes_live_jobs():
    jid = jobs.create_job("q", 5, False, "")
    jobs.finish_job(jid, stage="done")
    assert jobs.finish_if_running(jid, stage="error", error="late") is False
    assert jobs.get_job(jid).stage == "done"

    jid2 = jobs.create_job("q", 5, False, "")
    assert jobs.finish_if_running(jid2, stage="error", error="crash") is True
    assert (jobs.get_job(jid2).stage, jobs.get_job(jid2).error) == ("error", "crash")


def test_log_total_grows_past_the_kept_window():
    jid = jobs.create_job("q", 5, False, "")
    n = jobs._LOG_KEEP + 25
    for i in range(n):
        jobs.add_log(jid, "info", f"line {i}")
    j = jobs.get_job(jid)
    assert len(j.log) == jobs._LOG_KEEP, "log list is capped"
    assert j.log[-1].msg == f"line {n - 1}", "newest kept, oldest dropped"
    state = jobs.job_state(jid)
    assert state["log_total"] == n
    # The poll payload is a tail window; log_total lets the client tell how
    # many lines it has not rendered yet.
    assert state["log"][-1]["msg"] == f"line {n - 1}"
    assert len(state["log"]) <= 200


def test_job_state_elapsed_freezes_at_finish():
    jid = jobs.create_job("q", 5, False, "")
    with jobs._lock:
        jobs._store[jid].started_at = time.time() - 50
    jobs.finish_job(jid)
    s1 = jobs.job_state(jid)
    assert s1["finished_at"] is not None
    time.sleep(0.15)
    s2 = jobs.job_state(jid)
    assert s1["elapsed"] == s2["elapsed"], "a finished job's payload must be stable"
    assert 49 <= s2["elapsed"] <= 52


def test_prune_ttl_keys_on_finished_at():
    """A long job that only just finished must not be evicted as 'old'."""
    jid = jobs.create_job("long", 5, False, "")
    with jobs._lock:
        jobs._store[jid].started_at = time.time() - (jobs._DONE_TTL_S + 600)
    jobs.finish_job(jid)
    jobs.create_job("trigger prune", 5, False, "")
    assert jobs.get_job(jid) is not None


# ---------- _run_thread crash handling ----------

class _Fatal(BaseException):
    """Not an Exception: escapes a plain `except Exception`."""


def test_crashed_worker_finishes_its_job(monkeypatch):
    jid = jobs.create_job("q", 5, False, "")

    async def dies(job_id):
        raise _Fatal("killed")

    monkeypatch.setattr(jobs, "_run_job", dies)
    jobs._run_thread(jid)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"
    assert "killed" in (j.error or "")
    # The slot is released: a full house of new jobs fits again.
    for _ in range(jobs.MAX_ACTIVE_JOBS):
        jobs.create_job("fits", 5, False, "")


def test_worker_that_returns_without_finishing_is_closed(monkeypatch):
    jid = jobs.create_job("q", 5, False, "")

    async def forgets(job_id):
        return None

    monkeypatch.setattr(jobs, "_run_job", forgets)
    jobs._run_thread(jid)
    j = jobs.get_job(jid)
    assert j.done and j.stage == "error"


# ---------- _run_job ----------

def _sr(u):
    return SearchResult(url=u, title="t " + u, snippet="s")


def _doc(u, query="q", doc_id=None):
    return Document(id=doc_id or ("new-" + u), url=u, domain="d.example",
                    title="T " + u, search_query=query, crawled_at="t",
                    content_markdown="word " * 100, content_fit="word " * 100,
                    word_count=100)


def _wire(monkeypatch, urls, crawl_hook=None, search_hook=None):
    asyncio.run(storage.init_db())

    def fake_search(query, max_results=5):
        if search_hook:
            search_hook()
        return [_sr(u) for u in urls]

    async def fake_crawl(results, query, job_id, attempted=None, skip_on_cancel=False):
        if crawl_hook:
            crawl_hook(job_id)
        return [_doc(sr.url, query) for sr in results]

    monkeypatch.setattr(search, "web_search", fake_search)
    monkeypatch.setattr(crawler, "crawl_urls_with_progress", fake_crawl)


def _extraction(doc, prompt=""):
    return ExtractedData(id="x-" + doc.id, document_id=doc.id, model="m",
                         extracted_at="t", prompt=prompt, data_json="{}")


def test_run_job_uses_authoritative_ids_and_skips_failed_writes(monkeypatch):
    urls = ["http://a/1", "http://a/2", "http://a/3"]
    _wire(monkeypatch, urls)
    # A previous crawl of the same (url, query) already owns an id.
    asyncio.run(storage.upsert_document(_doc("http://a/1", doc_id="old-id")))

    real_upsert = storage.upsert_document

    async def flaky_upsert(doc):
        if doc.url == "http://a/2":
            raise RuntimeError("disk <full>")
        return await real_upsert(doc)

    monkeypatch.setattr(storage, "upsert_document", flaky_upsert)
    extracted = []

    def fake_extract(doc, prompt=""):
        extracted.append(doc.id)
        return _extraction(doc, prompt)

    monkeypatch.setattr(extractor, "extract_from_document", fake_extract)

    jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))

    j = jobs.get_job(jid)
    assert j.done and j.stage == "done", j.error
    assert j.document_ids == ["old-id", "new-http://a/3"]
    assert sorted(extracted) == ["new-http://a/3", "old-id"], \
        "extraction must use stored ids and skip the unstored doc"
    assert j.extract_total == 2 and j.extract_done == 2
    # The failure is surfaced, escaped.
    msgs = " ".join(entry.msg for entry in j.log)
    assert "disk &lt;full&gt;" in msgs and "<full>" not in msgs
    # Extractions reference a real document row.
    assert asyncio.run(storage.get_extractions_for_document("old-id"))


def test_run_job_extracts_concurrently(monkeypatch):
    """Two extractions must be in flight at once: each waits on a barrier
    that only opens when both are running."""
    _wire(monkeypatch, ["http://c/1", "http://c/2"])
    barrier = threading.Barrier(2, timeout=5)

    def rendezvous(doc, prompt=""):
        barrier.wait()
        return _extraction(doc, prompt)

    monkeypatch.setattr(extractor, "extract_from_document", rendezvous)
    jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert j.stage == "done" and j.extract_done == 2
    assert not any(entry.level == "err" for entry in j.log), [e.msg for e in j.log]


def test_extract_concurrency_is_shared():
    import agent_runner
    assert jobs.EXTRACT_CONCURRENCY == 4
    assert agent_runner.EXTRACT_CONCURRENCY == jobs.EXTRACT_CONCURRENCY


def test_cancel_before_crawl(monkeypatch):
    holder = {}
    crawled = []
    _wire(monkeypatch, ["http://a/1"],
          search_hook=lambda: jobs.request_cancel(holder["jid"]),
          crawl_hook=lambda job_id: crawled.append(job_id))
    holder["jid"] = jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert j.done and j.stage == "cancelled"
    assert crawled == []
    assert any("cancel" in entry.msg for entry in j.log)


def test_cancel_before_extraction_keeps_stored_docs(monkeypatch):
    _wire(monkeypatch, ["http://a/1", "http://a/2"],
          crawl_hook=lambda job_id: jobs.request_cancel(job_id))
    calls = []
    monkeypatch.setattr(extractor, "extract_from_document",
                        lambda d, p="": calls.append(d) or _extraction(d))
    jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert j.stage == "cancelled" and j.done
    assert calls == []
    assert len(j.document_ids) == 2, "crawled pages are still stored"


class _Page:
    """What crawl4ai's arun returns, for the real crawl_urls_with_progress."""

    def __init__(self, url):
        self.url = url
        self.success = True
        self.error_message = ""
        self.markdown = type("MD", (), {"raw_markdown": "word " * 120,
                                        "fit_markdown": "word " * 120})()
        self.links = {"internal": [], "external": []}
        self.metadata = {"title": "A real page"}
        self.redirected_url = None


def test_cancel_mid_crawl_skips_unfetched_pages_and_keeps_fetched_ones(monkeypatch):
    """Extraction off, so the crawl is the last stage: a cancel during it
    must stop the pages not fetched yet (marked skipped, never fetched), keep
    the ones already fetched, and end the job cancelled rather than done."""
    asyncio.run(storage.init_db())
    urls = [f"http://m.example/{i}" for i in range(7)] + ["http://m.example/7?q=<x>"]
    monkeypatch.setattr(search, "web_search",
                        lambda query, max_results=5: [_sr(u) for u in urls])
    holder = {}
    fetched = []

    class FakeCrawler:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def arun(self, url, config=None):
            fetched.append(url)
            if url == urls[0]:
                jobs.request_cancel(holder["jid"])   # flipped mid-batch
            await asyncio.sleep(0)
            return _Page(url)

    monkeypatch.setattr(crawler, "AsyncWebCrawler", FakeCrawler)
    holder["jid"] = jid = jobs.create_job("q", 8, False, "")
    asyncio.run(jobs._run_job(jid))

    j = jobs.get_job(jid)
    assert j.done and j.stage == "cancelled", (j.stage, j.error)
    status = {u.url: u.status for u in j.urls}
    # Only fetches already holding one of the crawler's 4 slots when the
    # cancel landed ran; every page queued behind them was skipped.
    assert urls[0] in fetched and len(fetched) <= 4
    assert {u for u, s in status.items() if s == "done"} == set(fetched)
    assert {u for u, s in status.items() if s == "skipped"} == set(urls) - set(fetched)
    assert j.crawl_done == len(urls), "skipped pages still complete the progress bar"
    # What was fetched is stored and kept.
    assert len(j.document_ids) == len(fetched)
    stored = asyncio.run(storage.get_all_documents())
    assert sorted(d.url for d in stored) == sorted(fetched)
    # The skip is reported, escaped.
    msgs = " ".join(entry.msg for entry in j.log)
    assert "skipped <code>http://m.example/7?q=&lt;x&gt;</code>" in msgs
    assert "<x>" not in msgs
    assert f"kept {len(fetched)} documents" in msgs


def test_cancel_between_document_extractions(monkeypatch):
    _wire(monkeypatch, ["http://a/1", "http://a/2", "http://a/3"])
    monkeypatch.setattr(jobs, "EXTRACT_CONCURRENCY", 1, raising=False)
    holder = {}
    calls = []

    def extract_then_cancel(doc, prompt=""):
        calls.append(doc.id)
        jobs.request_cancel(holder["jid"])
        return _extraction(doc, prompt)

    monkeypatch.setattr(extractor, "extract_from_document", extract_then_cancel)
    holder["jid"] = jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert len(calls) == 1, "no further document starts after a cancel"
    assert j.stage == "cancelled" and j.done


def test_no_results_finishes_done_with_reason(monkeypatch):
    _wire(monkeypatch, [])
    jid = jobs.create_job("q", 5, False, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert (j.done, j.stage, j.error) == (True, "done", "no search results")
    assert j.finished_at is not None


def test_cancel_after_every_extraction_started_still_finishes_done(monkeypatch):
    """Nothing was skipped, so the job completed: it is 'done', not 'cancelled'."""
    _wire(monkeypatch, ["http://a/1", "http://a/2"])
    barrier = threading.Barrier(2, timeout=5)
    holder = {}

    def both_running_then_cancel(doc, prompt=""):
        barrier.wait()          # both documents are past the cancel check
        jobs.request_cancel(holder["jid"])
        return _extraction(doc, prompt)

    monkeypatch.setattr(extractor, "extract_from_document", both_running_then_cancel)
    holder["jid"] = jid = jobs.create_job("q", 5, True, "")
    asyncio.run(jobs._run_job(jid))
    j = jobs.get_job(jid)
    assert (j.stage, j.extract_done) == ("done", 2)


def test_finish_job_documented_overwrite_semantics():
    """A later finish with a different stage overwrites stage/error but keeps
    the first finished_at; finish_if_running is the variant that sticks."""
    jid = jobs.create_job("q", 5, False, "")
    jobs.finish_job(jid, stage="done")
    first = jobs.get_job(jid).finished_at
    jobs.finish_job(jid, stage="cancelled", error="late")
    j = jobs.get_job(jid)
    assert (j.stage, j.error, j.finished_at) == ("cancelled", "late", first)
    assert jobs.finish_if_running(jid, stage="error") is False
    assert jobs.get_job(jid).stage == "cancelled"
