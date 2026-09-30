import asyncio
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from html import escape as _esc
from typing import Optional
from urllib.parse import urlparse


@dataclass
class JobUrl:
    url: str
    title: str = ""
    status: str = "pending"
    words: int = 0
    links_internal: int = 0
    links_external: int = 0
    error: Optional[str] = None


@dataclass
class JobLog:
    t: float
    level: str
    msg: str


@dataclass
class Job:
    id: str
    query: str
    max_results: int
    extract: bool
    extract_prompt: str
    started_at: float
    stage: str = "search"
    search_total: int = 0
    crawl_total: int = 0
    crawl_done: int = 0
    extract_total: int = 0
    extract_done: int = 0
    urls: list = field(default_factory=list)
    log: list = field(default_factory=list)
    document_ids: list = field(default_factory=list)
    error: Optional[str] = None
    done: bool = False
    # Wall-clock time the job reached a terminal stage (finish_job). Freezes
    # `elapsed` and anchors the finished-trace TTL.
    finished_at: Optional[float] = None
    # Monotonic count of every log line ever added. `log` itself is capped at
    # _LOG_KEEP, and the poll payload is a tail window, so clients use this to
    # know how many lines they have not rendered yet.
    log_total: int = 0
    # Budget telemetry (missions) + cooperative cancellation (missions and
    # one-shot crawls), read by the mission telemetry strip / crawl page.
    pass_num: int = 0
    sources_used: int = 0
    cancel_requested: bool = False


_store: dict[str, Job] = {}
_lock = threading.Lock()
_recent_job_ids: list[str] = []
_RECENT_LIMIT = 10

# Every live job is a daemon thread plus (during crawl) a headless Chromium.
# Cap what can be in flight, and evict old finished traces so _store cannot
# grow for the life of the process (security audit: unbounded job creation).
MAX_ACTIVE_JOBS = 6
_DONE_KEEP = 40                 # most recent finished traces kept for the UI
_DONE_TTL_S = 6 * 60 * 60      # ...or until they age out
_LOG_KEEP = 2000                # log lines kept per job (log_total keeps counting)

# LLM extractions run at once per job. Extraction is one independent call per
# document and was the slow tail of every run; the cap keeps a local model
# (and a hosted rate limit) from being swamped. Shared with agent_runner.
EXTRACT_CONCURRENCY = 4


class JobLimitReached(RuntimeError):
    """Raised when too many jobs are already running."""


def _ended_at(j: "Job") -> float:
    return j.finished_at or j.started_at


def _prune_locked() -> None:
    """Caller holds _lock. Drop finished jobs beyond the keep-window; pages
    degrade to their DB-rendered form exactly as after a restart. Age is
    measured from when a job finished, so a long run that just ended is not
    evicted as 'old' the moment it completes."""
    done = sorted((j for j in _store.values() if j.done),
                  key=_ended_at, reverse=True)
    now = time.time()
    for i, j in enumerate(done):
        if i >= _DONE_KEEP or now - _ended_at(j) > _DONE_TTL_S:
            _store.pop(j.id, None)


def _admit_locked() -> None:
    """Caller holds _lock. Refuse a new job when the in-flight cap is hit."""
    active = sum(1 for j in _store.values() if not j.done)
    if active >= MAX_ACTIVE_JOBS:
        raise JobLimitReached(
            f"{active} jobs already running (max {MAX_ACTIVE_JOBS}) — "
            "wait for one to finish")


def create_job(query: str, max_results: int, extract: bool, extract_prompt: str) -> str:
    job = Job(
        id=str(uuid.uuid4()),
        query=query,
        max_results=max_results,
        extract=extract,
        extract_prompt=extract_prompt,
        started_at=time.time(),
    )
    with _lock:
        _admit_locked()
        _prune_locked()
        _store[job.id] = job
        _recent_job_ids.insert(0, job.id)
        del _recent_job_ids[_RECENT_LIMIT:]
    return job.id


def create_mission_job(question: str, max_sources: int) -> str:
    """A live-trace Job for an agentic mission. Deliberately NOT added to the
    search sidebar (_recent_job_ids) — missions have their own /missions view."""
    job = Job(
        id=str(uuid.uuid4()),
        query=question,
        max_results=max_sources,
        extract=False,
        extract_prompt="",
        started_at=time.time(),
        stage="planning",
        crawl_total=max_sources,
    )
    with _lock:
        _admit_locked()
        _prune_locked()
        _store[job.id] = job
    return job.id


def get_sidebar_jobs() -> dict:
    with _lock:
        live = None
        previous = []
        for i, jid in enumerate(_recent_job_ids):
            j = _store.get(jid)
            if not j:
                continue
            entry = {
                "id": j.id,
                "query": j.query,
                "done": j.done,
                "running": not j.done,
                "has_documents": bool(j.document_ids),
            }
            if i == 0:
                live = entry
            else:
                previous.append(entry)
        return {"live": live, "previous": previous}


def get_in_memory_job_ids() -> set[str]:
    with _lock:
        return set(_store.keys())


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _store.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        for k, v in kwargs.items():
            setattr(j, k, v)


def _finish_locked(j: Job, stage: str, error: Optional[str]) -> None:
    j.done = True
    j.stage = stage
    if error is not None:
        j.error = error
    if j.finished_at is None:
        j.finished_at = time.time()


def finish_job(job_id: str, stage: str = "done", error: Optional[str] = None) -> None:
    """Move a job to a terminal stage and release its slot. The one way a job
    should end; `update_job(done=True)` leaves finished_at unset.

    Repeating a call with the same arguments changes nothing. A later call
    with a different stage (or a new error) does overwrite those fields;
    only finished_at is stamped once and kept. Use finish_if_running when an
    earlier terminal stage must stick (crash and cleanup paths)."""
    with _lock:
        j = _store.get(job_id)
        if j:
            _finish_locked(j, stage, error)


def finish_if_running(job_id: str, stage: str = "error",
                      error: Optional[str] = None) -> bool:
    """finish_job, but only for a job that has not already finished — the
    check and the write happen under one lock hold. For crash/cleanup paths
    that must not overwrite a job the worker ended properly. True if this
    call finished it."""
    with _lock:
        j = _store.get(job_id)
        if not j or j.done:
            return False
        _finish_locked(j, stage, error)
        return True


def inc_counter(job_id: str, field_name: str, by: int = 1) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        setattr(j, field_name, getattr(j, field_name) + by)


def add_log(job_id: str, level: str, msg: str) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        j.log.append(JobLog(round(time.time() - j.started_at, 2), level, msg))
        j.log_total += 1
        if len(j.log) > _LOG_KEEP:
            del j.log[:len(j.log) - _LOG_KEEP]


def add_urls(job_id: str, urls: list[JobUrl]) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        j.urls.extend(urls)


def update_url(job_id: str, url: str, **kwargs) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        for u in j.urls:
            if u.url == url:
                for k, v in kwargs.items():
                    setattr(u, k, v)
                return


def set_document_ids(job_id: str, ids: list[str]) -> None:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return
        j.document_ids = ids


def request_cancel(job_id: str) -> bool:
    """Ask a running job to stop. Cooperative: a mission stops at the next
    pass boundary (the current pass finishes and the brief is still
    synthesized from whatever was collected); a one-shot crawl stops before
    crawling, before extraction, or before its next document's extraction."""
    with _lock:
        j = _store.get(job_id)
        if not j or j.done:
            return False
        j.cancel_requested = True
        return True


def is_cancelled(job_id: str) -> bool:
    with _lock:
        j = _store.get(job_id)
        return bool(j and j.cancel_requested)


def job_state(job_id: str) -> Optional[dict]:
    with _lock:
        j = _store.get(job_id)
        if not j:
            return None
        # A finished job's elapsed is frozen, so its payload stops changing.
        end = j.finished_at if j.finished_at is not None else time.time()
        return {
            "id": j.id,
            "query": j.query,
            "stage": j.stage,
            "elapsed": round(end - j.started_at, 1),
            "finished_at": j.finished_at,
            "log_total": j.log_total,
            "search_total": j.search_total,
            "crawl_total": j.crawl_total,
            "crawl_done": j.crawl_done,
            "extract": j.extract,
            "extract_total": j.extract_total,
            "extract_done": j.extract_done,
            "urls": [asdict(u) for u in j.urls],
            "log": [asdict(l) for l in j.log[-200:]],
            "done": j.done,
            "error": j.error,
            "document_ids": list(j.document_ids),
            "pass_num": j.pass_num,
            "sources_used": j.sources_used,
            "cancel_requested": j.cancel_requested,
        }


def run_job_in_background(job_id: str) -> None:
    t = threading.Thread(target=_run_thread, args=(job_id,), daemon=True)
    t.start()


def _run_thread(job_id: str) -> None:
    """Thread body. Whatever happens inside — including a BaseException that
    slips past `except Exception`, or a code path that returns without
    finishing — the job ends in a terminal stage, so it never holds one of
    the MAX_ACTIVE_JOBS slots forever."""
    failure: Optional[str] = None
    try:
        asyncio.run(_run_job(job_id))
    except BaseException as e:  # noqa: BLE001 - the slot must be released
        traceback.print_exc()
        failure = str(e) or type(e).__name__
        add_log(job_id, "err", f"worker crashed: {_esc(failure)}")
    finally:
        finish_if_running(job_id, stage="error",
                          error=failure or "worker exited without finishing")


def _cancelled(job_id: str, note: str) -> None:
    add_log(job_id, "warn", f"cancelled by user · {note}")
    finish_job(job_id, stage="cancelled")


async def _run_job(job_id: str) -> None:
    from search import web_search
    from crawler import crawl_urls_with_progress
    from storage import upsert_document, insert_search, insert_extraction
    from models import SearchRecord
    from extractor import extract_from_document

    job = get_job(job_id)
    if not job:
        return

    try:
        update_job(job_id, stage="search")
        add_log(job_id, "info", f'searching duckduckgo for <em>"{_esc(job.query)}"</em>')
        search_results = web_search(job.query, max_results=job.max_results)
        update_job(job_id, search_total=len(search_results))

        if not search_results:
            add_log(job_id, "warn", "no results")
            finish_job(job_id, stage="done", error="no search results")
            return

        if is_cancelled(job_id):
            _cancelled(job_id, "before crawling")
            return

        domain_set = {urlparse(r.url).netloc for r in search_results}
        add_log(job_id, "ok", f"got <em>{len(search_results)}</em> results across <em>{len(domain_set)}</em> domains")

        add_urls(job_id, [JobUrl(url=sr.url, title=sr.title or sr.url) for sr in search_results])
        update_job(job_id, stage="crawl", crawl_total=len(search_results))
        add_log(job_id, "info", f"opening headless chromium · concurrency <em>4</em>")

        documents = await crawl_urls_with_progress(search_results, job.query, job_id)

        # upsert_document returns the id that is authoritative for the
        # (url, query) pair — an earlier crawl's id on a re-crawl, not
        # doc.id — so every later reference (document_ids, extractions) uses
        # it. A document that fails to store is reported and left out.
        stored = []
        for doc in documents:
            try:
                doc_id = await upsert_document(doc)
            except Exception as e:  # noqa: BLE001 - one bad row must not end the job
                traceback.print_exc()
                add_log(job_id, "err",
                        f"could not store <code>{_esc(doc.url)}</code>: {_esc(str(e)[:120])}")
                continue
            stored.append(doc if doc_id == doc.id else doc.model_copy(update={"id": doc_id}))
        set_document_ids(job_id, [d.id for d in stored])
        add_log(job_id, "ok", f"stored <em>{len(stored)}</em> documents")

        search_record = SearchRecord(
            id=str(uuid.uuid4()),
            query=job.query,
            executed_at=datetime.now(timezone.utc).isoformat(),
            result_count=len(stored),
            job_id=job_id,
        )
        await insert_search(search_record)

        if job.extract and stored:
            if is_cancelled(job_id):
                _cancelled(job_id, f"kept {len(stored)} documents, skipped extraction")
                return
            await _extract_all(job_id, job.extract_prompt, stored,
                               extract_from_document, insert_extraction)
            live = get_job(job_id)
            n_done = live.extract_done if live else len(stored)
            if n_done < len(stored):  # a cancel skipped at least one document
                _cancelled(job_id, f"extracted {n_done} of {len(stored)} documents")
                return

        add_log(job_id, "ok", "agent finished")
        finish_job(job_id, stage="done")
    except Exception as e:
        traceback.print_exc()
        add_log(job_id, "err", f"job failed: {_esc(str(e))}")
        finish_job(job_id, stage="error", error=str(e))


async def _extract_all(job_id: str, prompt: str, docs: list,
                       extract_from_document, insert_extraction) -> None:
    """Run extraction over `docs`, EXTRACT_CONCURRENCY at a time, each call on
    a worker thread (litellm is synchronous). A cancel stops any document that
    has not started yet; ones already in flight finish."""
    update_job(job_id, stage="extract", extract_total=len(docs))
    add_log(job_id, "info",
            f"running llm extraction on <em>{len(docs)}</em> documents "
            f"· concurrency <em>{EXTRACT_CONCURRENCY}</em>")
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def one(doc) -> None:
        async with sem:
            if is_cancelled(job_id):
                return
            add_log(job_id, "info", f"extracting <em>{_esc((doc.title or doc.domain)[:70])}</em>")
            try:
                extraction = await asyncio.to_thread(extract_from_document, doc, prompt)
                if extraction:
                    await insert_extraction(extraction)
                    add_log(job_id, "ok", f"extracted <code>{_esc(doc.domain)}</code>")
                else:
                    add_log(job_id, "warn", f"no extraction for <code>{_esc(doc.domain)}</code>")
            except Exception as e:  # noqa: BLE001 - one document, not the job
                traceback.print_exc()
                add_log(job_id, "err",
                        f"extraction failed for <code>{_esc(doc.domain)}</code>: "
                        f"{_esc(type(e).__name__)}")
            inc_counter(job_id, "extract_done")

    await asyncio.gather(*(one(d) for d in docs))
