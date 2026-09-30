"""Orchestration for agentic collection.

Two background entry points, split by the human approval gate:

    start_planning(mission_id)   -> plan, then status=awaiting_approval
    start_collection(mission_id) -> collect/assess/re-task loop, then brief

Both run in daemon threads (one asyncio loop each) and stream progress into the
in-memory job store so the existing live-log / SSE machinery works unchanged.
Mission status, requirements, and the brief are the durable source of truth in
SQLite; the job store only carries the live trace.
"""
import asyncio
import json
import threading
import traceback
from datetime import datetime, timezone
from html import escape as _esc

import jobs
from jobs import JobUrl, EXTRACT_CONCURRENCY
from search import web_search
from crawler import crawl_urls_with_progress
from agent_planner import build_collection_plan
from agent_assessor import assess_requirement
from brief import synthesize_brief, ordered_sources
from extractor import extract_from_document
from storage import (
    get_agent, get_mission, update_mission,
    insert_requirement, update_requirement, get_requirements_for_mission,
    upsert_document, link_mission_document, insert_extraction,
    get_requirement_documents, get_mission_documents,
    get_latest_finished_mission,
)

PER_QUERY_RESULTS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Thread launchers ---

def start_planning(mission_id: str) -> None:
    threading.Thread(target=_thread, args=(_run_planning, mission_id), daemon=True).start()


def start_collection(mission_id: str) -> None:
    threading.Thread(target=_thread, args=(_run_collection, mission_id), daemon=True).start()


def _mission_job_id(mission_id: str):
    try:
        mission = asyncio.run(get_mission(mission_id))
    except Exception:  # noqa: BLE001 - best effort; the caller has a fallback
        traceback.print_exc()
        return None
    return mission.job_id if mission else None


def _thread(coro_fn, mission_id: str) -> None:
    """Thread body for both stages. The mission's live job always ends in a
    terminal stage — on a crash (any BaseException, not just Exception), and
    on a path that returns without finishing it — so a dead worker never
    holds one of the job store's MAX_ACTIVE_JOBS slots."""
    # Read up front as well as at the end: if the crash was the database
    # going away, the lookup in `finally` fails too.
    job_id = _mission_job_id(mission_id)
    failure = None
    try:
        asyncio.run(coro_fn(mission_id))
    except BaseException as e:  # noqa: BLE001 - the slot must be released
        traceback.print_exc()
        failure = str(e) or type(e).__name__
        try:
            asyncio.run(update_mission(mission_id, status="error", error=failure,
                                       finished_at=_now()))
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
    finally:
        job_id = _mission_job_id(mission_id) or job_id
        if job_id:
            if failure:
                jobs.add_log(job_id, "err", f"worker crashed: {_esc(failure)}")
            jobs.finish_if_running(job_id, stage="error",
                                   error=failure or "worker exited without finishing")


# --- Stage 1: planning ---

async def _run_planning(mission_id: str) -> None:
    mission = await get_mission(mission_id)
    if not mission:
        return
    agent = await get_agent(mission.agent_id)
    job_id = mission.job_id

    await update_mission(mission_id, status="planning", started_at=_now())
    if job_id:
        jobs.update_job(job_id, stage="planning")
        jobs.add_log(job_id, "info", f'planning collection for <em>"{_esc(mission.question)}"</em>')

    try:
        requirements = await asyncio.to_thread(
            build_collection_plan, agent, mission_id, mission.question
        )
    except Exception as e:  # noqa: BLE001
        await update_mission(mission_id, status="error", error=str(e), finished_at=_now())
        if job_id:
            jobs.add_log(job_id, "err", f"planning failed: {_esc(str(e))}")
            jobs.finish_job(job_id, stage="error", error=str(e))
        return

    for req in requirements:
        await insert_requirement(req)

    plan_summary = [{"title": r.title, "description": r.description} for r in requirements]

    # A scheduled run has nobody at the keyboard, so it approves its own plan
    # and goes straight on to collecting. Interactive runs still stop here.
    budget = {}
    try:
        budget = json.loads(mission.budget_json or "{}")
    except json.JSONDecodeError:
        pass
    if budget.get("auto_approve"):
        await update_mission(mission_id, status="collecting",
                             plan_json=json.dumps(plan_summary))
        if job_id:
            jobs.add_log(job_id, "ok",
                         f"plan ready: <em>{len(requirements)}</em> requirements "
                         f"— auto-approved (scheduled run)")
        await _run_collection(mission_id)
        return

    await update_mission(
        mission_id, status="awaiting_approval", plan_json=json.dumps(plan_summary)
    )
    if job_id:
        jobs.add_log(job_id, "ok",
                     f"plan ready: <em>{len(requirements)}</em> requirements — awaiting approval")
        # Waiting on a human holds no worker, so it must not hold a job slot
        # either: a few plans left at the gate would otherwise block every new
        # job. Approval starts collection on a fresh job.
        jobs.finish_job(job_id, stage="awaiting_approval")


# --- Stage 2: collection loop ---

async def _run_collection(mission_id: str) -> None:
    mission = await get_mission(mission_id)
    if not mission:
        return
    agent = await get_agent(mission.agent_id)
    job_id = mission.job_id
    budget = json.loads(mission.budget_json or "{}")
    max_passes = int(budget.get("max_passes", agent.default_max_passes if agent else 4))
    max_sources = int(budget.get("max_sources", agent.default_max_sources if agent else 30))
    per_req_attempts = int(budget.get("per_req_attempts", agent.default_per_req_attempts if agent else 3))

    await update_mission(mission_id, status="collecting")
    if job_id:
        jobs.update_job(job_id, stage="collecting")
        jobs.add_log(job_id, "info",
                     f"collecting · budget {max_sources} sources / {max_passes} passes")

    collected: dict[str, str] = {}   # url -> doc_id, crawled by THIS run
    job_urls: set[str] = set()       # urls already shown in the live trace

    # What the mission already holds (a retask re-enters here). Those URLs are
    # never crawled again, but they are not charged to this run's budget
    # either: `collected` starts empty, so a retask gets the fresh budget the
    # route promises. `known` lets a held page that resurfaces in search be
    # linked to the requirement asking for it without re-fetching it.
    known: dict[str, str] = {d.url: d.id for d in await get_mission_documents(mission_id)}
    attempted: set[str] = set(known)

    # Fair share of the source budget per requirement, so one greedy
    # requirement can't starve the rest within a pass. The global max_sources
    # remains the hard ceiling.
    all_reqs = await get_requirements_for_mission(mission_id)
    per_req_cap = max(1, max_sources // max(1, len(all_reqs)))

    # Why a requirement that was never tried stayed unmet (see the end).
    stop_reason = "pass budget exhausted"
    for pass_num in range(1, max_passes + 1):
        reqs = await get_requirements_for_mission(mission_id)
        pending = [r for r in reqs if r.status == "pending"]
        if not pending:
            break
        # Cooperative stop: requested from the mission page between passes, so
        # the collected sources are still written up into a brief.
        if job_id and jobs.is_cancelled(job_id):
            jobs.add_log(job_id, "warn", "stop requested — finishing after this pass")
            stop_reason = "stopped by user"
            break
        if job_id:
            jobs.update_job(job_id, pass_num=pass_num)
            jobs.add_log(job_id, "info",
                         f"pass <em>{pass_num}</em> · {len(pending)} open requirements")

        for req in pending:
            if len(collected) >= max_sources:
                break
            try:
                await _collect_one(
                    mission, req, collected, job_urls, job_id,
                    max_sources, per_req_cap, per_req_attempts,
                    attempted=attempted, known=known,
                )
            except Exception as e:  # noqa: BLE001
                # Isolate a requirement's failure: burn one of its attempts and
                # carry on, rather than losing the whole mission (and the
                # sources already crawled) to one bad call.
                traceback.print_exc()
                await update_requirement(
                    req.id, attempts=req.attempts + 1,
                    status=("unmet" if req.attempts + 1 >= per_req_attempts else "pending"),
                    assessment_missing=f"collection failed: {type(e).__name__}",
                    assessment_confidence="unknown",
                )
                if job_id:
                    jobs.add_log(job_id, "err",
                                 f"error on <em>{_esc(req.title)}</em>: {_esc(type(e).__name__)}")

        if len(collected) >= max_sources:
            if job_id:
                jobs.add_log(job_id, "warn", "source budget reached")
            stop_reason = "source budget exhausted"
            break

    # Anything still pending after the pass budget is an unmet gap. One that
    # was never attempted says why, rather than looking like a failed search.
    for r in await get_requirements_for_mission(mission_id):
        if r.status == "pending":
            fields = {"status": "unmet"}
            if r.attempts == 0:
                fields["assessment_missing"] = f"not attempted: {stop_reason}"
            await update_requirement(r.id, **fields)

    # Optional LLM extraction over the collected sources (applies regardless of
    # how they were gathered).
    if budget.get("extract"):
        await _extract_sources(mission_id, budget.get("extract_prompt", ""), job_id)

    await _synthesize(mission_id, agent, job_id)


async def _collect_one(mission, req, collected, job_urls, job_id,
                       max_sources, per_req_cap, per_req_attempts,
                       attempted: set[str] | None = None,
                       known: dict[str, str] | None = None) -> None:
    """Search, crawl and assess a single requirement. Raising here costs this
    requirement an attempt, not the mission.

    `attempted` is mission-scoped: every URL already tried (requested or
    redirected-to, failed or junk included) and every URL the mission held
    before this run. None of them is fetched again. `known` maps the held
    URLs to their document ids."""
    mission_id = mission.id
    attempted = attempted if attempted is not None else set()
    known = known or {}
    if req.attempts >= per_req_attempts:
        await update_requirement(req.id, status="unmet")
        if job_id:
            jobs.add_log(job_id, "warn", f"unmet (capped): <em>{_esc(req.title)}</em>")
        return

    queries = json.loads(req.next_queries_json or "[]") or [mission.question]
    if job_id:
        jobs.add_log(job_id, "info", f"collecting: <em>{_esc(req.title)}</em>")

    # Search across this requirement's queries.
    results = []
    seen_q_urls: set[str] = set()
    for q in queries:
        for sr in await asyncio.to_thread(web_search, q, PER_QUERY_RESULTS):
            if sr.url not in seen_q_urls:
                seen_q_urls.add(sr.url)
                results.append(sr)

    # New URLs to crawl, bounded by this requirement's fair share and the
    # remaining global source budget.
    remaining = max_sources - len(collected)
    cap = max(0, min(per_req_cap, remaining))
    to_crawl = [sr for sr in results
                if sr.url not in collected and sr.url not in attempted][:cap]

    if to_crawl:
        fresh_for_trace = [JobUrl(url=sr.url, title=sr.title or sr.url)
                           for sr in to_crawl if sr.url not in job_urls]
        if job_id and fresh_for_trace:
            jobs.add_urls(job_id, fresh_for_trace)
            job_urls.update(u.url for u in fresh_for_trace)
        docs = await crawl_urls_with_progress(to_crawl, mission.question, job_id or "",
                                              attempted=attempted)
        for doc in docs:
            doc_id = await upsert_document(doc)
            collected[doc.url] = doc_id
            await link_mission_document(mission_id, req.id, doc_id)
        if job_id:
            jobs.update_job(job_id, sources_used=len(collected))

    # Link any already-held URLs that resurfaced for this requirement (crawled
    # earlier in this run, or before a retask), so assessment sees the full
    # picture without fetching them again.
    for sr in results:
        doc_id = collected.get(sr.url) or known.get(sr.url)
        if doc_id:
            await link_mission_document(mission_id, req.id, doc_id)

    req_docs = await get_requirement_documents(mission_id, req.id)
    assessment = await asyncio.to_thread(assess_requirement, req, req_docs)
    attempts = req.attempts + 1

    if assessment.satisfied:
        await update_requirement(
            req.id, status="satisfied", attempts=attempts,
            satisfied_doc_ids_json=json.dumps([d.id for d in req_docs]),
            assessment_missing="",
            assessment_confidence=assessment.confidence,
        )
        if job_id:
            jobs.add_log(job_id, "ok",
                         f"satisfied: <em>{_esc(req.title)}</em> · {len(req_docs)} sources")
    else:
        next_q = assessment.next_queries or queries
        status = "unmet" if attempts >= per_req_attempts else "pending"
        await update_requirement(
            req.id, status=status, attempts=attempts,
            next_queries_json=json.dumps(next_q),
            assessment_missing=assessment.missing,
            assessment_confidence=assessment.confidence,
        )
        if job_id:
            label = "unmet (capped)" if status == "unmet" else "gap remains"
            # confidence is an unconstrained LLM string — escape it like any
            # other untrusted interpolation (audit finding: the one gap in the
            # otherwise-consistent _esc invariant).
            jobs.add_log(job_id, "warn",
                         f"{label} ({_esc(str(assessment.confidence))}): <em>{_esc(req.title)}</em>")


async def _extract_sources(mission_id: str, prompt: str, job_id) -> None:
    docs = await get_mission_documents(mission_id)
    if job_id:
        jobs.add_log(job_id, "info",
                     f"extracting structured data from <em>{len(docs)}</em> sources "
                     f"· concurrency <em>{EXTRACT_CONCURRENCY}</em>")

    # Extraction is one independent LLM call per document and was the slow tail
    # of every mission; run a bounded number at once. The cap keeps a local
    # model (and a hosted rate limit) from being swamped.
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def one(doc):
        async with sem:
            return await asyncio.to_thread(extract_from_document, doc, prompt)

    results = await asyncio.gather(*(one(d) for d in docs), return_exceptions=True)

    done = 0
    for doc, ext in zip(docs, results):
        if isinstance(ext, Exception):
            if job_id:
                jobs.add_log(job_id, "warn",
                             f"extraction failed for <code>{_esc(doc.domain)}</code>")
            continue
        if ext:
            await insert_extraction(ext)
            done += 1
    if job_id:
        jobs.add_log(job_id, "ok", f"extraction complete · <em>{done}/{len(docs)}</em>")


async def _synthesize(mission_id: str, agent, job_id) -> None:
    await update_mission(mission_id, status="synthesizing")
    if job_id:
        jobs.update_job(job_id, stage="synthesizing")
        jobs.add_log(job_id, "info", "synthesizing brief")

    mission = await get_mission(mission_id)
    requirements = await get_requirements_for_mission(mission_id)
    docs = await get_mission_documents(mission_id)

    # Delta for the "morning brief": what this run found that the previous run
    # of the same agent on the same question did not. Diffed against that one
    # mission specifically — not against every mission ever — or a URL another
    # agent happened to crawl would wrongly count as "already seen". A mission
    # that records its parent (every scheduled run does) is diffed against
    # exactly that run; only a mission without one falls back to "the latest
    # finished run", which may be a different lineage's.
    new_urls: set[str] = set()
    if mission.parent_mission_id:
        prior = await get_mission(mission.parent_mission_id)
    else:
        prior = await get_latest_finished_mission(mission.agent_id, mission.question, mission_id)
    if prior:
        prior_urls = {d.url for d in await get_mission_documents(prior.id)}
        new_urls = {d.url for d in docs} - prior_urls
        if job_id:
            jobs.add_log(job_id, "info",
                         f"delta vs previous run: <em>{len(new_urls)}</em> new source(s)")

    # The order the brief's [n] markers refer to, fixed now and stored with
    # the brief, so the source rail cannot drift from the text if the
    # mission's documents change later (e.g. a retask adds sources).
    # synthesize_brief numbers `docs` with this same ordered_sources call.
    ordered = ordered_sources(docs)
    brief_md = await asyncio.to_thread(synthesize_brief, mission, requirements, docs, new_urls)

    n_sat = sum(1 for r in requirements if r.status == "satisfied")
    await update_mission(
        mission_id, status="done", brief_markdown=brief_md,
        brief_sources_json=json.dumps([d.id for d in ordered]), finished_at=_now(),
    )
    if job_id:
        jobs.add_log(job_id, "ok",
                     f"done · {n_sat}/{len(requirements)} requirements satisfied · "
                     f"{len(docs)} sources")
        jobs.finish_job(job_id, stage="done")
