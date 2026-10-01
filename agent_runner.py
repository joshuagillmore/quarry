"""Orchestration for agentic collection.

Two background entry points, split by the human approval gate:

    start_planning(mission_id, job_id)   -> plan, then status=awaiting_approval
    start_collection(mission_id, job_id) -> collect/assess/re-task loop, then brief

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
from config import settings
from jobs import JobUrl, EXTRACT_CONCURRENCY
from search import web_search_ex
from crawler import crawl_urls_with_progress
from agent_planner import build_collection_plan
from agent_assessor import assess_requirement
from brief import synthesize_brief, ordered_sources, brief_warnings
from extractor import extract_from_document
from storage import (
    get_agent, get_mission, update_mission,
    insert_requirement, update_requirement, get_requirements_for_mission,
    upsert_document, link_mission_document, insert_extraction,
    get_requirement_documents, get_mission_documents,
    get_latest_finished_mission, get_mission_llm_usage,
)

PER_QUERY_RESULTS = 5
# Per-requirement search history kept in requirements.search_stats_json.
SEARCH_STATS_KEEP = 40

STOPPED_BY_USER = "stopped by user"
TOKEN_BUDGET_REACHED = "token budget reached"
PASS_BUDGET_EXHAUSTED = "pass budget exhausted"
SOURCE_BUDGET_EXHAUSTED = "source budget exhausted"

# missions.stop_reason for a run that ended with requirements still open,
# keyed by the reason the loop logged. A run that left nothing pending
# stores "complete" when no unmet requirement has attempts left either,
# else it keeps the previous run's reason (see the end of _run_collection).
STOP_REASON_CODES = {
    PASS_BUDGET_EXHAUSTED: "pass_budget",
    SOURCE_BUDGET_EXHAUSTED: "source_budget",
    TOKEN_BUDGET_REACHED: "token_budget",
    STOPPED_BY_USER: "user_stop",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Thread launchers ---

def start_planning(mission_id: str, job_id: str | None = None) -> None:
    """Plan on a daemon thread. `job_id` is the live job this worker owns and
    must finish; give it whenever the caller created one."""
    threading.Thread(target=_thread, args=(_run_planning, mission_id, job_id),
                     daemon=True).start()


def start_collection(mission_id: str, job_id: str | None = None) -> None:
    """Collect on a daemon thread; `job_id` as for start_planning."""
    threading.Thread(target=_thread, args=(_run_collection, mission_id, job_id),
                     daemon=True).start()


def _mission_job_id(mission_id: str):
    try:
        mission = asyncio.run(get_mission(mission_id))
    except Exception:  # noqa: BLE001 - best effort; the caller has a fallback
        traceback.print_exc()
        return None
    return mission.job_id if mission else None


def _thread(coro_fn, mission_id: str, job_id: str | None = None) -> None:
    """Thread body for both stages. The mission's live job always ends in a
    terminal stage — on a crash (any BaseException, not just Exception), and
    on a path that returns without finishing it — so a dead worker never
    holds one of the job store's MAX_ACTIVE_JOBS slots.

    With `job_id`, that job is the one the coroutine reports to and the one
    finished, whatever the mission row says by then: the row may be gone
    (the mission was deleted, so there is nothing to look the job up from)
    or may already point at a newer job that another worker owns (a
    retask), which must not be touched here. Without it, both look the job
    up from the mission row."""
    given = job_id
    # Read up front as well as at the end: if the crash was the database
    # going away, the lookup in `finally` fails too.
    if not given:
        job_id = _mission_job_id(mission_id)
    failure = None
    try:
        asyncio.run(coro_fn(mission_id, given))
    except BaseException as e:  # noqa: BLE001 - the slot must be released
        traceback.print_exc()
        failure = str(e) or type(e).__name__
        try:
            asyncio.run(update_mission(mission_id, status="error", error=failure,
                                       finished_at=_now()))
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
    finally:
        if not given:
            job_id = _mission_job_id(mission_id) or job_id
        if job_id:
            if failure:
                jobs.add_log(job_id, "err", f"worker crashed: {_esc(failure)}")
            jobs.finish_if_running(job_id, stage="error",
                                   error=failure or "worker exited without finishing")


# --- Stage 1: planning ---

async def _run_planning(mission_id: str, job_id: str | None = None) -> None:
    """Plan; `job_id` is the job to report to (else the mission row's)."""
    mission = await get_mission(mission_id)
    if not mission:
        return
    agent = await get_agent(mission.agent_id)
    job_id = job_id or mission.job_id

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
        await _run_collection(mission_id, job_id)
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

def _token_budget(budget: dict) -> int:
    """The mission's LLM token budget: its own max_llm_tokens, else (unset,
    null or junk) the setting. 0 or less means unlimited."""
    raw = budget.get("max_llm_tokens")
    if raw is None:
        raw = settings.max_llm_tokens
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return max(0, int(settings.max_llm_tokens or 0))


async def _stop_reason(mission_id: str, job_id, token_budget: int,
                       mid_pass: bool) -> str | None:
    """Why collection must stop now, or None to carry on. Checked at every
    pass boundary and again before each requirement, so a Stop click or a
    spent token budget costs at most the requirement already in flight. The
    reason is logged here once; the caller breaks out and the requirements
    never reached say why (see the end of _run_collection)."""
    if job_id and jobs.is_cancelled(job_id):
        jobs.add_log(job_id, "warn",
                     "stop requested — skipping the rest of this pass" if mid_pass
                     else "stop requested — writing the brief from what was collected")
        return STOPPED_BY_USER
    if token_budget > 0:
        usage = await get_mission_llm_usage(mission_id)
        used = usage["prompt_tokens"] + usage["completion_tokens"]
        if used > token_budget:
            if job_id:
                jobs.add_log(job_id, "warn",
                             f"{TOKEN_BUDGET_REACHED} · {used:,} of {token_budget:,} tokens used")
            return TOKEN_BUDGET_REACHED
    return None


async def _run_collection(mission_id: str, job_id: str | None = None) -> None:
    """Collect, then brief; `job_id` is the job to report to (else the
    mission row's)."""
    mission = await get_mission(mission_id)
    if not mission:
        return
    agent = await get_agent(mission.agent_id)
    job_id = job_id or mission.job_id
    budget = json.loads(mission.budget_json or "{}")
    max_passes = int(budget.get("max_passes", agent.default_max_passes if agent else 4))
    max_sources = int(budget.get("max_sources", agent.default_max_sources if agent else 30))
    per_req_attempts = int(budget.get("per_req_attempts", agent.default_per_req_attempts if agent else 3))
    token_budget = _token_budget(budget)

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
    # linked to the requirement asking for it without re-fetching it, under
    # any of its names (`aliases`: other name -> stored url, seeded from the
    # names the crawler kept in each document's metadata).
    held = await get_mission_documents(mission_id)
    known: dict[str, str] = {d.url: d.id for d in held}
    aliases: dict[str, str] = {}
    for d in held:
        for name in _other_names(d):
            aliases.setdefault(name, d.url)
    attempted: set[str] = set(known) | set(aliases)

    # Fair share of the source budget per requirement, so one greedy
    # requirement can't starve the rest within a pass. The global max_sources
    # remains the hard ceiling.
    all_reqs = await get_requirements_for_mission(mission_id)
    per_req_cap = max(1, max_sources // max(1, len(all_reqs)))

    # Why a requirement that was never tried stayed unmet (see the end).
    stop_reason = PASS_BUDGET_EXHAUSTED
    for pass_num in range(1, max_passes + 1):
        reqs = await get_requirements_for_mission(mission_id)
        pending = [r for r in reqs if r.status == "pending"]
        if not pending:
            break
        # Cooperative stop (from the mission page) and the token budget are
        # checked here and before each requirement below; either way the
        # collected sources are still written up into a brief.
        halt = await _stop_reason(mission_id, job_id, token_budget, mid_pass=False)
        if halt:
            stop_reason = halt
            break
        if job_id:
            jobs.update_job(job_id, pass_num=pass_num)
            jobs.add_log(job_id, "info",
                         f"pass <em>{pass_num}</em> · {len(pending)} open requirements")

        for req in pending:
            if len(collected) >= max_sources:
                break
            halt = await _stop_reason(mission_id, job_id, token_budget, mid_pass=True)
            if halt:
                break
            try:
                await _collect_one(
                    mission, req, collected, job_urls, job_id,
                    max_sources, per_req_cap, per_req_attempts,
                    attempted=attempted, known=known, aliases=aliases,
                    pass_num=pass_num,
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

        if halt:
            stop_reason = halt
            break
        if len(collected) >= max_sources:
            if job_id:
                jobs.add_log(job_id, "warn", "source budget reached")
            stop_reason = SOURCE_BUDGET_EXHAUSTED
            break

    # Anything still pending after the pass budget is an unmet gap. One that
    # was never attempted says why, rather than looking like a failed search.
    final_reqs = await get_requirements_for_mission(mission_id)
    still_pending = [r for r in final_reqs if r.status == "pending"]
    for r in still_pending:
        fields = {"status": "unmet"}
        if r.attempts == 0:
            fields["assessment_missing"] = f"not attempted: {stop_reason}"
        await update_requirement(r.id, **fields)
    if still_pending:
        stop_code = STOP_REASON_CODES[stop_reason]
    elif any(r.status == "unmet" and r.attempts < per_req_attempts for r in final_reqs):
        # Nothing pending, yet requirements this run never worked still have
        # attempts left: a retask reopens only the one requirement, after an
        # earlier run stopped on a limit. That earlier stop still stands
        # (retask leaves stop_reason alone), so carry it forward; storing
        # "complete" would hide the leftovers from Resume.
        stop_code = mission.stop_reason
    else:
        # Every requirement satisfied or capped out, whichever limit this
        # run also reached on its last pass: there is nothing to resume.
        stop_code = "complete"

    # Optional LLM extraction over the collected sources (applies regardless of
    # how they were gathered). Not once the token budget stopped collection:
    # extraction costs one LLM call per source. The brief, a single call,
    # still runs so the mission ends with a write-up of what was collected.
    if budget.get("extract"):
        if stop_reason == TOKEN_BUDGET_REACHED:
            if job_id:
                jobs.add_log(job_id, "warn", f"skipping extraction: {TOKEN_BUDGET_REACHED}")
        else:
            await _extract_sources(mission_id, budget.get("extract_prompt", ""), job_id)

    await _synthesize(mission_id, agent, job_id, stop_reason=stop_code)


def _other_names(doc) -> list[str]:
    """The URLs besides doc.url that a stored page was crawled under, as the
    crawler recorded them in its metadata (requested_url, redirected_url)."""
    try:
        meta = json.loads(doc.metadata_json or "{}")
    except (TypeError, ValueError):
        return []
    if not isinstance(meta, dict):
        return []
    names = (meta.get("requested_url"), meta.get("redirected_url"))
    return [n for n in names if isinstance(n, str) and n and n != doc.url]


def _search_stats(req) -> list:
    """The requirement's stored search history, or [] when unset or corrupt."""
    try:
        stats = json.loads(req.search_stats_json or "[]")
    except (TypeError, ValueError):
        return []
    return stats if isinstance(stats, list) else []


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _search_note(entries: list[dict]) -> str:
    """One line for the assessor: what this pass's searches returned, e.g.
    "Search coverage this pass: 3 queries, 11 results (brave)". Engine
    names come from the configured backend list, never from a page."""
    total = sum(e["results"] for e in entries)
    engines = list(dict.fromkeys(e["engine"] for e in entries if e["engine"]))
    return (f"Search coverage this pass: {_count(len(entries), 'query', 'queries')}, "
            f"{_count(total, 'result', 'results')} "
            f"({', '.join(engines) if engines else 'no engine answered'})")


async def _collect_one(mission, req, collected, job_urls, job_id,
                       max_sources, per_req_cap, per_req_attempts,
                       attempted: set[str] | None = None,
                       known: dict[str, str] | None = None,
                       aliases: dict[str, str] | None = None,
                       pass_num: int = 1) -> None:
    """Search, crawl and assess a single requirement. Raising here costs this
    requirement an attempt, not the mission.

    All three maps are mission-scoped and shared across requirements:
    `attempted` holds every URL already tried under any name (requested,
    reported, redirected-to; failed and junk pages included) and every URL
    the mission held before this run — none is fetched again. `known` maps
    held URLs to document ids; `aliases` maps a page's other names to the URL
    its document is stored under, so a search result is resolved to its
    document whichever name it comes back as."""
    mission_id = mission.id
    attempted = attempted if attempted is not None else set()
    known = known or {}
    aliases = aliases if aliases is not None else {}
    if req.attempts >= per_req_attempts:
        await update_requirement(req.id, status="unmet")
        if job_id:
            jobs.add_log(job_id, "warn", f"unmet (capped): <em>{_esc(req.title)}</em>")
        return

    queries = json.loads(req.next_queries_json or "[]") or [mission.question]
    if job_id:
        jobs.add_log(job_id, "info", f"collecting: <em>{_esc(req.title)}</em>")

    # Search across this requirement's queries, recording what each one
    # returned and which engine answered (a zero-result query included:
    # that is the signal) before anything else can fail.
    results = []
    seen_q_urls: set[str] = set()
    this_pass: list[dict] = []
    for q in queries:
        found, engine = await asyncio.to_thread(web_search_ex, q, PER_QUERY_RESULTS)
        this_pass.append({"pass": pass_num, "query": q, "engine": engine,
                          "results": len(found)})
        for sr in found:
            if sr.url not in seen_q_urls:
                seen_q_urls.add(sr.url)
                results.append(sr)
    await update_requirement(req.id, search_stats_json=json.dumps(
        (_search_stats(req) + this_pass)[-SEARCH_STATS_KEEP:]))

    # New URLs to crawl, bounded by this requirement's fair share and the
    # remaining global source budget.
    remaining = max_sources - len(collected)
    cap = max(0, min(per_req_cap, remaining))
    to_crawl = [sr for sr in results
                if aliases.get(sr.url, sr.url) not in collected
                and sr.url not in attempted
                and aliases.get(sr.url, sr.url) not in attempted][:cap]

    if to_crawl:
        fresh_for_trace = [JobUrl(url=sr.url, title=sr.title or sr.url)
                           for sr in to_crawl if sr.url not in job_urls]
        if job_id and fresh_for_trace:
            jobs.add_urls(job_id, fresh_for_trace)
            job_urls.update(u.url for u in fresh_for_trace)
        docs = await crawl_urls_with_progress(to_crawl, mission.question, job_id or "",
                                              attempted=attempted, aliases=aliases)
        for doc in docs:
            doc_id = await upsert_document(doc)
            collected[doc.url] = doc_id
            await link_mission_document(mission_id, req.id, doc_id)
        if job_id:
            jobs.update_job(job_id, sources_used=len(collected))

    # Link any already-held URLs that resurfaced for this requirement (crawled
    # earlier in this run, or before a retask; under any of their names), so
    # assessment sees the full picture without fetching them again.
    for sr in results:
        stored_url = aliases.get(sr.url, sr.url)
        doc_id = collected.get(stored_url) or known.get(stored_url)
        if doc_id:
            await link_mission_document(mission_id, req.id, doc_id)

    req_docs = await get_requirement_documents(mission_id, req.id)
    assessment = await asyncio.to_thread(assess_requirement, req, req_docs,
                                         search_note=_search_note(this_pass))
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
            return await asyncio.to_thread(extract_from_document, doc, prompt, mission_id)

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


def _check_brief(mission, requirements, docs, brief_md: str, job_id) -> str | None:
    """brief_warnings as the JSON stored with the brief, logged as a count.
    `mission` carries the source order the brief was numbered with. A
    checker failure is logged and stored as None (not checked) rather than
    costing the mission the brief it has already paid for."""
    try:
        warnings = brief_warnings(mission, requirements, docs, brief_md)
    except Exception as e:  # noqa: BLE001 - the brief matters more than its checks
        traceback.print_exc()
        if job_id:
            jobs.add_log(job_id, "warn", f"brief checks failed: {_esc(type(e).__name__)}")
        return None
    if job_id:
        jobs.add_log(job_id, "warn" if warnings else "info",
                     f"brief checks: {len(warnings)} warning(s)")
    return json.dumps(warnings)


async def _synthesize(mission_id: str, agent, job_id,
                      stop_reason: str | None = None) -> None:
    """Write the brief and finish the mission. `stop_reason` (a
    STOP_REASON_CODES value or "complete") is stored in the same update as
    the brief, so a finished mission always says why its collection ended."""
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
    sources_json = json.dumps([d.id for d in ordered])
    warnings_json = _check_brief(mission.model_copy(update={"brief_sources_json": sources_json}),
                                 requirements, docs, brief_md, job_id)

    n_sat = sum(1 for r in requirements if r.status == "satisfied")
    await update_mission(
        mission_id, status="done", brief_markdown=brief_md,
        brief_sources_json=sources_json, brief_warnings_json=warnings_json,
        stop_reason=stop_reason, finished_at=_now(),
    )
    if job_id:
        jobs.add_log(job_id, "ok",
                     f"done · {n_sat}/{len(requirements)} requirements satisfied · "
                     f"{len(docs)} sources")
        jobs.finish_job(job_id, stage="done")
