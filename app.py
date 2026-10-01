import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from flask import (Flask, Response, render_template, request, redirect,
                   url_for, flash, session, stream_with_context)
from werkzeug.exceptions import SecurityError

import auth
import brief
import jobs
import scheduler
from config import (settings, save_overrides, known_models,
                    persistent_secret_key, active_api_key)
from search import web_search
from storage import (
    init_db, insert_search, insert_extraction,
    get_document, get_documents_by_search, get_all_documents,
    get_extractions_for_document, get_search_history,
    count_documents, count_searches, count_extractions, count_domains,
    get_doc_ids_with_extractions, get_search_history_enriched,
    get_related_documents, search_documents_fts,
    insert_agent, get_agent, list_agents, update_agent, delete_agent,
    insert_mission, update_mission, get_mission, list_missions,
    get_requirements_for_mission, get_mission_documents,
    get_missions_enriched, get_distinct_search_queries,
    insert_requirement, update_requirement, delete_requirement,
    get_requirement_documents, get_agent_track_records,
    reconcile_interrupted_missions, delete_mission,
    claim_mission_status, agent_has_active_mission,
    get_mission_llm_usage, get_mission_pair_documents,
)
from jobs import (
    create_job, get_job, job_state, run_job_in_background,
    get_sidebar_jobs, get_in_memory_job_ids, create_mission_job,
    request_cancel, JobLimitReached,
)
from agent_runner import start_planning, start_collection, _token_budget
from brief import linkify_citations
from scheduler import (start_scheduler, sync_agent_jobs, validate_cron,
                       describe_next_run, scheduled_jobs)
from markdown_render import render_markdown, to_plain_text, snippet
from models import SearchRecord, Agent, Mission, Requirement
from prompt_templates import build_persona

app = Flask(__name__)
# Stable across restarts (generated once into the data volume) so login
# sessions and flash messages survive a redeploy.
app.secret_key = persistent_secret_key()
auth.configure(app.secret_key)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
if settings.quarry_behind_proxy:
    # TLS-proxy deployment mode: trust one proxy hop so the login rate limiter
    # sees real client IPs (not the proxy collapsing everyone into one bucket),
    # and mark the session cookie Secure since TLS terminates upstream.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.config["SESSION_COOKIE_SECURE"] = True


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


def _trusted_hosts() -> list[str] | None:
    """Host-header allowlist; Flask 3.1's TRUSTED_HOSTS answers any other
    Host with 400. Always on while auth is off, where it is the DNS-rebinding
    guard: a hostile page that rebinds its own name to 127.0.0.1 still sends
    that name as Host. With a password set it applies only when the operator
    lists hosts in QUARRY_TRUSTED_HOSTS."""
    extra = [h.strip() for h in (settings.quarry_trusted_hosts or "").split(",")
             if h.strip()]
    if auth.enabled() and not extra:
        return None
    return ["localhost", "127.0.0.1", "[::1]"] + extra


app.config["TRUSTED_HOSTS"] = _trusted_hosts()

# Set by __main__ to the dev server's bind address; stays None under gunicorn.
_dev_server_host: str | None = None


def dev_server_exposed(host: str) -> bool:
    """True when the Flask dev server (python app.py) would bind beyond
    loopback with no password set."""
    return (host or "").strip() not in _LOOPBACK_HOSTS and not auth.enabled()


def exposure_reason() -> str:
    """Why this instance is reachable beyond localhost without a password,
    or '' when it is not (the footgun the security audit flagged)."""
    if auth.enabled():
        return ""
    bind = (settings.quarry_bind or "127.0.0.1").strip()
    if bind not in _LOOPBACK_HOSTS + ("",):
        return f"QUARRY_BIND={bind} publishes this app beyond localhost"
    if settings.quarry_behind_proxy:
        return "QUARRY_BEHIND_PROXY=true serves this app through a reverse proxy"
    if _dev_server_host is not None and dev_server_exposed(_dev_server_host):
        return f"FLASK_HOST={_dev_server_host} binds the dev server beyond localhost"
    return ""


def insecure_exposure() -> bool:
    return bool(exposure_reason())


def _warn_exposure(reason: str) -> None:
    print("=" * 70 + f"\n[SECURITY] {reason} WITHOUT a password.\n"
          "[SECURITY] Set QUARRY_PASSWORD in .env, or keep the app on "
          "127.0.0.1.\n" + "=" * 70,
          file=sys.stderr, flush=True)


if exposure_reason():
    _warn_exposure(exposure_reason())


@app.context_processor
def inject_globals():
    try:
        doc_ct = run_async(count_documents())
        search_ct = run_async(count_searches())
        extract_ct = run_async(count_extractions())
        llm_provider = settings.llm_provider
        llm_model = llm_provider.split("/")[-1] if "/" in llm_provider else llm_provider
        llm_vendor = llm_provider.split("/")[0] if "/" in llm_provider else llm_provider
        fast = settings.llm_provider_fast
        llm_fast_model = (fast.split("/")[-1] if "/" in fast else fast) if fast else "—"
        sidebar = get_sidebar_jobs()
        return dict(
            doc_count=doc_ct,
            search_count=search_ct,
            extraction_count=extract_ct,
            llm_vendor=llm_vendor.title(),
            llm_model=llm_model,
            llm_fast_model=llm_fast_model,
            default_results=settings.search_max_results,
            live_job=sidebar["live"],
            previous_jobs=sidebar["previous"],
            auth_enabled=auth.enabled(),
            insecure_exposure=insecure_exposure(),
            exposure_reason=exposure_reason(),
        )
    except Exception:
        return dict(
            doc_count=0, search_count=0, extraction_count=0,
            llm_vendor="Cohere", llm_model="command-a-03-2025",
            llm_fast_model="—", default_results=5,
            live_job=None, previous_jobs=[],
            auth_enabled=auth.enabled(),
            insecure_exposure=insecure_exposure(),
            exposure_reason=exposure_reason(),
        )


def run_async(coro):
    """Run a coroutine to completion from sync code (routes, startup).

    Routes run on plain threads with no event loop, so this is normally just
    asyncio.run. If a loop is already running on this thread (a caller that
    is itself async), asyncio.run would refuse, so run the coroutine on a
    fresh loop in a worker thread instead. get_running_loop, unlike
    get_event_loop, never warns or implicitly creates a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@app.url_defaults
def _static_cache_bust(endpoint, values):
    """Stamp static URLs with the file's mtime so a deploy can't leave users on
    a cached stylesheet (CSS changes otherwise need a manual hard refresh)."""
    if endpoint == "static" and "filename" in values:
        try:
            values["v"] = int(os.stat(
                os.path.join(app.static_folder, values["filename"])).st_mtime)
        except OSError:
            pass


@app.before_request
def reject_untrusted_host():
    """Flask records a Host outside TRUSTED_HOSTS as a routing exception but
    still runs before_request hooks before raising it. Registered first, so a
    rebinding request gets its 400 before DB init or the login redirect (which
    cannot even build a URL without the unbuilt URL adapter) runs."""
    if isinstance(request.routing_exception, SecurityError):
        raise request.routing_exception


@app.before_request
def block_cross_site_posts():
    """CSRF defense via origin checking (no tokens needed): browsers label
    cross-site requests with Sec-Fetch-Site / a mismatching Origin, so a
    malicious page can't blind-POST to this (unauthenticated, localhost-bound)
    app. Same-origin form posts and non-browser clients are unaffected."""
    if request.method != "POST":
        return None
    sfs = request.headers.get("Sec-Fetch-Site")
    if sfs and sfs not in ("same-origin", "same-site", "none"):
        return "Cross-site POST blocked", 403
    origin = request.headers.get("Origin")
    if origin:
        # "Origin: null" comes from sandboxed iframes / opaque origins — never
        # from a legitimate same-origin form post. Treat it as cross-site.
        from urllib.parse import urlparse
        if origin == "null" or urlparse(origin).netloc != request.host:
            return "Cross-site POST blocked", 403
    return None


_init_db_lock = threading.Lock()


def initialize() -> None:
    """One-time process startup: create/migrate the schema, fail missions
    orphaned by a previous process, start the scheduler.

    Idempotent and locked (double-checked), so gunicorn's post_worker_init
    can call it eagerly and the first request's ensure_db is then a no-op;
    concurrent first requests never run init_db (and its FTS rebuild) twice.
    """
    if getattr(app, "_db_initialized", False):
        return
    with _init_db_lock:
        if getattr(app, "_db_initialized", False):
            return
        run_async(init_db())
        # Safe here: no collection worker can be running yet in this process,
        # so any mission still in an in-flight state is one whose thread died
        # with a previous process.
        stale = run_async(reconcile_interrupted_missions())
        if stale:
            print(f"[STARTUP] marked {stale} interrupted mission(s) as failed",
                  file=sys.stderr, flush=True)
        # Started after the DB exists. Safe under the single gunicorn worker;
        # the Flask reloader would otherwise start two.
        if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            start_scheduler()
        app._db_initialized = True


@app.before_request
def ensure_db():
    initialize()


def _session_authed() -> bool:
    """The session carries the token for the current secret key + password.
    A bare True (older versions) or a token from before a password change
    does not count."""
    tok = session.get("authed")
    return isinstance(tok, str) and hmac.compare_digest(tok, auth.session_token())


_SAFE_NEXT = re.compile(r"/(?![/\\])[^\s\\]*")


def _safe_next(nxt: str) -> str:
    """A post-login redirect target, only ever a path on this site. Rejects
    anything a browser could normalise into another origin: '//host',
    '/\\host', and '/<tab>/host' (browsers strip tabs and newlines, and read
    backslashes as slashes)."""
    nxt = nxt or ""
    if _SAFE_NEXT.fullmatch(nxt):
        parts = urlsplit(nxt)
        if not parts.scheme and not parts.netloc:
            return nxt
    return url_for("index")


@app.before_request
def require_login():
    """When a password is configured, everything except the login page and
    static assets requires a session. Registered after ensure_db so startup
    reconciliation still runs on the first request either way."""
    if not auth.enabled():
        return None
    if request.endpoint in ("login", "static"):
        return None
    if _session_authed():
        return None
    if request.path.startswith("/api/"):
        return {"error": "authentication required"}, 401
    nxt = request.full_path if request.method == "GET" else None
    return redirect(url_for("login", next=nxt))


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


@app.route("/login", methods=["GET", "POST"])
def login():
    if not auth.enabled() or _session_authed():
        return redirect(url_for("index"))

    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        # Counted before verifying, atomically with the lockout check, so
        # parallel guesses cannot all get through the gap between the two.
        allowed, wait_s = auth.reserve_attempt(ip)
        if not allowed:
            flash(f"Too many attempts — try again in {wait_s // 60 + 1} minute(s).", "error")
            return render_template("login.html"), 429

        if auth.verify_password(request.form.get("password", "")):
            auth.clear_failures(ip)
            session.clear()  # fresh session id state on privilege change
            session["authed"] = auth.session_token()
            session.permanent = True
            # Only ever redirect within this app (no open redirect).
            return redirect(_safe_next(request.values.get("next", "")))

        flash("Wrong password.", "error")
        return render_template("login.html"), 401

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Signed out.", "success")
    return redirect(url_for("login") if auth.enabled() else url_for("index"))


@app.route("/")
def index():
    try:
        stats = {
            "docs": run_async(count_documents()),
            "searches": run_async(count_searches()),
            "extractions": run_async(count_extractions()),
            "domains": run_async(count_domains()),
        }
    except Exception:
        stats = None
    return render_template("index.html", stats=stats, agents=run_async(list_agents()),
                           default_max_llm_tokens=settings.max_llm_tokens)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()[:500]
    try:
        max_results = max(1, min(20, int(request.form.get("max_results", 5))))
    except (TypeError, ValueError):
        max_results = 5
    extract = request.form.get("extract") == "on"
    extract_prompt = request.form.get("extract_prompt", "").strip()[:5000]

    if not query:
        flash("Please enter a search query.", "error")
        return redirect(url_for("index"))

    try:
        job_id = create_job(query, max_results, extract, extract_prompt)
    except JobLimitReached as e:
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("index"))
    run_job_in_background(job_id)
    return redirect(url_for("crawl_view", job_id=job_id))


@app.route("/crawl/<job_id>")
def crawl_view(job_id):
    job = get_job(job_id)
    if not job:
        flash("Job not found.", "error")
        return redirect(url_for("index"))
    # The page renders from the same serialization the stream sends, so a
    # finished job reads right before the first message arrives.
    return render_template("crawl.html", job=job, state=job_state(job_id))


@app.route("/crawl/<job_id>/cancel", methods=["POST"])
def crawl_cancel(job_id):
    """Cooperative: the job checks for this between stages and between
    documents, then finishes as `cancelled` with what it already stored."""
    if request_cancel(job_id):
        flash("Cancelling — the crawl stops at its next checkpoint.", "info")
    else:
        flash("This crawl is not running.", "info")
    return redirect(url_for("crawl_view", job_id=job_id))


# Fields that change on every poll without anything having happened; left
# out of the SSE change check so an idle job does not send a message 4x/s.
_SSE_VOLATILE = ("elapsed", "finished_at")
_SSE_POLL_S = 0.25
# Unchanged polls before a keepalive comment: 60 x 0.25 s = 15 s of silence.
_SSE_KEEPALIVE_POLLS = 60


@app.route("/api/job/<job_id>")
def api_job(job_id):
    state = job_state(job_id)
    if not state:
        return {"error": "not found"}, 404
    return state


@app.route("/api/job/<job_id>/stream")
def api_job_stream(job_id):
    # 204 is the one answer that makes EventSource stop reconnecting: a job
    # that is gone (finished trace evicted, process restarted) stays gone.
    if job_state(job_id) is None:
        return Response(status=204)

    def gen():
        last_hash = None
        silent = 0
        # Cap the stream at ~10 minutes to bound resource use; the browser's
        # EventSource reconnects and resumes if the job is still running.
        for _ in range(2400):
            state = job_state(job_id)
            if not state:
                yield "event: gone\ndata: {}\n\n"
                return
            stable = {k: v for k, v in state.items() if k not in _SSE_VOLATILE}
            h = hashlib.md5(json.dumps(stable, sort_keys=True).encode()).hexdigest()
            if h != last_hash:
                yield f"data: {json.dumps(state)}\n\n"
                last_hash = h
                silent = 0
            else:
                silent += 1
                if silent >= _SSE_KEEPALIVE_POLLS:
                    # An SSE comment line: EventSource ignores it, but it keeps
                    # proxies with idle timeouts (often 60 s) from cutting a
                    # stream that is quiet through a long extraction.
                    yield ": keepalive\n\n"
                    silent = 0
            if state.get("done"):
                return
            time.sleep(_SSE_POLL_S)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/results/<job_id>")
def results_view(job_id):
    job = get_job(job_id)
    if not job:
        flash("Job not found.", "error")
        return redirect(url_for("index"))

    documents = []
    for doc_id in job.document_ids:
        d = run_async(get_document(doc_id))
        if d:
            documents.append(d)

    ext_ids = run_async(get_doc_ids_with_extractions())
    elapsed_s = int(time.time() - job.started_at)
    total_words = sum((d.word_count or 0) for d in documents)

    return render_template(
        "index.html",
        query=job.query,
        max_results=job.max_results,
        extract=job.extract,
        extract_prompt=job.extract_prompt,
        documents=documents,
        ext_ids=ext_ids,
        agents=run_async(list_agents()),
        default_max_llm_tokens=settings.max_llm_tokens,
        active_page="results",
        job_meta={
            "elapsed": elapsed_s,
            "extract_done": job.extract_done,
            "total_words": total_words,
            "crawl_total": job.crawl_total,
        },
    )


@app.route("/document/<doc_id>")
def document_view(doc_id):
    doc = run_async(get_document(doc_id))
    if not doc:
        flash("Document not found.", "error")
        return redirect(url_for("index"))

    extractions = run_async(get_extractions_for_document(doc_id))

    metadata = None
    if doc.metadata_json:
        try:
            metadata = json.dumps(json.loads(doc.metadata_json), indent=2)
        except json.JSONDecodeError:
            metadata = doc.metadata_json

    parsed_extractions = []
    for ext in extractions:
        parsed = None
        try:
            parsed = json.loads(ext.data_json) if ext.data_json else None
        except json.JSONDecodeError:
            parsed = None
        parsed_extractions.append({"ext": ext, "data": parsed})

    reading_min = max(1, round((doc.word_count or 0) / 220))

    related = run_async(get_related_documents(
        doc.id, doc.search_query or "", doc.domain or "", limit=3
    ))

    content_html = render_markdown(doc.content_markdown or "")
    content_fit_html = render_markdown(doc.content_fit) if doc.content_fit else ""
    content_plain = to_plain_text(doc.content_markdown or "")

    return render_template(
        "document.html",
        doc=doc,
        extractions=extractions,
        parsed_extractions=parsed_extractions,
        metadata=metadata,
        reading_min=reading_min,
        related=related,
        content_html=content_html,
        content_fit_html=content_fit_html,
        content_plain=content_plain,
    )


@app.route("/extract/<doc_id>", methods=["GET", "POST"])
def extract_document(doc_id):
    doc = run_async(get_document(doc_id))
    if not doc:
        flash("Document not found.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()[:5000]
        try:
            from extractor import extract_from_document
            extraction = extract_from_document(doc, prompt)
            if extraction:
                run_async(insert_extraction(extraction))
                flash("Extraction completed successfully.", "success")
            else:
                flash("Extraction returned no results.", "info")
        except Exception as e:
            flash(f"Extraction error: {str(e)}", "error")

    return redirect(url_for("document_view", doc_id=doc_id))


@app.route("/history")
def history():
    searches = run_async(get_search_history_enriched())
    missions = run_async(get_missions_enriched())
    agents = {a.id: a.name for a in run_async(list_agents())}

    events = []
    for s in searches:
        events.append({"kind": "search", "ts": s["executed_at"], **s})
    for m in missions:
        events.append({"kind": "mission", "ts": m["created_at"],
                       "agent_name": agents.get(m["agent_id"], "agent"), **m})
    events.sort(key=lambda e: e["ts"] or "", reverse=True)

    groups = {}
    for e in events:
        day = (e["ts"] or "")[:10]
        groups.setdefault(day, []).append(e)
    grouped = [(day, items) for day, items in groups.items()]
    return render_template(
        "history.html",
        grouped=grouped,
        total=len(events),
        live_job_ids=get_in_memory_job_ids(),
    )


PAGE_SIZE = 60
# Card previews need only the opening of each page, not the whole body.
PREVIEW_CHARS = 4000


@app.route("/documents")
def documents_list():
    search_filter = request.args.get("search", "").strip()
    mission_filter = request.args.get("mission", "").strip()
    full_text = request.args.get("q", "").strip()[:200]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1

    mission_obj = None
    total = None  # set only for the paged listings
    if mission_filter:
        # Mission filter takes precedence; full-text within a mission is handled
        # client-side by the page's filter (the same input, seeded with `q`).
        # Unpaged: a mission is bounded by its source budget.
        mission_obj = run_async(get_mission(mission_filter))
        documents = run_async(get_mission_documents(mission_filter))
    elif full_text:
        documents = run_async(search_documents_fts(full_text, search_filter or None))
    else:
        if search_filter:
            # No COUNT helper for one collection in storage: count metadata
            # rows (preview_chars=0 keeps every page body out of the result).
            total = len(run_async(get_documents_by_search(search_filter, preview_chars=0)))
        else:
            total = run_async(count_documents())
        page = min(page, max(1, math.ceil(total / PAGE_SIZE)))
        paging = dict(limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE,
                      preview_chars=PREVIEW_CHARS)
        if search_filter:
            documents = run_async(get_documents_by_search(search_filter, **paging))
        else:
            documents = run_async(get_all_documents(**paging))

    pager = None
    if total is not None and total > PAGE_SIZE:
        def page_url(n):
            args = {"page": n}
            if search_filter:
                args["search"] = search_filter
            return url_for("documents_list", **args)

        first = (page - 1) * PAGE_SIZE + 1
        pager = {
            "first": first,
            "last": first + len(documents) - 1,
            "total": total,
            "prev_url": page_url(page - 1) if page > 1 else None,
            "next_url": page_url(page + 1) if page * PAGE_SIZE < total else None,
        }

    ext_ids = run_async(get_doc_ids_with_extractions())
    domain_counts = {}
    for d in documents:
        domain_counts[d.domain] = domain_counts.get(d.domain, 0) + 1
    domain_counts = dict(sorted(domain_counts.items(), key=lambda x: -x[1])[:12])

    # Collection selector: one-shot searches + agentic missions.
    search_queries = run_async(get_distinct_search_queries())
    missions = run_async(list_missions())

    # Readable card previews: the fit-markdown is already pruned of nav, so
    # prefer it; either way strip the markup rather than showing its source.
    snippets = {
        d.id: snippet(d.content_fit or d.content_markdown or "")
        for d in documents
    }

    return render_template(
        "documents.html",
        documents=documents, snippets=snippets,
        ext_ids=ext_ids,
        domain_counts=domain_counts,
        search_filter=search_filter,
        mission_filter=mission_filter,
        mission_obj=mission_obj,
        # The one search input always shows `q`; full_text_query is set only
        # when the cards ARE full-text matches, which the client filter must
        # then leave alone (their hit may be in the body it cannot see).
        lib_query=full_text,
        full_text_query="" if mission_filter else full_text,
        search_queries=search_queries,
        missions=missions,
        pager=pager,
        total=total if total is not None else len(documents),
    )


# --- Agentic collection ---

@app.route("/agents")
def agents_list():
    agents = run_async(list_agents())
    next_runs = {j["agent_id"]: j["next_run"] for j in scheduled_jobs()}
    return render_template("agents.html", agents=agents,
                           records=run_async(get_agent_track_records()),
                           next_runs=next_runs, active_page="agents")


_AGENT_FORM_FIELDS = {
    "name": 120, "expertise": 500, "max_sources": 10, "max_passes": 10,
    "per_req_attempts": 10, "schedule_cron": 120, "schedule_question": 500,
    "persona_prompt": 5000,
}


def _agent_form_error(agent, message: str):
    """Re-render the form with what the user typed and why it was refused,
    rather than redirecting to an empty form. Values are bounded like the
    fields they came from; Jinja autoescapes them."""
    values = {k: request.form.get(k, "")[:n] for k, n in _AGENT_FORM_FIELDS.items()}
    return render_template("agent_form.html", agent=agent, form=values,
                           error=message, active_page="agents"), 400


@app.route("/agents/new", methods=["GET", "POST"])
def agent_new():
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:120]
        expertise = request.form.get("expertise", "").strip()[:500]
        if not name or not expertise:
            return _agent_form_error(None, "Name and area of expertise are required.")

        def _clamp(field, default, lo, hi):
            try:
                return max(lo, min(hi, int(request.form.get(field, default))))
            except (TypeError, ValueError):
                return default

        max_passes = _clamp("max_passes", 4, 1, 10)
        max_sources = _clamp("max_sources", 30, 1, 100)
        per_req = _clamp("per_req_attempts", 3, 1, 6)
        custom_persona = request.form.get("persona_prompt", "").strip()[:5000]
        persona = custom_persona or build_persona(expertise)

        cron = request.form.get("schedule_cron", "").strip()[:120]
        ok, err = validate_cron(cron)
        if not ok:
            return _agent_form_error(
                None, f"That schedule isn't a valid cron expression: {err}")
        sched_q = request.form.get("schedule_question", "").strip()[:500]

        agent = Agent(
            id=str(uuid.uuid4()), name=name, expertise=expertise,
            persona_prompt=persona, default_max_passes=max_passes,
            default_max_sources=max_sources, default_per_req_attempts=per_req,
            schedule_cron=cron or None, schedule_question=sched_q or None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        run_async(insert_agent(agent))
        sync_agent_jobs()
        flash(f"Agent “{name}” created.", "success")
        return redirect(url_for("agents_list"))

    return render_template("agent_form.html", active_page="agents")


@app.route("/agents/<agent_id>/edit", methods=["GET", "POST"])
def agent_edit(agent_id):
    agent = run_async(get_agent(agent_id))
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("agents_list"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()[:120]
        expertise = request.form.get("expertise", "").strip()[:500]
        if not name or not expertise:
            return _agent_form_error(agent, "Name and area of expertise are required.")

        def _clamp(field, default, lo, hi):
            try:
                return max(lo, min(hi, int(request.form.get(field, default))))
            except (TypeError, ValueError):
                return default

        # Blank persona regenerates from the (possibly changed) expertise.
        custom_persona = request.form.get("persona_prompt", "").strip()[:5000]
        persona = custom_persona or build_persona(expertise)

        cron = request.form.get("schedule_cron", "").strip()[:120]
        ok, err = validate_cron(cron)
        if not ok:
            return _agent_form_error(
                agent, f"That schedule isn't a valid cron expression: {err}")
        sched_q = request.form.get("schedule_question", "").strip()[:500]

        run_async(update_agent(
            agent_id,
            name=name, expertise=expertise, persona_prompt=persona,
            schedule_cron=cron or None, schedule_question=sched_q or None,
            default_max_passes=_clamp("max_passes", agent.default_max_passes, 1, 10),
            default_max_sources=_clamp("max_sources", agent.default_max_sources, 1, 100),
            default_per_req_attempts=_clamp("per_req_attempts", agent.default_per_req_attempts, 1, 6),
        ))
        sync_agent_jobs()
        flash(f"Agent “{name}” updated.", "success")
        return redirect(url_for("agents_list"))

    return render_template("agent_form.html", agent=agent, active_page="agents")


@app.route("/agents/<agent_id>/run-scheduled", methods=["POST"])
def agent_run_scheduled(agent_id):
    """Fire an agent's scheduled run immediately — so a morning brief can be
    tested without waiting for its cron window. Same unattended path as the
    scheduler: auto-approves and collects."""
    agent = run_async(get_agent(agent_id))
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("agents_list"))
    if not (agent.schedule_question or "").strip():
        flash("Set a standing question before running the schedule.", "error")
        return redirect(url_for("agent_edit", agent_id=agent_id))
    try:
        mission_id = scheduler.launch_scheduled_mission(agent_id)
    except JobLimitReached as e:
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("agents_list"))
    if mission_id is None:
        if not agent.active:
            why = f"{agent.name} is inactive, so its schedule does not run."
        elif run_async(agent_has_active_mission(agent_id)):
            why = (f"{agent.name} already has a mission running — "
                   "wait for it to finish, then run again.")
        else:
            why = f"{agent.name}'s scheduled question could not be started."
        flash(f"Nothing started: {why}", "info")
        return redirect(url_for("agents_list"))
    flash(f"Running {agent.name}'s scheduled question now — it approves its own plan.",
          "success")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/agents/<agent_id>/delete", methods=["POST"])
def agent_delete(agent_id):
    agent = run_async(get_agent(agent_id))
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("agents_list"))
    run_async(delete_agent(agent_id))
    sync_agent_jobs()
    flash(f"Agent “{agent.name}” deleted. Its past missions are kept under History.", "success")
    return redirect(url_for("agents_list"))


# Upper bound on a per-mission LLM token budget typed into the run form.
MAX_LLM_TOKENS_INPUT = 5_000_000


def _form_token_budget() -> int:
    """The run form's per-mission LLM token budget (prompt + completion),
    clamped to 0..MAX_LLM_TOKENS_INPUT. Blank, missing or unreadable input
    means the operator default, settings.max_llm_tokens (0 = unlimited)."""
    raw = (request.form.get("max_llm_tokens") or "").strip()[:20]
    try:
        return max(0, min(MAX_LLM_TOKENS_INPUT, int(raw)))
    except ValueError:
        return settings.max_llm_tokens


@app.route("/agents/<agent_id>/run", methods=["POST"])
def agent_run(agent_id):
    agent = run_async(get_agent(agent_id))
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("agents_list"))
    # Accept either "question" (agents page) or "query" (unified search bar).
    question = (request.form.get("question") or request.form.get("query") or "").strip()[:500]
    if not question:
        flash("Enter a question for the agent to research.", "error")
        return redirect(url_for("agents_list"))

    # Optional per-run budget overrides (from the unified bar); default to the
    # agent's saved values.
    def _clamp(field, default, lo, hi):
        try:
            return max(lo, min(hi, int(request.form.get(field, default))))
        except (TypeError, ValueError):
            return default

    max_sources = _clamp("max_sources", agent.default_max_sources, 1, 100)
    max_passes = _clamp("max_passes", agent.default_max_passes, 1, 10)
    per_req = _clamp("per_req_attempts", agent.default_per_req_attempts, 1, 6)
    max_llm_tokens = _form_token_budget()
    # LLM extraction applies to the collected sources regardless of mode.
    extract = request.form.get("extract") == "on"
    extract_prompt = request.form.get("extract_prompt", "").strip()[:5000]

    try:
        job_id = create_mission_job(question, max_sources)
    except JobLimitReached as e:
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("agents_list"))
    mission = Mission(
        id=str(uuid.uuid4()), agent_id=agent.id, question=question,
        status="planning", job_id=job_id,
        budget_json=json.dumps({
            "max_passes": max_passes,
            "max_sources": max_sources,
            "per_req_attempts": per_req,
            "max_llm_tokens": max_llm_tokens,
            "extract": extract,
            "extract_prompt": extract_prompt,
        }),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    run_async(insert_mission(mission))
    # Handed the job it runs under, so the worker releases that slot on every
    # exit path -- even if the mission row is deleted underneath it.
    start_planning(mission.id, job_id)
    return redirect(url_for("mission_view", mission_id=mission.id))


@app.route("/missions")
def missions_list():
    # Grouped, not one flat list: a mission awaiting approval is a call to
    # action and must not be buried under finished ones.
    missions = run_async(get_missions_enriched())
    agents = {a.id: a for a in run_async(list_agents())}
    groups = {"needs_you": [], "running": [], "finished": []}
    for m in missions:
        if m["status"] == "awaiting_approval":
            groups["needs_you"].append(m)
        elif m["status"] in ("planning", "collecting", "synthesizing"):
            groups["running"].append(m)
        else:
            groups["finished"].append(m)
    return render_template("missions.html", groups=groups, agents=agents,
                           total=len(missions), active_page="missions")


def _cite_bound(mission, numbered) -> int:
    """How many of the brief's [n] markers may become citation links.

    Without a stored order the brief numbered ordered_sources(docs), which is
    the rail itself, so every numbered entry is citable. With a stored order
    (brief_sources_json) the brief numbered exactly the stored ids. The rail
    keeps each stored slot where it was -- a document deleted since is a None
    slot, so later numbers never shift -- and then appends documents the
    brief never numbered (e.g. from a later retask). The bound is the number
    of the last stored slot whose document still exists: [n] past it is LLM
    noise or an uncited document and stays text, while a removed slot before
    it links to its "source no longer available" row. With no removed slot
    before a surviving one this equals the count of non-None stored slots;
    past a gap, that count would unlink sources the brief did cite."""
    try:
        stored = json.loads(mission.brief_sources_json or "null")
    except (ValueError, RecursionError):
        stored = None
    if not isinstance(stored, list):
        return len(numbered)
    stored_ids = {i for i in stored if isinstance(i, str)}
    bound = 0
    for n, d in numbered:
        if d is None:
            continue            # a stored slot whose document is gone
        if d.id not in stored_ids:
            break               # the appended, never-numbered documents
        bound = n
    return bound


_BRIEF_CHECK_LABELS = {
    "uncited_paragraph": "Uncited claim",
    "junk_citation": "Cites an unusable source",
    "requirement_unmentioned": "Requirement not addressed",
}
_MAX_BRIEF_CHECKS = 50


def _brief_checks(raw) -> list[dict]:
    """brief_warnings_json as display rows {"kind", "label", "detail"}. Never
    raises: non-list JSON, non-dict rows and non-string fields are dropped,
    values are bounded, and the template autoescapes them (a detail may quote
    LLM or page text)."""
    try:
        parsed = json.loads(raw) if raw else []
    except (ValueError, RecursionError):
        return []
    rows = []
    for item in parsed[:_MAX_BRIEF_CHECKS] if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind") if isinstance(item.get("kind"), str) else ""
        detail = item.get("detail") if isinstance(item.get("detail"), str) else ""
        if not kind and not detail:
            continue
        rows.append({"kind": kind[:60],
                     "label": _BRIEF_CHECK_LABELS.get(kind, kind[:60] or "Check"),
                     "detail": detail[:500]})
    return rows


def _search_passes(raw) -> list[dict]:
    """A requirement's search_stats_json as one line per pass, in the order
    the searches ran: {"run", "n", "queries", "results", "engines"}.

    Rows are appended chronologically, and every collection run (the first,
    then each retask) numbers its passes from 1 again. So consecutive rows
    with the same pass number are one line, and a pass number lower than the
    line before starts a new run: a retask's pass 1 never merges into the
    first run's. (A retask right after a run whose last search for this
    requirement was also pass 1 looks like more queries in that pass;
    nothing in a row marks the run.) engines are the distinct names that
    answered, first seen first (empty when none did). Rows that are not
    dicts or carry no integer pass are skipped; never raises."""
    try:
        parsed = json.loads(raw) if raw else []
    except (ValueError, RecursionError):
        return []
    lines: list[dict] = []
    run = 1
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("pass"))
        except (TypeError, ValueError, OverflowError):
            continue
        try:
            results = max(0, int(item.get("results") or 0))
        except (TypeError, ValueError, OverflowError):
            results = 0
        if not lines or lines[-1]["n"] != n:
            if lines and n < lines[-1]["n"]:
                run += 1
            lines.append({"run": run, "n": n, "queries": 0, "results": 0, "engines": []})
        line = lines[-1]
        line["queries"] += 1
        line["results"] += results
        engine = item.get("engine")
        if isinstance(engine, str) and engine and engine[:40] not in line["engines"]:
            line["engines"].append(engine[:40])
    return lines


_PURPOSE_ORDER = ("plan", "assess", "extract", "brief")


def _token_rows(usage: dict) -> list[dict]:
    """get_mission_llm_usage's by_purpose as rows in pipeline order (plan,
    assess, extract, brief, then anything else alphabetically)."""
    by = usage.get("by_purpose") or {}
    rank = {p: i for i, p in enumerate(_PURPOSE_ORDER)}
    return [
        {"purpose": p, "calls": by[p]["calls"], "prompt": by[p]["prompt_tokens"],
         "completion": by[p]["completion_tokens"],
         "total": by[p]["prompt_tokens"] + by[p]["completion_tokens"]}
        for p in sorted(by, key=lambda p: (rank.get(p, len(rank)), p))
    ]


def _budget_max_llm_tokens(budget: dict) -> int:
    """The token cap collection enforces: the mission's own budget_json
    max_llm_tokens, else settings.max_llm_tokens. 0 = none."""
    try:
        return max(0, int(budget.get("max_llm_tokens", settings.max_llm_tokens)))
    except (TypeError, ValueError, OverflowError):
        return max(0, settings.max_llm_tokens)


@app.route("/missions/<mission_id>")
def mission_view(mission_id):
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    requirements = run_async(get_requirements_for_mission(mission_id))
    documents = run_async(get_mission_documents(mission_id))
    agent = run_async(get_agent(mission.agent_id))
    ext_ids = run_async(get_doc_ids_with_extractions())

    # Number the sources exactly as brief.py numbered them for the LLM (the
    # order stored with the brief when there is one), so the [n] markers in
    # the brief bind to the right rail entry. Capped like the brief itself:
    # with a stored order, ordered_sources_for_mission appends every uncited
    # document too, and numbering must not run past what a brief can cite.
    # A None entry is a stored slot whose document is gone: it keeps its
    # number (the rail shows it as removed) and no document carries it.
    ordered = brief.ordered_sources_for_mission(mission, documents)
    numbered = list(enumerate(ordered[:brief.MAX_BRIEF_SOURCES], 1))
    doc_number = {d.id: n for n, d in numbered if d is not None}

    # Sanitize first (never bypassed), then turn [n] into citation controls.
    brief_html = ""
    if mission.brief_markdown:
        brief_html = linkify_citations(
            render_markdown(mission.brief_markdown), _cite_bound(mission, numbered))

    # Sources per requirement, carrying their citation number where they have
    # one, plus the queries the agent ran/will run (stored as JSON) and what
    # each pass of searching returned.
    req_sources, req_queries, req_search = {}, {}, {}
    for r in requirements:
        req_sources[r.id] = [
            {"doc": d, "n": doc_number.get(d.id)}
            for d in run_async(get_requirement_documents(mission_id, r.id))
        ]
        try:
            req_queries[r.id] = json.loads(r.next_queries_json or "[]")
        except json.JSONDecodeError:
            req_queries[r.id] = []
        req_search[r.id] = _search_passes(r.search_stats_json)

    budget = _mission_budget(mission)
    llm_usage = run_async(get_mission_llm_usage(mission_id))
    # The compare link only when there is a previous run to compare with.
    has_parent = bool(mission.parent_mission_id) and \
        run_async(get_mission(mission.parent_mission_id)) is not None

    live_state = job_state(mission.job_id) if mission.job_id else None
    return render_template(
        "mission.html", mission=mission, agent=agent,
        requirements=requirements, documents=documents,
        numbered_sources=numbered,
        rail_available=sum(1 for _n, d in numbered if d is not None),
        req_sources=req_sources, req_queries=req_queries, req_search=req_search,
        brief_html=brief_html, brief_checks=_brief_checks(mission.brief_warnings_json),
        ext_ids=ext_ids, budget=budget,
        llm_usage=llm_usage, token_rows=_token_rows(llm_usage),
        max_llm_tokens=_budget_max_llm_tokens(budget),
        resume=_resume_offer(mission, requirements, budget, agent,
                             llm_usage["prompt_tokens"] + llm_usage["completion_tokens"]),
        has_parent=has_parent,
        live_state=live_state,
        live=mission.job_id in get_in_memory_job_ids(),
        active_page="missions",
    )


def _unique_by_url(docs: list) -> list:
    """One document per URL, first occurrence kept: the same page stored
    under two search queries is one source."""
    seen, out = set(), []
    for d in docs:
        if d.url not in seen:
            seen.add(d.url)
            out.append(d)
    return out


@app.route("/missions/<mission_id>/compare")
def mission_compare(mission_id):
    """This run beside the run it follows (parent_mission_id, which the
    scheduler sets): both briefs, and the sources split by URL into new in
    this run, dropped since the previous one, and shared."""
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    if not mission.parent_mission_id:
        flash("This mission has no previous run to compare with.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))
    parent = run_async(get_mission(mission.parent_mission_id))
    if not parent:
        flash("The previous run was deleted, so there is nothing to compare with.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))

    child_docs, parent_docs = run_async(get_mission_pair_documents(mission_id, parent.id))
    child_docs, parent_docs = _unique_by_url(child_docs), _unique_by_url(parent_docs)
    child_urls = {d.url for d in child_docs}
    parent_urls = {d.url for d in parent_docs}
    delta = {
        "new": [d for d in child_docs if d.url not in parent_urls],
        "dropped": [d for d in parent_docs if d.url not in child_urls],
        "shared": [d for d in child_docs if d.url in parent_urls],
    }
    # Both briefs sanitized like every brief. Not linkified: [n] controls
    # bind to a source rail, and this page has none.
    return render_template(
        "mission_compare.html", mission=mission, parent=parent,
        agent=run_async(get_agent(mission.agent_id)),
        brief_html=render_markdown(mission.brief_markdown) if mission.brief_markdown else "",
        parent_brief_html=render_markdown(parent.brief_markdown) if parent.brief_markdown else "",
        delta=delta, active_page="missions",
    )


def _mission_budget(mission) -> dict:
    try:
        budget = json.loads(mission.budget_json or "{}")
    except json.JSONDecodeError:
        return {}
    return budget if isinstance(budget, dict) else {}


def _budget_max_sources(budget: dict) -> int:
    try:
        return max(1, int(budget.get("max_sources", 30)))
    except (TypeError, ValueError):
        return 30


def _budget_max_passes(budget: dict) -> int:
    try:
        return max(1, int(budget.get("max_passes", 4)))
    except (TypeError, ValueError, OverflowError):
        return 4


# --- resume after a limit ---

# missions.stop_reason values a Resume picks up from: the run ended on a
# limit, or on the user's Stop, with requirements still open.
_RESUMABLE_STOPS = ("pass_budget", "source_budget", "token_budget", "user_stop")

# Per limit: the budget_json key a Resume raises, its name in plain words,
# the most one Resume may add, and the unit shown beside its field. A user
# Stop raises nothing.
_RESUME_LIMITS = {
    "token_budget": ("max_llm_tokens", "token budget", MAX_LLM_TOKENS_INPUT, "tokens"),
    "source_budget": ("max_sources", "source budget", 100, "sources"),
    "pass_budget": ("max_passes", "pass budget", 10, "passes"),
}

_STOP_LABELS = {
    "token_budget": "Stopped: token budget reached",
    "source_budget": "Stopped: source budget reached",
    "pass_budget": "Stopped: pass budget reached",
    "user_stop": "Stopped by you",
}

# Prefill for the token field when the stored budget is 0 (unlimited); a
# mission with no token limit cannot stop on one, so this is a fallback.
_RESUME_TOKEN_PREFILL = 15000


def _attempt_cap(mission, agent=None) -> int:
    """The per-requirement attempt cap collection ran under, read as
    _run_collection reads it: the mission's budget_json per_req_attempts,
    else the agent's default (3 once the agent is gone). The agent is only
    looked up when the budget lacks the key and none was passed."""
    budget = _mission_budget(mission)
    if "per_req_attempts" not in budget and agent is None:
        agent = run_async(get_agent(mission.agent_id))
    default = agent.default_per_req_attempts if agent else 3
    try:
        return max(1, int(budget.get("per_req_attempts", default)))
    except (TypeError, ValueError, OverflowError):
        return max(1, int(default))


def _reopenable(mission, requirements, agent=None) -> list:
    """The requirements a Resume puts back to work: every one still
    pending, plus every unmet one with attempts left (attempts below the
    cap): never reached ("not attempted: ..."), or tried but still open
    when the run stopped. Satisfied and capped-out requirements never
    re-run. The one rule behind the route, the page and api_mission."""
    cap = _attempt_cap(mission, agent)
    return [r for r in requirements
            if r.status == "pending" or (r.status == "unmet" and r.attempts < cap)]


def _resumable(mission, requirements, agent=None) -> bool:
    """True when a finished mission stopped on a limit (or a Stop) and has
    something to reopen."""
    return (mission.status == "done" and mission.stop_reason in _RESUMABLE_STOPS
            and bool(_reopenable(mission, requirements, agent)))


def _limit_value(budget: dict, stop_reason: str) -> int:
    """The current value of the limit that stopped the mission (0 for an
    unlimited token budget)."""
    if stop_reason == "token_budget":
        return _budget_max_llm_tokens(budget)
    if stop_reason == "source_budget":
        return _budget_max_sources(budget)
    return _budget_max_passes(budget)


def _form_count(field: str, cap: int) -> int | None:
    """A Resume form number clamped to 1..cap, or None when the field is
    missing, blank or unreadable."""
    raw = (request.form.get(field) or "").strip()[:20]
    try:
        return max(1, min(cap, int(raw)))
    except ValueError:
        return None


def _tokens_spent(budget: dict, used: int) -> bool:
    """True when the mission has a token budget and has used all of it: a
    run resumed like that stops before its first requirement."""
    cap = _budget_max_llm_tokens(budget)
    return cap > 0 and used >= cap


def _token_prefill(budget: dict, used: int) -> int:
    """The token field's prefill: the stored budget plus however far usage
    has overshot it, so the resumed run gets a full budget of headroom (a
    run only stops at the first checkpoint after crossing its budget, so a
    token stop has always overshot). Clamped to 1..MAX_LLM_TOKENS_INPUT."""
    cap = _budget_max_llm_tokens(budget)
    if cap <= 0:
        return _RESUME_TOKEN_PREFILL
    return max(1, min(MAX_LLM_TOKENS_INPUT, cap + max(0, used - cap)))


def _resume_prefill(budget: dict, stop_reason: str, used: int) -> int:
    """What the `extra` field is prefilled with, and what a blank or
    unreadable `extra` means: the token prefill after a token stop, else
    the limit's current value, clamped like the field."""
    if stop_reason == "token_budget":
        return _token_prefill(budget, used)
    _key, _name, cap, _unit = _RESUME_LIMITS[stop_reason]
    return max(1, min(cap, _limit_value(budget, stop_reason)))


def _resume_offer(mission, requirements, budget: dict, agent=None,
                  used_tokens: int = 0) -> dict | None:
    """What the done-state Resume control shows, or None when the mission
    has nothing to resume. `fields` holds the number inputs: `extra` for
    the limit that stopped the run (none after a user Stop), and
    `extra_tokens` when the token budget is spent and the stop was not the
    token budget itself (that one's `extra` is already the token field).
    Wherever a token field shows, the label says how much was used."""
    if not _resumable(mission, requirements, agent):
        return None
    stop = mission.stop_reason
    n = len(_reopenable(mission, requirements, agent))
    token_cap = _budget_max_llm_tokens(budget)
    fields, usage_note = [], ""
    limit = _RESUME_LIMITS.get(stop)
    if limit:
        _key, name, cap, unit = limit
        fields.append({"input": "extra", "name": name, "unit": unit, "max": cap,
                       "value": _resume_prefill(budget, stop, used_tokens)})
    if stop == "token_budget" and token_cap > 0:
        usage_note = f" · {used_tokens:,} of {token_cap:,} used"
    elif _tokens_spent(budget, used_tokens):
        fields.append({"input": "extra_tokens", "name": "token budget", "unit": "tokens",
                       "max": MAX_LLM_TOKENS_INPUT,
                       "value": _token_prefill(budget, used_tokens)})
        usage_note = f" · {used_tokens:,} of {token_cap:,} tokens used"
    return {"label": f"{_STOP_LABELS[stop]}{usage_note} · "
                     f"{n} requirement{'' if n == 1 else 's'} still open",
            "fields": fields}


_MAX_PLAN_ROWS = 50   # a drafted plan is a handful of requirements


def _plan_edits(raw: str) -> list[dict]:
    """The gate's plan_json, normalised: dict rows only (at most
    _MAX_PLAN_ROWS), `id` a str or None, `title` a bounded str, `queries` a
    list of bounded strs or None when the row did not send one.

    Pure and never raises: an unreadable plan is treated like JS-off (no
    edits). json.loads raises RecursionError, not a ValueError, on deeply
    nested input such as '[' * 100000, which fits in a form post."""
    try:
        parsed = json.loads(raw) if raw else []
    except (ValueError, RecursionError):
        return []
    rows = []
    for item in parsed[:_MAX_PLAN_ROWS] if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        rid, title, desc = item.get("id"), item.get("title"), item.get("description")
        queries = item.get("queries")
        rows.append({
            "id": rid if isinstance(rid, str) else None,
            "title": title.strip()[:200] if isinstance(title, str) else "",
            "description": desc.strip()[:1000] if isinstance(desc, str) else "",
            "queries": ([q.strip()[:300] for q in queries if isinstance(q, str) and q.strip()][:8]
                        if isinstance(queries, list) else None),
            "dropped": bool(item.get("dropped")),
        })
    return rows


def _rollback(mission_id: str, undo: list) -> None:
    """Run a failed transition's undo steps, newest first. Each step is
    best-effort and logged on failure: this runs while another exception
    (often the same locked database) is already propagating, and one failed
    step must not skip the rest."""
    for what, step in reversed(undo):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            print(f"[MISSION] {mission_id}: could not {what}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)


@app.route("/missions/<mission_id>/approve", methods=["POST"])
def mission_approve(mission_id):
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    # The gate lets the user reword requirements, cut them, and edit the seeded
    # queries. JS serializes that into plan_json; with JS off the field is empty
    # and we approve the plan as drafted. Parsed before the claim: it is pure,
    # so nothing about a bad payload can strand a claimed mission.
    edits = _plan_edits(request.form.get("plan_json", "").strip())

    # Claim the transition before touching anything: of two concurrent
    # approvals (a double click, two tabs) exactly one moves the row, and only
    # that one edits the plan and starts a worker.
    if not run_async(claim_mission_status(mission_id, "awaiting_approval", "collecting")):
        flash("This mission is not awaiting approval.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))

    # From here on, any failure must undo the claim: a mission left in
    # `collecting` with no worker cannot be stopped or deleted until restart.
    undo = [("reopen the gate", lambda: run_async(
        claim_mission_status(mission_id, "collecting", "awaiting_approval")))]
    try:
        existing = {r.id: r for r in run_async(get_requirements_for_mission(mission_id))}
        drop_ids = {row["id"] for row in edits if row["dropped"] and row["id"] in existing}
        additions = [row for row in edits
                     if not row["dropped"] and row["id"] not in existing and row["title"]]
        # Decided before anything is deleted, so an all-dropped plan changes
        # nothing.
        kept = len(existing) - len(drop_ids) + len(additions)
        if kept == 0:
            _rollback(mission_id, undo)
            flash("A plan needs at least one requirement — nothing was approved.", "error")
            return redirect(url_for("mission_view", mission_id=mission_id))

        for rid in drop_ids:
            run_async(delete_requirement(rid))
        for row in edits:
            if row["dropped"]:
                continue
            if row["id"] in existing:
                fields = {}
                if row["title"]:
                    fields["title"] = row["title"]
                if row["queries"] is not None:
                    # An explicitly emptied query list means "search for the
                    # requirement itself", not "keep the queries I removed".
                    fields["next_queries_json"] = json.dumps(
                        row["queries"] or [row["title"] or existing[row["id"]].title])
                if fields:
                    run_async(update_requirement(row["id"], **fields))
            elif row["title"]:
                # A requirement the user added at the gate.
                run_async(insert_requirement(Requirement(
                    id=str(uuid.uuid4()), mission_id=mission_id, title=row["title"],
                    description=row["description"],
                    rationale="Added by you at the approval gate.",
                    status="pending", attempts=0,
                    next_queries_json=json.dumps(row["queries"] or [row["title"]]),
                )))
        # The planning trace finished at the gate; collection gets its own.
        job_id = create_mission_job(mission.question,
                                    _budget_max_sources(_mission_budget(mission)))
        undo.append(("release the new job", lambda: jobs.finish_job(
            job_id, stage="error", error="approval failed")))
        run_async(update_mission(mission_id, job_id=job_id))
        undo.append(("restore the planning trace", lambda: run_async(
            update_mission(mission_id, job_id=mission.job_id))))
        # Handed the job so the worker releases it on every exit path.
        start_collection(mission_id, job_id)
    except JobLimitReached as e:
        _rollback(mission_id, undo)
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))
    except BaseException:
        _rollback(mission_id, undo)
        raise

    msg = f"Plan approved — collecting against {kept} requirements."
    if drop_ids:
        msg += f" {len(drop_ids)} dropped."
    flash(msg, "success")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/missions/<mission_id>/stop", methods=["POST"])
def mission_stop(mission_id):
    """Cooperative stop, honoured before the next requirement: the runner
    finishes the requirement in flight, marks the rest unmet (one never
    tried says "not attempted: stopped by user"), then synthesizes a brief
    from whatever was collected."""
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    if mission.job_id and request_cancel(mission.job_id):
        flash("Stopping after the current requirement — the brief will still be written.", "info")
    else:
        flash("This mission is not running.", "info")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/missions/<mission_id>/delete", methods=["POST"])
def mission_delete(mission_id):
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    # Deleting a mission out from under its worker would leave the thread
    # writing rows for a mission that no longer exists. Only a live job means
    # a worker: a finished trace still in memory behind an in-flight status is
    # a stranded mission, and must stay deletable before its trace is evicted.
    live = get_job(mission.job_id) if mission.job_id else None
    if mission.status in ("planning", "collecting", "synthesizing") \
            and live is not None and not live.done:
        flash("This mission is still running — stop it first, then delete.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))
    # A trace that never finished (e.g. a plan discarded at the gate) would
    # otherwise keep holding a job slot after its mission is gone.
    if mission.job_id:
        job = get_job(mission.job_id)
        if job is not None and not job.done:
            jobs.finish_job(mission.job_id, stage="cancelled")
    run_async(delete_mission(mission_id))
    flash("Mission deleted. Its collected sources are still in the Library.", "success")
    return redirect(url_for("missions_list"))


@app.route("/missions/<mission_id>/requirements/<req_id>/retask", methods=["POST"])
def requirement_retask(mission_id, req_id):
    """Reopen a requirement the agent gave up on, with a query the user
    supplies, and put the mission back to work on it."""
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    if mission.status in ("planning", "collecting", "synthesizing"):
        flash("The agent is still working — wait for it to finish before re-tasking.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))
    if mission.status == "awaiting_approval":
        flash("Approve the plan first — re-tasking is for a mission that has finished.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))
    # A mission that has spent its token budget cannot collect again (the
    # runner stops before the first requirement), yet a retask would still
    # re-extract and re-write the brief: refuse it before anything changes.
    token_budget = _token_budget(_mission_budget(mission))
    if token_budget > 0:
        usage = run_async(get_mission_llm_usage(mission_id))
        used = usage["prompt_tokens"] + usage["completion_tokens"]
        if used >= token_budget:
            flash(f"This mission has used {used:,} of {token_budget:,} tokens; "
                  "raise the budget before re-tasking.", "error")
            return redirect(url_for("mission_view", mission_id=mission_id))

    req = next((r for r in run_async(get_requirements_for_mission(mission_id))
                if r.id == req_id), None)
    if not req:
        flash("Requirement not found.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))

    query = request.form.get("query", "").strip()[:300]
    if not query:
        flash("Enter a search query to re-task with.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))

    try:
        queries = json.loads(req.next_queries_json or "[]")
    except json.JSONDecodeError:
        queries = []
    if not isinstance(queries, list):
        queries = []
    if query not in queries:
        queries.append(query)

    # The job first (the old live trace is gone once the process restarts, so
    # the re-run gets its own): if the queue is full, nothing about the
    # mission or the requirement has changed yet.
    try:
        job_id = create_mission_job(mission.question,
                                    _budget_max_sources(_mission_budget(mission)))
    except JobLimitReached as e:
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))
    # From here on, any failure undoes what was done so far, newest first:
    # the job holds a slot, and a claimed mission with no worker behind it
    # cannot be stopped or deleted until restart.
    undo = [("release the new job", lambda: jobs.finish_job(
        job_id, stage="error", error="re-task failed"))]
    try:
        # Then claim the transition from the status we saw, so a concurrent
        # re-task (or approve) cannot start a second worker.
        if not run_async(claim_mission_status(mission_id, mission.status, "collecting")):
            jobs.finish_job(job_id, stage="cancelled")
            flash("This mission changed state in the meantime — nothing was re-tasked.",
                  "info")
            return redirect(url_for("mission_view", mission_id=mission_id))
        undo.append((f"restore status {mission.status}", lambda: run_async(
            claim_mission_status(mission_id, "collecting", mission.status))))
        run_async(update_mission(mission_id, job_id=job_id, error=None))
        undo.append(("restore the previous trace", lambda: run_async(update_mission(
            mission_id, job_id=mission.job_id, error=mission.error))))
        # A fresh attempt budget — the user explicitly asked for another round.
        run_async(update_requirement(
            req_id, status="pending", attempts=0, accepted_by_user=0,
            next_queries_json=json.dumps(queries[-8:]),
            assessment_missing="", assessment_confidence="",
        ))
        undo.append(("restore the requirement", lambda: run_async(update_requirement(
            req_id, status=req.status, attempts=req.attempts,
            accepted_by_user=req.accepted_by_user,
            next_queries_json=req.next_queries_json,
            assessment_missing=req.assessment_missing,
            assessment_confidence=req.assessment_confidence,
        ))))
        # Handed the job so the worker releases it on every exit path.
        start_collection(mission_id, job_id)
    except BaseException:
        _rollback(mission_id, undo)
        raise
    flash(f"Re-tasking “{req.title}” with a fresh attempt budget.", "success")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/missions/<mission_id>/requirements/<req_id>/accept", methods=["POST"])
def requirement_accept(mission_id, req_id):
    """Override the assessor and accept a requirement's coverage as-is. Recorded
    as a user decision so the UI never implies the assessor was satisfied."""
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    req = next((r for r in run_async(get_requirements_for_mission(mission_id))
                if r.id == req_id), None)
    if not req:
        flash("Requirement not found.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))
    run_async(update_requirement(req_id, status="satisfied", accepted_by_user=1))
    flash(f"“{req.title}” marked satisfied — recorded as your decision, not the assessor's.",
          "success")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/missions/<mission_id>/resume", methods=["POST"])
def mission_resume(mission_id):
    """Pick a mission back up after its collection stopped on a limit (or a
    Stop) with requirements still open: raise that limit, reopen every
    requirement with attempts left, and collect again on a fresh job.
    Sources already collected are kept, satisfied and capped-out
    requirements are never re-run, and the brief is rewritten as after a
    retask."""
    mission = run_async(get_mission(mission_id))
    if not mission:
        flash("Mission not found.", "error")
        return redirect(url_for("missions_list"))
    if mission.status != "done" or mission.stop_reason not in _RESUMABLE_STOPS:
        flash("This mission has nothing to resume.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))
    reopen = _reopenable(mission, run_async(get_requirements_for_mission(mission_id)))
    if not reopen:
        flash("Every requirement is satisfied or capped out — nothing to resume.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))

    # The raised budget, worked out before anything changes. max_llm_tokens
    # is cumulative across runs; max_sources/max_passes are per run.
    budget = _mission_budget(mission)
    token_cap = _budget_max_llm_tokens(budget)
    used = 0
    if token_cap > 0:
        usage = run_async(get_mission_llm_usage(mission_id))
        used = usage["prompt_tokens"] + usage["completion_tokens"]
    raised = dict(budget)
    notes = []
    limit = _RESUME_LIMITS.get(mission.stop_reason)
    if limit:
        key, name, cap, _unit = limit
        current = _limit_value(raised, mission.stop_reason)
        if current > 0:            # an unlimited (0) token budget stays unlimited
            extra = _form_count("extra", cap)
            if extra is None:      # blank or unreadable: what the field is prefilled with
                extra = _resume_prefill(budget, mission.stop_reason, used)
            raised[key] = current + extra
            notes.append(f"{name} raised to {raised[key]:,}")
    # Any other stop may raise the token budget as well (extra_tokens); a
    # token-budget stop's `extra` already is that raise.
    if mission.stop_reason != "token_budget" and token_cap > 0:
        extra_tokens = _form_count("extra_tokens", MAX_LLM_TOKENS_INPUT)
        if extra_tokens is not None:
            raised["max_llm_tokens"] = token_cap + extra_tokens
            notes.append(f"token budget raised to {raised['max_llm_tokens']:,}")
    # A spent token budget stops the resumed run before its first
    # requirement, yet the brief would still be re-written: unless the raise
    # takes the budget past what was used, refuse before anything changes.
    if token_cap > 0 and _tokens_spent(raised, used):
        flash(f"This mission has used {used:,} of its {token_cap:,}-token budget; "
              f"raise the token budget above {used:,} to resume.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))

    # The job first, as for a retask: if the queue is full, nothing about
    # the mission or its requirements has changed yet.
    try:
        job_id = create_mission_job(mission.question, _budget_max_sources(raised))
    except JobLimitReached as e:
        flash(f"Busy: {e}.", "error")
        return redirect(url_for("mission_view", mission_id=mission_id))
    # From here on, any failure undoes what was done so far, newest first:
    # the job holds a slot, and a claimed mission with no worker behind it
    # cannot be stopped or deleted until restart.
    undo = [("release the new job", lambda: jobs.finish_job(
        job_id, stage="error", error="resume failed"))]
    try:
        # Then claim the transition, so a concurrent resume (or retask)
        # cannot start a second worker.
        if not run_async(claim_mission_status(mission_id, "done", "collecting")):
            jobs.finish_job(job_id, stage="cancelled")
            flash("This mission changed state in the meantime — nothing was resumed.",
                  "info")
            return redirect(url_for("mission_view", mission_id=mission_id))
        undo.append(("restore status done", lambda: run_async(
            claim_mission_status(mission_id, "collecting", "done"))))
        run_async(update_mission(
            mission_id, budget_json=json.dumps(raised),
            resume_count=mission.resume_count + 1,
            error=None, stop_reason=None, job_id=job_id))
        undo.append(("restore the mission", lambda: run_async(update_mission(
            mission_id, budget_json=mission.budget_json,
            resume_count=mission.resume_count, error=mission.error,
            stop_reason=mission.stop_reason, job_id=mission.job_id))))
        # Reopened with their attempts kept, so each gets only the tries it
        # has left (one never reached has 0).
        for r in reopen:
            run_async(update_requirement(
                r.id, status="pending", assessment_missing="", assessment_confidence=""))
            undo.append((f"restore requirement {r.id}", lambda r=r: run_async(
                update_requirement(r.id, status=r.status,
                                   assessment_missing=r.assessment_missing,
                                   assessment_confidence=r.assessment_confidence))))
        # Handed the job so the worker releases it on every exit path.
        start_collection(mission_id, job_id)
    except BaseException:
        _rollback(mission_id, undo)
        raise
    n = len(reopen)
    flash(f"Resuming — {n} requirement{'' if n == 1 else 's'} reopened"
          f"{''.join(', ' + note for note in notes)}.", "success")
    return redirect(url_for("mission_view", mission_id=mission_id))


@app.route("/missions/<mission_id>/brief.md")
def mission_brief_export(mission_id):
    mission = run_async(get_mission(mission_id))
    if not mission or not mission.brief_markdown:
        flash("No brief to export yet.", "info")
        return redirect(url_for("mission_view", mission_id=mission_id))
    slug = re.sub(r"[^a-z0-9]+", "-", (mission.question or "brief").lower()).strip("-")[:60]
    return Response(
        mission.brief_markdown,
        mimetype="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug or "brief"}.md"'},
    )


@app.route("/api/mission/<mission_id>")
def api_mission(mission_id):
    mission = run_async(get_mission(mission_id))
    if not mission:
        return {"error": "not found"}, 404
    requirements = run_async(get_requirements_for_mission(mission_id))
    state = {
        "id": mission.id,
        "status": mission.status,
        "question": mission.question,
        "error": mission.error,
        "has_brief": bool(mission.brief_markdown),
        "requirements": [
            {"id": r.id, "title": r.title, "status": r.status,
             "attempts": r.attempts,
             "missing": r.assessment_missing or "",
             "confidence": r.assessment_confidence or ""}
            for r in requirements
        ],
        "satisfied": sum(1 for r in requirements if r.status == "satisfied"),
        "unmet": sum(1 for r in requirements if r.status == "unmet"),
        "total": len(requirements),
        "done": mission.status in ("done", "error"),
        "stop_reason": mission.stop_reason,
        "resumable": _resumable(mission, requirements),
    }
    # Prompt + completion across every recorded call; the page's LLM tokens
    # cell ([data-tele-tokens]) follows it live.
    usage = run_async(get_mission_llm_usage(mission_id))
    state["llm_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if mission.job_id:
        js = job_state(mission.job_id)
        if js:
            state["trace"] = {
                "stage": js["stage"], "elapsed": js["elapsed"],
                "log": js["log"],
                # Monotonic count of every log line: `log` is a sliding
                # window, so the page's render cursor keys off this.
                "log_total": js.get("log_total", len(js["log"])),
                "urls": js["urls"],
                "pass_num": js["pass_num"], "sources_used": js["sources_used"],
                "cancel_requested": js["cancel_requested"],
            }
    return state


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        vals = {
            "llm_provider": request.form.get("llm_provider", "").strip()[:200] or settings.llm_provider,
            # Fast tier may be intentionally blank (falls back to reasoning).
            "llm_provider_fast": request.form.get("llm_provider_fast", "").strip()[:200],
            "ollama_api_base": request.form.get("ollama_api_base", "").strip()[:300] or settings.ollama_api_base,
        }
        try:
            vals["search_max_results"] = max(1, min(20, int(request.form.get("search_max_results", settings.search_max_results))))
        except (TypeError, ValueError):
            vals["search_max_results"] = settings.search_max_results
        # Only overwrite the API key when a new one is supplied.
        key = request.form.get("llm_api_key", "").strip()
        if key:
            vals["llm_api_key"] = key
        save_overrides(vals)
        flash("Settings saved — applied immediately, no restart needed.", "success")
        return redirect(url_for("settings_page"))

    return render_template(
        "settings.html", s=settings, models=known_models(),
        has_key=bool(active_api_key()), active_page="settings",
    )


if __name__ == "__main__":
    app.debug = settings.flask_debug
    _dev_server_host = settings.flask_host
    if dev_server_exposed(_dev_server_host):
        _warn_exposure(f"FLASK_HOST={_dev_server_host} binds the dev server "
                       "beyond localhost")
    # The reloader's parent process only watches files and re-spawns the
    # child (WERKZEUG_RUN_MAIN=true), which serves; initialise there only.
    if not (app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true"):
        initialize()
    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )
