# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Quarry** — a single-file-per-module Flask app that turns a search query into a persisted library of cleaned web pages plus optional LLM extractions. Pipeline: **DuckDuckGo search → concurrent headless-Chromium crawl (crawl4ai) → optional per-document LiteLLM extraction → SQLite**.

## Commands

```bash
# Local dev
python -m venv .venv && . .venv/Scripts/activate   # bin/activate on macOS/Linux
pip install -r requirements.txt -c constraints.txt
crawl4ai-setup            # one-time: downloads Chromium for crawl4ai (~5 min)
cp .env.example .env      # set LLM_API_KEY (or change LLM_PROVIDER)
python app.py             # serves on 127.0.0.1:5000 by default (FLASK_HOST)

# Docker
docker compose up -d --build

# Tests (see the Tests section below for what this covers)
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with pytest --with Flask==3.1.3 \
  --with Werkzeug==3.1.8 --with pydantic==2.13.4 --with pydantic-settings==2.14.2 \
  --with aiosqlite==0.22.1 --with python-dotenv==1.2.2 --with Markdown==3.10.2 \
  --with bleach==6.4.0 --with APScheduler==3.11.3 \
  python -m pytest -q tests
```

There is **no linter or formatter** configured. `pytest.ini` supplies `pythonpath`, so the test command above needs no `PYTHONPATH` env var; `.github/workflows/tests.yml` runs it on every push and pull request.

## Architecture

The flow crosses a **sync/async boundary** that shapes most of the code:

- **Flask routes (`app.py`) are synchronous** but all storage and crawling is `async`. Routes call the `run_async(coro)` helper, which runs a coroutine to completion, spawning a thread-pool executor when an event loop is already running. Use `run_async()` for any DB call from a route — never `await` directly in a route.
- **Background jobs run in daemon threads.** `POST /search` → `create_job()` + `run_job_in_background()` spins a thread that calls `asyncio.run(_run_job(...))`. `_run_job` is the orchestrator: search → crawl-with-progress → store → optional extract → mark done.
- **The job store is in-memory global module state** (`_store`/`_lock`/`_recent_job_ids`). It holds the live URL stream, log lines, and the sidebar's "recent crawls" list. **It is wiped on process restart** — only the SQLite DB persists. The History page's "Live crawl" links auto-hide for jobs no longer in `_store` (`get_in_memory_job_ids`). All access goes through the module-level lock.
- **A job's terminal state goes through one function.** `finish_job(job_id, stage="done", error=None)` stamps `Job.finished_at` and is the only place a job is marked over — on success, on a crawl error, and when a mission reaches its approval gate. **The gate releases the job's slot**: hitting `awaiting_approval` calls `finish_job`, so a mission idling at the gate waiting on a human doesn't count against the bounded job-store limit; `POST /missions/<id>/approve` then creates a **fresh** job (new job_id) for the collection phase rather than reusing the planning job's slot.

### Progress streaming

The crawl page is driven by the SSE endpoint alone (`GET /api/job/<id>/stream`, capped at ~10 min) — there's no separate polling loop. Each event carries `log_total`, a monotonic count of every log line ever added to the job (not just what's currently buffered), so the frontend can tell what it's already seen across events without diffing. Once a job has been evicted from the in-memory store (see the bounded job store below), the stream ends with a `204` rather than an error — "gone" is an expected outcome for an old job, not a failure. `job_state()` is the single serialization point that converts a `Job` dataclass into the JSON the frontend consumes — keep it and the templates in sync.

### Two crawler functions — use the progress one

`crawler.py` has both `crawl_urls` (batch, no progress) and `crawl_urls_with_progress`. **The app only uses the latter.** It crawls with an `asyncio.Semaphore(4)` and reports per-URL state back into the job store via `update_url`/`add_log`/`inc_counter`. Crawl4ai produces two markdown variants per page: `raw_markdown` (stored as `content_markdown`) and `fit_markdown` (stored as `content_fit`); the extractor prefers `content_fit`.

### Storage (`storage.py`)

- **No connection pool** — every function opens its own `aiosqlite.connect()`. `DB_PATH` is resolved once and cached at module level (`get_db_path`).
- **`init_db()` is idempotent and self-migrating**: `CREATE TABLE IF NOT EXISTS`, an `ALTER TABLE ... ADD COLUMN job_id` wrapped in try/except, and an FTS5 virtual table `documents_fts`. If the FTS row count diverges from `documents`, it **rebuilds the whole FTS index**. It runs once via `app.initialize()` (see Deployment notes) rather than on every request.
- Documents are keyed by UUID but **`UNIQUE(url, search_query)`** — re-crawling the same URL under the same query replaces the row (`INSERT OR REPLACE`). FTS rows are deleted+reinserted alongside every document write to stay consistent.
- Full-text search input is tokenized and quoted by `_build_fts_query` before hitting `MATCH` to avoid FTS5 syntax injection.
- **Library listing is paginated at the SQL layer, not in Python.** `get_all_documents`/`get_documents_by_search` take `limit`/`offset`, and a `preview_chars` argument that — when set — has SQLite itself truncate `content_fit` (`substr(coalesce(content_fit, content_markdown), 1, preview_chars)`) and drops `content_markdown` from the row entirely, so a Library page of cards never pulls full document bodies over the wire.

### LLM extraction (`extractor.py`)

- Uses **LiteLLM** (`litellm.completion`) with `settings.llm_provider` as the model id. Switch vendors by changing `LLM_PROVIDER` (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`) and setting the vendor's API key env var.
- The key is provider-agnostic: `config.active_api_key()` returns `LLM_API_KEY`, falling back to the legacy `COHERE_API_KEY`. When it is empty no `api_key` kwarg is passed at all, so LiteLLM reads the vendor's own env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). Ollama needs no key.
- Content is **truncated to 20k chars** before the call. Output is parsed as JSON; on parse failure it's wrapped as `{"raw_response": ...}` rather than failing.

## Agentic collection (expert agents)

A second pipeline layered on top of the one-shot search. A saved **Agent**
(persona built from an area of expertise via `prompt_templates.build_persona`)
runs a **Mission** against a question using an intelligence-collection loop:
**plan → approve → collect/assess/re-task → synthesize brief.**

- **Two background stages split by the approval gate** (`agent_runner.py`):
  `start_planning` decomposes the question into `requirements` (EEIs) and stops
  at `status=awaiting_approval`; `POST /missions/<id>/approve` then launches
  `start_collection`, which loops searching → `crawl_urls_with_progress` →
  `assess_requirement` (gap analysis) → re-tasking the unmet gaps. Both run in
  daemon threads (`asyncio.run`), mirroring `jobs._run_thread`.
- **Completion is concrete, not vibes:** a mission is done when every requirement
  is `satisfied` or `unmet`. The circuit-breaker is `per_req_attempts` — a
  requirement still unmet after that many tries is marked `unmet` and the agent
  moves on. `max_passes`/`max_sources` are the global budget backstops (stored in
  `missions.budget_json`).
- **Source of truth is SQLite, not the job store.** Mission status, requirements,
  and the brief live in the new tables (`agents`, `missions`, `requirements`,
  `mission_documents`). The in-memory job store (`jobs.create_mission_job`) only
  carries the **live trace** (log lines), so the mission view degrades to a
  static DB-rendered page once the process restarts — same pattern as the History
  "Live crawl" auto-hide.
- **Use `storage.upsert_document`, not `insert_document`, for collected docs.**
  It refreshes a doc in place on `(url, search_query)` conflict and keeps the
  same id, so `mission_documents` never points at an orphaned id (plain
  `INSERT OR REPLACE` mints a new id on conflict).
- **LLM calls go through `llm.py`** (`chat` / `chat_json`), a thin wrapper over
  LiteLLM with robust JSON extraction (`_extract_json` handles ```json fences and
  prose). It supports a **two-tier model setup** via a `tier` arg:
  `"reasoning"` → `settings.llm_provider` (Cohere) for **planning** and
  **assessment**; `"fast"` → `settings.llm_provider_fast` for summarization-style
  work (**brief synthesis** and the **document extractor**). The fast tier is
  optional — empty `LLM_PROVIDER_FAST` falls back to the reasoning model.
  `_provider_kwargs` branches on the model id: Ollama models
  (`ollama/`, `ollama_chat/`) get `ollama_api_base` and no key; hosted providers
  get `active_api_key()`. A local Ollama is reachable from the container at
  `http://host.docker.internal:11434` on Docker Desktop. `model_for(tier)`
  resolves the id. **Fast-tier calls auto-fall back to the reasoning model** on
  provider failure (e.g. Ollama down); `chat_ex` returns
  `(text, model_that_actually_answered)` and `extractor` records that true
  producer on each extraction row. Every call passes `timeout=settings.llm_timeout_s`
  (default 120s) so a hung provider can't hold a mission worker — and its job
  slot — forever. An empty completion (200 OK, no content) is treated as a
  failure and raises, instead of quietly feeding an empty string into JSON
  parsing or the brief prompt. `_complete_retrying` only retries what looks
  transient (rate limits, dropped connections, Cohere's
  `NO_VALID_RESPONSE_GENERATED`); a non-transient error — a bad model id, an
  auth failure — fails on the first attempt rather than burning the retry
  budget on something a retry can't fix.
- **The mission view** (`templates/mission.html`) renders three states from
  `mission.status` — editable **approval gate**, **requirements matrix**, and
  **brief + citation-linked source rail**. It polls `GET /api/mission/<id>`
  (~1.2s), updates the matrix/telemetry in place, and **reloads on status
  change** so the server re-renders the gate / brief rather than duplicating
  that rendering in JS.
- **The approval gate is editable.** The DOM *is* the plan; on submit JS
  serializes titles, per-requirement queries, and dropped ids into a
  `plan_json` field. `POST /missions/<id>/approve` applies those edits
  (update/insert/`delete_requirement`) before `start_collection`. With JS off
  the field is empty and the plan is approved as drafted.
- **The assessor's reasoning is persisted, not discarded**: `assessment_missing`
  and `assessment_confidence` columns on `requirements` feed the matrix's gap /
  satisfied callouts. This was the single biggest gap in the old UI.
- **Citation numbering must match the brief, even after a restart.**
  `brief.ordered_sources_for_mission(mission, docs)` is the one ordering both
  the LLM prompt and the source rail use; `_synthesize` persists that exact
  order to `missions.brief_sources_json` so re-rendering the mission page (or
  restarting the process) can't silently renumber the source rail against a
  different ordering than the one the brief text actually cites.
  `brief.linkify_citations()` turns `[n]` into `.cite` controls **after**
  `render_markdown` sanitization (it only ever injects markup built from an
  integer it re-serializes, so the sanitizer is never weakened or bypassed).
- **Approve and re-task are atomic status transitions, not unconditional
  `UPDATE`s.** `storage.claim_mission_status(mission_id, from_status,
  to_status)` is a compare-and-set: it only flips the row if it's still in
  `from_status`, so a double-click on Approve (or a retried POST after a
  dropped connection) can't launch collection twice for the same mission.
- **Telemetry + cooperative stop** live in the job store (`pass_num`,
  `sources_used`, `cancel_requested`); `agent_runner` checks `jobs.is_cancelled`
  at pass boundaries so a stop still produces a brief from what was collected.
- **Static assets are cache-busted** by mtime (`@app.url_defaults`), because a
  deploy otherwise leaves users on a stale `style.css`.
- **Scheduling ("morning brief")** lives in `scheduler.py`. An agent with a
  `schedule_cron` **and** a `schedule_question` gets an APScheduler job
  (`coalesce`, `max_instances=1`, so a missed window fires once and runs never
  overlap). A scheduled mission carries `auto_approve` in `budget_json`:
  `_run_planning` sees it, skips the human gate, and goes straight to
  collecting — nobody is at the keyboard at 07:00. Each run links to the
  previous one via `parent_mission_id`, and the brief leads with what is new
  since that specific mission (not "any mission ever", or another agent's crawl
  of the same URL would wrongly count as seen). `POST /agents/<id>/run-scheduled`
  fires it now, so a brief can be tested without waiting for its cron window.
  One scheduler exists because the image runs exactly one gunicorn worker.
  **Cron expressions are evaluated in UTC**, not the host's local timezone —
  `0 7 * * *` fires at 07:00 UTC regardless of where the container runs.
  Before launching, `scheduler.launch_scheduled_mission` checks
  `storage.agent_has_active_mission` and skips the run (returning `None`)
  if the agent already has a mission in `planning`/`collecting`/`synthesizing`,
  so a slow morning brief doesn't get a second one stacked on top of it at the
  next cron tick.

### Collection quality (why these exist)

- **Search must not depend on one engine.** `search.py` walks
  `settings.search_backends` (default `auto,brave,bing,duckduckgo`) until one
  answers. Measured on this box: the `duckduckgo` backend returned **0** results
  for a query where brave/bing returned 4 in under a second. A single-engine
  search silently starves missions.
- **A "successful" crawl is often a captcha wall.** `content_quality.py` is the
  one place that judges real content vs. block/interstitial/near-empty pages,
  shared by the crawler (before storing — junk must not consume the source
  budget) and the assessor/brief (before prompting). `crawler.py` also enables
  crawl4ai's stealth options, since plain headless Chromium is trivially
  fingerprinted.
- **One flaky LLM call must not destroy a paid-for mission.** Three layers:
  `llm._complete_retrying` retries transient provider failures (Cohere's
  `NO_VALID_RESPONSE_GENERATED` is common), `assess_requirement` converts an
  exception into a "not assessed" verdict, and `agent_runner._collect_one` is
  wrapped so a requirement's failure costs it one attempt rather than the run.
- Extraction runs `EXTRACT_CONCURRENCY` documents at once; it was the slow tail
  of every mission.

### Tests

The `tests/` suite stubs `litellm`/`crawl4ai`/`ddgs` (`tests/stubs/`) so it runs
without the heavy crawl stack. `pytest.ini` puts the repo root and
`tests/stubs` on `pythonpath`, so no `PYTHONPATH` env var is needed:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with pytest --with Flask==3.1.3 \
  --with Werkzeug==3.1.8 --with pydantic==2.13.4 --with pydantic-settings==2.14.2 \
  --with aiosqlite==0.22.1 --with python-dotenv==1.2.2 --with Markdown==3.10.2 \
  --with bleach==6.4.0 --with APScheduler==3.11.3 \
  python -m pytest -q tests
```

`tests/conftest.py` is what makes that command safe to run against a real dev
box: at import time it pins env vars (so an exported `QUARRY_PASSWORD`, or a
developer's real `.env`, never reaches the code under test) and resets the
`config.settings` singleton to its declared defaults, pointing `DB_PATH` at a
throwaway session-scoped temp directory. An autouse per-test fixture then gives
each test its own fresh SQLite file, a clean job store and login rate limiter,
and zero retry/backoff delay, restoring the settings singleton afterward so a
test that saves Settings-page overrides can't leak them into the next test.

`test_agent_planner` / `test_agent_assessor` / `test_brief` monkeypatch the LLM
calls; `test_storage_smoke` exercises the real schema, `upsert_document`
id-reuse, the join-table queries, and FTS consistency against a temp SQLite file.
`test_templates_parse` parses every Jinja template, and `test_style_css` guards
the stylesheet against corruption — a stray shell line once landed in it, and
browsers silently drop **every rule after** a malformed one, so the file still
"contains" styles that never load.

`tests/qa_harness.py`, `qa_lifecycle.py`, `qa_mission_ui.py`, `qa_ollama.py`, and
`qa_schedule.py` are end-to-end drivers (not pytest): copy one into the running
container and execute it there, e.g. `docker compose cp ./tests/qa_mission_ui.py
web-researcher:/tmp/x.py && docker compose exec -T web-researcher python
/tmp/x.py`.

## Security-relevant invariants (preserve these)

- **All crawled markdown is rendered through `markdown_render.render_markdown`**, which runs `markdown` → `bleach.clean` (tag/attr/protocol allowlist) → `bleach.linkify` with `rel="noopener nofollow" target="_blank"` hardening. Never bypass this when displaying crawled content.
- Live-log messages built in `crawler.py`/`jobs.py` HTML-escape interpolated page data with `html.escape` (`_esc`). Keep doing this — log strings are injected into the DOM.
- Inputs are bounded at the route layer: query ≤500, extract prompt ≤5000, `max_results` clamped 1–20, full-text `q` ≤200. Jinja autoescape is on everywhere.
- **Auth is optional and env-only** (`auth.py`): setting `QUARRY_PASSWORD`
  (plaintext or Werkzeug hash) gates every route except `login`/`static` via a
  `before_request` guard — HTML gets a redirect, `/api/*` gets 401 JSON. The
  login page is **standalone on purpose** (base.html's sidebar leaks counts,
  queries, and model config pre-auth). Login is rate-limited per IP in memory.
  `QUARRY_PASSWORD` must never become a Settings-page override — an
  unauthenticated visitor could set or clear it.
- The session signing key persists in `data/secret_key` (0600) via
  `config.persistent_secret_key`, so logins survive restarts; `FLASK_SECRET_KEY`
  overrides. Cookies are HttpOnly + SameSite=Lax; security headers
  (frame-deny, nosniff, referrer-policy) go out on every response.
  `QUARRY_BEHIND_PROXY=true` (TLS reverse-proxy mode) adds ProxyFix (real
  client IPs for the login limiter) and the Secure cookie flag.
- **Exposure footgun guard:** `QUARRY_BIND` (Docker) or `FLASK_HOST`
  (`python app.py`) beyond loopback without `QUARRY_PASSWORD` triggers a
  startup stderr warning and a red banner in the UI (`insecure_exposure()` in
  `app.py`). `QUARRY_BEHIND_PROXY=true` does not silence it — the guard is
  about whether unauthenticated traffic can reach the app, not where TLS
  terminates.
- **`QUARRY_TRUSTED_HOSTS`** is a DNS-rebinding guard on the `Host` header:
  while no `QUARRY_PASSWORD` is set, any Host other than
  localhost/`127.0.0.1`/`[::1]` (or one listed here) is refused outright; with
  a password set, the list is enforced only if non-empty. List your LAN or
  Tailscale hostname here if you browse to Quarry under something other than
  localhost.
- **The job store is bounded** (audit fix): `MAX_ACTIVE_JOBS` in-flight jobs
  (each is a thread + Chromium), finished traces evicted beyond
  `_DONE_KEEP`/`_DONE_TTL_S`. `create_job`/`create_mission_job` raise
  `JobLimitReached`; routes flash "Busy". The scheduler's `_launch` creates the
  job before the mission row, so a full queue skips a scheduled fire cleanly.
- **Dependencies are pinned** via `constraints.txt` (a `pip freeze` of a
  verified image) used as `pip install -r requirements.txt -c constraints.txt`.
  To upgrade deliberately: bump/rebuild, verify, re-freeze.
- **Known accepted limitations** (documented, not bugs): prompt injection from
  crawled pages can skew assessor verdicts/brief wording (bounded — the plan is
  written before any crawling; output HTML is always sanitized); the crawler
  has no private-IP blocklist (inputs come from search engines; container +
  loopback bind bound the risk); no CSP (inline scripts everywhere; bleach is
  the XSS control). Never write `.env` with PowerShell `-Encoding utf8` — the
  BOM corrupts the first key (pydantic then rejects `﻿COHERE_API_KEY`).

## Config

All settings come from `.env` via Pydantic Settings (`config.py`). The singleton `settings` is imported across modules. Key vars: `LLM_API_KEY` (legacy alias `COHERE_API_KEY`), `LLM_PROVIDER`, `LLM_PROVIDER_FAST`, `OLLAMA_API_BASE`, `DB_PATH` (default `data/research.db`), `CRAWL_TIMEOUT` (ms), `LLM_TIMEOUT_S` (per-call LLM timeout, seconds, default 120), `FLASK_HOST/PORT/DEBUG` (`FLASK_HOST` defaults to `127.0.0.1`), `FLASK_SECRET_KEY`, `QUARRY_BIND` (compose-level publish interface), `QUARRY_TRUSTED_HOSTS` (extra Host-header names accepted beyond localhost). Never set `FLASK_DEBUG=true` on a network-reachable host (Werkzeug console is an RCE primitive).

**UI-editable overrides:** the Settings page (`/settings`) persists `llm_provider`, `llm_provider_fast`, `ollama_api_base`, `llm_api_key` (legacy `cohere_api_key` still read), and `search_max_results` to `data/settings.json` (`config.save_overrides`), which is layered over `.env` at import (`load_overrides`) and mutated live on save — **`settings.json` wins over `.env`** for those keys. `known_models()` accumulates every model id ever saved so the Settings dropdowns never lose a previously used value.

## Deployment notes

- Docker runs **gunicorn with exactly 1 worker** (× 16 gthread threads), configured in `gunicorn.conf.py` rather than CLI flags. Keep `workers = 1`: the live job/mission trace is module-global memory (`jobs._store`), so >1 worker splits state. gthread is required so long-lived SSE streams aren't killed by the worker timeout. `python app.py` remains the Flask dev server for local dev.
- **Startup runs once, in the right place.** `app.initialize()` is the
  idempotent, lock-guarded entrypoint — `init_db()`,
  `reconcile_interrupted_missions()`, `start_scheduler()` — that used to live
  behind `ensure_db`'s `@app.before_request`. `gunicorn.conf.py`'s
  `post_worker_init` calls it right after the single worker forks, so the
  scheduler and a clean DB are guaranteed before gunicorn reports the worker
  healthy, instead of racing the first inbound request through a lock.
  `python app.py` (the dev server) still triggers it on the first request, the
  way `ensure_db` used to.
- Compose publishes on **`127.0.0.1` by default** (`QUARRY_BIND`) because the app has no auth; a healthcheck hits `/` every 30s.
- This is a **single-user design**: the global job store and recent-crawls tracker are not safe for concurrent users.
- Docker: `entrypoint.sh` runs as root only to `chown` the bind-mounted `data/`, then drops to the non-root `app` user (UID 1000) via `gosu`. The Dockerfile installs Chromium OS deps as root (`playwright install-deps`) *before* downloading the browser binary as `app`, because `crawl4ai-setup`'s own dep step needs root and fails silently otherwise.
