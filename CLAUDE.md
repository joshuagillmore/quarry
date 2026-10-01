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
- **A job's terminal state goes through `finish_job` or `finish_if_running`.** `finish_job(job_id, stage="done", error=None)` stamps `Job.finished_at` and is how a worker marks its job over — on success, on a crawl error, on a cancel, and when a mission reaches its approval gate. `finish_if_running(job_id, stage="error", error=None)` does the same write only if the job has not already finished (check and write under one lock hold); the crash/cleanup paths (`jobs._run_thread`, `agent_runner`'s thread wrapper) use it so they never overwrite a stage the worker set properly. `update_job(done=True)` is not a substitute: it leaves `finished_at` unset. **The gate releases the job's slot**: hitting `awaiting_approval` calls `finish_job`, so a mission idling at the gate waiting on a human doesn't count against the bounded job-store limit; `POST /missions/<id>/approve` then creates a **fresh** job (new job_id) for the collection phase rather than reusing the planning job's slot.

### Progress streaming

The crawl page is driven by the SSE endpoint alone (`GET /api/job/<id>/stream`, capped at ~10 min) — there's no separate polling loop. Each event carries `log_total`, a monotonic count of every log line ever added to the job (not just what's currently buffered), so the frontend can tell what it's already seen across events without diffing. A job can be evicted from the in-memory store (see the bounded job store below) or lost to a restart: if it is already gone when the stream opens, the endpoint answers `204` (the one status that stops EventSource reconnecting); if it disappears mid-stream, the stream sends `event: gone` and ends, and the page shows the same "no longer available" notice. Either way "gone" is an expected outcome for an old job, not a failure. `job_state()` is the single serialization point that converts a `Job` dataclass into the JSON the frontend consumes — keep it and the templates in sync.

### Two crawler functions — use the progress one

`crawler.py` has both `crawl_urls` (batch, no progress) and `crawl_urls_with_progress`. **The app only uses the latter.** It crawls with an `asyncio.Semaphore(4)` and reports per-URL state back into the job store via `update_url`/`add_log`/`inc_counter`. The one-shot crawl (`_run_job`) passes `skip_on_cancel=True`: a cancel marks every page not fetched yet `skipped`, keeps the pages already fetched, and ends the job `cancelled`. Missions leave it off, because their stop is honoured before the next requirement: the requirement in flight finishes its crawl and assessment. Crawl4ai produces two markdown variants per page: `raw_markdown` (stored as `content_markdown`) and `fit_markdown` (stored as `content_fit`); the extractor prefers `content_fit`.
- **A failed or blocked crawl gets one plain-HTTP retry.** When a page's Chromium fetch fails or times out, or `content_quality.looks_like_block_page` rejects it, and `settings.crawl_fallback` is true (default), `crawl_urls_with_progress` fetches the URL once more (`_fetch_fallback_html`). It streams the response with `httpx.stream` (a desktop `User-Agent`, `follow_redirects=True`, `Accept-Encoding: gzip, deflate` pinned, never `br`) and judges it from the headers before any body is read: a non-`200` status, a non-HTML content type, an unsupported or stacked `Content-Encoding`, or a `Content-Length` past the 5 MB cap is refused without downloading. The body is read raw (`iter_raw`, so httpx never decodes it) through `_CappedBody`, which inflates gzip/deflate itself with zlib's `max_length` and so bounds the **decoded** size exactly. A wall-clock deadline (the crawl timeout) runs from before the request, checked after the final headers and after every chunk, since httpx's timeout is per read. All of that body work — fetching, inflating, decoding, the block-page check — runs off the event loop via `asyncio.to_thread`, and every scan of the page-controlled HTML (title, script/style stripping, rough word count) is linear. Only a page that passes is converted: `crawler.arun(url="raw:" + html, config=_conversion_config(final_url))`, a plain config with no browser-requiring flags (crawl4ai would otherwise render a `raw:` page in Chromium) and the final URL as `base_url`. The result then goes through the same `_page_document` junk gate as a normal crawl, so a thin or block page is still discarded. Rows fetched this way are stored under the fetch's final URL with `metadata_json["fetched_via"] = "fallback"`, and the URL's status becomes `done`, not a degraded state. The fallback never runs for `file:`/non-HTTP URLs, and its log lines are still `html.escape`d like the rest of the crawl log.

### Storage (`storage.py`)

- **No connection pool** — every function opens its own `aiosqlite.connect()`. `DB_PATH` is resolved once and cached at module level (`get_db_path`).
- **`init_db()` is idempotent and self-migrating**: `CREATE TABLE IF NOT EXISTS`, columns added to older tables through `_add_column` (which checks `PRAGMA table_info` first and only then runs `ALTER TABLE ... ADD COLUMN`, so a real ALTER failure surfaces instead of being swallowed), and an FTS5 virtual table `documents_fts`. If the FTS row count diverges from `documents`, it **rebuilds the whole FTS index**. It runs once via `app.initialize()` (see Deployment notes) rather than on every request.
- Documents are keyed by UUID but **`UNIQUE(url, search_query)`**. Both the one-shot crawl and mission collection write through `upsert_document`, a single atomic `INSERT ... ON CONFLICT(url, search_query) DO UPDATE ... RETURNING id` — re-crawling the same URL under the same query updates the row **in place and keeps its id** rather than replacing it. `insert_document` is a thin alias that returns that same id and raises on error. FTS rows are deleted+reinserted alongside every document write to stay consistent.
- Full-text search input is tokenized and quoted by `_build_fts_query` before hitting `MATCH` to avoid FTS5 syntax injection.
- **Library listing is paginated at the SQL layer, not in Python.** `get_all_documents`/`get_documents_by_search` take `limit`/`offset`, and a `preview_chars` argument that — when set — has SQLite itself truncate the preview (`substr(coalesce(nullif(content_fit, ''), content_markdown), 1, ?)`; the `nullif` matters because the crawler stores a missing `fit_markdown` as `''`, not `NULL`) and drops `content_markdown` from the row entirely, so a Library page of cards never pulls full document bodies over the wire.

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
  daemon threads (`asyncio.run`), mirroring `jobs._run_thread`. Both take an
  optional `job_id`; when one is given, the background thread finishes that
  exact job on every exit path — success, error, or the mission row having
  been deleted out from under it mid-run — instead of trying to look the job
  up from the mission. `mission_approve`, `requirement_retask`, `agent_run`,
  and `scheduler._launch` all pass the job id they themselves created.
- **Completion is concrete, not vibes:** a mission is done when every requirement
  is `satisfied` or `unmet`. The circuit-breaker is `per_req_attempts` — a
  requirement still unmet after that many tries is marked `unmet` and the agent
  moves on. `max_passes`/`max_sources` are the global budget backstops (stored in
  `missions.budget_json`).
- **A token budget is a stop condition, not a suggestion.** `_run_collection`
  checks `storage.get_mission_llm_usage(mission_id)`'s prompt+completion total
  against `budget.get("max_llm_tokens", settings.max_llm_tokens)` (`0` =
  unlimited) at each pass boundary and again before each requirement inside
  the pass — the same checkpoint the mid-pass `Stop` cancellation already
  used — and crossing it stops the loop and logs `token budget reached`.
  Collection stopped that way also skips extraction (`skipping extraction:
  token budget reached`, one LLM call per source), while the brief, a single
  call, still runs. Every requirement still pending is marked `unmet`, the
  same shape as any unmet requirement rather than a crash; one never tried
  (`attempts == 0`) gets `assessment_missing="not attempted: token budget
  reached"` (or `"not attempted: stopped by user"` after a Stop), while one
  already tried in an earlier pass keeps its last assessor gap text.
  `requirement_retask` refuses a mission whose spend has reached its budget,
  with a flash and no state change: the runner would stop before the first
  requirement, yet the retask would still re-extract and re-write the brief.
  The per-run override is the "LLM tokens" field in the **Agentic
  Crawl** panel of the Search page's run form (`max_llm_tokens`, 0–5,000,000,
  blank or non-numeric falling back to `settings.max_llm_tokens`); `agent_run`
  writes it into that mission's `budget_json`. There is no agent-level
  default — the `agents` table is untouched — so re-running a mission from
  its own page (which carries no budget forward) always gets the operator
  default, not whatever budget its previous run used.
- **A mission that stopped on a limit can be resumed.** `_run_collection`
  records why it ended in `missions.stop_reason`, written by `_synthesize`
  in the same `update_mission` as the brief: `complete` (nothing left
  pending — every requirement satisfied or capped out, even when a budget
  ran out on that same last pass), `pass_budget`, `source_budget`,
  `token_budget` or `user_stop`. It is NULL for missions finished before the
  column existed and while a mission is running or has failed. A `done`
  mission with one of the four limit reasons gets a **Resume** control in
  the done-state telemetry actions (`POST /missions/<id>/resume`, one
  `extra` field). It reopens every requirement still `pending` plus every
  `unmet` one with attempts left (`attempts < per_req_attempts`, from
  `budget_json`, else the agent's default) — never reached ("not
  attempted: ...") or tried but still open when the run stopped, which is
  how a `pass_budget` stop leaves them — clearing their assessment and
  keeping `attempts`, so each gets only its remaining tries. Satisfied and
  capped-out requirements are never re-run, and every collected source is
  kept. It raises the limit that stopped the run:
  `max_llm_tokens += extra` (cumulative across runs, 1–5,000,000; an
  unlimited 0 stays 0), `max_sources += extra` (1–100) or `max_passes +=
  extra` (1–10), the last two being per-run budgets; after `user_stop`
  nothing is raised. Each later run gets a raised `max_sources`/`max_passes`
  in full, so repeated resumes grow them with no overall cap (each resume
  is bounded only by its own clamp); that is accepted. Whatever the stop, a
  token budget (> 0) that is already used up must be raised too, or the
  resumed run would stop before its first requirement and still re-write
  the brief: the form adds an `extra_tokens` field (1–5,000,000) when
  usage ≥ budget — after a `token_budget` stop `extra` already is that
  field — and the route refuses, with nothing changed, unless the raise
  takes the budget past the tokens used. A token field is prefilled with
  `max_llm_tokens + max(0, used - max_llm_tokens)` (a run stops only at
  the first checkpoint after crossing its budget, so it has overshot), so
  the resumed run gets a full budget of headroom, and the label shows the
  usage ("13,921 of 8,000 used"). It
  increments `resume_count`, clears `error` and
  `stop_reason`, and follows the retask shape: job first (Busy on
  `JobLimitReached`), then `claim_mission_status(id, "done", "collecting")`,
  an undo list on any later failure, then `start_collection(id, job_id)`,
  so extraction re-runs if the mission has it and the brief is rewritten.
  `_reopenable`/`_resumable` are the one rule behind the page control,
  `api_mission`'s `resumable` flag and the route.
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
  budget on something a retry can't fix. `chat_ex` (and the `chat`/`chat_json`
  wrappers, which forward the same kwargs) take `purpose` and `mission_id`,
  time every provider call, and record one `LlmCall` row via
  `storage.insert_llm_call` for each call that actually returns — the
  fallback call included, under its own model id — while an attempt that
  raised and got retried is never recorded; a recording failure is logged to
  stderr and never raised, so a DB hiccup can't fail a mission over
  telemetry. Callers pass a purpose per call site: `"plan"` (planner),
  `"assess"` (assessor), `"brief"` (synthesis), `"extract"` (extractor,
  threaded through `_extract_sources` with the mission id when extraction
  runs inside a mission, `None` for the one-shot pipeline).
  `storage.get_mission_llm_usage(mission_id)` rolls those rows up per
  mission — totals plus a `by_purpose` breakdown — for the mission page's
  telemetry strip; `api_mission` also returns `llm_tokens` so the ~1.2s poll
  updates that cell without a page reload.
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
  The LLM prompt numbers sources with `brief.ordered_sources(docs)`, and
  `_synthesize` persists that exact order to `missions.brief_sources_json`.
  The source rail uses the stored order via `brief.ordered_sources_for_mission`:
  stored ids keep their slot even when the document behind one is gone — a
  missing id yields `None` in that position rather than being skipped, so a
  deleted or stranded source doesn't shift every later citation number down —
  followed by any documents the brief did not number (e.g. added by a later
  retask), capped at `brief.MAX_BRIEF_SOURCES`. Re-rendering the mission page
  (or restarting the process) therefore can't silently renumber the rail
  against a different ordering than the one the brief text actually cites.
  The mission page renders a `None` slot as a muted "source no longer
  available" row and excludes it from `doc_number`. `linkify_citations`'s
  bound (`app._cite_bound`) is the **position of the last surviving stored
  slot**, not the count of surviving slots — those differ once a slot is
  `None`: with a stored order `[d1, gone, d2]`, the count of surviving slots
  is 2, but bounding by that count would unlink `[3]` even though it's `d2`,
  a real, still-cited source. Bounding by position instead keeps `[3]`
  linked, while `[2]` — the removed slot — still resolves to its "source no
  longer available" rail row rather than falling through as plain text.
  Past the bound, `[n]` is either LLM noise or an uncited document and stays
  plain text.
  `brief.linkify_citations()` turns `[n]` into `.cite` controls **after**
  `render_markdown` sanitization (it only ever injects markup built from an
  integer it re-serializes, so the sanitizer is never weakened or bypassed).
- **The brief is checked for its own mistakes.** After `_synthesize` writes
  the brief, `brief.brief_warnings(mission, requirements, docs, brief_md)`
  flags three things: an uncited paragraph or bullet of real length (outside
  Coverage & Gaps), a `[n]` citation pointing at a source
  `content_quality.is_usable` would reject, and a requirement whose key terms
  never surface in the brief outside its Coverage & Gaps section (which
  restates every requirement by design; a degraded coverage-only brief is not
  checked for this). `[n]` resolves only within the stored order the brief
  was numbered with, never against documents a later retask appended after
  it. The prompt asks for inline `[n]` in the Summary as well as Key
  Findings, so the uncited check does not fire on every Summary. The
  warnings are stored as `missions.brief_warnings_json` in the same
  `update_mission` call that saves the brief, and `_synthesize` logs
  `brief checks: N warning(s)`. The mission page renders any of them as a
  "Brief checks" callout above the brief — a prompt to read closer, not a
  correctness guarantee.
- **A mission can be compared against its own history.** `GET
  /missions/<id>/compare` renders `templates/mission_compare.html`: both
  briefs side by side (each through `render_markdown`, un-linkified), the two
  questions and dates, and three source lists built from
  `storage.get_mission_pair_documents(mission_id, parent_id)` — new this run,
  dropped since the parent, and shared, deduped by URL (a page stored under
  two different search queries still counts once). It 404s-to-flash when the
  mission has no `parent_mission_id` or that parent mission is gone. The
  mission page links to it ("Compare with previous run") whenever a live
  parent exists —
  every scheduled run already has one via `parent_mission_id`.
- **Approve and re-task are atomic status transitions, not unconditional
  `UPDATE`s.** `storage.claim_mission_status(mission_id, from_status,
  to_status)` is a compare-and-set: it only flips the row if it's still in
  `from_status`, so a double-click on Approve (or a retried POST after a
  dropped connection) can't launch collection twice for the same mission.
- **Telemetry + cooperative stop** live in the job store (`pass_num`,
  `sources_used`, `cancel_requested`); `agent_runner` checks `jobs.is_cancelled`
  at each pass boundary and again before each requirement, so a stop costs at
  most the requirement in flight and still produces a brief from what was
  collected. The mission page's button says so: "Stop after the current
  requirement".
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
- **What search actually returned is recorded, not just acted on.** Every
  `_collect_one` search call goes through `search.web_search_ex`, which
  returns `(results, engine)` instead of just `results`; each call appends a
  `{"pass", "query", "engine", "results"}` entry to that requirement's
  `requirements.search_stats_json` (capped at the last 40 entries). The
  requirement detail view renders these as one line per pass (`pass 1 · 3
  queries · 11 results · brave`), and `assess_requirement` gets a
  `search_note` summarizing the pass so a gap verdict caused by thin search
  results reads differently from one caused by good search and bad sources.
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
- **The crawl fallback never lets httpx decode a body.** It reads only
  `iter_raw()`, the decoded size is bounded by `_CappedBody` (zlib
  `max_length`, so a gzip bomb stops at the 5 MB cap), and `Accept-Encoding`
  stays pinned to `gzip, deflate`: httpx's own decoders (br above all) have
  no output limit, so a few hundred wire bytes could expand to gigabytes in
  one read. Any other or stacked `Content-Encoding` is refused before the
  body is read.
- **Dependencies are pinned** via `constraints.txt` (a `pip freeze` of a
  verified image) used as `pip install -r requirements.txt -c constraints.txt`.
  To upgrade deliberately: bump/rebuild, verify, re-freeze.
- **Known accepted limitations** (documented, not bugs): prompt injection from
  crawled pages can skew assessor verdicts/brief wording (bounded — the plan is
  written before any crawling; output HTML is always sanitized); the crawler
  has no private-IP blocklist (inputs come from search engines; container +
  loopback bind bound the risk); no CSP (inline scripts everywhere; bleach is
  the XSS control). Never write `.env` with PowerShell `-Encoding utf8` — the
  BOM corrupts the first key (pydantic then rejects `﻿LLM_API_KEY`).

## Config

All settings come from `.env` via Pydantic Settings (`config.py`). The singleton `settings` is imported across modules. Key vars: `LLM_API_KEY` (legacy alias `COHERE_API_KEY`), `LLM_PROVIDER`, `LLM_PROVIDER_FAST`, `OLLAMA_API_BASE`, `DB_PATH` (default `data/research.db`), `CRAWL_TIMEOUT` (ms), `LLM_TIMEOUT_S` (per-call LLM timeout, seconds, default 120), `FLASK_HOST/PORT/DEBUG` (`FLASK_HOST` defaults to `127.0.0.1`), `FLASK_SECRET_KEY`, `QUARRY_BIND` (compose-level publish interface), `QUARRY_TRUSTED_HOSTS` (extra Host-header names accepted beyond localhost). Never set `FLASK_DEBUG=true` on a network-reachable host (Werkzeug console is an RCE primitive).

`.env` tolerates unknown keys (`extra="ignore"` — a vendor's own `OPENAI_API_KEY`-style variables
are fine) and a leading UTF-8 BOM. `env_ignore_empty=True` changes precedence: an explicitly empty
value — in `.env` or in the process environment — is treated as "not set" rather than "clear this",
so an exported `LLM_PROVIDER_FAST=` no longer overrides a non-empty value already in `.env`; it
just falls through to it.

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
  `python app.py` (the dev server) calls `initialize()` itself before
  `app.run` (with the reloader on, only in the serving child process).
  `ensure_db`'s `@app.before_request` remains as a backstop and is a no-op
  once `initialize()` has run.
- Compose publishes on **`127.0.0.1` by default** (`QUARRY_BIND`) since there's no login unless `QUARRY_PASSWORD` is set; a healthcheck hits `/` every 30s.
- This is a **single-user design**: the global job store and recent-crawls tracker are not safe for concurrent users.
- Docker: `entrypoint.sh` runs as root only to `chown` the bind-mounted `data/`, then drops to the non-root `app` user (UID 1000) via `gosu`. The Dockerfile installs Chromium OS deps as root (`playwright install-deps`) *before* downloading the browser binary as `app`, because `crawl4ai-setup`'s own dep step needs root and fails silently otherwise.
