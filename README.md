# Quarry

A small Flask app that turns a question into a searchable library of cleaned web pages and structured LLM extractions.

![Quarry search: ask a question, the agent searches, crawls, and distills](docs/img/hero-search.png)

<sub>Light theme. The UI ships both, switchable from the tweaks panel.</sub>

![Quarry search in the light theme](docs/img/hero-search-light.png)

The agent runs three stages end-to-end:

1. **Search**: DuckDuckGo text search via [`ddgs`](https://pypi.org/project/ddgs/).
2. **Crawl**: concurrent headless Chromium fetches via [`crawl4ai`](https://github.com/unclecode/crawl4ai), producing both raw and fit-markdown.
3. **Extract** *(optional)*: per-document LLM extraction via [`litellm`](https://github.com/BerriAI/litellm), so you can use any supported provider (or a fully local model). Output is JSON: summary, key facts, entities, topics, sentiment, or whatever your custom prompt asks for.

Everything is persisted to SQLite so you can re-open documents, re-run extractions, and revisit the search trail later.

## Quick start (Docker)

```bash
cp .env.example .env
# edit .env: pick an LLM_PROVIDER and set LLM_API_KEY (see "Choosing a provider")

docker compose up -d --build
```

Then open [http://localhost:5000](http://localhost:5000).

The first build is slow (~5 min) because `crawl4ai-setup` downloads Chromium. Subsequent rebuilds reuse cached layers.

The `data/` directory is bind-mounted, so `data/research.db` survives container rebuilds.

## Local dev (no Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt -c constraints.txt
crawl4ai-setup            # one-time: installs Chromium for crawl4ai
cp .env.example .env      # then edit
python app.py
```

`python app.py` binds `127.0.0.1:5000` by default (`FLASK_HOST`/`FLASK_PORT`), reachable only
from this machine; widening it is covered by the exposure guard below. `.env` tolerates unknown
keys (a vendor's own `OPENAI_API_KEY`-style variables) and a leading UTF-8 BOM, so re-saving it
from an editor that adds one won't break config parsing. Never write it with PowerShell's
`-Encoding utf8`, though — that adds a BOM *and* corrupts the first key. An explicitly empty
exported environment variable (e.g. `export LLM_PROVIDER_FAST=`) no longer overrides a non-empty
value already set in `.env` — empty now means "not set", not "clear this".

## Running tests

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with pytest --with Flask==3.1.3 \
  --with Werkzeug==3.1.8 --with pydantic==2.13.4 --with pydantic-settings==2.14.2 \
  --with aiosqlite==0.22.1 --with python-dotenv==1.2.2 --with Markdown==3.10.2 \
  --with bleach==6.4.0 --with APScheduler==3.11.3 \
  python -m pytest -q tests
```

No `PYTHONPATH` needed — `pytest.ini` sets `pythonpath` for you. The `--with` flags pin the
versions this was last verified against; `.github/workflows/tests.yml` runs the same command on
every push and pull request.

## Configuration

All settings come from `.env` (see `.env.example`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_API_KEY` | API key for the hosted provider you chose (not needed for Ollama) | *(empty)* |
| `LLM_PROVIDER` | Reasoning-tier model (planning, gap assessment) | `cohere/command-a-03-2025` |
| `LLM_PROVIDER_FAST` | Fast-tier model (brief synthesis, extraction); empty → reuse reasoning model | *(empty)* |
| `OLLAMA_API_BASE` | Ollama endpoint for `ollama/`-prefixed models | `http://host.docker.internal:11434` |
| `QUARRY_BIND` | Host interface Docker publishes on (`127.0.0.1` = this machine only) | `127.0.0.1` |
| `SEARCH_MAX_RESULTS` | Default result count for the search form | `5` |
| `DB_PATH` | SQLite file path | `data/research.db` |
| `CRAWL_TIMEOUT` | Per-page crawl timeout (ms) | `30000` |
| `CRAWL_FALLBACK` | Retry a failed or blocked page with a plain HTTP fetch instead of giving up on it (see "Collection reliability and telemetry" below) | `true` |
| `LLM_TIMEOUT_S` | Per-call LLM request timeout (seconds) — a hung provider can't hold a mission worker (and its job slot) forever | `120` |
| `MAX_LLM_TOKENS` | Default per-mission LLM token budget (prompt + completion); a mission stops cleanly once it crosses this. `0` = unlimited | `0` |
| `FLASK_HOST` / `FLASK_PORT` | Bind address. Loopback by default; widening it is covered by the exposure guard below | `127.0.0.1:5000` |
| `FLASK_DEBUG` | Flask debug + auto-reload | `false` |
| `FLASK_SECRET_KEY` | Override Flask session signing key. Empty → generated once into `data/secret_key` | *(empty)* |
| `QUARRY_PASSWORD` | Optional login password. Empty → no login | *(empty)* |
| `QUARRY_BEHIND_PROXY` | Behind a TLS-terminating reverse proxy (ProxyFix + Secure cookie) | `false` |
| `QUARRY_TRUSTED_HOSTS` | Extra hostnames accepted in the `Host` header, comma-separated, beyond localhost/`127.0.0.1`/`[::1]` (a DNS-rebinding guard) | *(empty)* |

### Choosing an LLM provider

Quarry calls LLMs through LiteLLM, so **no vendor is required** — pick one, set
`LLM_PROVIDER`, and put that vendor's key in `LLM_API_KEY`:

| Provider | `LLM_PROVIDER` | Get an API key |
| --- | --- | --- |
| Cohere | `cohere/command-a-03-2025` | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) |
| OpenAI | `openai/gpt-4o-mini` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | `anthropic/claude-sonnet-4-5` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| Google | `gemini/gemini-2.0-flash` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| Groq | `groq/llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) |
| **Ollama (local)** | `ollama_chat/qwen2.5:14b` | **none — runs on your machine** |

**No API key at all?** Install [Ollama](https://ollama.com), run
`ollama pull qwen2.5:14b`, set `LLM_PROVIDER=ollama_chat/qwen2.5:14b`, and leave
`LLM_API_KEY` empty. Everything works locally and free.

If you already use a vendor's standard variable (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, …), leave `LLM_API_KEY` blank and LiteLLM will pick it up.

**Two tiers.** `LLM_PROVIDER` handles planning and gap assessment; the optional
`LLM_PROVIDER_FAST` handles brief synthesis and extraction. Pointing the fast
tier at a local Ollama model keeps the high-volume calls free while a stronger
hosted model does the reasoning. If the fast tier fails (Ollama down), calls
fall back to the reasoning model automatically.

Providers and the key can also be changed at runtime on the **Settings** page
(`/settings`), which links to each vendor's key page. Those choices persist to
`data/settings.json` (never the repo), apply immediately, and take precedence
over `.env`.

### Using a local Ollama from Docker

`OLLAMA_API_BASE` defaults to `http://host.docker.internal:11434`, which only
works if Ollama listens on every interface (`0.0.0.0`). Ollama's own default
bind is `127.0.0.1` — loopback only — and a connection arriving through
`host.docker.internal` is not the same connection as one from the host's own
loopback interface, so the host refuses it even though the hostname resolves
fine. The symptom: `curl http://localhost:11434` works on the host, but
Quarry's container can't reach Ollama at all.

Two fixes:

1. **Rebind Ollama to `0.0.0.0`.** Set `OLLAMA_HOST=0.0.0.0:11434` wherever
   you start Ollama and restart it; leave `OLLAMA_API_BASE` at its default.
   This also makes Ollama reachable from other machines on your LAN, so only
   do it on a network you trust.
2. **Join Quarry to Ollama's own Docker network instead.** If Ollama runs in
   a container (its own, or part of another Compose stack), copy
   `docker-compose.ollama.example.yml`, fill in your Ollama container's name
   and network, and launch with both files:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.ollama.example.yml up -d --build
   ```

   Find the network name with:

   ```bash
   docker inspect <ollama-container-name> --format '{{json .NetworkSettings.Networks}}'
   ```

   The external network must already exist — Compose won't create one for
   you — and `OLLAMA_API_BASE` then addresses Ollama by its container name
   (Docker's built-in DNS resolves it) instead of going through the host.

## Features

- **End-to-end agent run** with a live progress page showing search results, in-flight crawls, and an agent log stream.
- **Library**: every crawled page, deduplicated by URL, with client-side filtering by domain/age/word-count and full-text search across titles + snippets.
- **History**: every search the agent ran, grouped by day. Each entry has:
  - **Live crawl**: opens the trace (URL stream + agent log) for that run, while the job is still in memory.
  - **View**: opens Library filtered to just the documents from that search (`/documents?search=...`).
  - **Re-run**: re-submits the same query.
- **Persistent sidebar**: "Live crawl" links to the most recent run (pulse dot when active); "Previous live crawls" lists earlier runs from the current process.
- **Per-document deep view**: markdown content, metadata, links, related documents from the same search/domain, and any extractions, plus an inline button to run a new extraction with a custom prompt.

![Library: every crawled page, deduplicated by URL, filterable by domain and searchable full-text](docs/img/library.png)

### Agentic collection

Beyond one-shot search, a saved **agent** runs a **mission**: it decomposes the
question into requirements, stops at an editable approval gate, then collects
autonomously, assessing each requirement, re-tasking the gaps, and synthesizing
a cited brief. Completion is concrete rather than vibes: a mission ends when
every requirement is satisfied or provably unmet, bounded by attempt and source
budgets.

An agent is just a name, an area of expertise, and a budget; the persona is
generated from the expertise. Give one a cron expression and a standing question
and it runs unattended, each brief leading with what changed since its own last
run.

![Expert agents: each with an area of expertise and its own collection budget](docs/img/agents.png)

![Missions: every agent run, with per-mission requirement coverage and status](docs/img/missions.png)

### Collection reliability and telemetry

- **Crawl fallback.** If a page's headless-Chromium crawl fails, times out, or
  comes back looking like a block/captcha page, Quarry retries it once with a
  plain HTTP fetch and feeds that HTML straight into the same markdown
  pipeline. Sites that only fail because Chromium's fingerprint gets blocked
  often succeed on the plain retry. Controlled by `CRAWL_FALLBACK` (default
  on); never attempted for local/non-HTTP URLs.
- **LLM token telemetry and budget.** Every planning, assessment, brief, and
  extraction call is logged (model, purpose, token counts, duration) and
  rolled up per mission on the mission page's telemetry strip, broken down by
  purpose. Set `MAX_LLM_TOKENS` to give a mission a token budget, counted in
  prompt + completion tokens (`0` = unlimited, the default). Once a mission's
  spend passes it, collection stops cleanly at the next pass or requirement
  boundary instead of mid-requirement, and extraction is skipped too; the
  brief is still written from what was collected, so the final spend can
  run a little past the budget. Requirements that were never tried are
  marked "not attempted: token budget reached"; one already tried in an
  earlier pass keeps its last gap note. A mission that has used up its
  budget can't be re-tasked. Override it per run with the "LLM tokens"
  field in the Agentic Crawl panel on the Search page (blank uses
  `MAX_LLM_TOKENS`); it's a per-run choice, not an agent setting, so
  re-running a mission from its own page doesn't carry its budget forward.
- **Brief quality checks.** After synthesis, the brief is checked for
  uncited long paragraphs, citations pointing at a source that turned out not
  to be usable, and requirements that never actually made it into the brief's
  prose. Any findings show up as a "Brief checks" callout above the brief — a
  prompt to read closer, not a hard failure.
- **Compare view.** A mission with a previous run (every scheduled "morning
  brief" has one) links to "Compare with previous run", showing both briefs
  side by side plus which sources are new, which dropped out, and which are
  shared between the two runs, deduped by URL.
- **Search signals.** Each requirement's detail view shows a line per
  collection pass — queries run, results returned, and which engine answered
  (e.g. `pass 1 · 3 queries · 11 results · brave`) — so a pass that quietly
  returned nothing is visible instead of looking identical to one that found
  nothing worth citing.
- **Stop lands mid-pass.** Clicking "Stop after the current requirement" on
  a running mission is honored before each requirement within the current
  pass, not just between passes. Requirements that were never tried show
  "not attempted: stopped by user"; one already tried in an earlier pass
  keeps its last gap note.

## Architecture

```
app.py            Flask routes + context processor; spawns background jobs.
config.py         Pydantic Settings → reads .env.
models.py         Pydantic models: SearchResult, Document, ExtractedData, SearchRecord.
search.py         DuckDuckGo wrapper.
crawler.py        crawl4ai async wrapper; emits per-URL progress to the job store.
extractor.py      LiteLLM call; tries to parse JSON, falls back to {"raw_response": ...}.
jobs.py           In-memory job store + threaded async runner.
                  Tracks recent job IDs for the sidebar.
storage.py        aiosqlite: documents / extractions / searches tables.
templates/        Jinja2: base.html (shell + sidebar) extended by per-page templates.
static/style.css  Hand-rolled CSS, supports light/dark themes + accent swatches.
```

### Job lifecycle

1. `POST /search` calls `create_job(...)` → returns a UUID and starts a daemon thread.
2. The thread runs `_run_job(job_id)` → search → crawl → optional extract → done.
3. The crawl page polls `GET /api/job/<id>` every 500ms, rendering URL rows and log lines incrementally.
4. On completion, the page stops polling and shows a "Jump to results" button (no auto-redirect).
5. The job record (URL stream + log + document IDs) lives in memory until the process restarts.

### Data model

```
documents       crawled pages, keyed by UUID, unique on (url, search_query)
extractions     LLM output per document, with the prompt that produced it
searches        one row per agent run, with job_id back-reference for trace lookup
```

## Security posture

Designed for **single-user local use** behind a firewall. Some specifics:

- **Optional password login.** By default there is no login and Docker publishes on `127.0.0.1` (`QUARRY_BIND`), so the app is reachable only from the host machine. Before widening `QUARRY_BIND`, set `QUARRY_PASSWORD` in `.env` — every page and API then requires sign-in (rate-limited, 30-day session, logout in the sidebar). The value can be a Werkzeug hash instead of plaintext. Even with a password set, prefer Tailscale/VPN over direct internet exposure.
- **Host check (DNS-rebinding guard).** With no password set, any request whose `Host` is not `localhost`, `127.0.0.1` or `[::1]` gets a `400` unless that host is listed in `QUARRY_TRUSTED_HOSTS`. Browsing by a LAN name or IP (`http://my-box:5000`, `http://192.168.1.20:5000`) therefore needs that setting. With a password set, the allowlist applies only when `QUARRY_TRUSTED_HOSTS` is non-empty.
- **Sessions survive restarts.** The signing key is generated once into `data/secret_key` (0600); set `FLASK_SECRET_KEY` to override.
- **Response hardening:** `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin` on every response.
- **Behind TLS?** Set `QUARRY_BEHIND_PROXY=true` when a TLS-terminating reverse proxy fronts the app: it enables ProxyFix (correct client IPs for login rate limiting) and the `Secure` cookie flag. Your proxy needs to forward the original host and scheme — for nginx:
  ```
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-Proto $scheme;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  proxy_set_header X-Forwarded-Host $host;
  ```
  Without these, ProxyFix can't recover the real client IP or scheme, and `QUARRY_TRUSTED_HOSTS` (above) sees the proxy's own hostname instead of the one the browser actually used. LAN exposure over plain HTTP still sends the password and session cookie in cleartext — use a TLS proxy or Tailscale.
- **Footgun guard:** widening exposure beyond loopback — via `QUARRY_BIND` (Docker) or `FLASK_HOST` (`python app.py`) — without a password logs a loud startup warning and shows a red banner in the UI. `QUARRY_BEHIND_PROXY=true` doesn't silence it: the guard is about whether unauthenticated traffic can reach the app, not where TLS terminates.
- **Bounded job store:** at most 6 crawl/mission jobs run concurrently (each is a thread + headless Chromium); finished live traces are evicted after a keep-window so memory can't grow unbounded.
- **Pinned dependencies:** `constraints.txt` (a `pip freeze` of a verified image) pins the full tree for reproducible builds.
- **`FLASK_DEBUG` defaults to `false`** in `.env.example`. Never set it to `true` on a host reachable from untrusted networks; Werkzeug's debugger console is an RCE primitive.
- **Crawled HTML is treated as untrusted.** Page titles, URLs, and error strings are HTML-escaped before being inserted into the live agent log. Server-side templates use Jinja autoescape throughout.
- **Inputs are bounded:** query ≤ 500 chars, extraction prompt ≤ 5000 chars, `max_results` clamped to 1–20.
- **Container runs as a non-root `app` user** (UID 1000). The bind-mounted `data/` directory must be writable by that UID on the host.
- **Prompt injection is possible**: the LLM extractor sees raw page content. Treat extraction output as suggestion, not ground truth. Don't pipe it into anything that auto-executes.
- **CSRF is enforced without tokens.** Every POST is checked against `Sec-Fetch-Site` (rejecting anything other than `same-origin`/`same-site`/`none`) and, as a fallback, the `Origin` header must match the request's own host — a mismatch, or the `null` origin sandboxed iframes send, is rejected with a 403. Combined with the SameSite=Lax session cookie, that blocks the relevant cross-site POST scenarios without a token in every form.
- **DDG returns external URLs only.** No allowlist on what the crawler will fetch; a crafted query could in theory point the crawler at a private network address. Out of scope today; consider an SSRF guard if you ever expose this.

## Upgrading

- **Everyone is signed out once on this release.** Login sessions are now bound to the password: the cookie carries a token derived from the signing key and `QUARRY_PASSWORD`, so cookies issued by earlier versions are invalid and each browser has to sign in again. Changing the password later signs every session out the same way.
- **LAN access without a password now needs `QUARRY_TRUSTED_HOSTS`.** See the host check under Security posture.

## Notes & caveats

- **In-memory job store.** URL streams, log entries, and the sidebar's "Previous live crawls" list reset when the Flask process restarts. The DB persists; the live trace does not. The History → Live Crawl button auto-hides for jobs no longer in memory.
- **DuckDuckGo rate limiting.** Heavy use can return zero results temporarily; the agent surfaces this as `no search results`.
- **Extraction context window.** Documents are truncated to ~20k chars before being sent to the LLM (see `extractor.py`).
- **Single-user design.** The job store and "recent crawls" tracker are global module state, which is fine for local use but not safe for multi-user deployments.
- **Production WSGI.** The Docker image runs gunicorn (1 worker × 16 threads). The worker count must stay at 1 because the live crawl/mission trace is held in process memory, so multiple workers would split state. `python app.py` still uses Flask's dev server for local development.

## License

Released under the MIT License. See [LICENSE](LICENSE).
