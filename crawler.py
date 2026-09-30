import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from html import escape as _esc, unescape
from urllib.parse import urlparse

import httpx
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from models import Document, SearchResult
from config import settings
from content_quality import looks_like_block_page


async def crawl_urls(search_results: list[SearchResult], search_query: str) -> list[Document]:
    browser_cfg = BrowserConfig(
        headless=True,
        browser_type="chromium",
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=50,
        page_timeout=settings.crawl_timeout,
    )

    documents = []
    urls = [sr.url for sr in search_results]

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun_many(urls=urls, config=run_cfg)

        for result in results:
            if not result.success:
                print(f"Failed to crawl {result.url}: {result.error_message}")
                continue

            parsed = urlparse(result.url)
            markdown_content = ""
            fit_content = ""

            if result.markdown:
                markdown_content = result.markdown.raw_markdown or ""
                fit_content = result.markdown.fit_markdown or ""

            internal_links = len(result.links.get("internal", [])) if result.links else 0
            external_links = len(result.links.get("external", [])) if result.links else 0

            metadata = {}
            if result.metadata:
                metadata = result.metadata if isinstance(result.metadata, dict) else {}

            doc = Document(
                id=str(uuid.uuid4()),
                url=result.url,
                domain=parsed.netloc,
                title=metadata.get("title", parsed.netloc),
                search_query=search_query,
                crawled_at=datetime.now(timezone.utc).isoformat(),
                content_markdown=markdown_content,
                content_fit=fit_content,
                word_count=len(markdown_content.split()) if markdown_content else 0,
                links_internal=internal_links,
                links_external=external_links,
                metadata_json=json.dumps(metadata),
            )
            documents.append(doc)

    return documents


def _arun_timeout_s() -> float:
    """Hard ceiling on one page, in seconds. crawl4ai's page_timeout bounds
    navigation, but not every stage after it (a wedged browser, a script that
    never settles), and one hung page would otherwise hold its semaphore slot
    — and the mission's worker and job slot — forever. Twice the navigation
    budget plus headroom for rendering and markdown extraction."""
    return settings.crawl_timeout / 1000 * 2 + 15


def _page_names(sr: SearchResult, result) -> list[str]:
    """Every URL one crawl answered to: the one requested, the one crawl4ai
    reports as `url` (0.9.x: the requested URL), and `redirected_url` (where
    the browser actually landed)."""
    names = [sr.url, getattr(result, "url", None), getattr(result, "redirected_url", None)]
    return [n for i, n in enumerate(names) if n and n not in names[:i]]


def _page_document(result, sr: SearchResult, search_query: str, url: str,
                   redirected: str | None, names: list[str],
                   aliases: dict[str, str] | None,
                   extra_meta: dict | None = None) -> tuple[Document | None, str]:
    """(Document stored under `url`, "") for a crawl result, or (None, why)
    when the page is a captcha wall or empty shell. A junk page "succeeds"
    as far as the browser is concerned; rejecting it here means it is never
    stored, never cited, and never charged to a mission's source budget.

    The page's other names (`names`, `redirected`) are kept in the
    document's metadata for a later run and, for an accepted page, mapped to
    `url` in `aliases` now."""
    parsed = urlparse(url)
    markdown_content = result.markdown.raw_markdown if result.markdown else ""
    fit_content = result.markdown.fit_markdown if result.markdown else ""
    internal_links = len(result.links.get("internal", [])) if result.links else 0
    external_links = len(result.links.get("external", [])) if result.links else 0
    metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
    title = metadata.get("title") or sr.title or parsed.netloc
    word_count = len(markdown_content.split()) if markdown_content else 0

    junk, why = looks_like_block_page(title, word_count)
    if junk:
        return None, why

    metadata.update(extra_meta or {})
    if sr.url != url:
        metadata["requested_url"] = sr.url
    if redirected and redirected != url:
        metadata["redirected_url"] = redirected
    if aliases is not None:
        for name in names:
            if name != url:
                aliases[name] = url

    return Document(
        id=str(uuid.uuid4()),
        url=url,
        domain=parsed.netloc,
        title=title,
        search_query=search_query,
        crawled_at=datetime.now(timezone.utc).isoformat(),
        content_markdown=markdown_content,
        content_fit=fit_content,
        word_count=word_count,
        links_internal=internal_links,
        links_external=external_links,
        metadata_json=json.dumps(metadata),
    ), ""


# --- Plain-HTTP fallback ---
#
# Some bot walls fingerprint the headless browser but let a plain HTTP GET
# through. When the browser fails a page or gets a block page, the page is
# fetched once more with httpx and that HTML is converted by crawl4ai
# ("raw:" URL), so it gets the same markdown and the same junk gate.

_FALLBACK_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,*/*",
}
# Bigger than any article; the body is converted in memory. Enforced on the
# decoded stream as it arrives, so a longer body is never read in full.
_FALLBACK_MAX_BYTES = 5_000_000
# The fallback's pre-check reads page-controlled HTML, so every scan here is
# linear: no pattern can run ahead past a "<" it would then backtrack over.
# (The first versions, `<[^>]*>` and lazy `.*?` spans, were quadratic: about
# 0.75 s at 40 KB and hours at 5 MB, with the GIL held.)
_TITLE_SCAN_CHARS = 64 * 1024
_TITLE_RE = re.compile(r"<title[^<>]*>([^<]*)</title", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^<>]*>")
_NON_TEXT_OPEN_RE = re.compile(r"<(script|style|noscript)\b", re.IGNORECASE)
_NON_TEXT_CLOSE_RE = {name: re.compile(f"</{name}", re.IGNORECASE)
                      for name in ("script", "style", "noscript")}


def _is_http_url(url: str) -> bool:
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(parts.netloc)


def _html_title(html: str) -> str:
    """The <title> text, looked for only near the top of the page."""
    m = _TITLE_RE.search(html, 0, _TITLE_SCAN_CHARS)
    return " ".join(unescape(m.group(1)).split())[:300] if m else ""


def _strip_non_text(html: str) -> str:
    """`html` without its script/style/noscript blocks. Each search starts
    where the last one ended, so the whole pass is linear; an opener with no
    closer stops the stripping there (the rest is kept as-is) instead of
    being searched past again."""
    parts: list[str] = []
    pos = 0
    while True:
        opener = _NON_TEXT_OPEN_RE.search(html, pos)
        if not opener:
            break
        closer = _NON_TEXT_CLOSE_RE[opener.group(1).lower()].search(html, opener.end())
        if not closer:
            break
        parts.append(html[pos:opener.start()])
        end = html.find(">", closer.end())
        pos = len(html) if end == -1 else end + 1
    parts.append(html[pos:])
    return " ".join(parts)


def _rough_word_count(html: str) -> int:
    return len(_TAG_RE.sub(" ", _strip_non_text(html)).split())


def _fallback_page_problem(html: str) -> str:
    """Why a fetched body is not worth converting (a block page, judged on
    its <title> and a rough word count), or "" when it is."""
    junk, why = looks_like_block_page(_html_title(html), _rough_word_count(html))
    return why if junk else ""


def _declared_length(response) -> int:
    try:
        return int(response.headers.get("content-length") or 0)
    except (TypeError, ValueError):
        return 0


def _decode(body: bytes, encoding: str | None) -> str:
    try:
        return body.decode(encoding or "utf-8", errors="replace")
    except LookupError:  # a charset Python does not know
        return body.decode("utf-8", errors="replace")


def _fetch_fallback_html(url: str) -> tuple[str, str, str]:
    """Worker-thread body of the fallback: (html, final url, "") from a
    plain GET of `url`, or ("", "", reason) when it is not a usable page.

    The response is streamed and judged as early as possible: status and
    content type from the headers before any body is read (a PDF link is
    never downloaded), a declared or actual size past _FALLBACK_MAX_BYTES
    aborts the read (so does a wall-clock deadline, since httpx's timeout is
    per read and a trickling body would never trip it), and only then is
    the body decoded and checked for a block page. All of it runs here, off
    the event loop: page-controlled bytes never hold up the app."""
    budget_s = settings.crawl_timeout / 1000
    deadline = time.monotonic() + budget_s
    with httpx.stream("GET", url, follow_redirects=True, timeout=budget_s,
                      headers=_FALLBACK_HEADERS) as response:
        if response.status_code != 200:
            return "", "", f"HTTP {response.status_code}"
        ctype = (response.headers.get("content-type") or "").lower()
        if "html" not in ctype:
            return "", "", f"not HTML ({ctype.split(';')[0].strip() or 'no content type'})"
        if _declared_length(response) > _FALLBACK_MAX_BYTES:
            return "", "", "page too large"
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > _FALLBACK_MAX_BYTES:
                return "", "", "page too large"
            if time.monotonic() > deadline:
                return "", "", f"timed out after {budget_s:g}s"
        html = _decode(bytes(body), response.encoding)
        final_url = str(response.url)
    problem = _fallback_page_problem(html)
    if problem:
        return "", "", problem
    return html, final_url, ""


async def _fallback_html(url: str) -> tuple[str, str, str]:
    """_fetch_fallback_html on a worker thread."""
    return await asyncio.to_thread(_fetch_fallback_html, url)


async def crawl_urls_with_progress(search_results: list[SearchResult], search_query: str,
                                   job_id: str, attempted: set[str] | None = None,
                                   aliases: dict[str, str] | None = None,
                                   skip_on_cancel: bool = False) -> list[Document]:
    """Crawl `search_results` with progress reported into the job store.

    One page can go by several URLs (the search result, the URL crawl4ai
    reports, the post-redirect URL). When `attempted` is given, every one of
    them that this call touched is added to it — failed and junk pages
    included — so a caller can avoid fetching the same page again under
    another name. When `aliases` is given, each name other than the stored
    document's `url` is mapped to it, so a caller can find the document from
    any of them. The extra names are also kept in the document's metadata
    (`requested_url`, `redirected_url`) for callers in a later run.

    A page the browser fails, times out on, or gets a block page for is
    fetched once more over plain HTTP when `settings.crawl_fallback` is on
    (http(s) URLs only); a page rescued that way is stored under the fetch's
    final URL with `"fetched_via": "fallback"` in its metadata.

    With `skip_on_cancel` (the one-shot crawl), a cancel requested on the job
    stops every page not fetched yet: it is marked "skipped" and not added to
    `attempted`; fetches already in flight finish and are returned (without
    a fallback fetch). Missions leave it off: their stop is honoured before
    the next requirement, and a skipped crawl would have the rest of the
    requirement's assessment run against sources that were never fetched."""
    from jobs import update_url, add_log, inc_counter, is_cancelled

    browser_cfg = BrowserConfig(headless=True, browser_type="chromium")
    # Stealth options: a plain headless Chromium is trivially fingerprinted, and
    # a large share of otherwise-good sources (Britannica, Stack Exchange, …)
    # answer with a Cloudflare challenge instead of content.
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=50,
        page_timeout=settings.crawl_timeout,
        simulate_user=True,
        override_navigator=True,
        user_agent_mode="random",
        remove_overlay_elements=True,
        mean_delay=0.4,
        max_range=0.8,
    )

    documents: list[Document] = []
    semaphore = asyncio.Semaphore(4)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        def done(sr: SearchResult, doc: Document) -> Document:
            update_url(
                job_id, sr.url,
                status="done", title=doc.title, words=doc.word_count,
                links_internal=doc.links_internal, links_external=doc.links_external,
            )
            inc_counter(job_id, "crawl_done")
            add_log(job_id, "ok", f"<em>{_esc((doc.title or '')[:80])}</em> · {doc.word_count:,}w")
            return doc

        def failed(sr: SearchResult, error: str) -> None:
            update_url(job_id, sr.url, status="error", error=error[:140])
            inc_counter(job_id, "crawl_done")
            return None

        async def fallback(sr: SearchResult, limit: float) -> tuple[Document | None, str]:
            html, final_url, reason = await asyncio.wait_for(_fallback_html(sr.url),
                                                             timeout=limit)
            if not html:
                return None, reason
            result = await asyncio.wait_for(
                crawler.arun(url="raw:" + html, config=run_cfg), timeout=limit)
            if not result.success:
                return None, f"conversion failed: {(result.error_message or 'unknown error')[:100]}"
            names = [n for n in dict.fromkeys((sr.url, final_url)) if n]
            doc, why = _page_document(result, sr, search_query, final_url, None, names,
                                      aliases, extra_meta={"fetched_via": "fallback"})
            if doc is not None and attempted is not None:
                attempted.update(names)
            return doc, why

        async def fail_or_fall_back(sr: SearchResult, error: str, limit: float) -> Document | None:
            """The browser could not produce a usable page: try the plain
            HTTP fallback once, then either store its page or record the
            failure. Never raises (bar cancellation): one page must not
            take down the whole crawl."""
            if (not settings.crawl_fallback or not _is_http_url(sr.url)
                    or (skip_on_cancel and is_cancelled(job_id))):
                return failed(sr, error)
            try:
                doc, reason = await fallback(sr, limit)
            except asyncio.TimeoutError:
                doc, reason = None, f"timed out after {limit:.0f}s"
            except Exception as e:  # noqa: BLE001 - reported below, never raised
                doc, reason = None, f"{type(e).__name__}: {str(e)[:100]}"
            if doc is None:
                add_log(job_id, "warn",
                        f"fallback failed: {_esc(reason)} (<code>{_esc(sr.url)}</code>)")
                return failed(sr, f"{error}; fallback: {reason}")
            add_log(job_id, "info", f"fallback fetched <code>{_esc(doc.url)}</code>")
            return done(sr, doc)

        async def crawl_one(sr: SearchResult) -> Document | None:
            async with semaphore:
                if skip_on_cancel and is_cancelled(job_id):
                    update_url(job_id, sr.url, status="skipped")
                    inc_counter(job_id, "crawl_done")
                    add_log(job_id, "warn",
                            f"skipped <code>{_esc(sr.url)}</code>: cancelled")
                    return None
                if attempted is not None:
                    attempted.add(sr.url)
                update_url(job_id, sr.url, status="fetching")
                add_log(job_id, "info", f"fetching <code>{_esc(sr.url)}</code>")
                limit = _arun_timeout_s()
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=sr.url, config=run_cfg), timeout=limit)
                    names = _page_names(sr, result)
                    if attempted is not None:
                        attempted.update(names)
                    if not result.success:
                        error = (result.error_message or "unknown error")[:140]
                        add_log(job_id, "err", f"failed <code>{_esc(sr.url)}</code>: {_esc(error)}")
                    else:
                        doc, error = _page_document(
                            result, sr, search_query, result.url,
                            getattr(result, "redirected_url", None), names, aliases)
                        if doc is not None:
                            return done(sr, doc)
                        add_log(job_id, "warn",
                                f"discarded <code>{_esc(sr.url)}</code>: {_esc(error)}")
                except asyncio.TimeoutError:
                    # str(TimeoutError()) is "", so say what happened.
                    error = f"timed out after {limit:.0f}s"
                    add_log(job_id, "err", f"gave up on <code>{_esc(sr.url)}</code>: {_esc(error)}")
                except Exception as e:
                    error = str(e)[:140]
                    add_log(job_id, "err", f"exception on <code>{_esc(sr.url)}</code>: {_esc(str(e)[:120])}")
                return await fail_or_fall_back(sr, error, limit)

        results = await asyncio.gather(*[crawl_one(sr) for sr in search_results])
        documents = [d for d in results if d is not None]

    return documents
