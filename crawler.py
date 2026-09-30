import asyncio
import json
import uuid
from datetime import datetime, timezone
from html import escape as _esc
from urllib.parse import urlparse

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


async def crawl_urls_with_progress(search_results: list[SearchResult], search_query: str,
                                   job_id: str, attempted: set[str] | None = None,
                                   aliases: dict[str, str] | None = None) -> list[Document]:
    """Crawl `search_results` with progress reported into the job store.

    One page can go by several URLs (the search result, the URL crawl4ai
    reports, the post-redirect URL). When `attempted` is given, every one of
    them that this call touched is added to it — failed and junk pages
    included — so a caller can avoid fetching the same page again under
    another name. When `aliases` is given, each name other than the stored
    document's `url` is mapped to it, so a caller can find the document from
    any of them. The extra names are also kept in the document's metadata
    (`requested_url`, `redirected_url`) for callers in a later run."""
    from jobs import update_url, add_log, inc_counter

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
        async def crawl_one(sr: SearchResult) -> Document | None:
            async with semaphore:
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
                        msg = (result.error_message or "unknown error")[:140]
                        update_url(job_id, sr.url, status="error", error=msg)
                        inc_counter(job_id, "crawl_done")
                        add_log(job_id, "err", f"failed <code>{_esc(sr.url)}</code>: {_esc(msg)}")
                        return None

                    parsed = urlparse(result.url)
                    markdown_content = result.markdown.raw_markdown if result.markdown else ""
                    fit_content = result.markdown.fit_markdown if result.markdown else ""
                    internal_links = len(result.links.get("internal", [])) if result.links else 0
                    external_links = len(result.links.get("external", [])) if result.links else 0
                    metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
                    title = metadata.get("title") or sr.title or parsed.netloc
                    word_count = len(markdown_content.split()) if markdown_content else 0

                    # A captcha wall or empty shell "succeeds" as far as the
                    # browser is concerned. Reject it here so it is never
                    # stored, never cited, and never charged to the mission's
                    # source budget.
                    junk, why = looks_like_block_page(title, word_count)
                    if junk:
                        update_url(job_id, sr.url, status="error", error=why)
                        inc_counter(job_id, "crawl_done")
                        add_log(job_id, "warn",
                                f"discarded <code>{_esc(sr.url)}</code>: {_esc(why)}")
                        return None

                    # The page's other names, resolvable now (aliases) and in
                    # a later run of the same mission (metadata).
                    redirected = getattr(result, "redirected_url", None)
                    if sr.url != result.url:
                        metadata["requested_url"] = sr.url
                    if redirected and redirected != result.url:
                        metadata["redirected_url"] = redirected
                    if aliases is not None:
                        for name in names:
                            if name != result.url:
                                aliases[name] = result.url

                    doc = Document(
                        id=str(uuid.uuid4()),
                        url=result.url,
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
                    )

                    update_url(
                        job_id, sr.url,
                        status="done", title=title, words=word_count,
                        links_internal=internal_links, links_external=external_links,
                    )
                    inc_counter(job_id, "crawl_done")
                    add_log(job_id, "ok", f"<em>{_esc(title[:80])}</em> · {word_count:,}w")
                    return doc
                except asyncio.TimeoutError:
                    # str(TimeoutError()) is "", so say what happened.
                    msg = f"timed out after {limit:.0f}s"
                    update_url(job_id, sr.url, status="error", error=msg)
                    inc_counter(job_id, "crawl_done")
                    add_log(job_id, "err", f"gave up on <code>{_esc(sr.url)}</code>: {_esc(msg)}")
                    return None
                except Exception as e:
                    update_url(job_id, sr.url, status="error", error=str(e)[:140])
                    inc_counter(job_id, "crawl_done")
                    add_log(job_id, "err", f"exception on <code>{_esc(sr.url)}</code>: {_esc(str(e)[:120])}")
                    return None

        results = await asyncio.gather(*[crawl_one(sr) for sr in search_results])
        documents = [d for d in results if d is not None]

    return documents
