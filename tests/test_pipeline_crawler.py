"""crawler.crawl_urls_with_progress: records every URL it tried (including
redirect targets) and never lets one hung page hold the crawl forever."""
import asyncio
import time

import pytest

import crawler
import jobs
from models import SearchResult


class _MD:
    def __init__(self, text):
        self.raw_markdown = text
        self.fit_markdown = text


class _Result:
    def __init__(self, url, success=True, title="A real page", words=120, error=""):
        self.url = url
        self.success = success
        self.error_message = error
        self.markdown = _MD("word " * words) if success else None
        self.links = {"internal": [], "external": []}
        self.metadata = {"title": title}


def _fake_crawler(behaviour):
    """behaviour: url -> async callable returning a _Result."""

    class FakeCrawler:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def arun(self, url, config=None):
            return await behaviour[url]()

    return FakeCrawler


def _job(urls):
    jid = jobs.create_job("q", 5, False, "")
    jobs.add_urls(jid, [jobs.JobUrl(url=u) for u in urls])
    return jid


def _srs(urls):
    return [SearchResult(url=u, title=u, snippet="") for u in urls]


def test_attempted_records_requested_and_final_urls(monkeypatch):
    async def redirected():
        return _Result("http://a.example/final")

    async def failed():
        return _Result("http://b.example/x", success=False, error="net::ERR")

    async def fine():
        return _Result("http://c.example/1")

    monkeypatch.setattr(crawler, "AsyncWebCrawler", _fake_crawler({
        "http://a.example/start": redirected,
        "http://b.example/x": failed,
        "http://c.example/1": fine,
    }))
    urls = ["http://a.example/start", "http://b.example/x", "http://c.example/1"]
    jid = _job(urls)
    attempted = {"http://already.example/"}
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs(urls), "q", jid, attempted=attempted))

    assert sorted(d.url for d in docs) == ["http://a.example/final", "http://c.example/1"]
    assert attempted == {"http://already.example/", "http://a.example/start",
                         "http://a.example/final", "http://b.example/x",
                         "http://c.example/1"}


def test_attempted_is_optional(monkeypatch):
    async def fine():
        return _Result("http://c.example/1")

    monkeypatch.setattr(crawler, "AsyncWebCrawler", _fake_crawler({"http://c.example/1": fine}))
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs(["http://c.example/1"]), "q", ""))
    assert [d.url for d in docs] == ["http://c.example/1"]


def test_hung_page_times_out(monkeypatch):
    async def hangs():
        await asyncio.sleep(30)
        return _Result("http://slow.example/")

    async def fine():
        return _Result("http://c.example/1")

    monkeypatch.setattr(crawler, "AsyncWebCrawler", _fake_crawler({
        "http://slow.example/": hangs, "http://c.example/1": fine}))
    monkeypatch.setattr(crawler, "_arun_timeout_s", lambda: 0.05)
    urls = ["http://slow.example/", "http://c.example/1"]
    jid = _job(urls)
    attempted = set()

    t0 = time.monotonic()
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs(urls), "q", jid, attempted=attempted))
    assert time.monotonic() - t0 < 5

    assert [d.url for d in docs] == ["http://c.example/1"]
    slow = next(u for u in jobs.get_job(jid).urls if u.url == "http://slow.example/")
    assert slow.status == "error" and "timed out" in (slow.error or "")
    assert "http://slow.example/" in attempted
    assert jobs.get_job(jid).crawl_done == 2


def test_timeout_derives_from_crawl_timeout(monkeypatch):
    monkeypatch.setattr(crawler.settings, "crawl_timeout", 30000)
    assert crawler._arun_timeout_s() == pytest.approx(75.0)
