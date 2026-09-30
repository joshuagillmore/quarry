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
    def __init__(self, url, success=True, title="A real page", words=120, error="",
                 redirected_url=None):
        self.url = url
        self.success = success
        self.error_message = error
        self.markdown = _MD("word " * words) if success else None
        self.links = {"internal": [], "external": []}
        self.metadata = {"title": title}
        self.redirected_url = redirected_url


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


def _pages(urls):
    def page(u):
        async def fetch():
            return _Result(u)
        return fetch
    return {u: page(u) for u in urls}


def test_cancel_skips_every_unfetched_page_when_asked(monkeypatch):
    urls = ["http://c.example/1", "http://c.example/2?q=<b>"]
    fetched = []

    class Recording(_fake_crawler(_pages(urls))):
        async def arun(self, url, config=None):
            fetched.append(url)
            return await super().arun(url, config)

    monkeypatch.setattr(crawler, "AsyncWebCrawler", Recording)
    jid = _job(urls)
    jobs.request_cancel(jid)
    attempted = set()
    docs = asyncio.run(crawler.crawl_urls_with_progress(
        _srs(urls), "q", jid, attempted=attempted, skip_on_cancel=True))

    assert docs == [] and fetched == []
    assert attempted == set(), "a skipped page was never tried; a later run may fetch it"
    job = jobs.get_job(jid)
    assert [u.status for u in job.urls] == ["skipped", "skipped"]
    assert job.crawl_done == 2
    msgs = " ".join(entry.msg for entry in job.log)
    assert "&lt;b&gt;" in msgs and "<b>" not in msgs


def test_cancel_does_not_skip_pages_by_default(monkeypatch):
    """Missions leave skip_on_cancel off: their stop is honoured before the
    next requirement, so the crawl in flight still fetches every page."""
    urls = ["http://c.example/1", "http://c.example/2"]
    monkeypatch.setattr(crawler, "AsyncWebCrawler", _fake_crawler(_pages(urls)))
    jid = _job(urls)
    jobs.request_cancel(jid)
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs(urls), "q", jid))
    assert sorted(d.url for d in docs) == urls
    assert [u.status for u in jobs.get_job(jid).urls] == ["done", "done"]


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


def test_aliases_map_every_name_to_the_stored_url(monkeypatch):
    """Two redirect shapes: result.url already final (a), and crawl4ai
    0.9.2's result.url == requested with the landing page in redirected_url
    (c). Every name resolves to the URL the document is stored under, and
    the other names are kept in metadata so a later run can seed them."""
    import json as _json

    async def url_is_final():
        return _Result("http://b.example/final")

    async def redirected_field():
        return _Result("http://c.example/asked", redirected_url="http://d.example/landed")

    async def plain():
        return _Result("http://e.example/1", redirected_url="http://e.example/1")

    monkeypatch.setattr(crawler, "AsyncWebCrawler", _fake_crawler({
        "http://a.example/start": url_is_final,
        "http://c.example/asked": redirected_field,
        "http://e.example/1": plain,
    }))
    urls = ["http://a.example/start", "http://c.example/asked", "http://e.example/1"]
    attempted, aliases = set(), {}
    docs = asyncio.run(crawler.crawl_urls_with_progress(
        _srs(urls), "q", _job(urls), attempted=attempted, aliases=aliases))

    by_url = {d.url: d for d in docs}
    assert set(by_url) == {"http://b.example/final", "http://c.example/asked",
                           "http://e.example/1"}
    assert aliases == {"http://a.example/start": "http://b.example/final",
                       "http://d.example/landed": "http://c.example/asked"}
    assert attempted == {"http://a.example/start", "http://b.example/final",
                         "http://c.example/asked", "http://d.example/landed",
                         "http://e.example/1"}
    meta_b = _json.loads(by_url["http://b.example/final"].metadata_json)
    meta_c = _json.loads(by_url["http://c.example/asked"].metadata_json)
    meta_e = _json.loads(by_url["http://e.example/1"].metadata_json)
    assert meta_b["requested_url"] == "http://a.example/start"
    assert meta_c["redirected_url"] == "http://d.example/landed"
    assert "requested_url" not in meta_e and "redirected_url" not in meta_e
    assert meta_b["title"] == "A real page", "page metadata is kept"


# ---------- plain-HTTP fallback ----------

import json  # noqa: E402
import threading  # noqa: E402

import httpx  # noqa: E402  (tests/stubs/httpx.py)

_GOOD_HTML = ("<html><head><title>Plain page</title></head><body>"
              + "<p>" + "fact " * 200 + "</p></body></html>")
_BLOCK_HTML = ("<html><head><title>Just a moment...</title></head>"
               "<body>Checking your browser</body></html>")


class _Stream:
    """Stands in for the response of `with httpx.stream(...) as r`: status,
    headers and url are there at once; the body only arrives when iterated,
    and `pulled` counts the bytes the caller actually read."""

    def __init__(self, status=200, ctype="text/html; charset=utf-8", text=_GOOD_HTML,
                 url="http://f.example/1", chunks=None, length=None, encoding="utf-8"):
        self.status_code = status
        self.headers = {"content-type": ctype}
        if length is not None:
            self.headers["content-length"] = str(length)
        self.url = url
        self.encoding = encoding
        self._chunks = chunks if chunks is not None else [text.encode()]
        self.pulled = 0
        self.body_read = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True
        return False

    def iter_bytes(self):
        self.body_read = True
        for chunk in self._chunks:
            self.pulled += len(chunk)
            yield chunk


def _raw_aware_crawler(browser, raw_calls):
    """arun(url) answers from `browser` for real URLs; a "raw:" URL is
    converted like crawl4ai would: markdown from the HTML's text."""
    import re

    class FakeCrawler(_fake_crawler(browser)):
        async def arun(self, url, config=None):
            if url.startswith("raw:"):
                html = url[len("raw:"):]
                raw_calls.append(html)
                text = re.sub(r"<[^>]+>", " ", html)
                title = re.search(r"<title>(.*?)</title>", html).group(1)
                res = _Result(url)
                res.markdown = _MD(" ".join(text.split()))
                res.metadata = {"title": title}
                return res
            return await super().arun(url, config)

    return FakeCrawler


def _fallback_case(monkeypatch, browser_result, http=None, url="http://f.example/1"):
    """Crawl one URL whose browser fetch gives `browser_result`; `http` is
    what httpx.stream returns (or raises). Returns (docs, job, calls, raw_calls)."""
    async def page():
        return browser_result

    calls, raw_calls = [], []

    def fake_stream(method, u, **kwargs):
        calls.append((method, u, kwargs))
        if isinstance(http, Exception):
            raise http
        return http

    monkeypatch.setattr(crawler.httpx, "stream", fake_stream)
    monkeypatch.setattr(crawler, "AsyncWebCrawler", _raw_aware_crawler({url: page}, raw_calls))
    jid = _job([url])
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs([url]), "q", jid))
    return docs, jobs.get_job(jid), calls, raw_calls


def _msgs(job):
    return " ".join(entry.msg for entry in job.log)


def _failed(url="http://f.example/1"):
    return _Result(url, success=False, error="net::ERR_BLOCKED")


def test_fallback_rescues_a_failed_page(monkeypatch):
    final = "http://f.example/landed?a=<b>"
    resp = _Stream(url=final)
    docs, job, calls, raw_calls = _fallback_case(monkeypatch, _failed(), resp)
    [doc] = docs
    assert doc.url == final, "stored under the fetch's final URL, not raw:"
    assert doc.title == "Plain page" and doc.word_count >= 200
    meta = json.loads(doc.metadata_json)
    assert meta["fetched_via"] == "fallback"
    assert meta["requested_url"] == "http://f.example/1"
    assert [u.status for u in job.urls] == ["done"]
    assert job.crawl_done == 1
    assert "fallback fetched <code>http://f.example/landed?a=&lt;b&gt;</code>" in _msgs(job)
    [(method, url, kwargs)] = calls
    assert (method, url) == ("GET", "http://f.example/1")
    assert kwargs["follow_redirects"] is True
    assert kwargs["timeout"] == crawler.settings.crawl_timeout / 1000
    assert "Mozilla/5.0" in kwargs["headers"]["User-Agent"]
    assert kwargs["headers"]["Accept"] == "text/html,*/*"
    assert raw_calls == [_GOOD_HTML]
    assert resp.closed, "the stream is always closed"


def test_fallback_rescues_a_block_page(monkeypatch):
    blocked = _Result("http://f.example/1", title="Just a moment...", words=20)
    docs, job, _calls, _raw = _fallback_case(monkeypatch, blocked, _Stream())
    assert [d.url for d in docs] == ["http://f.example/1"]
    assert "requested_url" not in json.loads(docs[0].metadata_json)
    assert [u.status for u in job.urls] == ["done"]


def test_fallback_decodes_with_the_declared_charset(monkeypatch):
    html = _GOOD_HTML.replace("Plain page", "Café page")
    resp = _Stream(chunks=[html.encode("latin-1")], encoding="latin-1")
    docs, _job, _calls, raw_calls = _fallback_case(monkeypatch, _failed(), resp)
    assert docs[0].title == "Café page" and raw_calls == [html]


def test_fallback_http_error_keeps_the_page_failed(monkeypatch):
    resp = _Stream(status=403)
    docs, job, _calls, raw_calls = _fallback_case(monkeypatch, _failed(), resp)
    assert docs == [] and raw_calls == []
    assert not resp.body_read, "an error status is judged from the headers alone"
    [u] = job.urls
    assert u.status == "error" and "net::ERR_BLOCKED" in u.error
    assert "fallback failed: HTTP 403" in _msgs(job)
    assert job.crawl_done == 1


def test_fallback_rejects_non_html_without_reading_the_body(monkeypatch):
    resp = _Stream(ctype="application/pdf", text="%PDF-1.7")
    docs, job, _calls, raw_calls = _fallback_case(monkeypatch, _failed(), resp)
    assert docs == [] and raw_calls == []
    assert not resp.body_read and resp.pulled == 0
    assert "fallback failed: not HTML" in _msgs(job)


def _endless(chunk_size):
    while True:
        yield b"x" * chunk_size


def test_fallback_aborts_an_oversized_body_at_the_cap(monkeypatch):
    chunk = 256 * 1024
    resp = _Stream(chunks=_endless(chunk))
    docs, job, _calls, raw_calls = _fallback_case(monkeypatch, _failed(), resp)
    assert docs == [] and raw_calls == []
    assert "fallback failed: page too large" in _msgs(job)
    assert crawler._FALLBACK_MAX_BYTES < resp.pulled <= crawler._FALLBACK_MAX_BYTES + chunk, \
        "reading stops at the first chunk past the cap"
    assert resp.closed


def test_fallback_rejects_a_declared_oversize_without_reading(monkeypatch):
    resp = _Stream(length=crawler._FALLBACK_MAX_BYTES + 1)
    docs, job, _calls, _raw = _fallback_case(monkeypatch, _failed(), resp)
    assert docs == [] and not resp.body_read
    assert "fallback failed: page too large" in _msgs(job)


def test_fallback_gives_up_on_a_body_that_never_ends_in_time(monkeypatch):
    import time as _time

    def trickle():
        while True:
            _time.sleep(0.01)
            yield b"<p>slow</p>"

    monkeypatch.setattr(crawler.settings, "crawl_timeout", 100)   # 0.1 s budget
    resp = _Stream(chunks=trickle())
    t0 = _time.monotonic()
    docs, job, _calls, _raw = _fallback_case(monkeypatch, _failed(), resp)
    assert _time.monotonic() - t0 < 3
    assert docs == [] and "fallback failed: timed out" in _msgs(job)
    assert resp.closed


def test_fallback_rejects_a_block_page_body(monkeypatch):
    docs, job, _calls, raw_calls = _fallback_case(monkeypatch, _failed(),
                                                  _Stream(text=_BLOCK_HTML))
    assert docs == [] and raw_calls == [], "a block page is never converted"
    assert "fallback failed:" in _msgs(job) and "just a moment" in _msgs(job)
    assert job.urls[0].status == "error"


def test_fallback_body_is_judged_off_the_event_loop(monkeypatch):
    """Decoding and scanning a page-controlled body must not hold the event
    loop (the whole app's) — it happens in the fetch's worker thread."""
    seen = []
    real = crawler._fallback_page_problem

    def spy(html):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(html)

    monkeypatch.setattr(crawler, "_fallback_page_problem", spy)
    docs, _job, _calls, _raw = _fallback_case(monkeypatch, _failed(), _Stream())
    assert docs and seen == [False]


# Short ids: pytest puts the test id in an environment variable, and a 200 KB
# body would overflow it on Windows.
@pytest.mark.parametrize("body", [
    pytest.param("<" * 200_000, id="lt-run"),
    pytest.param("<script>" * 25_000, id="unclosed-scripts"),
    pytest.param("<style" * 40_000, id="unclosed-style-openers"),
    pytest.param("<title" * 40_000, id="title-openers"),
    pytest.param("<title>" + "a" * 200_000, id="unclosed-title"),
    pytest.param("<a" * 100_000, id="unclosed-tags"),
    pytest.param("<" + "a" * 200_000, id="one-endless-tag"),
    pytest.param("<script>x</script>" * 10_000 + "<style>", id="scripts-then-unclosed"),
])
def test_fallback_page_check_is_linear_on_hostile_bodies(body):
    import time as _time
    t0 = _time.monotonic()
    crawler._fallback_page_problem(body)
    assert _time.monotonic() - t0 < 0.5


def test_fallback_page_check_still_reads_title_and_text():
    assert crawler._fallback_page_problem(_GOOD_HTML) == ""
    assert "just a moment" in crawler._fallback_page_problem(_BLOCK_HTML)
    # Script and style text is not page text; an unclosed opener stops the
    # stripping rather than scanning on.
    scripted = ("<title>Real</title><script>" + "var x; " * 500 + "</script>"
                + "<p>" + "word " * 10 + "</p>")
    assert crawler._fallback_page_problem(scripted) == "only 11 words of content"


def test_fallback_conversion_still_goes_through_the_junk_gate(monkeypatch):
    class ThinCrawler(_fake_crawler({})):
        async def arun(self, url, config=None):
            if url.startswith("raw:"):
                return _Result(url, title="Thin", words=10)  # converts to almost nothing
            return _failed()

    monkeypatch.setattr(crawler.httpx, "stream", lambda m, u, **k: _Stream())
    monkeypatch.setattr(crawler, "AsyncWebCrawler", ThinCrawler)
    jid = _job(["http://f.example/1"])
    docs = asyncio.run(crawler.crawl_urls_with_progress(_srs(["http://f.example/1"]), "q", jid))
    assert docs == []
    assert "fallback failed: only 10 words of content" in _msgs(jobs.get_job(jid))
    assert jobs.get_job(jid).urls[0].status == "error"


def test_fallback_disabled_by_setting(monkeypatch):
    monkeypatch.setattr(crawler.settings, "crawl_fallback", False)
    docs, job, calls, _raw = _fallback_case(monkeypatch, _failed(), _Stream())
    assert docs == [] and calls == []
    assert "fallback" not in _msgs(job)
    assert job.urls[0].status == "error"


def test_fallback_never_fetches_non_http_urls(monkeypatch):
    docs, job, calls, _raw = _fallback_case(
        monkeypatch, _failed("file:///etc/passwd"), _Stream(), url="file:///etc/passwd")
    assert docs == [] and calls == []


def test_fallback_network_error_is_reported_not_raised(monkeypatch):
    docs, job, _calls, _raw = _fallback_case(
        monkeypatch, _failed(), httpx.ConnectError("connection refused"))
    assert docs == []
    assert "fallback failed: ConnectError" in _msgs(job)
    assert job.urls[0].status == "error" and job.crawl_done == 1


def test_fallback_skipped_after_a_one_shot_cancel(monkeypatch):
    async def fails_then_cancel():
        jobs.request_cancel(jid)
        return _failed()

    calls = []
    monkeypatch.setattr(crawler.httpx, "stream", lambda m, u, **k: calls.append(u) or _Stream())
    monkeypatch.setattr(crawler, "AsyncWebCrawler",
                        _fake_crawler({"http://f.example/1": fails_then_cancel}))
    jid = _job(["http://f.example/1"])
    docs = asyncio.run(crawler.crawl_urls_with_progress(
        _srs(["http://f.example/1"]), "q", jid, skip_on_cancel=True))
    assert docs == [] and calls == []
