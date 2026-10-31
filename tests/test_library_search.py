"""The Library's full-text search goes through ranking.hybrid_search (FTS
fused with the dense index and reranked where those are on) and says how the
matches were ranked beside the match count; without embeddings it is plain
FTS, labelled "full-text". The toolbar offers an export of the search or
mission collection in view."""
import asyncio
import re

import pytest

import config
import ranking
import storage
from models import Agent, Document, Mission


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def client():
    import app as app_mod
    app_mod.initialize()
    return app_mod.app.test_client()


def _seed_docs(n, query="alpha", start=0):
    async def go():
        for i in range(start, start + n):
            await storage.upsert_document(Document(
                id=f"d{i:04d}", url=f"https://site{i % 7}.example/p{i}",
                domain=f"site{i % 7}.example", title=f"Doc {i}", search_query=query,
                crawled_at=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                content_markdown="An ordinary sentence about the topic that reads like prose. " * 3,
                word_count=100))
    _run(go())


def _cards(html):
    return html.count('class="doc-card"')


def _page_sub(html):
    m = re.search(r'<p class="page-sub">(.*?)</p>', html, re.S)
    assert m, "no page-sub line"
    return m.group(1)


def _mode_label(html):
    m = re.search(r'<span class="mono" id="libSearchMode"[^>]*>(.*?)</span>', html, re.S)
    return m.group(1).strip() if m else None


def _patch_hybrid(monkeypatch, mode, pick=None):
    """Replace ranking.hybrid_search with one that records its arguments and
    answers every stored document (or `pick` of them) under `mode`."""
    calls = []

    async def fake(query, search_filter, *, limit=100):
        calls.append((query, search_filter, limit))
        docs = await storage.get_all_documents()
        if pick is not None:
            # In `pick`'s order: the page must keep the ranking it was given.
            by_id = {d.id: d for d in docs}
            docs = [by_id[i] for i in pick if i in by_id]
        return docs, mode

    monkeypatch.setattr(ranking, "hybrid_search", fake)
    return calls


def test_full_text_query_uses_hybrid_search_and_shows_the_rerank_mode(client, monkeypatch):
    _seed_docs(3)
    monkeypatch.setattr(config.settings, "rerank_model", "cohere/rerank-v5.0-pro")
    calls = _patch_hybrid(monkeypatch, "hybrid+rerank", pick=["d0002", "d0000"])

    html = client.get("/documents?q=ordinary&search=alpha").get_data(as_text=True)
    assert calls == [("ordinary", "alpha", 100)]
    # The cards are what hybrid_search returned, in its order.
    assert _cards(html) == 2
    assert re.findall(r'href="/document/(d\d{4})" class="doc-card"', html) == ["d0002", "d0000"]
    sub = _page_sub(html)
    assert "<strong>2</strong> matches" in sub
    assert _mode_label(html) == "ranked: hybrid + rerank · cohere/rerank-v5.0-pro"
    # The one search input still carries the query for refining.
    assert 'const FTS_QUERY = "ordinary";' in html


def test_hybrid_without_rerank_is_labelled_hybrid(client, monkeypatch):
    _seed_docs(2)
    _patch_hybrid(monkeypatch, "hybrid")
    html = client.get("/documents?q=ordinary").get_data(as_text=True)
    assert _cards(html) == 2
    assert _mode_label(html) == "hybrid"
    assert "rerank" not in _page_sub(html)


def test_full_text_query_keeps_its_200_char_bound(client, monkeypatch):
    _seed_docs(1)
    calls = _patch_hybrid(monkeypatch, "fts")
    assert client.get("/documents?q=" + "z" * 300).status_code == 200
    assert calls and len(calls[0][0]) == 200
    # No search filter: hybrid_search gets None, not "".
    assert calls[0][1] is None


def test_without_embeddings_the_search_is_plain_full_text(client, monkeypatch):
    # hybrid_search runs for real: no embed model, no reranker -> FTS only,
    # and the page says so.
    _seed_docs(3)
    monkeypatch.setattr(config.settings, "embed_model", "")
    monkeypatch.setattr(config.settings, "embed_query_model", "")
    monkeypatch.setattr(config.settings, "rerank_model", "")

    html = client.get("/documents?q=ordinary").get_data(as_text=True)
    assert _cards(html) == 3
    assert "<strong>3</strong> matches" in _page_sub(html)
    assert _mode_label(html) == "full-text"
    assert "hybrid" not in _page_sub(html) and "rerank" not in _page_sub(html)

    html = client.get("/documents?q=zzzznomatch").get_data(as_text=True)
    assert _cards(html) == 0
    assert "No pages match" in html


def test_mode_label_is_shown_only_for_full_text_matches(client, monkeypatch):
    _seed_docs(2)
    calls = _patch_hybrid(monkeypatch, "hybrid")
    assert _mode_label(client.get("/documents").get_data(as_text=True)) is None
    assert _mode_label(client.get("/documents?search=alpha").get_data(as_text=True)) is None
    # Inside a mission `q` is a client-side filter, never a search.
    _run(storage.insert_agent(Agent(id="a1", name="Ada", expertise="x",
                                    persona_prompt="p", created_at="t")))
    _run(storage.insert_mission(Mission(
        id="4c1f1c2e-8f3e-4c47-9a2e-0f3f3c1d9b10", agent_id="a1", question="Q?",
        status="done", created_at="t")))
    html = client.get("/documents?mission=4c1f1c2e-8f3e-4c47-9a2e-0f3f3c1d9b10&q=ordinary"
                      ).get_data(as_text=True)
    assert _mode_label(html) is None
    assert calls == []


def _export_form(html):
    m = re.search(r'<form id="libExport" method="post" action="/documents/export"[^>]*>(.*?)</form>',
                  html, re.S)
    return m.group(1) if m else None


def test_toolbar_offers_an_export_of_the_search_or_mission_in_view(client):
    _seed_docs(2, query="alpha")
    _run(storage.insert_agent(Agent(id="a1", name="Ada", expertise="x",
                                    persona_prompt="p", created_at="t")))
    mid = "4c1f1c2e-8f3e-4c47-9a2e-0f3f3c1d9b10"
    _run(storage.insert_mission(Mission(id=mid, agent_id="a1", question="Q?",
                                        status="done", created_at="t")))
    _run(storage.link_mission_document(mid, "r", "d0000"))

    # Unfiltered: nothing to export as one collection.
    html = client.get("/documents").get_data(as_text=True)
    assert _export_form(html) is None and "Export to corpus" not in html

    html = client.get("/documents?search=alpha").get_data(as_text=True)
    form = _export_form(html)
    assert form is not None and 'name="search" value="alpha"' in form
    assert 'name="mission"' not in form
    assert re.search(r'<button type="submit" form="libExport"[^>]*>\s*(<svg.*?</svg>\s*)?Export to corpus',
                     html, re.S)

    html = client.get(f"/documents?mission={mid}").get_data(as_text=True)
    form = _export_form(html)
    assert form is not None and f'name="mission" value="{mid}"' in form
    assert 'name="search"' not in form


def test_export_form_escapes_the_search_query(client):
    _seed_docs(1, query='al"pha<b>')
    html = client.get('/documents?search=al%22pha%3Cb%3E').get_data(as_text=True)
    form = _export_form(html)
    assert form is not None
    assert 'value="al&#34;pha&lt;b&gt;"' in form
    assert "<b>" not in form
