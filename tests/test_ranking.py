"""ranking.py (WP5) against the litellm stub and a real temp SQLite.

The stub reranks by the share of the query's tokens a string contains
(ties by index) and embeds a text to a unit vector seeded from that text
alone, so a query embeds to exactly the vector stored for a chunk when the
stored vector was made from the same string. Every ordering asserted here
follows from those two rules; nothing monkeypatches the ranking itself
except the tests of the degrade rule, which make the provider fail."""
import asyncio

import litellm
import pytest

import brief
import embeddings
import ranking
import storage
from config import settings
from models import Document, SearchResult

RERANK = "cohere/rerank-v5.0-pro"
PRO = "cohere/embed-v5.0-pro"
FAST = "cohere/embed-v5.0-fast"
DIM = 8


class AuthenticationError(Exception):
    """Named like litellm's: not retried, so one failed attempt."""


@pytest.fixture(autouse=True)
def _models(monkeypatch):
    """Rerank on, embeddings off, unless a test says otherwise."""
    litellm.reset_calls()
    monkeypatch.setattr(settings, "rerank_model", RERANK)
    monkeypatch.setattr(settings, "embed_model", "")
    monkeypatch.setattr(settings, "embed_query_model", FAST)
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "llm_api_key", "k")
    yield
    litellm.reset_calls()


def _run(coro):
    return asyncio.run(coro)


def _doc(doc_id, body, *, words=500, title=None, query="Q", url=None):
    return Document(id=doc_id, url=url or f"http://x/{doc_id}", domain="x",
                    title=title if title is not None else f"title {doc_id}",
                    search_query=query, crawled_at="2026-01-01",
                    content_markdown=body, content_fit="", word_count=words)


def _sr(title, snippet=""):
    return SearchResult(url=f"http://s/{title}", title=title, snippet=snippet)


def _rank_lines(capsys):
    return [line for line in capsys.readouterr().err.splitlines() if "[RANK]" in line]


QUERY = "solar battery storage"
# A: every query term, but split across two heading-bounded chunks, so its
# best chunk holds two of three terms. B: all three in one chunk.
A_BODY = "# One\n\nsolar panels on the roof\n\n# Two\n\nbattery storage in the garage"
B_BODY = "solar battery storage in one paragraph"


# ---------- interface ----------

def test_interface_constants_and_dataclass():
    assert ranking.DENSE_TOP_K == 50 and ranking.RERANK_POOL == 30
    r = ranking.RankedDoc(doc=_doc("d", "x"), score=0.5, excerpt="x", chunk_id="d#0")
    assert (r.score, r.excerpt, r.chunk_id) == (0.5, "x", "d#0")
    assert ranking.rank_for_brief.__defaults__ == (brief.MAX_BRIEF_SOURCES,)


# ---------- rank_search_results ----------

def test_search_results_reordered_before_a_slice_is_taken():
    results = [_sr("dogs", "chase cars"), _sr("cats", "on mats"), _sr("mats", "rubber")]
    out = ranking.rank_search_results("cats mats", results, mission_id="m")
    assert [r.title for r in out] == ["cats", "mats", "dogs"]
    assert [r.title for r in out[:2]] == ["cats", "mats"] != [r.title for r in results[:2]]
    [kw] = litellm.rerank_calls
    assert kw["documents"] == ["dogs\nchase cars", "cats\non mats", "mats\nrubber"]
    assert kw["query"] == "cats mats"
    assert out is not results and all(isinstance(r, SearchResult) for r in out)


def test_search_results_unchanged_when_rerank_disabled_or_failing(monkeypatch, capsys):
    results = [_sr("dogs"), _sr("cats mats"), _sr("mats")]
    monkeypatch.setattr(settings, "rerank_model", "")
    capsys.readouterr()
    assert ranking.rank_search_results("cats mats", results) == results
    assert litellm.rerank_calls == []
    assert len(capsys.readouterr().err.strip().splitlines()) == 1   # try_rerank's own line

    monkeypatch.setattr(settings, "rerank_model", RERANK)

    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", boom)
    capsys.readouterr()
    assert ranking.rank_search_results("cats mats", results) == results
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_search_results_never_raise_and_keep_unranked_entries(monkeypatch, capsys):
    results = [_sr("a"), _sr("b"), _sr("c")]
    # A provider that returns only part of the list, plus a nonsense index.
    monkeypatch.setattr(embeddings, "try_rerank", lambda q, d, **kw: [(2, 1.0), (99, 0.9)])
    assert [r.title for r in ranking.rank_search_results("q", results)] == ["c", "a", "b"]

    def explode(q, d, **kw):
        raise ValueError("bug")

    monkeypatch.setattr(embeddings, "try_rerank", explode)
    capsys.readouterr()
    assert ranking.rank_search_results("q", results) == results
    assert len(_rank_lines(capsys)) == 1

    assert ranking.rank_search_results("q", []) == []
    assert ranking.rank_search_results("q", [results[0]]) == [results[0]]


# ---------- rank_documents ----------

def test_rank_documents_best_chunk_excerpt_usable_first():
    junk, a, b = _doc("J", "tiny", words=5), _doc("A", A_BODY), _doc("B", B_BODY)
    out = ranking.rank_documents(QUERY, [junk, a, b], top_n=3, excerpt_chars=20, mission_id="m")
    assert [r.doc.id for r in out] == ["B", "A", "J"]
    best_b, best_a, j = out
    assert best_b.score == 1.0 and best_b.chunk_id == "B#0"
    assert best_b.excerpt == B_BODY[:20]
    assert best_a.score == pytest.approx(2 / 3) and best_a.chunk_id == "A#1"
    assert best_a.excerpt == "Two\n\nbattery storage in the garage"[:20]
    assert (j.score, j.chunk_id, j.excerpt) == (0.0, None, "tiny")
    # Only usable documents were sent, as indexed_text(title, chunk.text).
    [kw] = litellm.rerank_calls
    assert kw["documents"] == [
        "title A\n\nOne\n\nsolar panels on the roof",
        "title B\n\nsolar battery storage in one paragraph",
        "title A\n\nTwo\n\nbattery storage in the garage",
    ]


def test_rank_documents_appends_unusable_only_while_slots_remain():
    junk, a, b = _doc("J", "tiny", words=5), _doc("A", A_BODY), _doc("B", B_BODY)
    assert [r.doc.id for r in ranking.rank_documents(QUERY, [junk, a, b], top_n=2,
                                                      excerpt_chars=50)] == ["B", "A"]
    assert [r.doc.id for r in ranking.rank_documents(QUERY, [junk, a, b], top_n=None,
                                                      excerpt_chars=50)] == ["B", "A", "J"]
    # Two junk pages keep their input order behind the ranked ones.
    junk2 = _doc("K", "also tiny", words=3)
    out = ranking.rank_documents(QUERY, [junk2, junk, a], top_n=5, excerpt_chars=50)
    assert [r.doc.id for r in out] == ["A", "K", "J"]


def test_rank_documents_caps_chunks_round_robin():
    a = _doc("A", "\n\n".join(f"# h{i}\n\npara {i}" for i in range(250)))
    b = _doc("B", "# x\n\nfirst\n\n# y\n\nsecond\n\n# z\n\nthird")
    out = ranking.rank_documents("third", [a, b], top_n=2, excerpt_chars=10)
    sent = litellm.last_rerank_kwargs["documents"]
    assert len(sent) == ranking.MAX_RERANK_CHUNKS == 200
    assert sent[:4] == ["title A\n\nh0\n\npara 0", "title B\n\nx\n\nfirst",
                        "title A\n\nh1\n\npara 1", "title B\n\ny\n\nsecond"]
    assert "title B\n\nz\n\nthird" in sent, "every doc represented before any gets a 4th slot"
    assert [r.doc.id for r in out] == ["B", "A"] and out[0].chunk_id == "B#2"


def test_rank_documents_none_on_disabled_failure_or_nothing_usable(monkeypatch, capsys):
    docs = [_doc("A", A_BODY), _doc("B", B_BODY)]
    monkeypatch.setattr(settings, "rerank_model", "")
    assert ranking.rank_documents(QUERY, docs, top_n=2, excerpt_chars=50) is None
    assert litellm.rerank_calls == []

    monkeypatch.setattr(settings, "rerank_model", RERANK)

    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", boom)
    capsys.readouterr()
    assert ranking.rank_documents(QUERY, docs, top_n=2, excerpt_chars=50) is None
    assert len(capsys.readouterr().err.strip().splitlines()) == 1

    monkeypatch.setattr(litellm, "rerank", lambda **kw: (_ for _ in ()).throw(AssertionError))
    assert ranking.rank_documents(QUERY, [_doc("J", "tiny", words=5)],
                                  top_n=2, excerpt_chars=50) is None, "no usable doc: no call"
    assert ranking.rank_documents(QUERY, [], top_n=2, excerpt_chars=50) is None

    def explode(q, d, **kw):
        raise ValueError("bug")

    monkeypatch.setattr(embeddings, "try_rerank", explode)
    capsys.readouterr()
    assert ranking.rank_documents(QUERY, docs, top_n=2, excerpt_chars=50) is None
    assert len(_rank_lines(capsys)) == 1


# ---------- rank_for_brief ----------

class _M:
    def __init__(self, question="Q?", mission_id="m"):
        self.question = question
        self.id = mission_id


def test_brief_order_equals_ordered_sources_when_disabled_or_failing(monkeypatch):
    docs = [_doc("J", "tiny", words=5), _doc("A", A_BODY), _doc("B", B_BODY)]
    monkeypatch.setattr(settings, "rerank_model", "")
    assert ranking.rank_for_brief(_M(QUERY), docs) == brief.ordered_sources(docs)
    assert ranking.rank_for_brief(_M(QUERY), docs, max_docs=2) == brief.ordered_sources(docs, 2)
    assert litellm.rerank_calls == []

    monkeypatch.setattr(settings, "rerank_model", RERANK)

    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", boom)
    assert ranking.rank_for_brief(_M(QUERY), docs) == brief.ordered_sources(docs)

    def explode(*a, **kw):
        raise ValueError("bug")

    monkeypatch.setattr(ranking, "rank_documents", explode)
    assert ranking.rank_for_brief(_M(QUERY), docs) == brief.ordered_sources(docs)
    assert ranking.rank_for_brief(_M(QUERY), []) == []


def test_brief_order_reranked_usable_first_then_unusable_truncated():
    junk, a, b = _doc("J", "tiny", words=5), _doc("A", A_BODY), _doc("B", B_BODY)
    out = ranking.rank_for_brief(_M(QUERY, "m7"), [junk, a, b])
    assert [d.id for d in out] == ["B", "A", "J"]
    assert all(isinstance(d, Document) for d in out)
    assert [d.id for d in ranking.rank_for_brief(_M(QUERY), [junk, a, b], max_docs=2)] == ["B", "A"]
    assert [d.id for d in ranking.rank_for_brief(_M(QUERY), [junk, a, b], max_docs=1)] == ["B"]
    assert litellm.rerank_calls[0]["query"] == QUERY
    # The brief order differs from today's only in the usable part's order.
    assert sorted(d.id for d in out) == sorted(d.id for d in brief.ordered_sources([junk, a, b]))


# ---------- hybrid_search ----------

def _store(*docs):
    _run(storage.init_db())
    for d in docs:
        _run(storage.upsert_document(d))


def _ids(docs):
    return [d.id for d in docs]


# FTS5 bm25 prefers the shorter document at equal term frequency, so the
# short page with the terms scattered (A) outranks the long page with them
# together (B); the reranker's best-chunk view says the opposite.
A_LONGISH = A_BODY
B_LONG = B_BODY + "\n\n" + " ".join(f"filler{i}" for i in range(120))


def test_hybrid_falls_back_to_fts_without_embeddings(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "")
    a, b, x = _doc("A", A_LONGISH), _doc("B", B_LONG), _doc("X", "giraffe facts")
    _store(a, b, x)
    expected = _ids(_run(storage.search_documents_fts(QUERY, None)))
    assert expected == ["A", "B"]

    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (expected, "fts")
    assert litellm.embedding_calls == [] and litellm.rerank_calls == []

    # Embeddings on but nothing embedded yet: still FTS, and no query embed.
    monkeypatch.setattr(settings, "embed_model", PRO)
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (expected, "fts")
    assert litellm.embedding_calls == []

    docs, mode = _run(ranking.hybrid_search(QUERY, "Q", limit=1))
    assert (_ids(docs), mode) == (["A"], "fts")
    assert _run(ranking.hybrid_search("nomatch", None)) == ([], "fts")


def _query_vector(text):
    """The vector embed_query will produce for `text` (stub: seeded from the
    text), stored for a chunk so that chunk is a perfect dense hit."""
    return embeddings.vector_to_blob(embeddings.embed_query(text))


def test_rrf_surfaces_a_dense_only_hit(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "")
    monkeypatch.setattr(settings, "embed_model", PRO)
    a, b, x = _doc("A", "zebra crossing rules"), _doc("B", "zebra crossing rules " + "w " * 100), \
        _doc("X", "giraffe facts", query="other")
    _store(a, b, x)
    assert _ids(_run(storage.search_documents_fts("zebra", None))) == ["A", "B"]
    _run(storage.upsert_chunk_embeddings([("X#0", PRO, DIM, _query_vector("zebra"))]))
    litellm.reset_calls()

    docs, mode = _run(ranking.hybrid_search("zebra", None))
    # A: FTS rank 1; X: dense rank 1 (same RRF score, FTS wins the tie); B: FTS rank 2.
    assert (_ids(docs), mode) == (["A", "X", "B"], "hybrid")
    assert docs[1].content_markdown == "giraffe facts", "dense-only hit loaded with its body"
    [kw] = litellm.embedding_calls
    assert kw["model"] == FAST and kw["input_type"] == "search_query" and kw["dimensions"] == DIM

    # The search filter bounds the dense arm too: X is under another query.
    _run(storage.upsert_chunk_embeddings([("B#0", PRO, DIM, _query_vector("zebra"))]))
    docs, mode = _run(ranking.hybrid_search("zebra", "Q"))
    assert mode == "hybrid" and "X" not in _ids(docs)
    assert _ids(docs) == ["B", "A"], "B is in both lists now and overtakes A"
    assert _run(ranking.hybrid_search("zebra", "nothing-here")) == ([], "fts")


def test_dense_only_hit_whose_document_vanished_is_dropped(monkeypatch):
    monkeypatch.setattr(settings, "rerank_model", "")
    monkeypatch.setattr(settings, "embed_model", PRO)
    a, x = _doc("A", "zebra crossing rules"), _doc("X", "giraffe facts")
    _store(a, x)
    _run(storage.upsert_chunk_embeddings([("X#0", PRO, DIM, _query_vector("zebra"))]))
    monkeypatch.setattr(storage, "get_documents_by_ids", _no_docs)
    docs, mode = _run(ranking.hybrid_search("zebra", None))
    assert (_ids(docs), mode) == (["A"], "hybrid")


async def _no_docs(ids):
    return []


def test_dense_stage_failure_keeps_fts_with_one_line(monkeypatch, capsys):
    monkeypatch.setattr(settings, "rerank_model", "")
    monkeypatch.setattr(settings, "embed_model", PRO)
    a, x = _doc("A", "zebra crossing rules"), _doc("X", "giraffe facts")
    _store(a, x)
    _run(storage.upsert_chunk_embeddings([("X#0", PRO, DIM, _query_vector("zebra"))]))
    stub_embedding = litellm.embedding

    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "embedding", boom)
    capsys.readouterr()
    docs, mode = _run(ranking.hybrid_search("zebra", None))
    assert (_ids(docs), mode) == (["A"], "fts")
    lines = _rank_lines(capsys)
    assert len(lines) == 1 and "dense" in lines[0] and "AuthenticationError" in lines[0]

    # A stored vector of the wrong width is ignored, not fatal.
    monkeypatch.setattr(litellm, "embedding", stub_embedding)
    _run(storage.upsert_chunk_embeddings([("A#0", PRO, 4, b"\0" * 16)]))
    docs, mode = _run(ranking.hybrid_search("zebra", None))
    assert (_ids(docs), mode) == (["A", "X"], "hybrid")


def test_rerank_applied_to_the_pool(monkeypatch):
    a, b = _doc("A", A_LONGISH), _doc("B", B_LONG)
    _store(a, b)
    monkeypatch.setattr(settings, "rerank_model", "")
    assert _ids(_run(ranking.hybrid_search(QUERY, None))[0]) == ["A", "B"]

    monkeypatch.setattr(settings, "rerank_model", RERANK)
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (["B", "A"], "fts+rerank")
    [kw] = litellm.rerank_calls
    assert kw["query"] == QUERY
    # Round-robin: A's first chunk, then B's.
    assert kw["documents"][1].startswith("title B\n\nsolar battery storage in one paragraph")
    assert all(isinstance(d, Document) and d.content_markdown for d in docs)

    # With a dense arm as well the mode says so.
    monkeypatch.setattr(settings, "embed_model", PRO)
    _run(storage.upsert_chunk_embeddings([("A#0", PRO, DIM, _query_vector(QUERY))]))
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (["B", "A"], "hybrid+rerank")


def test_rerank_pool_is_bounded_and_the_rest_keep_their_order(monkeypatch):
    monkeypatch.setattr(ranking, "RERANK_POOL", 2)
    a, b = _doc("A", A_LONGISH), _doc("B", B_LONG)
    c = _doc("C", B_BODY + "\n\n" + " ".join(f"more{i}" for i in range(200)))
    _store(a, b, c)
    monkeypatch.setattr(settings, "rerank_model", "")
    assert _ids(_run(ranking.hybrid_search(QUERY, None))[0]) == ["A", "B", "C"]

    monkeypatch.setattr(settings, "rerank_model", RERANK)
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (["B", "A", "C"], "fts+rerank")
    [kw] = litellm.rerank_calls
    assert not any("title C" in d for d in kw["documents"]), "C was outside the pool"


def test_rerank_stage_failure_keeps_the_previous_stage(monkeypatch, capsys):
    a, b = _doc("A", A_LONGISH), _doc("B", B_LONG)
    _store(a, b)

    def boom(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", boom)
    capsys.readouterr()
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (["A", "B"], "fts")
    assert len(capsys.readouterr().err.strip().splitlines()) == 1  # try_rerank's line

    def explode(*a, **kw):
        raise ValueError("bug")

    monkeypatch.setattr(ranking, "rank_documents", explode)
    capsys.readouterr()
    docs, mode = _run(ranking.hybrid_search(QUERY, None))
    assert (_ids(docs), mode) == (["A", "B"], "fts")
    assert len(_rank_lines(capsys)) == 1


def test_hybrid_never_raises_even_when_fts_does(monkeypatch, capsys):
    async def boom(query, search_filter=None):
        raise RuntimeError("db gone")

    monkeypatch.setattr(storage, "search_documents_fts", boom)
    capsys.readouterr()
    assert _run(ranking.hybrid_search(QUERY, None)) == ([], "fts")
    assert len(_rank_lines(capsys)) == 1
