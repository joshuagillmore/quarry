"""Storage for chunks, dense embeddings and corpus-export state: the three
WP3 tables, the upsert_document chunk hook, and the query helpers the
embedder, ranker and exporter use. Real temp SQLite per test (conftest).

Chunk boundaries are chunking.py's business (tests/test_chunking.py); here a
one-paragraph-per-chunk double stands in for `storage.chunking` wherever a
test needs to know exactly which chunks a body becomes."""
import asyncio
import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest

import storage
from models import Agent, Document, DocumentExport, LlmCall, Mission, Requirement


def _sql(query, args=()):
    db = sqlite3.connect(storage.DB_PATH)
    try:
        return db.execute(query, args).fetchall()
    finally:
        db.close()


def _columns(table):
    return {row[1] for row in _sql(f"PRAGMA table_info({table})")}


def _index_column_sets(table):
    out = set()
    for row in _sql(f"PRAGMA index_list({table})"):
        out.add(tuple(r[2] for r in _sql(f"PRAGMA index_info('{row[1]}')")))
    return out


def _doc(doc_id, body, url="http://x/1", query="Q", title="t", fit=None, crawled_at="t"):
    return Document(id=doc_id, url=url, domain="x", title=title, search_query=query,
                    crawled_at=crawled_at, content_markdown=body, content_fit=fit)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blob(*values):
    return struct.pack(f"<{len(values)}f", *values)


def _unblob(blob):
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


@dataclass(frozen=True)
class _Chunk:
    idx: int
    section: Optional[str]
    text: str


def _paragraph_chunks(doc):
    body = doc.content_fit or doc.content_markdown or ""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    return [_Chunk(idx=i, section=None, text=p) for i, p in enumerate(paras)]


@pytest.fixture
def paragraph_chunking(monkeypatch):
    """One chunk per blank-line paragraph, so a test can say which chunks a
    body has and which of them a re-crawl changes."""
    double = SimpleNamespace(document_chunks=_paragraph_chunks, chunk_sha256=_sha)
    monkeypatch.setattr(storage, "chunking", double)
    return double


def _run(coro):
    return asyncio.run(coro)


# --- DDL ---------------------------------------------------------------------------

def test_init_db_creates_the_chunk_tables_and_is_idempotent():
    _run(storage.init_db())
    _run(storage.init_db())
    assert _columns("chunks") == {"chunk_id", "document_id", "idx", "section", "text",
                                  "char_len", "content_sha256"}
    assert _columns("chunk_embeddings") == {"chunk_id", "model", "dim", "vector", "created_at"}
    assert _columns("document_exports") == {"document_id", "collection", "content_sha256",
                                            "embed_model", "n_chunks", "exported_at"}
    assert ("document_id",) in _index_column_sets("chunks")
    assert ("document_id", "idx") in _index_column_sets("chunks")       # UNIQUE
    assert ("chunk_id", "model") in _index_column_sets("chunk_embeddings")
    assert ("document_id", "collection") in _index_column_sets("document_exports")


def test_init_db_adds_the_tables_to_a_db_that_predates_them():
    """An existing library (documents + FTS already consistent) gains the
    tables without touching its rows or triggering the FTS rebuild."""
    db = sqlite3.connect(storage.DB_PATH)
    try:
        db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, url TEXT NOT NULL, "
                   "domain TEXT NOT NULL, title TEXT, search_query TEXT, "
                   "crawled_at TEXT NOT NULL, content_markdown TEXT, content_fit TEXT, "
                   "word_count INTEGER DEFAULT 0, links_internal INTEGER DEFAULT 0, "
                   "links_external INTEGER DEFAULT 0, metadata_json TEXT, "
                   "UNIQUE(url, search_query))")
        db.execute("INSERT INTO documents (id, url, domain, search_query, crawled_at, "
                   "content_markdown) VALUES ('old', 'http://x/old', 'x', 'Q', 't', 'old body')")
        db.commit()
    finally:
        db.close()
    _run(storage.init_db())
    assert {"chunks", "chunk_embeddings", "document_exports"} <= {
        r[0] for r in _sql("SELECT name FROM sqlite_master WHERE type = 'table'")}
    # A document stored before chunking existed simply has no chunks yet.
    assert _run(storage.get_document_chunks("old")) == []
    assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
    assert [d.id for d in _run(storage.search_documents_fts("old"))] == ["old"]


# --- upsert_document hook ------------------------------------------------------------

def test_upsert_document_stores_the_documents_chunks(paragraph_chunking):
    _run(storage.init_db())
    doc_id = _run(storage.upsert_document(_doc("d1", "first para\n\nsecond para")))
    rows = _run(storage.get_document_chunks(doc_id))
    assert [(r.chunk_id, r.document_id, r.idx, r.section, r.text, r.char_len) for r in rows] == [
        ("d1#0", "d1", 0, None, "first para", 10),
        ("d1#1", "d1", 1, None, "second para", 11),
    ]
    assert rows[0].content_sha256 == _sha("first para")
    assert rows[1].content_sha256 == _sha("second para")


def test_upsert_document_uses_the_real_chunker_by_default():
    """Without the double: the chunks are whatever chunking.document_chunks
    says, keyed by that module's own sha, so WP2's boundary rules and
    storage never drift apart."""
    import chunking
    _run(storage.init_db())
    body = "# Intro\n\nA paragraph of text.\n\n# Next\n\nAnother paragraph."
    doc_id = _run(storage.upsert_document(_doc("d1", body)))
    expected = chunking.document_chunks(_doc("d1", body))
    rows = _run(storage.get_document_chunks(doc_id))
    assert [(r.idx, r.section, r.text) for r in rows] == [
        (c.idx, c.section, c.text) for c in expected]
    assert len(rows) >= 1
    assert all(r.content_sha256 == chunking.chunk_sha256(r.text) for r in rows)
    assert all(r.chunk_id == f"d1#{r.idx}" for r in rows)


def test_re_upsert_rebuilds_chunks_under_the_same_document_id(paragraph_chunking):
    """A re-crawl of the same (url, search_query) keeps the document id, so
    its chunks are replaced under that id: no chunks for the discarded
    doc.id, no leftovers from the earlier body."""
    _run(storage.init_db())
    first = _run(storage.upsert_document(_doc("d1", "a\n\nb\n\nc")))
    again = _run(storage.upsert_document(_doc("dOTHER", "a\n\nz")))
    assert first == again == "d1"
    rows = _run(storage.get_document_chunks("d1"))
    assert [(r.chunk_id, r.text) for r in rows] == [("d1#0", "a"), ("d1#1", "z")]
    assert _run(storage.get_document_chunks("dOTHER")) == []
    assert _sql("SELECT COUNT(*) FROM chunks")[0][0] == 2


def test_empty_body_stores_no_chunks_and_clears_old_ones(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "a\n\nb")))
    assert len(_run(storage.get_document_chunks("d1"))) == 2
    _run(storage.upsert_document(_doc("d1", "")))
    assert _run(storage.get_document_chunks("d1")) == []


def test_upsert_keeps_fts_consistent_alongside_chunks(paragraph_chunking):
    """The chunk hook sits inside upsert_document's transaction; the FTS
    delete+reinsert it already did must be unchanged by it."""
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "hello zebrafish\n\nworld")))
    _run(storage.upsert_document(_doc("d2", "completely new\n\ncontent")))
    assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
    assert _run(storage.search_documents_fts("zebrafish")) == []
    assert [d.id for d in _run(storage.search_documents_fts("completely"))] == ["d1"]
    # init_db's consistency check sees documents == fts rows and leaves both.
    _run(storage.init_db())
    assert _sql("SELECT COUNT(*) FROM documents_fts")[0][0] == 1
    assert len(_run(storage.get_document_chunks("d1"))) == 2


# --- embeddings across a re-crawl ----------------------------------------------------

def test_recrawl_keeps_unchanged_embeddings_and_drops_changed_or_gone(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "same\n\nold\n\ngone")))
    _run(storage.upsert_chunk_embeddings([
        ("d1#0", "m", 2, _blob(1.0, 0.0)),
        ("d1#1", "m", 2, _blob(0.0, 1.0)),
        ("d1#2", "m", 2, _blob(0.5, 0.5)),
        ("d1#0", "other-model", 2, _blob(9.0, 9.0)),
    ]))
    # chunk 0 unchanged, chunk 1's text changed, chunk 2 no longer exists.
    _run(storage.upsert_document(_doc("d1", "same\n\nnew")))

    rows = _run(storage.get_document_chunks("d1"))
    assert [(r.chunk_id, r.text) for r in rows] == [("d1#0", "same"), ("d1#1", "new")]
    kept = _sql("SELECT chunk_id, model FROM chunk_embeddings ORDER BY chunk_id, model")
    assert kept == [("d1#0", "m"), ("d1#0", "other-model")]
    assert _run(storage.get_chunk_embeddings("m")) == [("d1#0", "d1", _blob(1.0, 0.0))]
    missing = _run(storage.get_chunks_missing_embeddings("m"))
    assert [c.chunk_id for c in missing] == ["d1#1"]


def test_recrawl_drops_the_embedding_of_a_chunk_whose_text_moved(paragraph_chunking):
    """Same text at a different idx is a different chunk_id: the old slot's
    vector described other text and must not be reused for it."""
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "alpha\n\nbeta")))
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 1, _blob(1.0)),
                                          ("d1#1", "m", 1, _blob(2.0))]))
    _run(storage.upsert_document(_doc("d1", "beta\n\nalpha")))
    assert _sql("SELECT COUNT(*) FROM chunk_embeddings")[0][0] == 0
    assert [c.chunk_id for c in _run(storage.get_chunks_missing_embeddings("m"))] == [
        "d1#0", "d1#1"]


def test_replace_document_chunks_is_the_same_core_as_the_upsert_hook(paragraph_chunking):
    """Rebuilding chunks for a document stored before chunking existed goes
    through the same keep/drop rule, in one transaction."""
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "p1\n\np2")))
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 1, _blob(1.0)),
                                          ("d1#1", "m", 1, _blob(2.0))]))
    _run(storage.replace_document_chunks(
        "d1", [_Chunk(0, "Sec", "p1"), _Chunk(1, None, "p2 changed")]))
    rows = _run(storage.get_document_chunks("d1"))
    assert [(r.chunk_id, r.section, r.text) for r in rows] == [
        ("d1#0", "Sec", "p1"), ("d1#1", None, "p2 changed")]
    assert _sql("SELECT chunk_id FROM chunk_embeddings") == [("d1#0",)]


# --- embeddings: roundtrip and filters -----------------------------------------------

def test_float32_vectors_roundtrip_as_little_endian_bytes(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "one")))
    values = [0.25, -1.5, 3.0e-3, 1024.0]
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 4, _blob(*values))]))
    (chunk_id, doc_id, blob), = _run(storage.get_chunk_embeddings("m"))
    assert (chunk_id, doc_id) == ("d1#0", "d1")
    assert isinstance(blob, bytes) and len(blob) == 16
    assert _unblob(blob) == pytest.approx(values)
    assert blob[:4] == struct.pack("<f", 0.25)   # little-endian, not native/big
    assert _sql("SELECT dim FROM chunk_embeddings") == [(4,)]


def test_upsert_chunk_embeddings_replaces_in_place_and_checks_the_blob_size(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "one")))
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 2, _blob(1.0, 2.0))]))
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 2, _blob(3.0, 4.0))]))
    assert _run(storage.get_chunk_embeddings("m")) == [("d1#0", "d1", _blob(3.0, 4.0))]
    with pytest.raises(ValueError):
        _run(storage.upsert_chunk_embeddings([("d1#0", "m", 3, _blob(1.0, 2.0))]))
    with pytest.raises(ValueError):
        _run(storage.upsert_chunk_embeddings([("d1#0", "m", 0, b"")]))
    assert _run(storage.upsert_chunk_embeddings([])) is None


def test_get_chunks_missing_embeddings_filters_by_model_documents_and_limit(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "a\n\nb", url="http://x/1")))
    _run(storage.upsert_document(_doc("d2", "c\n\nd", url="http://x/2")))
    _run(storage.upsert_document(_doc("d3", "e", url="http://x/3")))
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 1, _blob(1.0)),
                                          ("d2#1", "m", 1, _blob(1.0)),
                                          ("d3#0", "other", 1, _blob(1.0))]))
    ids = lambda rows: [c.chunk_id for c in rows]  # noqa: E731
    assert ids(_run(storage.get_chunks_missing_embeddings("m"))) == ["d1#1", "d2#0", "d3#0"]
    assert ids(_run(storage.get_chunks_missing_embeddings("other"))) == [
        "d1#0", "d1#1", "d2#0", "d2#1"]
    assert ids(_run(storage.get_chunks_missing_embeddings("m", ["d3", "d1", "d1"]))) == [
        "d1#1", "d3#0"]
    assert ids(_run(storage.get_chunks_missing_embeddings("m", limit=2))) == ["d1#1", "d2#0"]
    assert ids(_run(storage.get_chunks_missing_embeddings("m", ["d1", "d3"], limit=1))) == ["d1#1"]
    assert _run(storage.get_chunks_missing_embeddings("m", [])) == []
    assert _run(storage.get_chunks_missing_embeddings("m", limit=0)) == []
    row = _run(storage.get_chunks_missing_embeddings("m", ["d1"]))[0]
    assert (row.document_id, row.idx, row.text, row.char_len) == ("d1", 1, "b", 1)
    assert row.content_sha256 == _sha("b")


def test_get_chunk_embeddings_by_documents_in_chunk_order(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "a\n\nb", url="http://x/1")))
    _run(storage.upsert_document(_doc("d2", "c", url="http://x/2")))
    _run(storage.upsert_chunk_embeddings([("d1#1", "m", 1, _blob(2.0)),
                                          ("d1#0", "m", 1, _blob(1.0)),
                                          ("d2#0", "m", 1, _blob(3.0)),
                                          ("d2#0", "other", 1, _blob(9.0))]))
    assert _run(storage.get_chunk_embeddings("m")) == [
        ("d1#0", "d1", _blob(1.0)), ("d1#1", "d1", _blob(2.0)), ("d2#0", "d2", _blob(3.0))]
    assert _run(storage.get_chunk_embeddings("m", ["d2", "nope"])) == [("d2#0", "d2", _blob(3.0))]
    assert _run(storage.get_chunk_embeddings("m", [])) == []
    assert _run(storage.get_chunk_embeddings("unknown")) == []


def test_get_chunks_for_documents_keys_every_requested_id(paragraph_chunking):
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("d1", "a\n\nb", url="http://x/1")))
    _run(storage.upsert_document(_doc("d2", "c", url="http://x/2")))
    out = _run(storage.get_chunks_for_documents(["d2", "d1", "missing", "d2"]))
    assert list(out) == ["d2", "d1", "missing"]
    assert [c.chunk_id for c in out["d1"]] == ["d1#0", "d1#1"]
    assert [c.text for c in out["d2"]] == ["c"]
    assert out["missing"] == []
    assert _run(storage.get_chunks_for_documents([])) == {}


def test_get_embedding_coverage(paragraph_chunking):
    _run(storage.init_db())
    assert _run(storage.get_embedding_coverage("m")) == {
        "model": "m", "documents": 0, "documents_chunked": 0, "documents_embedded": 0,
        "chunks": 0, "embedded": 0, "missing": 0}
    _run(storage.upsert_document(_doc("d1", "a\n\nb", url="http://x/1")))
    _run(storage.upsert_document(_doc("d2", "c", url="http://x/2")))
    _run(storage.upsert_document(_doc("d3", "", url="http://x/3")))   # nothing to chunk
    _run(storage.upsert_chunk_embeddings([("d1#0", "m", 1, _blob(1.0)),
                                          ("d2#0", "m", 1, _blob(1.0)),
                                          ("d1#1", "other", 1, _blob(1.0))]))
    assert _run(storage.get_embedding_coverage("m")) == {
        "model": "m", "documents": 3, "documents_chunked": 2, "documents_embedded": 1,
        "chunks": 3, "embedded": 2, "missing": 1}


# --- export state --------------------------------------------------------------------

def test_document_export_rows_roundtrip_per_collection():
    _run(storage.init_db())
    _run(storage.upsert_document_export(DocumentExport(
        document_id="d1", collection="quarry", content_sha256="abc",
        embed_model="cohere/embed-v5.0-pro", n_chunks=3, exported_at="2026-10-31T00:00:00+00:00")))
    _run(storage.upsert_document_export(DocumentExport(
        document_id="d2", collection="quarry", content_sha256="def", embed_model="m")))
    _run(storage.upsert_document_export(DocumentExport(
        document_id="d1", collection="elsewhere", content_sha256="zzz", embed_model="m")))

    got = _run(storage.get_document_exports(["d2", "d1", "d9", "d1"], "quarry"))
    assert set(got) == {"d1", "d2"}
    assert got["d1"] == DocumentExport(
        document_id="d1", collection="quarry", content_sha256="abc",
        embed_model="cohere/embed-v5.0-pro", n_chunks=3, exported_at="2026-10-31T00:00:00+00:00")
    assert got["d2"].n_chunks == 0 and "T" in got["d2"].exported_at  # defaults filled
    assert _run(storage.get_document_exports([], "quarry")) == {}
    assert _run(storage.list_exported_document_ids("quarry")) == ["d1", "d2"]
    assert _run(storage.list_exported_document_ids("elsewhere")) == ["d1"]
    assert _run(storage.list_exported_document_ids("nope")) == []

    # Re-export updates in place: still one row per (document, collection).
    _run(storage.upsert_document_export(DocumentExport(
        document_id="d1", collection="quarry", content_sha256="abc2", embed_model="m2",
        n_chunks=4, exported_at="later")))
    row = _run(storage.get_document_exports(["d1"], "quarry"))["d1"]
    assert (row.content_sha256, row.embed_model, row.n_chunks, row.exported_at) == (
        "abc2", "m2", 4, "later")
    assert _sql("SELECT COUNT(*) FROM document_exports")[0][0] == 3

    _run(storage.delete_document_exports(["d1", "d9"], "quarry"))
    assert _run(storage.list_exported_document_ids("quarry")) == ["d2"]
    assert _run(storage.list_exported_document_ids("elsewhere")) == ["d1"]  # untouched
    _run(storage.delete_document_exports([], "quarry"))
    assert _run(storage.list_exported_document_ids("quarry")) == ["d2"]


# --- get_documents_by_ids ------------------------------------------------------------

def test_get_documents_by_ids_preserves_input_order_with_full_bodies():
    _run(storage.init_db())
    for i in (1, 2, 3):
        _run(storage.upsert_document(_doc(f"d{i}", f"body {i}", url=f"http://x/{i}",
                                          fit=f"fit {i}", crawled_at=f"2026-01-0{i}")))
    docs = _run(storage.get_documents_by_ids(["d3", "d1", "missing", "d3", "d2"]))
    assert [d.id for d in docs] == ["d3", "d1", "d2"]
    assert docs[1].content_markdown == "body 1" and docs[1].content_fit == "fit 1"
    assert _run(storage.get_documents_by_ids([])) == []
    assert _run(storage.get_documents_by_ids(["nope"])) == []


def test_get_documents_by_ids_batches_large_id_lists():
    _run(storage.init_db())
    _run(storage.upsert_document(_doc("real", "b", url="http://x/real")))
    ids = [f"fake{i}" for i in range(storage._IN_BATCH * 2 + 7)] + ["real"]
    assert [d.id for d in _run(storage.get_documents_by_ids(ids))] == ["real"]


# --- llm usage: exclude_purposes -----------------------------------------------------

def _call(mission_id, purpose, prompt, completion):
    return LlmCall(mission_id=mission_id, purpose=purpose, tier="reasoning",
                   model="m", prompt_tokens=prompt, completion_tokens=completion)


def test_get_mission_llm_usage_excludes_purposes_from_totals_only():
    _run(storage.init_db())

    async def seed():
        for call in (_call("m1", "plan", 100, 20), _call("m1", "assess", 300, 40),
                     _call("m1", "embed", 5000, 0), _call("m1", "embed", 6000, 0),
                     _call("m1", "rerank", 0, 0)):
            await storage.insert_llm_call(call)
    _run(seed())

    full = _run(storage.get_mission_llm_usage("m1"))
    assert (full["calls"], full["prompt_tokens"], full["completion_tokens"]) == (5, 11400, 60)

    usage = _run(storage.get_mission_llm_usage("m1", exclude_purposes=("embed", "rerank")))
    assert (usage["calls"], usage["prompt_tokens"], usage["completion_tokens"]) == (2, 400, 60)
    # by_purpose still shows the whole spend, excluded purposes included.
    assert usage["by_purpose"] == full["by_purpose"]
    assert usage["by_purpose"]["embed"] == {"calls": 2, "prompt_tokens": 11000,
                                            "completion_tokens": 0}
    assert usage["by_purpose"]["rerank"]["calls"] == 1

    # Excluding a purpose with no rows, or excluding nothing, changes nothing.
    assert _run(storage.get_mission_llm_usage("m1", exclude_purposes=("nope",))) == full
    assert _run(storage.get_mission_llm_usage("m1", exclude_purposes=())) == full
    everything = _run(storage.get_mission_llm_usage(
        "m1", exclude_purposes=("plan", "assess", "embed", "rerank")))
    assert (everything["calls"], everything["prompt_tokens"]) == (0, 0)
    assert everything["by_purpose"] == full["by_purpose"]


# --- requirement -> document map ---------------------------------------------------

def test_get_requirement_document_map():
    async def run():
        await storage.init_db()
        await storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                         persona_prompt="p", created_at="t"))
        await storage.insert_mission(Mission(id="m1", agent_id="a1", question="Q",
                                             status="done", created_at="t"))
        await storage.insert_mission(Mission(id="m2", agent_id="a1", question="Q",
                                             status="done", created_at="t"))
        for rid in ("r1", "r2", "r3"):
            await storage.insert_requirement(Requirement(id=rid, mission_id="m1", title=rid))
        await storage.insert_requirement(Requirement(id="r-m2", mission_id="m2", title="x"))
        await storage.link_mission_document("m1", "r2", "dB")
        await storage.link_mission_document("m1", "r1", "dA")
        await storage.link_mission_document("m1", "r2", "dA")
        await storage.link_mission_document("m1", "r2", "dB")   # duplicate link: ignored
        await storage.link_mission_document("m2", "r-m2", "dZ")  # another mission
        return (await storage.get_requirement_document_map("m1"),
                await storage.get_requirement_document_map("m2"),
                await storage.get_requirement_document_map("nope"))

    m1, m2, none = _run(run())
    assert m1 == {"r1": ["dA"], "r2": ["dB", "dA"], "r3": []}
    assert list(m1) == ["r1", "r2", "r3"]            # plan order
    assert m2 == {"r-m2": ["dZ"]}
    assert none == {}
