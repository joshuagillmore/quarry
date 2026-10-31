"""vector_export: the dense-only Qdrant export shaped for Coherence.

An in-memory QdrantClient is injected through `client=`; vectors come from
the litellm stub (deterministic unit vectors, `dimensions=1024`), the
documents from the real temp SQLite that conftest gives every test."""
import asyncio
import json
import os
import threading
import time
import uuid

import litellm
import numpy as np
import pytest
from qdrant_client import QdrantClient, models as qm

import chunking
import storage
import vector_export
from config import settings
from models import Agent, Document, Mission, Requirement

DIM = 1024
COLLECTION = "quarry_test"
PRO = "cohere/embed-v5.0-pro"


@pytest.fixture(autouse=True)
def _export_settings(monkeypatch, tmp_path):
    litellm.reset_calls()
    monkeypatch.setattr(settings, "embed_model", PRO)
    monkeypatch.setattr(settings, "embed_query_model", "cohere/embed-v5.0-fast")
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "export_dir", str(tmp_path / "export"))
    monkeypatch.setattr(settings, "qdrant_export_collection", COLLECTION)
    monkeypatch.setattr(settings, "export_clearance_rank", 0)
    monkeypatch.setattr(settings, "export_classification", "NATO UNCLASSIFIED")
    monkeypatch.setattr(settings, "export_compartments", "")
    monkeypatch.setattr(settings, "export_releasability", "")
    asyncio.run(storage.init_db())
    yield
    litellm.reset_calls()


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def _run(coro):
    return asyncio.run(coro)


def _doc(doc_id, body, *, url="https://example.com/page", query="q1", title="Title",
         crawled_at="2026-08-11T06:48:00+00:00", metadata=None, domain="example.com"):
    return Document(id=doc_id, url=url, domain=domain, title=title, search_query=query,
                    crawled_at=crawled_at, content_markdown=body, content_fit="",
                    word_count=len(body.split()),
                    metadata_json=json.dumps(metadata) if metadata else None)


def _store(doc):
    return _run(storage.upsert_document(doc))


def _export(ids, client, **kw):
    kw.setdefault("check", False)
    return _run(vector_export.export_documents(ids, client=client, **kw))


def _points(client, collection=COLLECTION):
    records, _next = client.scroll(collection, limit=1000, with_payload=True,
                                   with_vectors=True)
    return records


def _doc_points(client, doc_id):
    return [r for r in _points(client)
            if r.payload.get("doc_id") == vector_export.coherence_doc_id(doc_id)]


# --- collection ------------------------------------------------------------------

def test_collection_created_with_named_dense_vector_and_indexes(client, monkeypatch):
    indexed = []
    real = client.create_payload_index

    def spy(collection_name, field_name, field_schema=None, **kw):
        indexed.append((field_name, field_schema))
        return real(collection_name=collection_name, field_name=field_name,
                    field_schema=field_schema, **kw)

    monkeypatch.setattr(client, "create_payload_index", spy)
    vector_export.ensure_collection(client, COLLECTION, DIM)

    info = client.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    assert set(vectors) == {"dense"}, "one named vector, so a sparse arm can be added later"
    assert vectors["dense"].size == DIM
    assert vectors["dense"].distance == qm.Distance.COSINE
    assert info.config.params.sparse_vectors in (None, {}), "dense-only"
    assert dict(indexed) == {
        "clearance_rank": qm.PayloadSchemaType.INTEGER,
        "compartments": qm.PayloadSchemaType.KEYWORD,
        "releasability": qm.PayloadSchemaType.KEYWORD,
        "doc_id": qm.PayloadSchemaType.KEYWORD,
        "cluster": qm.PayloadSchemaType.KEYWORD,
    }
    # Idempotent: a second call neither fails nor re-creates.
    vector_export.ensure_collection(client, COLLECTION, DIM)
    assert client.collection_exists(COLLECTION)


def test_ensure_collection_rejects_dim_mismatch_and_never_deletes(client, monkeypatch):
    client.create_collection(COLLECTION, vectors_config={
        "dense": qm.VectorParams(size=8, distance=qm.Distance.COSINE)})
    client.upsert(COLLECTION, points=[qm.PointStruct(
        id=str(uuid.uuid4()), vector={"dense": [1.0] + [0.0] * 7},
        payload={"doc_id": "foreign", "is_product": True})], wait=True)

    def forbidden(*a, **k):
        raise AssertionError("delete_collection must never be called")

    monkeypatch.setattr(client, "delete_collection", forbidden)
    with pytest.raises(vector_export.ExportError, match="dense size 8"):
        vector_export.ensure_collection(client, COLLECTION, DIM)
    assert client.count(COLLECTION).count == 1, "the existing point survived"

    # The export as a whole refuses too, before any write.
    doc_id = _store(_doc("d1", "some body text"))
    with pytest.raises(vector_export.ExportError):
        _export([doc_id], client)
    assert client.count(COLLECTION).count == 1
    assert _run(storage.get_document_exports([doc_id], COLLECTION)) == {}
    assert not os.path.exists(os.path.join(settings.export_dir, f"manifest-{COLLECTION}.json"))


def test_unnamed_vector_collection_is_refused(client):
    client.create_collection(COLLECTION, vectors_config=qm.VectorParams(
        size=DIM, distance=qm.Distance.COSINE))
    with pytest.raises(vector_export.ExportError, match="no 'dense' vector"):
        vector_export.ensure_collection(client, COLLECTION, DIM)
    assert client.collection_exists(COLLECTION)


def test_unreachable_qdrant_raises_before_any_write():
    doc_id = _store(_doc("d1", "some body text"))
    client = vector_export.make_client(url="http://127.0.0.1:1")
    with pytest.raises(vector_export.ExportError, match="unreachable"):
        _export([doc_id], client)
    assert _run(storage.get_document_exports([doc_id], COLLECTION)) == {}
    assert litellm.embedding_calls == [], "no vectors bought for an export that cannot land"
    assert not os.path.exists(os.path.join(settings.export_dir, f"manifest-{COLLECTION}.json"))


def test_make_client_passes_key_only_when_set(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeClient)
    monkeypatch.setattr(settings, "qdrant_url", "http://q:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    vector_export.make_client()
    assert seen == {"url": "http://q:6333", "timeout": 30, "check_compatibility": False}
    seen.clear()
    monkeypatch.setattr(settings, "qdrant_api_key", "k1")
    vector_export.make_client()
    assert seen["api_key"] == "k1"


# --- points and payload ------------------------------------------------------------

def test_points_carry_payload_and_uuid5_ids(client):
    body = "# Intro\n\nHello world paragraph."
    doc = _doc("d1", body, url="https://Example.com/path?a=1",
               metadata={"published_date": "2026-08-01"})
    doc_id = _store(doc)
    _run(storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                    persona_prompt="p", created_at="t")))
    _run(storage.insert_mission(Mission(id="m1", agent_id="a1", question="Q",
                                        status="done", created_at="t")))
    _run(storage.insert_requirement(Requirement(id="r1", mission_id="m1", title="T")))
    _run(storage.link_mission_document("m1", "r1", doc_id))

    result = _export([doc_id], client)
    assert result.exported == [doc_id]
    assert result.chunks == 1

    points = _doc_points(client, doc_id)
    assert len(points) == 1
    point = points[0]
    chunk_id = "QRY-d1#0"
    assert str(point.id) == str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
    assert set(point.vector) == {"dense"} and len(point.vector["dense"]) == DIM

    chunk_text = "Intro\n\nHello world paragraph."
    expected_vec = litellm._unit_vector(chunking.indexed_text("Title", chunk_text), DIM)
    assert np.allclose(point.vector["dense"], expected_vec, atol=1e-6), \
        "the vector is of indexed_text(title, chunk.text)"

    p = point.payload
    assert p["chunk_id"] == chunk_id
    assert p["doc_id"] == "QRY-d1"
    assert p["title"] == "Title"
    assert p["text"] == chunk_text, "section-prefixed, no title"
    assert p["section"] == "Intro"
    assert p["kind"] == "body"
    assert p["clearance_rank"] == 0 and isinstance(p["clearance_rank"], int)
    assert p["classification"] == "NATO UNCLASSIFIED"
    assert p["compartments"] == []
    assert p["releasability"] == []
    assert p["int_type"] == "OSINT"
    assert p["trust_tier"] == "EXTERNAL_UNVERIFIED"
    assert p["originator"] == "example.com"
    assert p["source_designator"] == "https://Example.com/path?a=1"
    assert p["source_url"] == "https://Example.com/path?a=1"
    assert p["date_time_group"] == "110648Z AUG 26"
    assert p["published_date"] == "2026-08-01"
    assert p["source_reliability"] == "C"
    assert p["info_credibility"] == 3
    assert p["provenance"] == {"url": "https://Example.com/path?a=1",
                               "fetched_at": "2026-08-11T06:48:00+00:00",
                               "sha256": chunking.document_digest(doc)}
    assert p["doc_type"] == "web_page"
    assert p["cluster"] == "q1"
    assert p["topics"] == []
    assert p["decoy_for"] == []
    assert p["embedding"] == {"model": PRO, "dim": DIM, "input_type": "search_document"}
    assert p["quarry"] == {
        "document_id": "d1", "search_query": "q1", "domain": "example.com",
        "word_count": doc.word_count, "crawled_at": "2026-08-11T06:48:00+00:00",
        "mission_ids": ["m1"], "exported_by": vector_export.EXPORTED_BY,
    }
    assert "is_product" not in p
    assert not p["doc_id"].startswith(("canary-", "contra-", "OSINT:"))


def test_markings_come_from_settings(client, monkeypatch):
    monkeypatch.setattr(settings, "export_clearance_rank", 2)
    monkeypatch.setattr(settings, "export_classification", "NATO SECRET")
    monkeypatch.setattr(settings, "export_compartments", "ALPHA, BETA")
    monkeypatch.setattr(settings, "export_releasability", "NATO")
    doc_id = _store(_doc("d1", "body text here"))
    _export([doc_id], client)
    p = _doc_points(client, doc_id)[0].payload
    assert p["clearance_rank"] == 2
    assert p["classification"] == "NATO SECRET"
    assert p["compartments"] == ["ALPHA", "BETA"]
    assert p["releasability"] == ["NATO"]


def test_published_date_falls_back_to_crawl_date():
    doc = _doc("d1", "x", metadata={"published_date": "August 2026"})
    chunk = chunking.Chunk(idx=0, section=None, text="x")
    payload = vector_export.build_payload(doc, chunk, vector_export.Markings(), "sha")
    assert payload["published_date"] == "2026-08-11"
    assert payload["title"] == "Title"
    untitled = _doc("d2", "x", title="")
    assert vector_export.build_payload(untitled, chunk, vector_export.Markings(),
                                       "sha")["title"] == "example.com"


@pytest.mark.parametrize("iso, dtg", [
    ("2026-08-11T06:48:00+00:00", "110648Z AUG 26"),
    ("2026-08-11T06:48:00Z", "110648Z AUG 26"),
    ("2026-08-11T06:48:00", "110648Z AUG 26"),
    ("2026-08-11T08:48:00+02:00", "110648Z AUG 26"),
    ("2025-01-02T23:05:59.123456+00:00", "022305Z JAN 25"),
    ("not a date", ""),
    ("", ""),
])
def test_date_time_group(iso, dtg):
    assert vector_export.date_time_group(iso) == dtg


# --- change detection ------------------------------------------------------------

def test_unchanged_skipped_changed_deleted_then_reupserted(client):
    doc_id = _store(_doc("d1", "# A\n\nfirst\n\n# B\n\nsecond"))
    first = _export([doc_id], client)
    assert first.exported == [doc_id] and first.chunks == 2
    assert {r.payload["chunk_id"] for r in _doc_points(client, doc_id)} == \
        {"QRY-d1#0", "QRY-d1#1"}
    calls_after_first = len(litellm.embedding_calls)

    again = _export([doc_id], client)
    assert again.skipped == [doc_id] and again.exported == [] and again.chunks == 0
    assert len(litellm.embedding_calls) == calls_after_first, "nothing embedded"
    assert len(_doc_points(client, doc_id)) == 2

    # Re-crawl under the same (url, search_query): same id, one chunk now.
    assert _store(_doc("d1-new-uuid", "# A\n\nfirst changed")) == doc_id
    third = _export([doc_id], client)
    assert third.exported == [doc_id] and third.chunks == 1
    points = _doc_points(client, doc_id)
    assert [r.payload["chunk_id"] for r in points] == ["QRY-d1#0"], "no stale point"
    assert points[0].payload["text"] == "A\n\nfirst changed"
    row = _run(storage.get_document_exports([doc_id], COLLECTION))[doc_id]
    assert row.n_chunks == 1 and row.embed_model == PRO
    assert row.content_sha256 == chunking.document_digest(_run(storage.get_document(doc_id)))


def test_force_rewrites_and_model_change_reexports(client, monkeypatch):
    doc_id = _store(_doc("d1", "body text"))
    _export([doc_id], client)
    assert _export([doc_id], client, force=True).exported == [doc_id]
    assert len(_doc_points(client, doc_id)) == 1
    monkeypatch.setattr(settings, "embed_model", "cohere/embed-v4.0")
    assert _export([doc_id], client).exported == [doc_id], "a new model means new vectors"
    assert _doc_points(client, doc_id)[0].payload["embedding"]["model"] == "cohere/embed-v4.0"


def test_stored_vectors_reused_not_rebought(client):
    doc_id = _store(_doc("d1", "body text"))
    chunk = _run(storage.get_document_chunks(doc_id))[0]
    stored = [0.0] * DIM
    stored[3] = 1.0
    _run(storage.upsert_chunk_embeddings([
        (chunk.chunk_id, PRO, DIM, vector_export.embeddings.vector_to_blob(stored))]))
    litellm.reset_calls()

    result = _export([doc_id], client)
    assert result.exported == [doc_id]
    assert litellm.embedding_calls == [], "the stored vector was reused"
    assert np.allclose(_doc_points(client, doc_id)[0].vector["dense"], stored)


def test_embedded_vectors_are_stored_back(client):
    doc_id = _store(_doc("d1", "body text"))
    assert _run(storage.get_chunk_embeddings(PRO, [doc_id])) == []
    _export([doc_id], client)
    rows = _run(storage.get_chunk_embeddings(PRO, [doc_id]))
    assert len(rows) == 1 and len(rows[0][2]) == DIM * 4


def test_chunks_rebuilt_for_a_document_without_them(client):
    doc_id = _store(_doc("d1", "para one\n\npara two"))
    _run(storage.replace_document_chunks(doc_id, []))
    assert _run(storage.get_document_chunks(doc_id)) == []
    result = _export([doc_id], client)
    assert result.exported == [doc_id] and result.chunks == 1
    assert len(_run(storage.get_document_chunks(doc_id))) == 1


def test_empty_body_skipped_and_unknown_id_ignored(client):
    doc_id = _store(_doc("d1", ""))
    result = _export([doc_id, "no-such-doc"], client)
    assert result.skipped == [doc_id] and result.exported == [] and result.failed == {}
    assert client.count(COLLECTION).count == 0


# --- robustness -------------------------------------------------------------------

def test_foreign_points_in_collection_survive(client):
    vector_export.ensure_collection(client, COLLECTION, DIM)
    foreign_id = str(uuid.uuid4())
    vec = [0.0] * DIM
    vec[0] = 1.0
    client.upsert(COLLECTION, points=[qm.PointStruct(
        id=foreign_id, vector={"dense": vec},
        payload={"doc_id": "crs-IF11130", "is_product": True, "chunk_id": "crs-IF11130#0"})],
        wait=True)
    d1 = _store(_doc("d1", "body one", url="https://a/1"))
    d2 = _store(_doc("d2", "body two", url="https://a/2"))
    _export([d1, d2], client)
    _store(_doc("d1x", "body one changed", url="https://a/1"))
    _export([d1, d2], client, force=True)

    kept = client.retrieve(COLLECTION, ids=[foreign_id], with_payload=True)
    assert len(kept) == 1
    assert kept[0].payload == {"doc_id": "crs-IF11130", "is_product": True,
                               "chunk_id": "crs-IF11130#0"}
    assert client.count(COLLECTION).count == 3


def test_per_document_failure_is_captured_not_raised(client, monkeypatch):
    real = vector_export.embeddings.embed_texts

    def flaky(texts, **kw):
        if any("BOOM" in t for t in texts):
            raise RuntimeError("provider refused")
        return real(texts, **kw)

    monkeypatch.setattr(vector_export.embeddings, "embed_texts", flaky)
    good = _store(_doc("d1", "fine body", url="https://a/1"))
    bad = _store(_doc("d2", "BOOM body", url="https://a/2"))
    result = _export([good, bad], client)
    assert result.exported == [good]
    assert list(result.failed) == [bad]
    assert "RuntimeError: provider refused" in result.failed[bad]
    assert _run(storage.get_document_exports([good, bad], COLLECTION)).keys() == {good}
    assert len(_doc_points(client, bad)) == 0


def test_failed_upsert_batch_leaves_no_export_row(client, monkeypatch):
    doc_id = _store(_doc("d1", "body text"))

    def refuse(*a, **k):
        raise RuntimeError("storage full")

    monkeypatch.setattr(client, "upsert", refuse)
    result = _export([doc_id], client)
    assert result.exported == [] and doc_id in result.failed
    assert "storage full" in result.failed[doc_id]
    assert _run(storage.get_document_exports([doc_id], COLLECTION)) == {}


def test_batches_of_400_with_wait(client, monkeypatch):
    # 3 docs x 150 one-paragraph-sections each -> 450 points -> two batches.
    body = "\n\n".join(f"# S{i}\n\np{i}" for i in range(150))
    ids = [_store(_doc(f"d{n}", body, url=f"https://a/{n}")) for n in range(3)]
    sizes = []
    real = client.upsert

    def spy(collection_name, points, wait=None, **kw):
        sizes.append((len(points), wait))
        return real(collection_name=collection_name, points=points, wait=wait, **kw)

    monkeypatch.setattr(client, "upsert", spy)
    result = _export(ids, client)
    assert result.chunks == 450
    assert sizes == [(400, True), (50, True)]


def test_empty_embed_model_is_an_export_error(client, monkeypatch):
    doc_id = _store(_doc("d1", "body"))
    monkeypatch.setattr(settings, "embed_model", "")
    with pytest.raises(vector_export.ExportError, match="EMBED_MODEL"):
        _export([doc_id], client)


def test_bad_collection_name_refused(client):
    doc_id = _store(_doc("d1", "body"))
    with pytest.raises(vector_export.ExportError, match="invalid collection"):
        _export([doc_id], client, collection="bad name; drop")


# --- manifest ---------------------------------------------------------------------

def test_manifest_written(client):
    d1 = _store(_doc("d1", "# A\n\none\n\n# B\n\ntwo", url="https://a/1"))
    d2 = _store(_doc("d2", "", url="https://a/2"))
    result = _export([d1, d2], client)
    path = os.path.join(settings.export_dir, f"manifest-{COLLECTION}.json")
    assert result.manifest_path == path
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["collection"] == COLLECTION
    assert manifest["embed_model"] == PRO
    assert manifest["dim"] == DIM
    assert manifest["vectors"] == ["dense"]
    assert manifest["n_docs"] == 1
    assert manifest["n_chunks"] == 2
    assert manifest["exported_at"] == result.finished_at
    by_id = {d["document_id"]: d for d in manifest["docs"]}
    assert by_id[d1]["status"] == "exported" and by_id[d1]["n_chunks"] == 2
    assert by_id[d1]["doc_id"] == "QRY-d1"
    assert by_id[d2]["status"] == "skipped" and by_id[d2]["reason"] == "empty body"


# --- background + status ---------------------------------------------------------

def test_background_export_runs_once_at_a_time_and_reports_status(client, monkeypatch):
    monkeypatch.setattr(vector_export, "make_client", lambda *a, **k: client)
    doc_id = _store(_doc("d1", "body text"))

    assert vector_export._export_lock.acquire(blocking=False)
    try:
        assert vector_export.start_export_in_background([doc_id]) is False
    finally:
        vector_export._export_lock.release()

    assert vector_export.start_export_in_background([doc_id]) is True
    deadline = time.monotonic() + 10
    while vector_export.export_status()["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
    status = vector_export.export_status()
    assert status["running"] is False
    assert status["error"] is None
    assert status["collection"] == COLLECTION
    assert status["result"]["exported"] == [doc_id]
    assert status["result"]["check"] is None
    assert not vector_export._export_lock.locked()
    assert threading.active_count() >= 1


def test_background_export_error_releases_lock(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no qdrant")

    monkeypatch.setattr(vector_export, "make_client", boom)
    doc_id = _store(_doc("d1", "body text"))
    assert vector_export.start_export_in_background([doc_id]) is True
    deadline = time.monotonic() + 10
    while vector_export.export_status()["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
    status = vector_export.export_status()
    assert status["running"] is False
    assert "no qdrant" in status["error"]
    assert not vector_export._export_lock.locked()


# --- CLI --------------------------------------------------------------------------

def test_cli_dry_run_writes_nothing(monkeypatch, capsys):
    doc_id = _store(_doc("d1", "body text"))

    def forbidden(*a, **k):
        raise AssertionError("dry run must not contact Qdrant")

    monkeypatch.setattr(vector_export, "make_client", forbidden)
    assert vector_export.main(["--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "export" in out and doc_id in out and "nothing written" in out
    assert _run(storage.get_document_exports([doc_id], COLLECTION)) == {}
    assert litellm.embedding_calls == []
    assert not os.path.exists(settings.export_dir)


def test_cli_exports_a_search_and_a_doc(client, monkeypatch, capsys):
    monkeypatch.setattr(vector_export, "make_client", lambda *a, **k: client)
    d1 = _store(_doc("d1", "one", url="https://a/1", query="alpha"))
    d2 = _store(_doc("d2", "two", url="https://a/2", query="beta"))
    assert vector_export.main(["--search", "alpha", "--no-check"]) == 0
    assert _run(storage.get_document_exports([d1, d2], COLLECTION)).keys() == {d1}
    assert vector_export.main(["--doc", d2, "--collection", COLLECTION, "--no-check"]) == 0
    assert _run(storage.get_document_exports([d1, d2], COLLECTION)).keys() == {d1, d2}
    assert "exported 1" in capsys.readouterr().out


def test_cli_unknown_mission_is_an_error(capsys):
    assert vector_export.main(["--mission", "nope"]) == 2
    assert "no mission" in capsys.readouterr().err


# --- export rows never outlive the points (review F1) -----------------------------

def test_failed_forced_rerun_leaves_no_export_row_so_the_next_run_reexports(client, monkeypatch):
    """A forced run deletes the old points by filter before its upsert; when
    that upsert is refused, the document's export row must go with the
    points, or the next plain run skips a document the collection no longer
    holds."""
    doc_id = _store(_doc("d1", "body text"))
    assert _export([doc_id], client).exported == [doc_id]
    assert len(_doc_points(client, doc_id)) == 1

    real_upsert = client.upsert

    def refuse(*a, **k):
        raise RuntimeError("storage full")

    monkeypatch.setattr(client, "upsert", refuse)
    forced = _export([doc_id], client, force=True)
    assert doc_id in forced.failed
    assert _doc_points(client, doc_id) == [], "old points deleted before the refused upsert"
    assert _run(storage.get_document_exports([doc_id], COLLECTION)) == {}, \
        "no row may say 'exported' while the collection holds no point for it"

    monkeypatch.setattr(client, "upsert", real_upsert)
    again = _export([doc_id], client)
    assert again.exported == [doc_id] and again.skipped == []
    assert len(_doc_points(client, doc_id)) == 1


def test_recreated_collection_is_repopulated(client):
    """A collection that was wiped (dropped and recreated by the export)
    holds nothing: every export row for that collection name is stale, so
    the next run re-exports instead of skipping as unchanged."""
    d1 = _store(_doc("d1", "body text", url="https://example.com/1"))
    d2 = _store(_doc("d2", "other text", url="https://example.com/2"))
    assert sorted(_export([d1, d2], client).exported) == sorted([d1, d2])
    assert client.collection_exists(COLLECTION)

    client.delete_collection(COLLECTION)
    res = _export([d1], client)
    assert res.exported == [d1] and res.skipped == []
    assert len(_doc_points(client, d1)) == 1
    # The rows of documents outside this run were cleared too, so d2 is
    # re-exported when it next comes round rather than skipped forever.
    assert _run(storage.list_exported_document_ids(COLLECTION)) == [d1]
    assert _export([d2], client).exported == [d2]


def test_ensure_collection_reports_whether_it_created_the_collection(client):
    assert vector_export.ensure_collection(client, COLLECTION, DIM) is True
    assert vector_export.ensure_collection(client, COLLECTION, DIM) is False


# --- the auto-export shares the lock and status (review F7) -------------------------

def test_run_export_locked_holds_the_lock_and_status_for_the_whole_run(client, monkeypatch):
    seen = {}

    async def fake_export(doc_ids, **kw):
        status = vector_export.export_status()
        seen["running"] = status["running"]
        seen["mission_id"] = status["mission_id"]
        seen["doc_count"] = status["doc_count"]
        seen["locked"] = vector_export._export_lock.locked()
        # A background export asked for mid-run must be refused.
        seen["background"] = vector_export.start_export_in_background(list(doc_ids))
        seen["kw"] = kw
        return vector_export.ExportResult(collection=kw.get("collection") or "c",
                                          exported=list(doc_ids))

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    res = vector_export.run_export_locked(["d1", "d1", "d2"], mission_id="m1")
    assert res is not None and res.exported == ["d1", "d2"]
    assert seen["running"] is True and seen["locked"] is True
    assert seen["mission_id"] == "m1" and seen["doc_count"] == 2
    assert seen["background"] is False
    assert seen["kw"]["mission_id"] == "m1" and seen["kw"]["collection"] == COLLECTION
    status = vector_export.export_status()
    assert status["running"] is False and status["error"] is None
    assert status["result"]["exported"] == ["d1", "d2"]
    assert not vector_export._export_lock.locked()


def test_run_export_locked_returns_none_when_an_export_is_running(monkeypatch, capsys):
    calls = []

    async def fake_export(doc_ids, **kw):
        calls.append(list(doc_ids))
        return vector_export.ExportResult(collection="c")

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    assert vector_export._export_lock.acquire(blocking=False)
    try:
        assert vector_export.run_export_locked(["d1"], mission_id="m1") is None
    finally:
        vector_export._export_lock.release()
    assert calls == []
    assert "another export is running" in capsys.readouterr().err


def test_run_export_locked_releases_the_lock_and_records_the_error_on_failure(monkeypatch):
    async def fake_export(doc_ids, **kw):
        raise vector_export.ExportError("no qdrant")

    monkeypatch.setattr(vector_export, "export_documents", fake_export)
    with pytest.raises(vector_export.ExportError):
        vector_export.run_export_locked(["d1"], mission_id="m1")
    status = vector_export.export_status()
    assert status["running"] is False and "no qdrant" in status["error"]
    assert not vector_export._export_lock.locked()


# --- the Qdrant URL's userinfo never reaches a message (review F10) -------------------

def test_probe_error_redacts_userinfo_in_the_qdrant_url(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "http://user:s3cret@qdrant.internal:6333")

    class Down:
        def get_collections(self):
            raise RuntimeError("cannot connect to http://user:s3cret@qdrant.internal:6333/")

    with pytest.raises(vector_export.ExportError) as err:
        vector_export._probe(Down())
    text = str(err.value)
    assert "s3cret" not in text and "user:" not in text
    assert "Qdrant unreachable at http://qdrant.internal:6333" in text
    assert "RuntimeError: cannot connect to http://qdrant.internal:6333/" in text


def test_redact_userinfo():
    assert vector_export._redact_userinfo("http://u:p@host:6333") == "http://host:6333"
    assert vector_export._redact_userinfo("https://k@host/x") == "https://host/x"
    assert vector_export._redact_userinfo("http://host:6333") == "http://host:6333"
    assert vector_export._redact_userinfo("a http://u:p@h:1 and https://u2:p2@h2") == \
        "a http://h:1 and https://h2"
