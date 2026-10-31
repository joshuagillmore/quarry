"""embeddings.py (WP1) against the litellm stub: Cohere Embed 5 ids and
params on the wire, provider branching, 96-text batches with one LlmCall
row each, retry policy, float32 storage, and the disabled switch.

Written before embeddings.py existed (WP0): a ModuleNotFoundError on
`import embeddings` means WP1 has not landed, not that these are wrong."""
import asyncio
import sqlite3
import struct

import litellm
import pytest

import config
import embeddings
import llm
import storage
from config import settings
from models import Document

PRO = "cohere/embed-v5.0-pro"
FAST = "cohere/embed-v5.0-fast"
DIM = 1024


class AuthenticationError(Exception):
    """Named like litellm's: llm._NON_RETRYABLE matches on class name."""


class RateLimitError(Exception):
    """Named like litellm's transient error: retried."""


@pytest.fixture(autouse=True)
def _cohere_v5(monkeypatch):
    """Explicit, distinctive settings so an assertion cannot pass by
    coinciding with a default."""
    litellm.reset_calls()
    monkeypatch.setattr(settings, "embed_model", PRO)
    monkeypatch.setattr(settings, "embed_query_model", FAST)
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "embed_timeout_s", 33)
    monkeypatch.setattr(settings, "llm_api_key", "test-embed-key")
    monkeypatch.setattr(settings, "cohere_api_key", "")
    monkeypatch.setattr(llm, "RETRY_DELAY_S", 0)
    yield
    litellm.reset_calls()


def _sql(query, args=()):
    db = sqlite3.connect(storage.DB_PATH)
    try:
        return db.execute(query, args).fetchall()
    finally:
        db.close()


def _texts(sent):
    """The stub records `input` exactly as sent; litellm accepts a str or a
    list, so normalise for comparison."""
    return [sent] if isinstance(sent, str) else list(sent)


# ---------- defaults (WP4) ----------

def test_default_models_are_cohere_v5():
    """conftest resets every settings field to its declared default, so a
    fresh Settings and the singleton agree. Read the declared defaults, not
    the (monkeypatched) singleton."""
    fields = config.Settings.model_fields
    assert fields["embed_model"].get_default() == PRO
    assert fields["embed_query_model"].get_default() == FAST
    assert fields["rerank_model"].get_default() == "cohere/rerank-v5.0-pro"
    assert fields["embed_dim"].get_default() == DIM
    assert fields["embed_timeout_s"].get_default() == 60
    assert fields["embed_tokens_count_toward_budget"].get_default() is False


# ---------- what goes on the wire ----------

def test_embed_documents_sends_v5_pro_search_document_and_1024_dims():
    vectors = embeddings.embed_texts(["alpha one", "beta two"])
    [kw] = litellm.embedding_calls
    assert kw["model"] == PRO
    assert _texts(kw["input"]) == ["alpha one", "beta two"]
    assert kw["input_type"] == "search_document"
    assert kw["dimensions"] == DIM
    assert kw["timeout"] == settings.embed_timeout_s == 33
    assert kw["api_key"] == "test-embed-key", "hosted provider gets the configured key"
    assert kw.get("drop_params") is True
    assert "api_base" not in kw
    assert len(vectors) == 2 and all(len(v) == DIM for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert vectors[0] == litellm._unit_vector("alpha one", DIM), "order and content preserved"
    assert vectors[1] == litellm._unit_vector("beta two", DIM)


def test_embed_query_sends_v5_fast_search_query_same_dims():
    """Shared space: documents indexed with Pro, queries embedded with Fast
    at the same dimensionality."""
    vector = embeddings.embed_query("what is alpha")
    [kw] = litellm.embedding_calls
    assert kw["model"] == FAST
    assert _texts(kw["input"]) == ["what is alpha"]
    assert kw["input_type"] == "search_query"
    assert kw["dimensions"] == DIM
    assert kw["timeout"] == 33
    assert kw["api_key"] == "test-embed-key"
    assert len(vector) == DIM and isinstance(vector[0], float)
    assert vector == litellm._unit_vector("what is alpha", DIM)


def test_embed_query_uses_embed_model_when_query_model_empty(monkeypatch):
    monkeypatch.setattr(settings, "embed_query_model", "")
    embeddings.embed_query("q")
    [kw] = litellm.embedding_calls
    assert kw["model"] == PRO
    assert kw["input_type"] == "search_query"


def test_ollama_model_gets_api_base_and_no_cohere_params(monkeypatch):
    monkeypatch.setattr(settings, "embed_model", "ollama/nomic-embed-text")
    monkeypatch.setattr(settings, "ollama_api_base", "http://ollama.test:11434")
    vectors = embeddings.embed_texts(["alpha one"])
    [kw] = litellm.embedding_calls
    assert kw["model"] == "ollama/nomic-embed-text"
    assert kw["api_base"] == "http://ollama.test:11434"
    assert "api_key" not in kw
    assert "input_type" not in kw, "Cohere-only parameter"
    assert "dimensions" not in kw, "Cohere-only parameter"
    assert kw["timeout"] == 33
    # The stub embeds at its default width when no `dimensions` was sent,
    # so the vector length is independent evidence the param stayed home.
    assert len(vectors) == 1 and len(vectors[0]) == 8


# ---------- batching + telemetry ----------

def test_batches_of_96_one_llmcall_each_with_tokens():
    asyncio.run(storage.init_db())
    assert embeddings.EMBED_BATCH == 96
    texts = [f"doc {i} has four" for i in range(200)]     # 4 words each
    vectors = embeddings.embed_texts(texts, mission_id="m1")

    assert [len(_texts(c["input"])) for c in litellm.embedding_calls] == [96, 96, 8]
    assert _texts(litellm.embedding_calls[0]["input"]) == texts[:96]
    assert _texts(litellm.embedding_calls[1]["input"]) == texts[96:192]
    assert _texts(litellm.embedding_calls[2]["input"]) == texts[192:]
    assert all(c["model"] == PRO and c["dimensions"] == DIM
               for c in litellm.embedding_calls)

    assert len(vectors) == 200
    for i in (0, 95, 96, 191, 192, 199):
        assert vectors[i] == litellm._unit_vector(texts[i], DIM), f"vector {i} out of order"

    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage["calls"] == 3
    assert usage["by_purpose"]["embed"] == {
        "calls": 3, "prompt_tokens": 800, "completion_tokens": 0}
    rows = _sql("SELECT purpose, tier, model, mission_id, prompt_tokens, "
                "completion_tokens FROM llm_calls ORDER BY created_at, rowid")
    assert rows == [
        ("embed", "embed", PRO, "m1", 384, 0),
        ("embed", "embed", PRO, "m1", 384, 0),
        ("embed", "embed", PRO, "m1", 32, 0),
    ]
    assert all(r[0] >= 0 for r in _sql("SELECT duration_ms FROM llm_calls"))


def test_llmcall_without_mission_has_null_mission_id():
    asyncio.run(storage.init_db())
    embeddings.embed_texts(["one two"])
    assert _sql("SELECT mission_id, purpose, prompt_tokens FROM llm_calls") == [
        (None, "embed", 2)]


# ---------- retry policy ----------

def test_non_retryable_fails_first_attempt_retryable_retries_once(monkeypatch):
    asyncio.run(storage.init_db())
    real = litellm.embedding
    attempts = []

    def auth_fails(**kw):
        attempts.append(kw["model"])
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "embedding", auth_fails)
    with pytest.raises(AuthenticationError):
        embeddings.embed_texts(["x"], mission_id="m1")
    assert len(attempts) == 1, "a bad key is not retried"
    assert _sql("SELECT COUNT(*) FROM llm_calls") == [(0,)], "a failed attempt is never recorded"

    attempts.clear()

    def flaky(**kw):
        attempts.append(kw["model"])
        if len(attempts) == 1:
            raise RateLimitError("429")
        return real(**kw)

    monkeypatch.setattr(litellm, "embedding", flaky)
    vectors = embeddings.embed_texts(["x y z"], mission_id="m1")
    assert len(attempts) == 2, "one transient failure, one successful retry"
    assert vectors == [litellm._unit_vector("x y z", DIM)]
    assert _sql("SELECT purpose, prompt_tokens FROM llm_calls") == [("embed", 3)], \
        "only the attempt that returned is recorded"

    attempts.clear()

    def always_limited(**kw):
        attempts.append(kw["model"])
        raise RateLimitError("429")

    monkeypatch.setattr(litellm, "embedding", always_limited)
    with pytest.raises(RateLimitError):
        embeddings.embed_texts(["x"])
    assert len(attempts) == llm.RETRY_ATTEMPTS, "the retry budget, then give up"


# ---------- storage path ----------

def _doc(doc_id="d-embed-1"):
    body = "\n\n".join(f"Paragraph {i} about alpha and beta " * 20 for i in range(6))
    return Document(id=doc_id, url="http://x.example/p", domain="x.example",
                    title="Alpha Beta Page", search_query="alpha beta",
                    crawled_at="2026-10-31T00:00:00+00:00",
                    content_markdown=body, content_fit=body, word_count=600)


def test_embed_pending_documents_stores_float32_blobs():
    if not hasattr(storage, "replace_document_chunks"):
        pytest.skip("WP3 storage not merged")
    chunking = pytest.importorskip("chunking", reason="WP2 chunking not merged")

    asyncio.run(storage.init_db())
    doc = _doc()
    doc_id = asyncio.run(storage.upsert_document(doc))
    chunks = chunking.document_chunks(doc)
    assert chunks, "fixture body must produce at least one chunk"
    # After WP3 the upsert itself writes these; calling it again is a no-op.
    asyncio.run(storage.replace_document_chunks(doc_id, chunks))

    n = asyncio.run(embeddings.embed_pending_documents([doc_id], mission_id="m1"))
    assert n == len(chunks)

    expected = {c.idx: litellm._unit_vector(chunking.indexed_text(doc.title, c.text), DIM)
                for c in chunks}
    sent = [t for call in litellm.embedding_calls for t in _texts(call["input"])]
    assert sent == [chunking.indexed_text(doc.title, c.text) for c in chunks], \
        "the ONE string that is embedded is indexed_text(title, chunk.text)"
    assert all(c["input_type"] == "search_document" and c["model"] == PRO
               for c in litellm.embedding_calls)

    rows = _sql("""SELECT c.idx, e.model, e.dim, e.vector
                   FROM chunk_embeddings e JOIN chunks c ON c.chunk_id = e.chunk_id
                   WHERE c.document_id = ? ORDER BY c.idx""", (doc_id,))
    assert [r[0] for r in rows] == sorted(expected)
    for idx, model, dim, blob in rows:
        assert (model, dim) == (PRO, DIM)
        assert isinstance(blob, bytes) and len(blob) == 4 * DIM, "little-endian float32"
        decoded = struct.unpack(f"<{DIM}f", blob)
        assert decoded == pytest.approx(expected[idx], abs=1e-6)
        roundtrip = embeddings.blob_to_vector(blob, DIM)
        assert len(roundtrip) == DIM
        assert list(map(float, roundtrip)) == pytest.approx(expected[idx], abs=1e-6)
        assert embeddings.vector_to_blob(expected[idx]) == blob

    stored = asyncio.run(storage.get_chunk_embeddings(PRO, [doc_id]))
    assert sorted(cid for cid, _d, _b in stored) == sorted(c.chunk_id for c in
                                                            asyncio.run(storage.get_document_chunks(doc_id)))
    assert all(d == doc_id for _c, d, _b in stored)

    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage["by_purpose"]["embed"]["calls"] == len(litellm.embedding_calls) >= 1
    assert usage["by_purpose"]["embed"]["prompt_tokens"] > 0

    # Nothing pending any more: no provider call, 0 embedded.
    before = len(litellm.embedding_calls)
    assert asyncio.run(embeddings.embed_pending_documents([doc_id], mission_id="m1")) == 0
    assert len(litellm.embedding_calls) == before


def test_embed_pending_documents_failure_logs_and_returns_zero(monkeypatch, capsys):
    if not hasattr(storage, "replace_document_chunks"):
        pytest.skip("WP3 storage not merged")
    chunking = pytest.importorskip("chunking", reason="WP2 chunking not merged")
    asyncio.run(storage.init_db())
    doc = _doc("d-embed-2")
    doc_id = asyncio.run(storage.upsert_document(doc))
    asyncio.run(storage.replace_document_chunks(doc_id, chunking.document_chunks(doc)))

    def down(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "embedding", down)
    assert asyncio.run(embeddings.embed_pending_documents([doc_id], mission_id="m1")) == 0
    err = capsys.readouterr().err
    assert "AuthenticationError" in err
    assert _sql("SELECT COUNT(*) FROM chunk_embeddings") == [(0,)]


# ---------- disabled ----------

def test_embed_disabled_when_model_empty(monkeypatch, capsys):
    monkeypatch.setattr(settings, "embed_model", "")
    assert embeddings.embed_enabled() is False
    with pytest.raises(embeddings.EmbeddingDisabled):
        embeddings.embed_texts(["x"])
    with pytest.raises(embeddings.EmbeddingDisabled):
        embeddings.embed_query("x")
    assert litellm.embedding_calls == [], "no provider call when disabled"
    assert issubclass(embeddings.EmbeddingDisabled, RuntimeError)

    asyncio.run(storage.init_db())
    assert asyncio.run(embeddings.embed_pending_documents(["no-such-doc"])) == 0
    assert litellm.embedding_calls == []
    assert capsys.readouterr().err.strip() != "", "disabled is logged, not silent"


def test_embed_enabled_when_model_set():
    assert embeddings.embed_enabled() is True
