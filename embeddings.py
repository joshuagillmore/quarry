"""Dense embeddings and reranking through LiteLLM, the way llm.py does chat.

The one place that knows how we talk to an embedding or rerank model: the
configured model ids, the Cohere-only wire parameters, a request timeout,
the same retry policy as llm.py, and one `llm_calls` row per provider call
that returned (tier "embed" or "rerank", so the mission page can show the
spend and the budget can leave it out). ranking.py, the vector export and
the pipeline hooks all go through here.

THE DEGRADE RULE

  An empty model setting means off, and a failed call is never a mission's
  problem. `embed_texts`/`embed_query`/`rerank` raise — `EmbeddingDisabled`
  / `RerankDisabled` when switched off, the provider's error otherwise — so a
  caller that needs a vector finds out. The two entry points the pipeline
  calls, `try_rerank` and `embed_pending_documents`, never raise: they fall
  back to today's behaviour (None, i.e. keep the existing order; 0 chunks
  embedded) and print exactly one stderr line naming the exception class,
  so the trace shows why ranking was skipped without the run failing.

THE ``indexed_text`` INVARIANT

  `embed_pending_documents` embeds `chunking.indexed_text(title, chunk.text)`
  and nothing else, so the vector stored for a chunk is of the same string
  every other path (the export, a query-time re-embed) would build. See
  chunking.py's module docstring for why there is exactly one such function.

Vectors are stored as little-endian float32 (`vector_to_blob`), the shape
Qdrant and numpy both read without conversion; storage.py only checks the
byte length and never imports numpy.
"""
import asyncio
import sys
import time
from typing import Iterable, Optional

import litellm
import numpy as np

import chunking
import llm
import storage
from config import settings


class EmbeddingDisabled(RuntimeError):
    """`EMBED_MODEL` is empty: embeddings are switched off."""


class RerankDisabled(RuntimeError):
    """`RERANK_MODEL` is empty: reranking is switched off."""


class BadEmbeddingResponse(RuntimeError):
    """The provider answered 200 but not with one vector per input. Raised
    inside the retry loop, like llm.EmptyCompletion, so a transient glitch
    gets one more try instead of misaligning vectors with their texts."""


# Cohere's /v2/embed accepts 96 texts per request; one batch per call keeps a
# document's chunks in as few provider calls (and telemetry rows) as possible.
EMBED_BATCH = 96
# Cohere Rerank 5's per-document token cap; longer documents are truncated
# server-side rather than rejected. 2048 covers a 2000-char chunk with room.
RERANK_MAX_TOKENS_PER_DOC = 2048

# Only Cohere's embed endpoint takes these; LiteLLM maps `dimensions` to
# Cohere's output_dimension. Other providers return their native width and
# have no search_document/search_query distinction.
_COHERE_EMBED_PREFIX = "cohere/"


def embed_enabled() -> bool:
    return bool((settings.embed_model or "").strip())


def rerank_enabled() -> bool:
    return bool((settings.rerank_model or "").strip())


def _call_retrying(fn, model: str, what: str):
    """llm._complete_retrying's policy for an embed/rerank call: retry a
    transient failure llm.RETRY_ATTEMPTS times with llm.RETRY_DELAY_S
    backoff, raise a permanent one (llm._NON_RETRYABLE, by class name) at
    once. Returns (response, duration_ms) of the attempt that returned, so a
    failed attempt is never recorded."""
    last = None
    for attempt in range(llm.RETRY_ATTEMPTS):
        started = time.monotonic()
        try:
            response = fn()
            return response, int((time.monotonic() - started) * 1000)
        except Exception as e:  # noqa: BLE001 - re-raised below if final
            last = e
            if type(e).__name__ in llm._NON_RETRYABLE:
                raise
            if attempt + 1 < llm.RETRY_ATTEMPTS:
                print(f"[{what}] {model} failed ({type(e).__name__}); retrying",
                      file=sys.stderr, flush=True)
                time.sleep(llm.RETRY_DELAY_S * (attempt + 1))
    raise last


def _field(item, name: str):
    """One field of a response row, which LiteLLM gives as a dict for some
    providers and an object for others."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _vectors(response, expected: int) -> list[list[float]]:
    """The embedding rows of a response, in input order, as plain floats."""
    data = list(getattr(response, "data", None) or [])
    if len(data) != expected:
        raise BadEmbeddingResponse(
            f"expected {expected} embeddings, got {len(data)}")
    # Rows carry their input index; order by it in case a provider does not
    # return them in input order, keeping the given order when none does.
    indexed = [(_field(row, "index"), row) for row in data]
    if all(isinstance(i, int) for i, _row in indexed):
        indexed.sort(key=lambda pair: pair[0])
    vectors = []
    for _i, row in indexed:
        vector = _field(row, "embedding")
        if not vector:
            raise BadEmbeddingResponse("provider returned an empty embedding")
        vectors.append([float(x) for x in vector])
    return vectors


def _embed_batch(model: str, batch: list[str], dim: int, input_type: str,
                 purpose: str, mission_id: Optional[str]) -> list[list[float]]:
    """One provider call for up to EMBED_BATCH texts, recorded as one
    LlmCall with the prompt tokens the provider reported (an embedding has
    no completion tokens)."""
    params = {}
    if model.startswith(_COHERE_EMBED_PREFIX):
        params = {"dimensions": dim, "input_type": input_type}

    def call():
        response = litellm.embedding(
            model=model,
            input=batch,
            # Without a timeout a hung provider holds a mission worker (and
            # its job slot) forever.
            timeout=settings.embed_timeout_s,
            # A provider that does not know a parameter gets it dropped
            # rather than the request rejected.
            drop_params=True,
            **params,
            **llm._provider_kwargs(model),
        )
        return _vectors(response, len(batch)), response

    (vectors, response), duration_ms = _call_retrying(call, model, "EMBED")
    prompt_tokens = llm._token_count(getattr(response, "usage", None), "prompt_tokens")
    llm._record(purpose, "embed", model, mission_id, (prompt_tokens, 0), duration_ms)
    return vectors


def embed_texts(texts: Iterable[str], *, model: Optional[str] = None,
                dim: Optional[int] = None, input_type: str = "search_document",
                purpose: str = "embed", mission_id: Optional[str] = None
                ) -> list[list[float]]:
    """Embed `texts` with `model` (default settings.embed_model), one vector
    per text in the given order, in batches of EMBED_BATCH. `dim` (default
    settings.embed_dim) and `input_type` go on the wire only for cohere/
    models. Raises EmbeddingDisabled when embeddings are off, the provider's
    error when a batch fails after retries."""
    if not embed_enabled():
        raise EmbeddingDisabled("EMBED_MODEL is empty")
    model = (model or settings.embed_model).strip()
    dim = int(dim or settings.embed_dim)
    texts = list(texts)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(_embed_batch(model, texts[start:start + EMBED_BATCH],
                                    dim, input_type, purpose, mission_id))
    return vectors


def embed_query(text: str, *, mission_id: Optional[str] = None) -> list[float]:
    """Embed one query for search against document vectors: the query model
    when one is configured (Embed 5 Pro and Fast share a space, so a cheaper
    model can answer queries), else the document model, as
    input_type="search_query" at the document dimensionality."""
    if not embed_enabled():
        raise EmbeddingDisabled("EMBED_MODEL is empty")
    model = (settings.embed_query_model or "").strip() or settings.embed_model
    return embed_texts([text], model=model, input_type="search_query",
                       mission_id=mission_id)[0]


def rerank(query: str, documents: Iterable[str], *, top_n: Optional[int] = None,
           model: Optional[str] = None,
           max_tokens_per_doc: int = RERANK_MAX_TOKENS_PER_DOC,
           purpose: str = "rerank", mission_id: Optional[str] = None
           ) -> list[tuple[int, float]]:
    """Rank `documents` against `query`: (index into documents, relevance
    score) pairs, best first, ties by lower index, cut to `top_n` when
    given. Recorded as one zero-token LlmCall (rerank providers report no
    usage; the row is for call counts and duration). Raises RerankDisabled
    when reranking is off, the provider's error when the call fails after
    retries."""
    model = (model or settings.rerank_model or "").strip()
    if not model:
        raise RerankDisabled("RERANK_MODEL is empty")
    documents = list(documents)

    def call():
        return litellm.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            max_tokens_per_doc=max_tokens_per_doc,
            timeout=settings.embed_timeout_s,
            **llm._provider_kwargs(model),
        )

    response, duration_ms = _call_retrying(call, model, "RERANK")
    llm._record(purpose, "rerank", model, mission_id, (0, 0), duration_ms)
    ranked = [(int(_field(r, "index")), float(_field(r, "relevance_score") or 0.0))
              for r in (getattr(response, "results", None) or [])]
    # Providers return best-first already; sorting here is what callers rely
    # on, so it does not depend on each provider keeping that promise.
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def try_rerank(query: str, documents: Iterable[str], **kw
               ) -> Optional[list[tuple[int, float]]]:
    """rerank, or None when reranking is off or the call failed. The degrade
    rule for every ranking slot: the caller keeps its existing order, and the
    trace gets one line saying why."""
    try:
        return rerank(query, documents, **kw)
    except Exception as e:  # noqa: BLE001 - ranking never fails a mission
        print(f"[RERANK] skipped ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
        return None


def vector_to_blob(vector) -> bytes:
    """Little-endian float32 bytes of a vector, the stored form."""
    return np.asarray(vector, dtype="<f4").tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    """The float32 vector stored by vector_to_blob, as a read-only view."""
    if len(blob) != int(dim) * 4:
        raise ValueError(
            f"{len(blob)} bytes is not {dim} little-endian float32 values")
    return np.frombuffer(blob, dtype="<f4", count=int(dim))


def _embed_pending(doc_ids: list[str], mission_id: Optional[str]) -> int:
    """The blocking body of embed_pending_documents: runs off the event
    loop, so the provider call does not stall the crawl, and the storage
    coroutines it needs run to completion on a loop of their own (there is
    none in this thread), the way llm._run_insert does."""
    model = settings.embed_model.strip()
    # The width a stored vector must have to count as present: the
    # configured one for cohere/ models (the only ones `dimensions` goes on
    # the wire for, so a changed EMBED_DIM re-embeds every chunk), any for
    # a provider that returns its native width.
    dim = int(settings.embed_dim) if model.startswith(_COHERE_EMBED_PREFIX) else None
    embedded = 0
    while True:
        pending = asyncio.run(storage.get_chunks_missing_embeddings(model, doc_ids, dim=dim))
        if not pending:
            return embedded
        docs = asyncio.run(storage.get_documents_by_ids(
            {chunk.document_id for chunk in pending}))
        titles = {doc.id: chunking.document_title(doc) for doc in docs}
        # A chunk whose document vanished mid-run still gets a vector (it is
        # dropped with its chunk rows); an empty title is the honest text.
        texts = [chunking.indexed_text(titles.get(chunk.document_id, ""), chunk.text)
                 for chunk in pending]
        vectors = embed_texts(texts, model=model, mission_id=mission_id)
        asyncio.run(storage.upsert_chunk_embeddings([
            (chunk.chunk_id, model, len(vector), vector_to_blob(vector))
            for chunk, vector in zip(pending, vectors)
        ]))
        embedded += len(pending)
        # The query is capped; a document set larger than one page is
        # finished on the next round, which finds only what is still missing.


async def embed_pending_documents(doc_ids: Iterable[str], *,
                                  mission_id: Optional[str] = None) -> int:
    """Embed every chunk of `doc_ids` that has no vector for the configured
    document model yet, storing one chunk_embeddings row each, and return
    how many were embedded. Chunks already embedded (an unchanged chunk of a
    re-crawled page keeps its row) cost no provider call.

    Never raises: embeddings off, or any failure, is one stderr line and 0,
    with nothing written for the failed round, so a crawl or mission goes on
    exactly as it would without embeddings."""
    if not embed_enabled():
        print("[EMBED] skipped: EMBED_MODEL is empty", file=sys.stderr, flush=True)
        return 0
    try:
        return await asyncio.to_thread(_embed_pending, list(doc_ids), mission_id)
    except Exception as e:  # noqa: BLE001 - embedding never fails a mission
        print(f"[EMBED] failed ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
        return 0
