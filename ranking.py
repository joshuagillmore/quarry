"""The ranking slots: where a reranker or the dense index reorders what the
pipeline would otherwise take in search-engine, newest-first or FTS order.

Four entry points, one per place the pipeline ranks something:

  rank_search_results  search-engine candidates before `_collect_one` caps them
  rank_documents       chunk-level rerank of collected docs; best chunk = excerpt
  rank_for_brief       the one order the brief prompt AND brief_sources_json use
  hybrid_search        the Library's full-text box: FTS ⊕ dense via RRF, reranked

THE DEGRADE RULE (every function, no exceptions)

  An empty model setting, a failed provider call, a missing index, a bug in
  here: the caller gets exactly what it would have had before this module
  existed, plus one stderr line. Nothing raises into a caller.
  `rank_search_results` returns the input order; `rank_documents` returns
  None (its callers keep their own ordering and excerpts); `rank_for_brief`
  returns `brief.ordered_sources`; `hybrid_search` returns the last stage
  that worked, FTS at worst. `embeddings.try_rerank` already prints its own
  line when the rerank itself is off or failed, so the lines printed here
  cover only what went wrong around it.

Rerank strings are built with `chunking.indexed_text(title, chunk.text)`,
the one string every embedding path builds, so what the reranker judges is
what the dense index stored. Chunks come from `chunking.document_chunks` on
the in-memory document (the same function `upsert_document` stores through),
so a document that is not in the database yet ranks like one that is.
"""
import asyncio
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import brief
import chunking
import embeddings
import storage
from config import settings
from content_quality import is_usable
from models import Document, SearchResult

# Dense arm of hybrid_search: how many chunk hits (by cosine) are collapsed
# into documents before fusion with the FTS list.
DENSE_TOP_K = 50
# The most chunk vectors the dense arm loads into memory per search, newest
# documents first (storage orders by crawled_at DESC): a library past this
# many chunks is searched densely over its most recent ones, with one
# stderr line saying so, rather than loading every vector it holds.
DENSE_MAX_CHUNKS = 20000
# How many fused documents the reranker sees; the rest keep their fused order.
RERANK_POOL = 30
# Rerank strings per rank_documents call. Round-robin over documents, so a
# long page cannot crowd a short one out of the pool.
MAX_RERANK_CHUNKS = 200
# Reciprocal rank fusion constant (Cormack et al.): a low rank in either list
# counts, a top rank in one list does not drown the other.
RRF_K = 60


@dataclass
class RankedDoc:
    """One document after a chunk-level rerank: the document, its best
    chunk's relevance score, that chunk's text cut to the caller's excerpt
    length, and the chunk's id (f"{doc.id}#{idx}", storage's chunk_id) or
    None when the excerpt is the body head rather than a scored chunk."""
    doc: Document
    score: float
    excerpt: str
    chunk_id: Optional[str]


def _log(what: str, e: BaseException) -> None:
    print(f"[RANK] {what} skipped ({type(e).__name__}: {e})",
          file=sys.stderr, flush=True)


def _body_head(doc: Document, excerpt_chars: int) -> str:
    return chunking.document_body(doc)[:max(0, int(excerpt_chars))]


# ---------------------------------------------------------------------------
# search-engine candidates
# ---------------------------------------------------------------------------

def rank_search_results(query: str, results: list[SearchResult], *,
                        mission_id: Optional[str] = None) -> list[SearchResult]:
    """Reorder search-engine results by rerank relevance of title + snippet to
    `query`, best first, so a cap taken afterwards keeps the most relevant
    candidates rather than the engine's first page.

    Degrades to the input order (a copy) when reranking is off, the call
    fails (try_rerank's one line), or anything here raises (one line). A
    result the provider left out of its ranking keeps its place after the
    ranked ones, in input order; fewer than two results are never sent."""
    results = list(results)
    if len(results) < 2:
        return results
    try:
        texts = [f"{(r.title or '').strip()}\n{(r.snippet or '').strip()}".strip()
                 for r in results]
        ranked = embeddings.try_rerank(query, texts, mission_id=mission_id)
        if ranked is None:
            return results
        order: list[int] = []
        seen: set[int] = set()
        for index, _score in ranked:
            if 0 <= index < len(results) and index not in seen:
                order.append(index)
                seen.add(index)
        order.extend(i for i in range(len(results)) if i not in seen)
        return [results[i] for i in order]
    except Exception as e:  # noqa: BLE001 - ranking never fails a mission
        _log("search-result rerank", e)
        return results


# ---------------------------------------------------------------------------
# collected documents (assessor sources, brief order, hybrid rerank pool)
# ---------------------------------------------------------------------------

def _chunk_candidates(docs: list[Document], cap: int = MAX_RERANK_CHUNKS
                      ) -> list[tuple[int, Optional[str], str, str]]:
    """(doc position, chunk_id, chunk text, rerank string) for up to `cap`
    chunks of `docs`, round-robin over documents (every doc's first chunk,
    then every doc's second, ...), so each document is represented before
    any gets a second slot. A document without chunks (no body) gets one
    title-only candidate with chunk_id None, so it is still scored."""
    per_doc: list[list[tuple[Optional[str], str, str]]] = []
    for doc in docs:
        title = chunking.document_title(doc)
        chunks = chunking.document_chunks(doc)
        if chunks:
            per_doc.append([(f"{doc.id}#{c.idx}", c.text,
                             chunking.indexed_text(title, c.text)) for c in chunks])
        else:
            per_doc.append([(None, "", chunking.indexed_text(title, ""))])
    out: list[tuple[int, Optional[str], str, str]] = []
    depth = 0
    while len(out) < cap:
        added = False
        for pos, chunks in enumerate(per_doc):
            if depth < len(chunks):
                out.append((pos, *chunks[depth]))
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        depth += 1
    return out


def rank_documents(query: str, docs: list[Document], *, top_n: Optional[int],
                   excerpt_chars: int, mission_id: Optional[str] = None
                   ) -> Optional[list[RankedDoc]]:
    """Chunk-level rerank of `docs` against `query`.

    Every usable document (content_quality.is_usable) is split with
    chunking.document_chunks and up to MAX_RERANK_CHUNKS strings
    (indexed_text(title, chunk.text), round-robin per document) go to the
    reranker in one call. Each document's best-scoring chunk becomes its
    excerpt (chunk text cut to `excerpt_chars`) and its score; documents are
    ordered by that score, best first. If fewer than `top_n` usable documents
    exist, unusable ones follow in their input order with score 0.0, the
    body head as excerpt and chunk_id None, so a short-but-valid page is
    never invisible to a caller that had room for it. `top_n` None keeps
    every document.

    Returns None — the caller keeps its own order and excerpts — when
    reranking is off or the call failed (try_rerank's one line), when there
    is no usable document to rank (no call is made), or when anything here
    raises (one line)."""
    try:
        docs = list(docs)
        usable = [d for d in docs if is_usable(d)]
        unusable = [d for d in docs if not is_usable(d)]
        if not usable:
            return None
        candidates = _chunk_candidates(usable)
        ranked = embeddings.try_rerank(
            query, [text for _p, _c, _t, text in candidates], mission_id=mission_id)
        if ranked is None:
            return None

        best: dict[int, RankedDoc] = {}
        ordered: list[RankedDoc] = []
        for index, score in ranked:
            if not 0 <= index < len(candidates):
                continue
            pos, chunk_id, chunk_text, _text = candidates[index]
            if pos in best:
                continue
            excerpt = (chunk_text if chunk_id is not None
                       else _body_head(usable[pos], excerpt_chars))
            item = RankedDoc(doc=usable[pos], score=float(score),
                             excerpt=excerpt[:max(0, int(excerpt_chars))],
                             chunk_id=chunk_id)
            best[pos] = item
            ordered.append(item)
        # A document none of whose chunks came back (a provider that cut the
        # list) is still a usable source: it follows the scored ones.
        for pos, doc in enumerate(usable):
            if pos not in best:
                ordered.append(RankedDoc(doc=doc, score=0.0,
                                         excerpt=_body_head(doc, excerpt_chars),
                                         chunk_id=None))
        limit = len(docs) if top_n is None else max(0, int(top_n))
        out = ordered[:limit]
        for doc in unusable:
            if len(out) >= limit:
                break
            out.append(RankedDoc(doc=doc, score=0.0,
                                 excerpt=_body_head(doc, excerpt_chars), chunk_id=None))
        return out
    except Exception as e:  # noqa: BLE001 - ranking never fails a mission
        _log("document rerank", e)
        return None


def rank_for_brief(mission, docs: list[Document],
                   max_docs: int = brief.MAX_BRIEF_SOURCES) -> list[Document]:
    """The one source order the brief is numbered with: both the prompt's
    [n] block and missions.brief_sources_json must come from this list, or
    the rail would cite a different document than the text.

    With reranking on: rank_documents against the mission question, usable
    documents by best-chunk relevance first, then unusable ones in input
    order, cut to `max_docs`. With reranking off, no usable document, a
    failed call, or any error here (one line): exactly
    brief.ordered_sources(docs, max_docs), today's order."""
    fallback = brief.ordered_sources(docs, max_docs)
    if not embeddings.rerank_enabled():
        return fallback
    try:
        question = str(getattr(mission, "question", "") or "")
        ranked = rank_documents(question, docs, top_n=max_docs, excerpt_chars=0,
                                mission_id=getattr(mission, "id", None))
        if ranked is None:
            return fallback
        return [r.doc for r in ranked][:max_docs]
    except Exception as e:  # noqa: BLE001 - the brief is written either way
        _log("brief source rerank", e)
        return fallback


# ---------------------------------------------------------------------------
# Library search: FTS ⊕ dense (RRF) → rerank
# ---------------------------------------------------------------------------

def _dense_doc_hits(rows: list[tuple[str, str, bytes]], qvec: list[float],
                    top_k: int = DENSE_TOP_K) -> list[tuple[str, str, float]]:
    """(document_id, chunk_id, cosine) for the documents behind the `top_k`
    best chunk vectors, best first, one entry per document (its best chunk).
    Rows whose blob is not the query's width are ignored; an unparseable one
    raises, which the caller turns into "dense unavailable"."""
    q = np.asarray(qvec, dtype=np.float32)
    dim = int(q.shape[0])
    if dim == 0:
        return []
    kept = [(cid, did, blob) for cid, did, blob in rows if len(blob) == dim * 4]
    if not kept:
        return []
    matrix = np.stack([embeddings.blob_to_vector(blob, dim) for _c, _d, blob in kept])
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    qn = float(np.linalg.norm(q)) or 1.0
    scores = (matrix @ q) / (norms * qn)
    order = np.argsort(-scores, kind="stable")[:max(1, int(top_k))]
    hits: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for i in order:
        cid, did, _blob = kept[int(i)]
        if did in seen:
            continue
        seen.add(did)
        hits.append((did, cid, float(scores[int(i)])))
    return hits


def _rrf(fts_ids: list[str], dense_ids: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion of two id lists: score = Σ 1/(k + rank) over
    the lists a document appears in; ties go to the better FTS rank, then
    the better dense rank, so the result is deterministic."""
    fts_rank = {doc_id: r for r, doc_id in enumerate(fts_ids, 1)}
    dense_rank = {doc_id: r for r, doc_id in enumerate(dense_ids, 1)}
    ids = list(fts_ids) + [d for d in dense_ids if d not in fts_rank]
    big = len(ids) + 1

    def key(doc_id: str):
        score = 0.0
        if doc_id in fts_rank:
            score += 1.0 / (k + fts_rank[doc_id])
        if doc_id in dense_rank:
            score += 1.0 / (k + dense_rank[doc_id])
        return (-score, fts_rank.get(doc_id, big), dense_rank.get(doc_id, big))

    return sorted(ids, key=key)


async def _dense_stage(query: str, search_filter: Optional[str],
                       fts_docs: list[Document], limit: int) -> Optional[list[Document]]:
    """FTS ⊕ dense. None when there is no dense index to fuse with (no
    stored vectors for the configured model within the filter), so the
    caller keeps the FTS mode. Raises on any failure; the caller degrades."""
    model = (settings.embed_model or "").strip()
    ids: Optional[list[str]] = None
    if search_filter:
        ids = [d.id for d in await storage.get_documents_by_search(
            search_filter, preview_chars=0)]
        if not ids:
            return None
    # Is there anything to fuse with at all? One row is enough to know, and
    # it saves the query embedding when nothing is embedded yet.
    if not await storage.get_chunk_embeddings(model, ids, limit=1):
        return None
    qvec = await asyncio.to_thread(embeddings.embed_query, query)
    # Only vectors of the query's width can match it, and at most
    # DENSE_MAX_CHUNKS of them (the most recent), so the arm is bounded.
    cap = max(1, int(DENSE_MAX_CHUNKS))
    rows = await storage.get_chunk_embeddings(model, ids, dim=len(qvec), limit=cap)
    if not rows:
        return None
    if len(rows) == cap:
        print(f"[RANK] dense arm capped at {cap} most recent chunks",
              file=sys.stderr, flush=True)
    hits = _dense_doc_hits(rows, qvec)
    if not hits:
        return None
    fts_ids = [d.id for d in fts_docs]
    fused = _rrf(fts_ids, [doc_id for doc_id, _cid, _s in hits])[:limit]
    by_id = {d.id: d for d in fts_docs}
    missing = [doc_id for doc_id in fused if doc_id not in by_id]
    if missing:
        for doc in await storage.get_documents_by_ids(missing):
            by_id[doc.id] = doc
    # An id whose document is gone (deleted since it was embedded) is dropped.
    return [by_id[doc_id] for doc_id in fused if doc_id in by_id]


async def hybrid_search(query: str, search_filter: Optional[str], *,
                        limit: int = 100) -> tuple[list[Document], str]:
    """Library full-text search in up to three stages, returning (documents,
    mode) where mode names the last stage that worked:

      "fts"            storage.search_documents_fts(query, search_filter)
      "hybrid"         + dense: cosine over the stored chunk vectors of the
                       configured embed model and the query's width (within
                       the search filter; at most DENSE_MAX_CHUNKS, newest
                       documents first, one stderr line when that cap binds),
                       top DENSE_TOP_K chunks collapsed to documents, fused
                       with the FTS list by reciprocal rank fusion (k =
                       RRF_K), cut to `limit`; dense-only hits get their
                       bodies loaded
      "<mode>+rerank"  + the top RERANK_POOL documents reranked by their
                       best chunk (rank_documents); the rest keep their order

    The dense stage runs only when embeddings are on and a vector exists to
    fuse with; the rerank stage only when reranking is on. A stage that
    fails leaves the previous stage's result and mode and prints one stderr
    line; the FTS stage failing gives ([], "fts"). Never raises."""
    limit = max(0, int(limit))
    try:
        docs = await storage.search_documents_fts(query, search_filter)
    except Exception as e:  # noqa: BLE001 - the Library page always renders
        _log("full-text search", e)
        docs = []
    docs = docs[:limit]
    mode = "fts"

    if embeddings.embed_enabled():
        try:
            fused = await _dense_stage(query, search_filter, docs, limit)
            if fused is not None:
                docs, mode = fused, "hybrid"
        except Exception as e:  # noqa: BLE001 - keep the FTS result
            _log("dense search", e)

    if embeddings.rerank_enabled() and docs:
        try:
            pool, rest = docs[:RERANK_POOL], docs[RERANK_POOL:]
            ranked = await asyncio.to_thread(
                rank_documents, query, pool, top_n=len(pool), excerpt_chars=0)
            if ranked is not None:
                docs, mode = [r.doc for r in ranked] + rest, mode + "+rerank"
        except Exception as e:  # noqa: BLE001 - keep the fused result
            _log("search rerank", e)

    return docs, mode
