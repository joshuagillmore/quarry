"""Export stored documents into a Qdrant collection shaped for Coherence.

One point per chunk, dense vector only (a named vector "dense", size
settings.embed_dim, cosine), payload per the WP6 mapping table so that
Coherence's ACL filter (clearance_rank / compartments / releasability) and
its citation resolver (chunk_id / doc_id) read it as one of their own
corpora. The vector of a chunk is the embedding of
``chunking.indexed_text(title, chunk.text)`` — the same string the in-app
index embeds — so a vector already stored in ``chunk_embeddings`` for
``settings.embed_model`` is reused as is and never re-bought.

WHAT THE EXPORT PROMISES

  Idempotent on content. A document whose ``document_exports`` row carries
  the same ``chunking.document_digest`` and embedding model is skipped;
  a changed one has its old points deleted by a ``doc_id`` filter and its
  new points upserted, so a re-crawl never leaves a stale chunk behind.
  ``--force`` re-writes everything.

  Never destructive beyond its own documents. ``ensure_collection`` creates
  the collection when missing and refuses (``ExportError``) when an existing
  one has a differently sized dense vector; it never calls
  ``delete_collection``. Points this module did not write (anything without
  a ``QRY-`` doc_id, Coherence's own product points) are never touched.

  Fails loudly before writing, quietly per document after. An unreachable
  Qdrant, an empty EMBED_MODEL or a bad collection name raise ``ExportError``
  before any point or export row is written. Once writing, one document's
  failure (an embedding call, a refused batch) is captured in
  ``ExportResult.failed`` and the rest go on; nothing here raises into a
  mission.

  Reports on the corpus it wrote. With ``check=True`` and a mission id the
  export runs ``corpus_check.run`` against the populated collection and
  attaches the report, so "exported" also says whether the corpus retrieves
  what the mission found.

``qdrant_client`` is imported inside the functions that need it, so this
module imports (and ``export_status`` answers) without the package installed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import traceback
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Optional
from urllib.parse import urlsplit

import chunking
import embeddings
import storage
from config import csv_list, export_dir, settings
from models import Document, DocumentExport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from corpus_check import CheckReport


class ExportError(RuntimeError):
    """The export could not start, or the collection cannot take these
    vectors. Raised before any point or export row is written."""


# Coherence document ids are prefixed per corpus; QRY- is Quarry's. Never
# `canary-`, `contra-` or `OSINT:`, which Coherence treats specially.
QRY_PREFIX = "QRY-"
UPSERT_BATCH = 400
VECTOR_NAME = "dense"
EXPORTED_BY = "quarry/vector_export"

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Payload fields Coherence filters or looks points up by.
_PAYLOAD_INDEXES = (
    ("clearance_rank", "integer"),
    ("compartments", "keyword"),
    ("releasability", "keyword"),
    ("doc_id", "keyword"),
    ("cluster", "keyword"),
)


@dataclass
class Markings:
    """The classification markings stamped on every exported point, in the
    shape Coherence's ACL filter reads (an int rank, list caveats)."""
    clearance_rank: int = 0
    classification: str = "NATO UNCLASSIFIED"
    compartments: list[str] = field(default_factory=list)
    releasability: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls) -> "Markings":
        return cls(
            clearance_rank=int(settings.export_clearance_rank),
            classification=(settings.export_classification or "").strip()
            or "NATO UNCLASSIFIED",
            compartments=csv_list(settings.export_compartments),
            releasability=csv_list(settings.export_releasability),
        )


@dataclass
class ExportResult:
    """What one export did. `exported`/`skipped` are document ids; `failed`
    maps a document id to why it was not written; `chunks` counts the
    points upserted."""
    collection: str
    exported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0
    stats_path: Optional[str] = None
    manifest_path: Optional[str] = None
    check: Optional["CheckReport"] = None
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["check"] = asdict(self.check) if self.check is not None else None
        return out

    def summary(self) -> str:
        line = (f"exported {len(self.exported)} · skipped {len(self.skipped)}"
                f" · failed {len(self.failed)} · {self.chunks} chunks"
                f" → '{self.collection}'")
        if self.check is not None:
            if self.check.error:
                line += f" · check failed: {self.check.error}"
            else:
                line += (f" · check recall@{self.check.k_effective}"
                         f" {self.check.recall_at_k:.2f} mrr {self.check.mrr:.2f}"
                         f" over {self.check.n_measured} measured queries"
                         f" ({'pass' if self.check.passed else 'FAIL'})")
        return line


# --- ids and payload ------------------------------------------------------------------

def coherence_doc_id(document_id: str) -> str:
    return f"{QRY_PREFIX}{document_id}"


def coherence_chunk_id(document_id: str, idx: int) -> str:
    return f"{coherence_doc_id(document_id)}#{int(idx)}"


def point_id(chunk_id: str) -> str:
    """Qdrant wants an int or a UUID; a uuid5 of the chunk id is stable, so
    re-exporting the same chunk overwrites its point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _parse_utc(iso_utc: str) -> Optional[datetime]:
    text = (iso_utc or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def date_time_group(iso_utc: str) -> str:
    """Military DTG of an ISO-8601 instant: "110648Z AUG 26" (day, HHMM,
    Zulu, month, two-digit year). "" when the input does not parse."""
    dt = _parse_utc(iso_utc)
    if dt is None:
        return ""
    return f"{dt:%d%H%M}Z {dt:%b}".upper() + f" {dt:%y}"


def _published_date(doc: Document) -> str:
    """metadata_json["published_date"] when it is a YYYY-MM-DD, else the
    crawl date."""
    try:
        meta = json.loads(doc.metadata_json) if doc.metadata_json else {}
    except (TypeError, ValueError):
        meta = {}
    published = meta.get("published_date") if isinstance(meta, dict) else None
    if isinstance(published, str) and _DATE_RE.match(published.strip()):
        return published.strip()
    dt = _parse_utc(doc.crawled_at)
    if dt is not None:
        return dt.strftime("%Y-%m-%d")
    text = (doc.crawled_at or "")[:10]
    return text if _DATE_RE.match(text) else ""


def _hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def build_payload(doc: Document, chunk, markings: Markings, digest: str,
                  mission_ids: Iterable[str] = (),
                  model: Optional[str] = None, dim: Optional[int] = None) -> dict:
    """The point payload for one chunk of `doc`, per the WP6 mapping table.
    `chunk` is a storage ChunkRow or a chunking.Chunk (idx, section, text);
    its text is stored section-prefixed and without the title."""
    title = chunking.document_title(doc)
    return {
        "chunk_id": coherence_chunk_id(doc.id, chunk.idx),
        "doc_id": coherence_doc_id(doc.id),
        "title": title,
        "text": chunk.text,
        "section": chunk.section,
        "kind": "body",
        "clearance_rank": int(markings.clearance_rank),
        "classification": markings.classification,
        "compartments": list(markings.compartments),
        "releasability": list(markings.releasability),
        "int_type": "OSINT",
        "trust_tier": "EXTERNAL_UNVERIFIED",
        "originator": _hostname(doc.url) or doc.domain,
        "source_designator": doc.url,
        "source_url": doc.url,
        "date_time_group": date_time_group(doc.crawled_at),
        "published_date": _published_date(doc),
        "source_reliability": "C",
        "info_credibility": 3,
        "provenance": {"url": doc.url, "fetched_at": doc.crawled_at, "sha256": digest},
        "doc_type": "web_page",
        "cluster": doc.search_query,
        "topics": [],
        "decoy_for": [],
        "embedding": {
            "model": model if model is not None else settings.embed_model,
            "dim": int(dim if dim is not None else settings.embed_dim),
            "input_type": "search_document",
        },
        "quarry": {
            "document_id": doc.id,
            "search_query": doc.search_query,
            "domain": doc.domain,
            "word_count": int(doc.word_count or 0),
            "crawled_at": doc.crawled_at,
            "mission_ids": list(mission_ids),
            "exported_by": EXPORTED_BY,
        },
    }


# --- Qdrant -------------------------------------------------------------------------

def make_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """A QdrantClient for settings.qdrant_url (or `url`). The api key goes
    on the wire only when one is set; the server version check is off, so a
    Qdrant a minor version apart from the client still answers."""
    from qdrant_client import QdrantClient
    key = (api_key if api_key is not None else settings.qdrant_api_key or "").strip()
    kwargs = {"url": url or settings.qdrant_url, "timeout": 30,
              "check_compatibility": False}
    if key:
        kwargs["api_key"] = key
    return QdrantClient(**kwargs)


def _dense_size(info) -> Optional[int]:
    """The size of an existing collection's "dense" vector, None when it
    has no vector of that name (an unnamed single vector counts as none)."""
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        params = vectors.get(VECTOR_NAME)
        return int(params.size) if params is not None else None
    return None


def ensure_collection(client, collection: str, dim: int) -> bool:
    """Create `collection` with a named dense vector of `dim` when it does
    not exist, and make sure the payload indexes Coherence filters on are
    there. Returns True when the collection was created here (so the caller
    knows it is empty), False when it already existed. An existing
    collection whose dense vector is not `dim` wide (or that has no "dense"
    vector) is refused with ExportError. Nothing is ever deleted here: a
    wrong collection is the operator's to fix."""
    from qdrant_client import models as qm

    if not _COLLECTION_RE.match(collection or ""):
        raise ExportError(f"invalid collection name {collection!r}")
    dim = int(dim)
    if dim <= 0:
        raise ExportError(f"invalid vector size {dim}")
    if client.collection_exists(collection):
        info = client.get_collection(collection)
        have = _dense_size(info)
        if have != dim:
            what = "no 'dense' vector" if have is None else f"dense size {have}"
            raise ExportError(
                f"collection {collection!r} exists with {what}, not {dim}; "
                f"refusing to touch it (never deleted)")
        existing = set((getattr(info, "payload_schema", None) or {}).keys())
        created = False
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config={VECTOR_NAME: qm.VectorParams(
                size=dim, distance=qm.Distance.COSINE)},
        )
        existing = set()
        created = True
    schema = {"integer": qm.PayloadSchemaType.INTEGER,
              "keyword": qm.PayloadSchemaType.KEYWORD}
    with warnings.catch_warnings():
        # The in-memory client warns that it has no payload indexes; the
        # export does not depend on them, Coherence's filters only go faster.
        warnings.simplefilter("ignore", UserWarning)
        for name, kind in _PAYLOAD_INDEXES:
            if name in existing:
                continue
            client.create_payload_index(
                collection_name=collection, field_name=name,
                field_schema=schema[kind], wait=True)
    return created


_USERINFO_RE = re.compile(r"(?i)\b(https?://)[^/\s@]*@")


def _redact_userinfo(text: str) -> str:
    """`http://user:pass@host:6333` -> `http://host:6333`, for every URL in
    `text`: a message that names the Qdrant URL (an error, a status field,
    a trace line) must not carry its credentials."""
    return _USERINFO_RE.sub(r"\1", text or "")


def _probe(client) -> None:
    """One round-trip that proves Qdrant answers; anything else is an
    ExportError before a single write. The message names the URL without
    its userinfo, and the provider's own text is redacted the same way."""
    try:
        client.get_collections()
    except Exception as e:  # noqa: BLE001 - the exact class is qdrant's business
        raise ExportError(_redact_userinfo(
            f"Qdrant unreachable at {settings.qdrant_url}: "
            f"{type(e).__name__}: {e}")) from e


def _delete_doc_points(client, collection: str, doc_ids: list[str]) -> None:
    from qdrant_client import models as qm
    if not doc_ids:
        return
    client.delete(
        collection_name=collection,
        points_selector=qm.Filter(must=[qm.FieldCondition(
            key="doc_id", match=qm.MatchAny(any=doc_ids))]),
        wait=True,
    )


def _upsert_points(client, collection: str, points: list) -> None:
    client.upsert(collection_name=collection, points=points, wait=True)


# --- the export ---------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _say(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        try:
            log(message)
            return
        except Exception:  # noqa: BLE001 - logging never fails the export
            pass
    print(f"[EXPORT] {message}", file=sys.stderr, flush=True)


def _export_target(doc: Document, row: Optional[DocumentExport], digest: str,
                   model: str, force: bool) -> bool:
    """True when `doc` must be (re)written: no export row, a changed
    digest, a different embedding model, or `force`."""
    if force or row is None:
        return True
    return not (row.content_sha256 == digest and row.embed_model == model)


async def _chunks_for(doc: Document) -> list:
    """The stored chunks of `doc`, rebuilt from its body when a document
    stored before chunking existed has none."""
    rows = await storage.get_document_chunks(doc.id)
    if rows:
        return rows
    await storage.replace_document_chunks(doc.id, chunking.document_chunks(doc))
    return await storage.get_document_chunks(doc.id)


async def _vectors_for(doc: Document, chunks: list, model: str, dim: int,
                       mission_id: Optional[str]) -> dict[str, list[float]]:
    """chunk_id -> vector for every chunk of `doc`: stored chunk_embeddings
    rows of the right model and width first, the rest embedded now (one
    provider call per 96 texts) and stored back for the next export."""
    have: dict[str, list[float]] = {}
    for chunk_id, _doc_id, blob in await storage.get_chunk_embeddings(model, [doc.id]):
        if len(blob) == dim * 4:
            have[chunk_id] = embeddings.blob_to_vector(blob, dim).tolist()
    missing = [c for c in chunks if c.chunk_id not in have]
    if missing:
        title = chunking.document_title(doc)
        texts = [chunking.indexed_text(title, c.text) for c in missing]
        vectors = await asyncio.to_thread(
            embeddings.embed_texts, texts, model=model, dim=dim,
            input_type="search_document", purpose="embed", mission_id=mission_id)
        rows = []
        for chunk, vector in zip(missing, vectors):
            if len(vector) != dim:
                raise ExportError(
                    f"{model} returned {len(vector)} dims for {chunk.chunk_id}, "
                    f"the collection takes {dim}")
            have[chunk.chunk_id] = [float(x) for x in vector]
            rows.append((chunk.chunk_id, model, dim,
                         embeddings.vector_to_blob(vector)))
        await storage.upsert_chunk_embeddings(rows)
    return have


def _write_json_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def export_documents(doc_ids: Iterable[str], *, collection: Optional[str] = None,
                           force: bool = False, client=None,
                           mission_id: Optional[str] = None,
                           log: Optional[Callable[[str], None]] = None,
                           check: bool = True) -> ExportResult:
    """Export the given documents into `collection` (default
    settings.qdrant_export_collection). See the module docstring for what is
    skipped, deleted and written. `client` is injected by tests
    (QdrantClient(":memory:")); `log` receives one line per event, else
    stderr. Raises ExportError only before the first write; per-document
    problems land in the result's `failed`."""
    from qdrant_client import models as qm

    started = _now()
    collection = (collection or settings.qdrant_export_collection or "").strip()
    if not _COLLECTION_RE.match(collection):
        raise ExportError(f"invalid collection name {collection!r}")
    model = (settings.embed_model or "").strip()
    if not model:
        raise ExportError("EMBED_MODEL is empty: nothing to export vectors with")
    dim = int(settings.embed_dim)
    result = ExportResult(collection=collection, started_at=started)

    if client is None:
        try:
            client = make_client()
        except Exception as e:  # noqa: BLE001 - missing package, bad url
            raise ExportError(f"cannot create Qdrant client: {type(e).__name__}: {e}") from e
    await asyncio.to_thread(_probe, client)
    created = await asyncio.to_thread(ensure_collection, client, collection, dim)
    if created:
        # A collection that had to be created holds nothing: whatever
        # document_exports says this collection received (a run before it
        # was wiped and recreated) is stale, so clear it before deciding
        # what is "unchanged", or those documents are skipped forever.
        stale = await storage.list_exported_document_ids(collection)
        if stale:
            await storage.delete_document_exports(stale, collection)
            _say(log, f"collection '{collection}' created empty: "
                      f"{len(stale)} export row(s) cleared, re-exporting")

    ids = list(dict.fromkeys(doc_ids))
    docs = await storage.get_documents_by_ids(ids)
    exports = await storage.get_document_exports([d.id for d in docs], collection)
    mission_map = await storage.get_document_mission_ids([d.id for d in docs])
    markings = Markings.from_settings()
    manifest_docs: list[dict] = []

    # Phase 1: decide, chunk and embed. A document that fails here is never
    # deleted from the collection: its old points (if any) stay valid.
    prepared: list[tuple[Document, str, list]] = []   # (doc, digest, points)
    for doc in docs:
        entry = {"document_id": doc.id, "doc_id": coherence_doc_id(doc.id),
                 "url": doc.url, "title": chunking.document_title(doc)}
        if not chunking.document_body(doc).strip():
            result.skipped.append(doc.id)
            manifest_docs.append({**entry, "status": "skipped", "reason": "empty body",
                                  "n_chunks": 0})
            continue
        digest = chunking.document_digest(doc)
        entry["sha256"] = digest
        if not _export_target(doc, exports.get(doc.id), digest, model, force):
            result.skipped.append(doc.id)
            manifest_docs.append({**entry, "status": "skipped", "reason": "unchanged",
                                  "n_chunks": exports[doc.id].n_chunks})
            continue
        try:
            chunks = await _chunks_for(doc)
            vectors = await _vectors_for(doc, chunks, model, dim, mission_id)
            points = [
                qm.PointStruct(
                    id=point_id(coherence_chunk_id(doc.id, c.idx)),
                    vector={VECTOR_NAME: vectors[c.chunk_id]},
                    payload=build_payload(doc, c, markings, digest,
                                          mission_ids=mission_map.get(doc.id, []),
                                          model=model, dim=dim),
                )
                for c in chunks
            ]
        except Exception as e:  # noqa: BLE001 - one document, not the export
            reason = f"{type(e).__name__}: {e}"
            result.failed[doc.id] = reason
            manifest_docs.append({**entry, "status": "failed", "error": reason,
                                  "n_chunks": 0})
            _say(log, f"export failed for {doc.id}: {reason}")
            continue
        prepared.append((doc, digest, points))

    # Phase 2: delete the changed documents' old points by filter, then
    # upsert the new ones in batches. A refused batch fails every document
    # with a point in it; those get no export row, so the next run redoes
    # them.
    if prepared:
        try:
            await asyncio.to_thread(
                _delete_doc_points, client, collection,
                [coherence_doc_id(doc.id) for doc, _d, _p in prepared])
            # The points are gone: the export row must not outlive them, or
            # a refused upsert below leaves a row saying "exported" for a
            # document the collection no longer holds (a forced run would
            # then be skipped as unchanged next time). The row comes back
            # once the upsert succeeds.
            await storage.delete_document_exports(
                [doc.id for doc, _d, _p in prepared], collection)
        except Exception as e:  # noqa: BLE001
            reason = f"delete failed: {type(e).__name__}: {e}"
            for doc, _digest, _points in prepared:
                result.failed[doc.id] = reason
                manifest_docs.append({"document_id": doc.id,
                                      "doc_id": coherence_doc_id(doc.id),
                                      "url": doc.url, "status": "failed",
                                      "error": reason, "n_chunks": 0})
            prepared = []
            _say(log, reason)

    all_points: list[tuple[str, object]] = [
        (doc.id, p) for doc, _d, points in prepared for p in points]
    failed_docs: set[str] = set()
    for start in range(0, len(all_points), UPSERT_BATCH):
        batch = all_points[start:start + UPSERT_BATCH]
        try:
            await asyncio.to_thread(_upsert_points, client, collection,
                                    [p for _doc_id, p in batch])
            result.chunks += len(batch)
        except Exception as e:  # noqa: BLE001
            reason = f"upsert failed: {type(e).__name__}: {e}"
            for doc_id, _p in batch:
                failed_docs.add(doc_id)
                result.failed[doc_id] = reason
            _say(log, reason)

    for doc, digest, points in prepared:
        entry = {"document_id": doc.id, "doc_id": coherence_doc_id(doc.id),
                 "url": doc.url, "title": chunking.document_title(doc),
                 "sha256": digest, "n_chunks": len(points)}
        if doc.id in failed_docs:
            manifest_docs.append({**entry, "status": "failed",
                                  "error": result.failed[doc.id]})
            continue
        await storage.upsert_document_export(DocumentExport(
            document_id=doc.id, collection=collection, content_sha256=digest,
            embed_model=model, n_chunks=len(points)))
        result.exported.append(doc.id)
        manifest_docs.append({**entry, "status": "exported"})

    result.finished_at = _now()
    manifest = {
        "collection": collection,
        "exported_at": result.finished_at,
        "embed_model": model,
        "dim": dim,
        "vectors": [VECTOR_NAME],
        "n_docs": len(result.exported),
        "n_chunks": result.chunks,
        "skipped": len(result.skipped),
        "failed": len(result.failed),
        "mission_id": mission_id,
        "markings": asdict(markings),
        "docs": manifest_docs,
    }
    manifest_path = os.path.join(export_dir(), f"manifest-{collection}.json")
    try:
        _write_json_atomic(manifest_path, manifest)
        result.manifest_path = manifest_path
    except OSError as e:
        _say(log, f"manifest not written ({type(e).__name__}: {e})")
    _say(log, result.summary())

    if check and mission_id:
        import corpus_check
        result.check = await corpus_check.run(mission_id, collection=collection,
                                              client=client)
        result.finished_at = _now()
        _say(log, result.summary())
    return result


# --- background thread + status ---------------------------------------------------------

# Held for the whole of a background export: one at a time, the way the job
# store bounds crawls. Status reads take _state_lock, never this one.
_export_lock = threading.Lock()
_state_lock = threading.Lock()
_state: dict = {
    "running": False, "collection": None, "mission_id": None, "doc_count": 0,
    "started_at": None, "finished_at": None, "result": None, "error": None,
}


def _set_state(**fields) -> None:
    with _state_lock:
        _state.update(fields)


def _run_export_thread(doc_ids: list[str], collection: Optional[str],
                       mission_id: Optional[str]) -> None:
    """Thread body, mirroring jobs._run_thread: whatever happens, the lock
    is released and the status says how it ended."""
    try:
        result = asyncio.run(export_documents(
            doc_ids, collection=collection, mission_id=mission_id))
        _set_state(running=False, finished_at=_now(), result=result.to_dict(),
                   error=None)
    except BaseException as e:  # noqa: BLE001 - the lock must be released
        traceback.print_exc()
        _set_state(running=False, finished_at=_now(),
                   error=f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
    finally:
        _export_lock.release()


def start_export_in_background(doc_ids: Iterable[str], *, collection: Optional[str] = None,
                               mission_id: Optional[str] = None) -> bool:
    """Run export_documents on a daemon thread. False (and nothing started)
    when an export is already running; True once the thread is off."""
    ids = list(dict.fromkeys(doc_ids))
    if not _export_lock.acquire(blocking=False):
        return False
    collection = (collection or settings.qdrant_export_collection or "").strip()
    _set_state(running=True, collection=collection, mission_id=mission_id,
               doc_count=len(ids), started_at=_now(), finished_at=None,
               result=None, error=None)
    try:
        threading.Thread(target=_run_export_thread, args=(ids, collection, mission_id),
                         daemon=True, name="vector-export").start()
    except BaseException:
        _set_state(running=False, error="could not start export thread")
        _export_lock.release()
        raise
    return True


def run_export_locked(doc_ids: Iterable[str], *, collection: Optional[str] = None,
                      mission_id: Optional[str] = None) -> Optional[ExportResult]:
    """export_documents under the same one-at-a-time lock and status as the
    background thread, run to completion in the calling thread (its own
    event loop via asyncio.run, so call it off any running loop, e.g.
    through asyncio.to_thread). None, with one stderr line, when an export
    is already running — the caller logs the skip; nothing is queued. An
    error is recorded in the status and re-raised; whatever happens the
    lock is released and the status says the run is over."""
    ids = list(dict.fromkeys(doc_ids))
    if not _export_lock.acquire(blocking=False):
        print("[EXPORT] skipped: another export is running", file=sys.stderr, flush=True)
        return None
    collection = (collection or settings.qdrant_export_collection or "").strip()
    _set_state(running=True, collection=collection, mission_id=mission_id,
               doc_count=len(ids), started_at=_now(), finished_at=None,
               result=None, error=None)
    try:
        result = asyncio.run(export_documents(
            ids, collection=collection, mission_id=mission_id))
        _set_state(running=False, finished_at=_now(), result=result.to_dict(),
                   error=None)
        return result
    except BaseException as e:  # noqa: BLE001 - the status must say how it ended
        _set_state(running=False, finished_at=_now(),
                   error=f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
        raise
    finally:
        _export_lock.release()


def export_status() -> dict:
    """The last/current background export: running flag, what it covers,
    when, its ExportResult (dict, with the check report) or its error."""
    with _state_lock:
        return json.loads(json.dumps(_state))


# --- CLI ------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vector_export",
        description="Export Quarry documents into a Coherence-shaped Qdrant collection.")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--mission", metavar="ID", help="every document a mission collected")
    what.add_argument("--search", metavar="QUERY", help="every document stored under a search query")
    what.add_argument("--doc", metavar="ID", nargs="+", help="specific document ids")
    what.add_argument("--all", action="store_true", help="the whole library")
    parser.add_argument("--collection", metavar="NAME",
                        help=f"target collection (default: {settings.qdrant_export_collection})")
    parser.add_argument("--force", action="store_true",
                        help="re-write documents whose export row is up to date")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be exported; write nothing, contact nothing")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the corpus self-test after exporting")
    return parser


async def _resolve_doc_ids(args) -> list[str]:
    if args.mission:
        if await storage.get_mission(args.mission) is None:
            raise ExportError(f"no mission {args.mission!r}")
        return [d.id for d in await storage.get_mission_documents(args.mission)]
    if args.search:
        return [d.id for d in await storage.get_documents_by_search(args.search)]
    if args.doc:
        return list(dict.fromkeys(args.doc))
    return [d.id for d in await storage.get_all_documents()]


async def _dry_run(doc_ids: list[str], collection: str, force: bool) -> int:
    model = (settings.embed_model or "").strip()
    docs = await storage.get_documents_by_ids(doc_ids)
    exports = await storage.get_document_exports([d.id for d in docs], collection)
    would, skip = 0, 0
    for doc in docs:
        if not chunking.document_body(doc).strip():
            verdict = "skip (empty body)"
            skip += 1
        elif _export_target(doc, exports.get(doc.id), chunking.document_digest(doc),
                            model, force):
            verdict = "export"
            would += 1
        else:
            verdict = "skip (unchanged)"
            skip += 1
        print(f"{verdict:22} {doc.id}  {doc.url}")
    print(f"dry run: {would} to export, {skip} to skip → '{collection}' "
          f"(model {model or 'NONE'}, dim {settings.embed_dim}); nothing written")
    return 0


async def _missions_touching(doc_ids: Iterable[str]) -> list[str]:
    """Ids of every mission that linked any of `doc_ids`, in mission order."""
    wanted = set(doc_ids)
    out = []
    for mission in await storage.list_missions(limit=10_000):
        req_map = await storage.get_requirement_document_map(mission.id)
        if any(doc_id in wanted for ids in req_map.values() for doc_id in ids):
            out.append(mission.id)
    return out


async def _cli(args) -> int:
    await storage.init_db()
    collection = (args.collection or settings.qdrant_export_collection or "").strip()
    doc_ids = await _resolve_doc_ids(args)
    if not doc_ids:
        print("no documents matched; nothing to export", file=sys.stderr)
        return 0
    if args.dry_run:
        return await _dry_run(doc_ids, collection, args.force)

    import corpus_check
    run_check = not args.no_check
    result = await export_documents(
        doc_ids, collection=collection, force=args.force,
        mission_id=args.mission or None, log=lambda m: print(m, file=sys.stderr),
        # --mission checks itself; --all checks every mission it touched below.
        check=run_check and bool(args.mission))
    for doc_id, reason in result.failed.items():
        print(f"failed {doc_id}: {reason}", file=sys.stderr)
    ok = not result.failed
    reports = [result.check] if result.check is not None else []
    if run_check and args.all:
        for mission_id in await _missions_touching(doc_ids):
            reports.append(await corpus_check.run(mission_id, collection=collection))
    for report in reports:
        if report.error:
            print(f"check {report.mission_id}: error {report.error}")
            ok = False
            continue
        print(f"check {report.mission_id}: recall@{report.k_effective} "
              f"{report.recall_at_k:.2f} mrr {report.mrr:.2f} "
              f"over {report.n_measured} measured queries "
              f"({'pass' if report.passed else 'FAIL'}) → {report.eval_dir}")
        ok = ok and report.passed
    print(result.summary())
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_cli(args))
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
