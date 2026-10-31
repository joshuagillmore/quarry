"""The corpus self-test: did the collection we just wrote retrieve what the
mission found?

After an export, each mission query is embedded (search_query side) and
run against the collection; the chunk hits are collapsed to documents and
scored against the mission's own record of which documents answered which
requirement. The result is a regression tripwire, not ground truth: the
grades come from Quarry's assessor, so the number says "the corpus
retrieves what the mission used", nothing more. It is labelled that way in
the report.

WHAT IS JUDGED

  ``Q00`` is the mission question. ``Q01..Qnn`` are the requirements, one
  each, in plan order, phrased as the first of the requirement's
  ``next_queries_json`` (the query the collector actually ran) or its title.
  A document linked to a requirement is graded 2 when that requirement was
  satisfied, 1 when it was linked but the requirement stayed unmet, and a
  document linked only to other requirements of the same mission is graded
  0 for this query (an explicit non-relevant row, as Coherence's qrels
  carry). ``Q00`` takes every linked document at its best grade.

  recall@k counts the grade >= 1 documents in the top k; the reciprocal
  rank is of the first grade >= 2 document. Each mean is over the queries
  that have something to measure (a requirement with no sources contributes
  nothing to recall, one with no satisfied source nothing to MRR).

The eval files are written in Coherence's ``evals/graded.py`` shape --
``queries.yaml`` declaring its ``collection``, ``qrels.jsonl`` rows of
``query_id/doc_id/grade`` -- so the same judgments can be re-run there.
``queries.yaml`` is written by hand (JSON-escaped scalars are valid YAML);
there is no YAML dependency.

``run`` never raises into an export: any failure is one stderr line and a
report with ``passed=False`` and ``error`` set.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import embeddings
import storage
from config import export_dir, settings
from models import Mission, Requirement
from vector_export import VECTOR_NAME, coherence_doc_id

RELEVANT_GRADE = 1   # counts for recall
GOLD_GRADE = 2       # counts for reciprocal rank
# Chunk hits collapse to documents; fetching more than k hits is what lets
# k distinct documents surface when one document owns several top chunks.
HIT_MULTIPLIER = 4


@dataclass
class QueryCheck:
    query_id: str
    text: str
    expected: list[str]        # QRY- doc ids graded >= 1
    hits: list[str]            # QRY- doc ids retrieved, best first, top k
    recall_at_k: float
    rr: float


@dataclass
class CheckReport:
    collection: str
    mission_id: str
    k: int
    queries: list[QueryCheck] = field(default_factory=list)
    recall_at_k: float = 0.0
    mrr: float = 0.0
    passed: bool = False
    eval_dir: str = ""
    error: Optional[str] = None


# --- the eval set ----------------------------------------------------------------------

def _first_query(req: Requirement) -> str:
    try:
        queries = json.loads(req.next_queries_json) if req.next_queries_json else []
    except (TypeError, ValueError):
        queries = []
    for q in queries if isinstance(queries, list) else []:
        if isinstance(q, str) and q.strip():
            return q.strip()
    return (req.title or "").strip()


def build_eval_set(mission: Mission, requirements: list[Requirement],
                   req_doc_map: dict[str, list[str]],
                   collection: Optional[str] = None) -> tuple[dict, list[dict]]:
    """(queries.yaml-shaped dict, qrels rows) for a mission. See the module
    docstring for the grading rule."""
    collection = collection or settings.qdrant_export_collection
    queries = [{"query_id": "Q00", "topic": "mission", "intent_type": "synthesis",
                "text": (mission.question or "").strip()}]
    qrels: list[dict] = []
    reqs = list(requirements)
    grade_of = {req.id: (GOLD_GRADE if req.status == "satisfied" else RELEVANT_GRADE)
                for req in reqs}
    linked = {req.id: list(dict.fromkeys(req_doc_map.get(req.id, []))) for req in reqs}

    best: dict[str, int] = {}
    for req in reqs:
        for doc_id in linked[req.id]:
            best[doc_id] = max(best.get(doc_id, 0), grade_of[req.id])
    for doc_id, grade in best.items():
        qrels.append({"query_id": "Q00", "doc_id": coherence_doc_id(doc_id),
                      "grade": grade})

    for n, req in enumerate(reqs, start=1):
        query_id = f"Q{n:02d}"
        queries.append({"query_id": query_id, "topic": (req.title or "").strip(),
                        "intent_type": "lookup", "text": _first_query(req)})
        own = set(linked[req.id])
        for doc_id in linked[req.id]:
            qrels.append({"query_id": query_id, "doc_id": coherence_doc_id(doc_id),
                          "grade": grade_of[req.id]})
        # Grade-0 rows only for documents the same mission linked elsewhere,
        # so the file names what the corpus holds without listing it all.
        for doc_id in best:
            if doc_id not in own:
                qrels.append({"query_id": query_id, "doc_id": coherence_doc_id(doc_id),
                              "grade": 0})

    doc = {
        "version": 1,
        "corpus": collection,
        "collection": collection,
        "mission_id": mission.id,
        "queries": queries,
    }
    return doc, qrels


# --- scoring ---------------------------------------------------------------------------

def score_hits(hits: list[str], grades: dict[str, int], k: int) -> tuple[float, float]:
    """(recall@k over grade >= 1, reciprocal rank of the first grade >= 2)
    for one query's document hits, best first. recall is 0.0 when nothing
    is expected; rr is 0.0 when no gold document is in the top k."""
    top = hits[:k]
    expected = {d for d, g in grades.items() if g >= RELEVANT_GRADE}
    recall = (len(expected & set(top)) / len(expected)) if expected else 0.0
    rr = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if grades.get(doc_id, 0) >= GOLD_GRADE:
            rr = 1.0 / rank
            break
    return recall, rr


def _collapse(points) -> list[str]:
    """Chunk hits (best first) -> distinct doc_ids in first-seen order."""
    out: list[str] = []
    seen = set()
    for p in points:
        payload = getattr(p, "payload", None) or {}
        doc_id = payload.get("doc_id")
        if isinstance(doc_id, str) and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


# --- files -----------------------------------------------------------------------------

def _yaml_scalar(value) -> str:
    """A YAML scalar for a str/int/bool/None. Strings are emitted
    double-quoted with JSON escapes, which YAML's double-quoted style reads
    identically (yaml.safe_load gives the same str back)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def queries_yaml(doc: dict) -> str:
    lines = [
        "# Corpus self-test queries written by Quarry (corpus_check.py).",
        "# Grades come from the mission's own assessor: this judges whether the",
        "# collection retrieves what the mission used, not ground truth.",
        "",
    ]
    for key in ("version", "corpus", "collection", "mission_id"):
        if key in doc:
            lines.append(f"{key}: {_yaml_scalar(doc[key])}")
    lines.append("")
    lines.append("queries:")
    for q in doc.get("queries", []):
        first = True
        for key in ("query_id", "topic", "intent_type", "text"):
            prefix = "  - " if first else "    "
            lines.append(f"{prefix}{key}: {_yaml_scalar(q.get(key))}")
            first = False
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".check.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def eval_dir_for(collection: str, mission_id: str) -> str:
    return os.path.join(export_dir(), "eval", collection, mission_id)


def write_eval_files(eval_dir: str, queries_doc: dict, qrels: list[dict],
                     report: CheckReport) -> None:
    _write_text_atomic(os.path.join(eval_dir, "queries.yaml"), queries_yaml(queries_doc))
    _write_text_atomic(os.path.join(eval_dir, "qrels.jsonl"),
                       "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qrels))
    payload = asdict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["floor"] = float(settings.corpus_check_floor)
    payload["note"] = ("Grades come from the mission's own assessor: a regression "
                       "tripwire for the exported corpus, not ground truth.")
    _write_text_atomic(os.path.join(eval_dir, "report.json"),
                       json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# --- run -------------------------------------------------------------------------------

def _query(client, collection: str, vector: list[float], limit: int):
    return client.query_points(
        collection_name=collection, query=vector, using=VECTOR_NAME,
        limit=limit, with_payload=["doc_id"])


async def _run(mission_id: str, collection: str, client, k: int) -> CheckReport:
    report = CheckReport(collection=collection, mission_id=mission_id, k=k,
                         eval_dir=eval_dir_for(collection, mission_id))
    mission = await storage.get_mission(mission_id)
    if mission is None:
        raise LookupError(f"no mission {mission_id!r}")
    requirements = await storage.get_requirements_for_mission(mission_id)
    req_doc_map = await storage.get_requirement_document_map(mission_id)
    queries_doc, qrels = build_eval_set(mission, requirements, req_doc_map, collection)

    grades: dict[str, dict[str, int]] = {}
    for row in qrels:
        grades.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["grade"])

    if client is None:
        from vector_export import make_client
        client = make_client()

    recalls: list[float] = []
    rrs: list[float] = []
    for q in queries_doc["queries"]:
        text = q["text"]
        query_grades = grades.get(q["query_id"], {})
        expected = sorted(d for d, g in query_grades.items() if g >= RELEVANT_GRADE)
        if not text:
            report.queries.append(QueryCheck(q["query_id"], text, expected, [], 0.0, 0.0))
            continue
        vector = await asyncio.to_thread(embeddings.embed_query, text,
                                         mission_id=mission_id)
        response = await asyncio.to_thread(_query, client, collection,
                                           [float(x) for x in vector],
                                           max(1, k * HIT_MULTIPLIER))
        hits = _collapse(getattr(response, "points", None) or [])[:k]
        recall, rr = score_hits(hits, query_grades, k)
        report.queries.append(QueryCheck(q["query_id"], text, expected, hits, recall, rr))
        if expected:
            recalls.append(recall)
        if any(g >= GOLD_GRADE for g in query_grades.values()):
            rrs.append(rr)

    report.recall_at_k = (sum(recalls) / len(recalls)) if recalls else 0.0
    report.mrr = (sum(rrs) / len(rrs)) if rrs else 0.0
    report.passed = report.recall_at_k >= float(settings.corpus_check_floor)
    write_eval_files(report.eval_dir, queries_doc, qrels, report)
    return report


async def run(mission_id: str, *, collection: Optional[str] = None, client=None,
              k: Optional[int] = None) -> CheckReport:
    """Check `collection` (default settings.qdrant_export_collection) against
    mission `mission_id`'s record at `k` (default settings.corpus_check_k),
    writing queries.yaml / qrels.jsonl / report.json under
    export_dir/eval/<collection>/<mission_id>/. Never raises: a failure is
    one stderr line and a report with passed=False and `error` set."""
    collection = (collection or settings.qdrant_export_collection or "").strip()
    k = int(k if k is not None else settings.corpus_check_k)
    k = max(1, k)
    try:
        return await _run(mission_id, collection, client, k)
    except Exception as e:  # noqa: BLE001 - the check never fails an export
        print(f"[CHECK] {collection} / {mission_id} failed ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
        return CheckReport(collection=collection, mission_id=mission_id, k=k,
                           eval_dir=eval_dir_for(collection, mission_id),
                           passed=False, error=f"{type(e).__name__}: {e}")
