"""corpus_check: the self-test run on the corpus an export just populated.

The eval set is built from a mission's own requirement -> document links;
the retrieval runs against an in-memory QdrantClient filled by
vector_export with the litellm stub's deterministic vectors. Because the
stub embeds identical text to identical vectors, a query equal to a stored
chunk's indexed text retrieves that chunk at cosine 1.0 -- the
self-consistent corpus -- and permuting the stored vectors breaks exactly
that, which is how the check proves it can fail."""
import asyncio
import json
import os

import litellm
import pytest
from qdrant_client import QdrantClient, models as qm

import chunking
import corpus_check
import storage
import vector_export
from config import settings
from models import Agent, Document, Mission, Requirement

DIM = 1024
COLLECTION = "quarry_test"


@pytest.fixture(autouse=True)
def _check_settings(monkeypatch, tmp_path):
    litellm.reset_calls()
    monkeypatch.setattr(settings, "embed_model", "cohere/embed-v5.0-pro")
    monkeypatch.setattr(settings, "embed_query_model", "cohere/embed-v5.0-fast")
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "export_dir", str(tmp_path / "export"))
    monkeypatch.setattr(settings, "qdrant_export_collection", COLLECTION)
    monkeypatch.setattr(settings, "corpus_check_k", 10)
    monkeypatch.setattr(settings, "corpus_check_floor", 0.0)
    asyncio.run(storage.init_db())
    yield
    litellm.reset_calls()


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def _run(coro):
    return asyncio.run(coro)


def _mission(mid="m1", question="Q", status="done"):
    return Mission(id=mid, agent_id="a1", question=question, status=status, created_at="t")


def _req(rid, mid="m1", title="T", status="pending", queries=None):
    return Requirement(id=rid, mission_id=mid, title=title, status=status,
                       next_queries_json=json.dumps(queries) if queries else None)


def _doc(doc_id, title, body, url):
    return Document(id=doc_id, url=url, domain="example.com", title=title,
                    search_query="q", crawled_at="2026-08-11T06:48:00+00:00",
                    content_markdown=body, content_fit="")


def _seed_mission(mission, requirements, links):
    """Insert an agent, the mission, its requirements and (req, doc) links."""
    async def go():
        await storage.insert_agent(Agent(id="a1", name="N", expertise="x",
                                         persona_prompt="p", created_at="t"))
        await storage.insert_mission(mission)
        for req in requirements:
            await storage.insert_requirement(req)
        for req_id, doc_id in links:
            await storage.link_mission_document(mission.id, req_id, doc_id)
    _run(go())


# --- eval set shape -----------------------------------------------------------------

def test_build_eval_set_matches_coherence_format():
    mission = _mission(question="What is happening?")
    reqs = [_req("r1", title="First EEI", status="satisfied", queries=["first query"]),
            _req("r2", title="Second EEI", status="unmet")]
    req_map = {"r1": ["d1", "d2"], "r2": ["d3"]}

    doc, qrels = corpus_check.build_eval_set(mission, reqs, req_map)

    assert doc["version"] == 1
    assert doc["collection"] == COLLECTION and doc["corpus"] == COLLECTION
    assert [q["query_id"] for q in doc["queries"]] == ["Q00", "Q01", "Q02"]
    for q in doc["queries"]:
        assert set(q) >= {"query_id", "topic", "intent_type", "text"}
    assert doc["queries"][0]["text"] == "What is happening?"
    assert doc["queries"][1] == {"query_id": "Q01", "topic": "First EEI",
                                 "intent_type": "lookup", "text": "first query"}
    assert doc["queries"][2]["text"] == "Second EEI", "no queries recorded -> the title"
    assert all(set(r) == {"query_id", "doc_id", "grade"} for r in qrels)
    assert all(r["doc_id"].startswith("QRY-") for r in qrels)


def test_grades_follow_the_requirement_map():
    mission = _mission()
    reqs = [_req("r1", status="satisfied"), _req("r2", status="unmet"), _req("r3")]
    req_map = {"r1": ["d1", "d2"], "r2": ["d2", "d3"], "r3": []}
    _doc_, qrels = corpus_check.build_eval_set(mission, reqs, req_map)
    grades = {}
    for r in qrels:
        grades.setdefault(r["query_id"], {})[r["doc_id"]] = r["grade"]

    # Q00: every linked doc at its best grade (d2 is satisfied via r1).
    assert grades["Q00"] == {"QRY-d1": 2, "QRY-d2": 2, "QRY-d3": 1}
    # Q01 satisfied: its docs 2; the mission's other docs 0.
    assert grades["Q01"] == {"QRY-d1": 2, "QRY-d2": 2, "QRY-d3": 0}
    # Q02 unmet: its docs 1 (d2 is 1 here even though r1 satisfied it).
    assert grades["Q02"] == {"QRY-d2": 1, "QRY-d3": 1, "QRY-d1": 0}
    # Q03 has no sources: only grade-0 rows for the mission's other docs.
    assert grades["Q03"] == {"QRY-d1": 0, "QRY-d2": 0, "QRY-d3": 0}
    # Grade-0 rows name only documents of this mission, so the file stays small.
    assert {r["doc_id"] for r in qrels} == {"QRY-d1", "QRY-d2", "QRY-d3"}


def test_queries_yaml_is_well_formed_and_round_trips():
    mission = _mission(question='Line one\nwith "quotes" and: colons')
    doc, _qrels = corpus_check.build_eval_set(
        mission, [_req("r1", title="T: one", queries=["q #1"])], {"r1": []})
    text = corpus_check.queries_yaml(doc)
    assert text.startswith("#")
    assert "\nversion: 1\n" in text
    assert f'\ncollection: "{COLLECTION}"\n' in text
    assert '\n  - query_id: "Q00"\n' in text
    assert '    text: "Line one\\nwith \\"quotes\\" and: colons"\n' in text
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(text)
    assert loaded["collection"] == COLLECTION
    assert loaded["queries"] == doc["queries"]


# --- arithmetic -----------------------------------------------------------------------

def test_recall_and_mrr_arithmetic():
    grades = {"a": 2, "b": 1, "c": 0}
    assert corpus_check.score_hits(["c", "b", "a"], grades, 3) == (1.0, pytest.approx(1 / 3))
    assert corpus_check.score_hits(["b", "a"], grades, 2) == (1.0, 0.5)
    assert corpus_check.score_hits(["c", "x", "b", "a"], grades, 2) == (0.0, 0.0)
    assert corpus_check.score_hits(["b", "x"], grades, 2) == (0.5, 0.0), \
        "a grade-1 hit counts for recall, not for the reciprocal rank"
    assert corpus_check.score_hits(["a"], {}, 5) == (0.0, 0.0)


def test_chunk_hits_collapse_to_document_order():
    class P:
        def __init__(self, payload):
            self.payload = payload

    points = [P({"doc_id": "QRY-b"}), P({"doc_id": "QRY-b"}), P({"doc_id": "QRY-a"}),
              P({}), P({"doc_id": "QRY-b"}), P({"doc_id": "QRY-c"})]
    assert corpus_check._collapse(points) == ["QRY-b", "QRY-a", "QRY-c"]


# --- against a populated corpus -----------------------------------------------------

def _self_consistent_corpus(client):
    """One mission whose question and whose one requirement's query are
    exactly the indexed text of its one source, plus three distractor
    documents in the same collection. Returns the mission doc id."""
    title, body = "Alpha Report", "alpha body text one."
    target = chunking.indexed_text(title, body)
    d1 = _run(storage.upsert_document(_doc("d1", title, body, "https://a/1")))
    distractors = [
        _run(storage.upsert_document(_doc(f"x{n}", f"Other {n}", f"other body {n}.",
                                          f"https://x/{n}")))
        for n in range(3)]
    _seed_mission(_mission(question=target),
                  [_req("r1", title="Alpha", status="satisfied", queries=[target])],
                  [("r1", d1)])
    result = _run(vector_export.export_documents([d1, *distractors], client=client,
                                                 check=False))
    assert len(result.exported) == 4 and result.failed == {}
    return d1


def test_self_consistent_corpus_scores_perfectly(client):
    d1 = _self_consistent_corpus(client)
    report = _run(corpus_check.run("m1", client=client, k=1))
    assert report.error is None
    assert report.collection == COLLECTION and report.mission_id == "m1" and report.k == 1
    assert [q.query_id for q in report.queries] == ["Q00", "Q01"]
    for q in report.queries:
        assert q.expected == [f"QRY-{d1}"]
        assert q.hits == [f"QRY-{d1}"]
        assert q.recall_at_k == 1.0 and q.rr == 1.0
    assert report.recall_at_k == 1.0 and report.mrr == 1.0
    assert report.passed is True


def test_shuffled_vectors_fail_the_check(client, monkeypatch):
    """The honesty test: the same corpus with its vectors permuted between
    points must score badly, or the check could not fail."""
    monkeypatch.setattr(settings, "corpus_check_floor", 0.5)
    _self_consistent_corpus(client)
    before = _run(corpus_check.run("m1", client=client, k=1))
    assert before.recall_at_k == 1.0 and before.passed is True

    records, _ = client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
    assert len(records) == 4
    rotated = records[1:] + records[:1]
    client.upsert(COLLECTION, points=[
        qm.PointStruct(id=r.id, vector=other.vector, payload=r.payload)
        for r, other in zip(records, rotated)], wait=True)

    after = _run(corpus_check.run("m1", client=client, k=1))
    assert after.error is None
    assert after.recall_at_k < 0.5
    assert after.mrr < 0.5
    assert after.passed is False


def test_report_files_written(client):
    d1 = _self_consistent_corpus(client)
    report = _run(corpus_check.run("m1", client=client))
    eval_dir = os.path.join(settings.export_dir, "eval", COLLECTION, "m1")
    assert report.eval_dir == eval_dir
    assert sorted(os.listdir(eval_dir)) == ["qrels.jsonl", "queries.yaml", "report.json"]
    with open(os.path.join(eval_dir, "qrels.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows == [{"query_id": "Q00", "doc_id": f"QRY-{d1}", "grade": 2},
                    {"query_id": "Q01", "doc_id": f"QRY-{d1}", "grade": 2}]
    with open(os.path.join(eval_dir, "queries.yaml"), encoding="utf-8") as f:
        assert f'collection: "{COLLECTION}"' in f.read()
    with open(os.path.join(eval_dir, "report.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["recall_at_k"] == 1.0 and saved["passed"] is True
    assert saved["k"] == 10 and saved["floor"] == 0.0
    # Four documents in the collection: recall is measured at k_effective = 3.
    assert saved["k_effective"] == 3 and saved["n_docs_in_collection"] == 4
    assert saved["n_measured"] == 2
    assert saved["queries"][0]["query_id"] == "Q00"
    assert "not ground truth" in saved["note"]


def test_queries_use_the_search_query_side(client):
    _self_consistent_corpus(client)
    litellm.reset_calls()
    _run(corpus_check.run("m1", client=client))
    assert len(litellm.embedding_calls) == 2
    for call in litellm.embedding_calls:
        assert call["model"] == "cohere/embed-v5.0-fast"
        assert call["input_type"] == "search_query"
        assert call["dimensions"] == DIM


def test_export_attaches_the_check(client):
    title, body = "Alpha Report", "alpha body text one."
    target = chunking.indexed_text(title, body)
    d1 = _run(storage.upsert_document(_doc("d1", title, body, "https://a/1")))
    _seed_mission(_mission(question=target),
                  [_req("r1", status="satisfied", queries=[target])], [("r1", d1)])
    result = _run(vector_export.export_documents([d1], client=client, mission_id="m1",
                                                 check=True))
    assert result.check is not None
    assert result.check.mission_id == "m1" and result.check.recall_at_k == 1.0
    # One document in the collection: k_effective is capped at 1.
    assert result.check.k == 10 and result.check.k_effective == 1
    assert "check recall@1 1.00 mrr 1.00 over 2 measured queries (pass)" in result.summary()
    status = result.to_dict()
    assert status["check"]["passed"] is True


# --- never raises ---------------------------------------------------------------------

def test_missing_mission_is_a_failed_report_not_an_exception(client, capsys):
    report = _run(corpus_check.run("nope", client=client))
    assert report.passed is False
    assert "LookupError" in report.error
    assert report.queries == []
    assert "[CHECK]" in capsys.readouterr().err


def test_a_failing_query_never_raises_into_the_export(client, monkeypatch):
    d1 = _run(storage.upsert_document(_doc("d1", "T", "body", "https://a/1")))
    _seed_mission(_mission(), [_req("r1", status="satisfied")], [("r1", d1)])

    def broken(*a, **k):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(corpus_check.embeddings, "embed_query", broken)
    result = _run(vector_export.export_documents([d1], client=client, mission_id="m1",
                                                 check=True))
    assert result.exported == [d1], "the export itself succeeded"
    assert result.check.passed is False
    assert "embedder down" in result.check.error
    assert "check failed" in result.summary()
    assert not os.path.exists(os.path.join(settings.export_dir, "eval")), \
        "no half-written eval files"


# --- the self-test cannot pass trivially (review F3) ---------------------------------

def test_k_is_capped_below_the_collection_size_so_recall_is_not_free(client, monkeypatch):
    """Six documents in the collection and the default k=10: before the cap
    every query returned the whole collection, so recall@10 was 1.0 whatever
    the vectors held. k_effective = min(k, n_docs - 1) leaves the worst hit
    out, and a document whose vector is the opposite of the query is that
    worst hit, so recall drops below 1.0 for the two queries that expect it."""
    monkeypatch.setattr(settings, "corpus_check_floor", 0.75)
    query = "the one query both Q00 and Q01 ask"
    ids = [_run(storage.upsert_document(_doc(f"d{i}", f"Doc {i}", f"body {i} words here.",
                                             f"https://a/{i}"))) for i in range(6)]
    _seed_mission(_mission(question=query),
                  [_req("r1", title="EEI", status="satisfied", queries=[query])],
                  [("r1", ids[0]), ("r1", ids[1])])
    res = _run(vector_export.export_documents(ids, client=client, check=False))
    assert len(res.exported) == 6

    qvec = embeddings_for_query(query)
    worst = [r for r in client.scroll(COLLECTION, limit=100, with_payload=True)[0]
             if r.payload["doc_id"] == f"QRY-{ids[0]}"]
    assert len(worst) == 1
    client.upsert(COLLECTION, points=[qm.PointStruct(
        id=worst[0].id, vector={"dense": [-x for x in qvec]}, payload=worst[0].payload)],
        wait=True)

    report = _run(corpus_check.run("m1", client=client))
    assert report.error is None
    assert report.k == 10 and report.n_docs_in_collection == 6 and report.k_effective == 5
    assert report.n_measured == 2
    for q in report.queries:
        assert len(q.hits) == 5 and f"QRY-{ids[0]}" not in q.hits
        assert q.recall_at_k == 0.5
    assert report.recall_at_k == 0.5
    assert report.passed is False


def embeddings_for_query(text):
    import embeddings
    return [float(x) for x in embeddings.embed_query(text)]


def test_nothing_measured_is_not_a_pass(client):
    d1 = _run(storage.upsert_document(_doc("d1", "T", "body", "https://a/1")))
    _seed_mission(_mission(), [_req("r1", title="t", status="unmet")], [])
    _run(vector_export.export_documents([d1], client=client, check=False))
    report = _run(corpus_check.run("m1", client=client))
    assert report.n_measured == 0
    assert report.passed is False
    assert report.error == "nothing measured"
    assert report.n_docs_in_collection == 1


def test_summary_and_cli_lines_say_what_was_measured(client, capsys):
    d1 = _self_consistent_corpus(client)
    result = _run(vector_export.export_documents([d1], client=client, mission_id="m1",
                                                 check=True))
    assert result.check.n_docs_in_collection == 4 and result.check.k_effective == 3
    assert "check recall@3 1.00 mrr 1.00 over 2 measured queries (pass)" in result.summary()


# --- the YAML is pure ASCII and round-trips odd strings (review F8) -------------------

# Each odd character sits mid-string: build_eval_set strips the question and
# titles, and Python counts NEL (U+0085) as whitespace, so a trailing one
# would be gone before the YAML writer ever saw it.
ODD_STRINGS = ["del \x7f here", "nel \x85 here", "c1 \x9f here", "sep \u2028 here",
               "emoji \U0001F600 here", "bom \ufeff here", "tab\there", "- leading dash",
               "key: value", "'single'", 'back\\slash "quoted" \\u0085 literal']


@pytest.mark.parametrize("s", ODD_STRINGS)
def test_yaml_scalars_are_ascii(s):
    text = corpus_check.queries_yaml(corpus_check.build_eval_set(
        _mission(question=s), [_req("r1", title=s)], {"r1": []})[0])
    assert text.isascii(), repr(text)


@pytest.mark.parametrize("s", ODD_STRINGS)
def test_yaml_scalars_round_trip_odd_strings(s):
    yaml = pytest.importorskip("yaml")
    text = corpus_check.queries_yaml(corpus_check.build_eval_set(
        _mission(question=s), [_req("r1", title=s)], {"r1": []})[0])
    loaded = yaml.safe_load(text)
    assert loaded["queries"][0]["text"] == s
    assert loaded["queries"][1]["text"] == s and loaded["queries"][1]["topic"] == s


# --- the query is the one the collector ran (review F9) ------------------------------

def test_first_query_prefers_the_last_search_the_collector_ran():
    req = _req("r1", title="The title", queries=["planned first", "planned second"])
    req.search_stats_json = json.dumps([
        {"pass": 1, "query": "ran in pass one", "engine": "brave", "results": 3},
        {"pass": 2, "query": "ran in pass two", "engine": "bing", "results": 5},
    ])
    assert corpus_check._first_query(req) == "ran in pass two"
    doc, _ = corpus_check.build_eval_set(_mission(), [req], {"r1": []})
    assert doc["queries"][1]["text"] == "ran in pass two"
    # No search recorded yet: the first planned query, then the title.
    req.search_stats_json = None
    assert corpus_check._first_query(req) == "planned first"
    req.search_stats_json = "not json"
    assert corpus_check._first_query(req) == "planned first"
    req.search_stats_json = json.dumps([{"pass": 1, "query": "   ", "engine": "x", "results": 0}])
    assert corpus_check._first_query(req) == "planned first"
    req.next_queries_json = None
    assert corpus_check._first_query(req) == "The title"
