# Quarry → Coherence (Fathom) corpus export: the handoff

This is the contract between Quarry's `vector_export.py` and whoever makes
Coherence (renamed Fathom by the user; the rename is not on the coherence
repo's `main`, whose head is from 2026-09-04) serve the collection it writes.
Everything in the first half is what Quarry already does and tests; the
second half is what still has to change in the coherence repo, with the
file and line references that were read when this was written.

The one-line summary: the export writes a **dense-only** Qdrant collection
named `quarry`, one point per chunk, embedded with **Cohere Embed 5**
(`embed-v5.0-pro`, 1024 dims) and carrying Coherence's payload shape.
Coherence `main` cannot serve it yet, because it always queries a BM25 arm
and refuses to start without that arm's statistics file, and because its
query embedder defaults to Embed v4.

## 1. Collection shape

| | |
|---|---|
| Collection name | `QDRANT_EXPORT_COLLECTION`, default `quarry` (`[A-Za-z0-9_-]+`; anything else is an `ExportError`) |
| Vectors | **one named vector, `dense`**: `VectorParams(size=EMBED_DIM, distance=COSINE)`, `EMBED_DIM` default 1024. No sparse vector. Named rather than anonymous so a `bm25` sparse arm can be added later with `update_collection` instead of re-creating the collection |
| Point id | `str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))` — deterministic from the chunk id, so a re-export of the same chunk overwrites its own point |
| Payload indexes | `clearance_rank` INTEGER; `compartments`, `releasability`, `doc_id`, `cluster` KEYWORD (`vector_export._PAYLOAD_INDEXES`); created on an existing collection too when missing |
| Vector content | the embedding of `chunking.indexed_text(title, chunk.text)` = `f"{title}\n\n{chunk_text}"`, `input_type="search_document"`, `dimensions=1024`. The title is embedded but is **not** in the payload `text` |

What `ensure_collection` promises: create-if-missing with the shape above;
on an existing collection, verify the `dense` vector is exactly `EMBED_DIM`
wide (an anonymous single vector counts as "no dense vector") and refuse
with `ExportError` otherwise; **never** `delete_collection`. A wrong
collection is the operator's to fix by hand.

## 2. Ids

| | form | example |
|---|---|---|
| `doc_id` | `QRY-<quarry document uuid>` | `QRY-3f0c…` |
| `chunk_id` | `QRY-<uuid>#<idx>`, `idx` from 0 in document order | `QRY-3f0c…#4` |
| Qdrant point id | `uuid5(NAMESPACE_URL, chunk_id)` | — |

`QRY-` is Quarry's corpus prefix, the way Coherence's corpora use `crs-`
and friends. The export never emits `canary-`, `contra-` or `OSINT:` ids and
never sets `is_product`, because Coherence treats those specially
(`retrieval/store.scope_to_source_reporting` excludes `is_product=True`
points from source retrieval, so every Quarry point is source reporting).

Quarry document ids are stable across re-crawls: `storage.upsert_document`
updates in place on `(url, search_query)` and keeps the id, so `doc_id` and
`chunk_id` survive a re-crawl; only the chunk *content* behind them changes.

## 3. Payload

Verified against `vector_export.build_payload` and
`tests/test_vector_export.py::test_points_carry_payload_and_uuid5_ids`
(one assertion per row).

| key | value | notes |
|---|---|---|
| `chunk_id` | `QRY-<doc.id>#<idx>` | read by Coherence's `Hit` (`store.py:561`) and `get_chunk` (`store.py:265`) |
| `doc_id` | `QRY-<doc.id>` | read by `Hit`; the export's delete-by-filter key; keyword-indexed |
| `title` | page title, else its domain (`chunking.document_title`) | read by `Hit` |
| `text` | the chunk text: `f"{section}\n\n{body}"` when a heading is in force, **no title** | read by `Hit`; what a brief quotes |
| `section` | the ATX heading in force, or `null` | |
| `kind` | `"body"` | markdown has no footnote typing |
| `clearance_rank` | int, `EXPORT_CLEARANCE_RANK` (default `0`) | `acl.build_filter` `Range(lte=identity.clearance_rank)` (`acl.py:81-84`) |
| `classification` | `EXPORT_CLASSIFICATION` (default `"NATO UNCLASSIFIED"`) | `Hit.classification` |
| `compartments` | list, `EXPORT_COMPARTMENTS` split on commas (default `[]`) | `acl.py:88-91` `must_not MatchAny(denied)` |
| `releasability` | list, `EXPORT_RELEASABILITY` split on commas (default `[]`) | `acl.py:93-106` nested filter; an empty list is unrestricted on this axis |
| `int_type` | `"OSINT"` | |
| `trust_tier` | `"EXTERNAL_UNVERIFIED"` | |
| `originator` | URL hostname, lower-cased (falls back to `doc.domain`) | |
| `source_designator` | the page URL | |
| `source_url` | the page URL | |
| `date_time_group` | military DTG of `crawled_at`, e.g. `"110648Z AUG 26"`; `""` if it does not parse | |
| `published_date` | `metadata_json["published_date"]` when it is `YYYY-MM-DD`, else the crawl date | |
| `source_reliability` | `"C"` | `Hit` grading string (`store.py:143-145`) |
| `info_credibility` | `3` (int) | same |
| `provenance` | `{"url", "fetched_at": crawled_at, "sha256": document_digest}` | `document_digest = sha256(indexed_text(title, body))` |
| `doc_type` | `"web_page"` | |
| `cluster` | the Quarry `search_query` the page was stored under | keyword-indexed |
| `topics` | `[]` | |
| `decoy_for` | `[]` | |
| `embedding` | see below | |
| `quarry` | see below | |

### The `embedding` sub-object

```json
"embedding": {"model": "cohere/embed-v5.0-pro", "dim": 1024, "input_type": "search_document"}
```

Recorded so a consumer can pick the matching query-side model: the corpus
is in the **Embed 5** space, and Embed 5 Pro (documents) and Fast (queries)
share it. `model` is the LiteLLM id Quarry used (`cohere/…` prefix);
Coherence's own setting takes the bare Cohere id (`embed-v5.0-fast`).

### The `quarry` sub-object

```json
"quarry": {
  "document_id": "<uuid>", "search_query": "<query>", "domain": "example.com",
  "word_count": 1234, "crawled_at": "2026-08-11T06:48:00+00:00",
  "mission_ids": ["<mission uuid>", "…"], "exported_by": "quarry/vector_export"
}
```

Quarry-side provenance; Coherence reads none of it. `mission_ids` is every
mission that linked the document, in link order.

### What is NOT in the payload

No `translated_from` (so `Hit.group_id` defaults to `doc_id`), no
`is_product`, no `page`/`bbox`, no `span_ids`. If Coherence code assumes any
of these exist for every point, it needs a `.get`.

## 4. What the export guarantees

All of this is tested in `tests/test_vector_export.py` with a
`QdrantClient(":memory:")` injected through `export_documents(client=…)`.

- **Idempotent on content.** A document whose `document_exports` row (SQLite)
  carries the same `document_digest` and `embed_model` is skipped;
  `--force` re-writes it. A changed document has its old points **deleted by
  filter** (`doc_id ∈ changed ids`, `wait=True`) and its new points upserted
  in batches of 400 with `wait=True`, so a re-crawl never leaves a stale
  chunk behind and a re-run of the same mission is mostly `skipped` with no
  duplicate points.
- **Stored vectors are reused.** A chunk already embedded in Quarry's
  `chunk_embeddings` for the same model and width is exported as is; only
  missing chunks are embedded (and stored back for next time).
- **Foreign points survive.** Anything the export did not write — a point
  without a `QRY-` doc_id, a Coherence product point with `is_product: true`
  — is never touched (`test_foreign_points_in_collection_survive`).
- **Fails loudly before writing, quietly per document after.** Qdrant
  unreachable, empty `EMBED_MODEL`, a bad collection name or a vector-size
  mismatch raise `ExportError` before any point or export row is written.
  After that, one document's failure (an embedding call, a refused batch)
  lands in `ExportResult.failed`; a document that failed in the prepare
  phase keeps its old points; a document in a refused upsert batch gets no
  export row, so the next run redoes it.
- **A manifest is written** to `<EXPORT_DIR>/manifest-<collection>.json`
  (default `data/export/`): `{collection, exported_at, embed_model, dim,
  vectors: ["dense"], n_docs, n_chunks, skipped, failed, mission_id,
  markings, docs: [{document_id, doc_id, url, title, sha256, n_chunks,
  status: exported|skipped|failed, reason|error}]}`.

## 5. Running the export

Prerequisites: `EMBED_MODEL` set (default `cohere/embed-v5.0-pro`, needing
`LLM_API_KEY`) and a reachable Qdrant.

```bash
# bundled Qdrant (data in ./data/qdrant, published on 127.0.0.1:6333 only)
docker compose --profile qdrant up -d --build       # and QDRANT_URL=http://qdrant:6333 in .env
# or point QDRANT_URL / QDRANT_API_KEY at an existing Qdrant

# then, inside the container (docker compose exec web-researcher …) or the venv:
python -m vector_export --mission <mission-id>        # that mission's sources + self-test
python -m vector_export --search "<query>"            # one search's pages
python -m vector_export --doc <id> [<id> …]
python -m vector_export --all [--no-check]            # the whole library; checks every mission it touched
python -m vector_export --all --dry-run               # lists export/skip per document; contacts nothing
python -m vector_export --mission <id> --force        # re-write even unchanged documents
```

Exit status: 0 ok; 1 when a document failed or a self-test did not pass;
2 on `ExportError`. In the app, `EXPORT_ON_MISSION_DONE=true` (also on the
Settings page) exports and checks a mission's sources as soon as its brief
is written (`agent_runner._export_sources`, one trace line, never a mission
failure); **Export to corpus** on a finished mission page (`POST
/missions/<id>/export`) and the Library toolbar's export button for a
search or mission filter (`POST /documents/export`) hand the document set to
`vector_export.start_export_in_background`, which runs one export at a
time; `GET /api/export/status` returns the current or last run with its
check report, and the mission page's state line (`exported n of total →
quarry · last 2026-08-11 06:48Z · export running`) follows
`document_exports` through the page's poll.

Markings come from `EXPORT_CLEARANCE_RANK`, `EXPORT_CLASSIFICATION`,
`EXPORT_COMPARTMENTS`, `EXPORT_RELEASABILITY`. Use compartment and
releasability names Coherence's lattice (`load_lattice()`) knows, or leave
them empty: `build_filter` denies by the complement of the analyst's set
over the lattice, so a name the lattice has never heard of is neither
granted nor denied by the store-side filter.

## 6. The self-test and its eval files

After a `--mission` export (or `--all`, per mission touched), `corpus_check.run`
asks: *does the collection we just wrote retrieve what the mission found?*
It writes three files to `<EXPORT_DIR>/eval/<collection>/<mission-id>/`.

### `queries.yaml`

```yaml
# Corpus self-test queries written by Quarry (corpus_check.py).
# Grades come from the mission's own assessor: this judges whether the
# collection retrieves what the mission used, not ground truth.

version: 1
corpus: "quarry"
collection: "quarry"
mission_id: "<mission uuid>"

queries:
  - query_id: "Q00"
    topic: "mission"
    intent_type: "synthesis"
    text: "<the mission question>"

  - query_id: "Q01"
    topic: "<requirement title>"
    intent_type: "lookup"
    text: "<first of the requirement's next_queries_json, else its title>"
```

Scalars are JSON-escaped double-quoted strings, which YAML's double-quoted
style reads identically (`yaml.safe_load` round-trips them;
`test_queries_yaml_is_well_formed_and_round_trips`). The keys are the ones
`evals/graded.py` uses: `collection` for the resolver, `queries[].query_id`,
`intent_type`, `text`.

### `qrels.jsonl`

One JSON object per line, `{"query_id", "doc_id", "grade"}`, `doc_id` being
the payload's `QRY-…` doc id:

```
{"query_id": "Q00", "doc_id": "QRY-3f0c…", "grade": 2}
{"query_id": "Q01", "doc_id": "QRY-3f0c…", "grade": 2}
{"query_id": "Q01", "doc_id": "QRY-9a1e…", "grade": 0}
```

Grades: **2** = the document is linked to that requirement and the
requirement ended `satisfied`; **1** = linked but the requirement stayed
`unmet`; **0** = the same mission linked the document only to other
requirements (an explicit non-relevant row, as Coherence's qrels carry).
`Q00` takes every linked document at its best grade. There is no grade 3.

### `report.json`

The `CheckReport` dataclass plus `generated_at`, `floor` and a `note`:
`{collection, mission_id, k, queries: [{query_id, text, expected: [doc ids
graded ≥ 1], hits: [top-k doc ids], recall_at_k, rr}], recall_at_k, mrr,
passed, eval_dir, error}`. recall@k is over grade ≥ 1 documents; the
reciprocal rank is of the first grade ≥ 2 document; each mean is over the
queries that have something to measure. `passed = recall_at_k >=
CORPUS_CHECK_FLOOR` (default `0.0`: report only).

**What the number means.** The grades come from Quarry's own assessor —
which documents it linked to which requirement, and whether it judged the
requirement satisfied — not from a human judgment. So the check measures
"the corpus retrieves what the mission used", a regression tripwire for the
export, not retrieval quality against ground truth. Its honesty is tested:
a corpus whose vectors are shuffled scores ≈ 0
(`tests/test_corpus_check.py::test_shuffled_vectors_fail_the_check`).

### Registering the files in `evals/graded.py`

`EVAL_SETS` (`evals/graded.py:91-100`) is a list of
`(queries_path, implied_collection)` pairs resolved by `resolve_eval_set`
(`:103-134`): it loads each `queries.yaml`, reads `collection` from the file
(falling back to the implied name), and takes `qrels.jsonl` from the same
directory. Paths are relative to the coherence repo root (`REPO`), so:

1. copy one mission's `queries.yaml` and `qrels.jsonl` into the coherence
   repo, e.g. `corpus/quarry/eval/`;
2. add a row:

   ```python
   EVAL_SETS = [
       (REPO / "corpus" / "realistic" / "eval" / "queries.yaml", None),
       (REPO / "corpus" / "eval_queries.yaml", "kestrel"),
       (REPO / "corpus" / "kestrel-ml" / "eval" / "queries.yaml", None),
       # Quarry's exported corpus: the file declares `collection: quarry` and
       # its qrels sit beside it. Grades are the mission's own assessor's.
       (REPO / "corpus" / "quarry" / "eval" / "queries.yaml", None),
   ]
   ```

3. run `QDRANT_COLLECTION=quarry uv run python -m evals.graded` once the
   dense-only mode below exists (`main()` at `graded.py:163-165` runs
   `search(mode="bm25")` and `mode="hybrid"` as well as `dense`, both of
   which hit `get_bm25()` today).

Two things to know when comparing numbers. Every Quarry mission produces
its own set with ids `Q00…Qnn`, so two missions' files cannot be
concatenated without re-prefixing the ids. And `graded.py` counts grade ≥ 2
as relevant (`RELEVANT = 2`, `graded.py:54`) where Quarry's own check counts
≥ 1: Coherence's recall over these files is recall of the *satisfied*
requirements' sources only.

## 7. What Coherence `main` needs changed, and where

### 7.1 A dense-only retrieval mode

Today both arms are always consulted and the BM25 arm is mandatory:

- `src/coherence/retrieval/store.py:74-99` — `get_bm25()` loads
  `var/bm25-<collection>.json` and raises `FileNotFoundError` when it is
  missing ("refusing to serve another collection's lexical statistics").
  That refusal is right for Coherence's own corpora and wrong for one that
  has no lexical arm at all.
- `src/coherence/retrieval/store.py:313-…` — `search(identity, query, *,
  mode="hybrid", …)`. The `"dense"` branch (`:343-354`) already queries
  `using="dense"` only. The `"hybrid"` branch (`:370-…`, `get_bm25()` at `:371`) calls it
  unconditionally, builds a dense `Prefetch` and, when the query has
  vocabulary, a `bm25` `Prefetch` (`:383-400`), then fetches each arm
  separately and fuses in code (`:447-…`).
- `src/coherence/retrieval/pipeline.py:179` — `retrieve()` hard-codes
  `mode="hybrid"`.
- `src/coherence/api/main.py:159-170` — `lifespan` calls `get_bm25()` at
  startup precisely so a missing stats file fails at the start line rather
  than at the first draft. Lines 119-158 already verify the collection
  exists and is non-empty and log it.

What the follow-up needs: a notion of "this collection has no sparse arm",
decided once at startup — either from the collection itself
(`get_store().get_collection(s.qdrant_collection).config.params.sparse_vectors`
is empty / has no `bm25`) or an explicit setting such as
`COHERENCE_RETRIEVAL_MODE=dense` — and then (a) `lifespan` skips the
`get_bm25()` probe for such a collection, (b) `search()` in `"hybrid"` mode
builds only the dense prefetch (the in-code fusion over one list is that
list), or `retrieve()` passes `mode="dense"`, and (c) `evals/graded.py`
skips or labels the `bm25`/`hybrid` arms. The ACL filter must stay on the
dense arm exactly as it is now (`store.py:383-392`); nothing about access
control changes.

### 7.2 Query embeddings must be Embed 5

- `src/coherence/settings.py:214` — `model_embed` defaults to
  `"embed-v4.0"` (`COHERE_MODEL_EMBED`); `:382` `embed_dim` defaults to 1024
  (`COHERE_EMBED_DIM`).
- `src/coherence/retrieval/store.py:188-197` — `embed_query()` embeds with
  `s.model_embed`, `input_type="search_query"`, `output_dimension=s.embed_dim`.

The `quarry` collection is embedded with `embed-v5.0-pro` at 1024 dims
(every point's `embedding` payload says so). Embed 5 Pro and Fast share a
space; v4 and v5 are not stated to be compatible. So when serving `quarry`:
`COHERE_MODEL_EMBED=embed-v5.0-fast` (or `-pro`), `COHERE_EMBED_DIM=1024`.
Two cautions:

- `model_embed` is global. `src/coherence/retrieval/external.py:61-65`
  indexes the external collection with the same setting, and the model id is
  part of the cassette request fingerprint, so changing it in a shared `.env`
  invalidates recorded cassettes for every other corpus. The per-corpus
  place is `config/corpus_profiles.yaml`, but `model_embed` is not in
  `corpus_profiles.ALLOWED` (`src/coherence/corpus_profiles.py:65-71`,
  which lists `model_draft … model_fallback`, `retrieval_limit`,
  `rerank_top_k`, `pool_size`). Adding `model_embed` there is consistent
  with the ruling the module quotes — it changes what the system *does*,
  not what *counts as good* — and would let a `quarry:` profile pin the
  query model without touching the others. Precedence stays env > profile
  > default.
- `COHERE_MODEL_RERANK` defaults to `rerank-v4.0-pro` (`settings.py:225`);
  Quarry uses `rerank-v5.0-pro` internally, but Coherence's reranker runs
  over the text it retrieved, so the two need not match. Leave it unless a
  measurement says otherwise.

### 7.3 `"quarry"` in `LIVE_COLLECTIONS`

The guard that stops an indexer deleting a collection it does not own is a
name set kept in three places and pinned equal by test:

- `src/coherence/ingest/indexer.py:173-180` — a dict `collection → owning
  indexer` (for the refusal message); add
  `"quarry": "Quarry's vector_export.py (external repo)"`.
- `corpus/kestrel-ml/index.py:91-98` and `corpus/multilingual/index.py:101-108`
  — plain sets; add `"quarry"`.
- `tests/test_kestrel_ml.py:83-98` pins all three equal to its `LIVE`
  constant (`:48`); add `"quarry"` there too.

Without this, `QDRANT_COLLECTION=quarry uv run python -m coherence.ingest.indexer`
would delete the exported corpus and refill it with KESTREL fixtures —
exactly the hole the guard exists for.

### 7.4 Serving it

`.env` in the coherence repo: `QDRANT_URL` pointing at the same Qdrant
Quarry wrote to, `QDRANT_COLLECTION=quarry` (pinned in `.env`, not the
shell — `main.py:119-139` explains why), plus the Embed 5 settings above.
`lifespan` will refuse an empty collection, so export first. The analyst
identities in Coherence's lattice need `clearance_rank ≥ EXPORT_CLEARANCE_RANK`
(0 by default: everyone) and, if `EXPORT_COMPARTMENTS`/`EXPORT_RELEASABILITY`
were set, the matching caveats.

## 8. Manual end-to-end verification

From the plan's verification list, on the user's box:

1. `docker compose --profile qdrant up -d --build` with
   `QDRANT_URL=http://qdrant:6333`, `EXPORT_ON_MISSION_DONE=true`, defaults
   otherwise (Cohere key in `LLM_API_KEY`).
2. Run a mission to completion. Expected in the trace: `reranked N
   candidates → crawling top K` (logged only when a requirement had more
   candidates than its cap and the reranker changed their order),
   `embedded N chunks · cohere/embed-v5.0-pro` after each requirement's
   crawl, `brief sources reranked` (only when the reranked order differs
   from the newest-first one), and the export line
   `exported N · skipped M · failed 0 · K chunks → 'quarry' · check recall@10 0.xx mrr 0.xx (pass)`.
   `python -m vector_export --mission <id>` prints the same summary line on
   stderr.
3. Inspect `data/export/manifest-quarry.json` (every document `exported`,
   `n_chunks` > 0) and `data/export/eval/quarry/<mission>/report.json`
   (`recall_at_k`, `mrr`, per-query `hits` vs `expected`).
4. Inspect one point as Coherence's ACL filter reads it, e.g.
   `curl -s localhost:6333/collections/quarry/points/scroll -d '{"limit":1,"with_payload":true}'`:
   `clearance_rank` is an int, `compartments`/`releasability` are lists,
   `chunk_id` resolves to `doc_id` + `#idx`, `vector` is absent from the
   payload and the point's `vectors` has only `dense` (1024 floats).
5. Re-run the same mission (or the same `--mission` export): the summary is
   mostly `skipped`, and `curl -s localhost:6333/collections/quarry` shows
   the same `points_count` (no duplicates; changed documents replaced).
6. Coherence serving the collection waits on §7.

## 9. Open items

- Coherence cannot serve a dense-only collection until §7.1 lands. The
  collection's named `dense` vector leaves room to add a `bm25` sparse arm
  later without re-creating it, if the BM25 decision is reversed; Quarry
  would then also need to write that arm and the `var/bm25-quarry.json`
  statistics file.
- Payload choices that are Quarry defaults, exposed as settings where they
  matter: `cluster = search_query`, `published_date` falling back to the
  crawl date, `source_reliability`/`info_credibility` fixed at `C`/`3`,
  markings from `EXPORT_*`. Change them in `build_payload` if Coherence
  needs otherwise; the payload test pins every row, so a change is visible.
- The self-test's grades are the assessor's; a human-judged qrels for the
  `quarry` collection would be a different, better measurement.
