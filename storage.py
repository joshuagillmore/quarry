import os
import json
import aiosqlite
from datetime import datetime, timezone
from typing import Iterable, Optional
import chunking
from models import (Document, ExtractedData, SearchRecord, Agent, Mission,
                    Requirement, LlmCall, ChunkRow, DocumentExport)

DB_PATH = None

# Rows per `IN (...)` clause. SQLite's variable limit is 32766 on current
# builds but 999 on older ones; batching keeps every query under both.
_IN_BATCH = 500

# Mission states with a live worker thread behind them. `awaiting_approval` is
# deliberately absent: it rests on the user, not on a thread.
_IN_FLIGHT_MISSION_STATUSES = ("planning", "collecting", "synthesizing")

# Columns update_agent/update_mission/update_requirement may set. Their SQL is
# built from the keyword names, so anything else is refused outright. `id` is
# excluded: rows are never re-keyed.
_AGENT_COLUMNS = frozenset({
    "name", "expertise", "persona_prompt", "default_max_passes",
    "default_max_sources", "default_per_req_attempts", "schedule_cron",
    "schedule_question", "active", "created_at",
})
_MISSION_COLUMNS = frozenset({
    "agent_id", "question", "status", "plan_json", "budget_json",
    "brief_markdown", "brief_sources_json", "brief_warnings_json", "job_id",
    "parent_mission_id", "error", "created_at", "started_at", "finished_at",
    "stop_reason", "resume_count",
})
_REQUIREMENT_COLUMNS = frozenset({
    "mission_id", "title", "description", "rationale", "status", "attempts",
    "next_queries_json", "satisfied_doc_ids_json", "assessment_missing",
    "assessment_confidence", "accepted_by_user", "search_stats_json",
})


def get_db_path() -> str:
    global DB_PATH
    if DB_PATH is None:
        from config import settings
        DB_PATH = settings.db_path
    return DB_PATH


async def _add_column(db, table: str, col: str, decl: str) -> None:
    """Add a column to a table created before the column existed. Checks
    PRAGMA table_info first, so an ALTER that fails for a real reason surfaces
    instead of being swallowed along with the expected "already there" case.
    `table`/`col`/`decl` are code constants, never user input."""
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        existing = {row[1] for row in await cur.fetchall()}
    if col not in existing:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _validated_columns(fields: dict, allowed: frozenset, table: str) -> str:
    """The `col = ?, ...` SET clause for an update, refusing any key that is
    not a known column (the clause is built from the keyword names)."""
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(f"unknown {table} column(s): {', '.join(unknown)}")
    return ", ".join(f"{k} = ?" for k in fields)


async def init_db():
    db_path = get_db_path()
    # A bare filename (DB_PATH=research.db) has no directory component.
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                title TEXT,
                search_query TEXT,
                crawled_at TEXT NOT NULL,
                content_markdown TEXT,
                content_fit TEXT,
                word_count INTEGER DEFAULT 0,
                links_internal INTEGER DEFAULT 0,
                links_external INTEGER DEFAULT 0,
                metadata_json TEXT,
                UNIQUE(url, search_query)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id),
                model TEXT,
                extracted_at TEXT NOT NULL,
                prompt TEXT,
                data_json TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                result_count INTEGER DEFAULT 0,
                job_id TEXT
            )
        """)
        await _add_column(db, "searches", "job_id", "TEXT")
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id UNINDEXED,
                title,
                domain,
                url,
                content,
                tokenize='porter unicode61'
            )
        """)
        # --- Agentic collection tables ---
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                expertise TEXT,
                persona_prompt TEXT,
                default_max_passes INTEGER DEFAULT 4,
                default_max_sources INTEGER DEFAULT 30,
                default_per_req_attempts INTEGER DEFAULT 3,
                schedule_cron TEXT,
                schedule_question TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        # The standing question a scheduled ("morning brief") run researches.
        await _add_column(db, "agents", "schedule_question", "TEXT")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id),
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_json TEXT,
                budget_json TEXT,
                brief_markdown TEXT,
                brief_sources_json TEXT,
                brief_warnings_json TEXT,
                job_id TEXT,
                parent_mission_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stop_reason TEXT,
                resume_count INTEGER DEFAULT 0
            )
        """)
        # The citation order the brief was written against, and the quality
        # checks run on it (see models.Mission).
        await _add_column(db, "missions", "brief_sources_json", "TEXT")
        await _add_column(db, "missions", "brief_warnings_json", "TEXT")
        # Why the last collection run ended, and how often it was resumed.
        await _add_column(db, "missions", "stop_reason", "TEXT")
        await _add_column(db, "missions", "resume_count", "INTEGER DEFAULT 0")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS requirements (
                id TEXT PRIMARY KEY,
                mission_id TEXT NOT NULL REFERENCES missions(id),
                title TEXT NOT NULL,
                description TEXT,
                rationale TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                next_queries_json TEXT,
                satisfied_doc_ids_json TEXT,
                assessment_missing TEXT,
                assessment_confidence TEXT,
                accepted_by_user INTEGER DEFAULT 0,
                search_stats_json TEXT
            )
        """)
        # The assessor's gap reasoning was previously computed and discarded;
        # these columns persist it for the requirements matrix, and
        # search_stats_json records what each search returned. ALTERs are for
        # DBs created before the columns existed.
        for _col, _type in (("assessment_missing", "TEXT"),
                            ("assessment_confidence", "TEXT"),
                            ("accepted_by_user", "INTEGER DEFAULT 0"),
                            ("search_stats_json", "TEXT")):
            await _add_column(db, "requirements", _col, _type)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS mission_documents (
                mission_id TEXT NOT NULL,
                requirement_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (mission_id, requirement_id, document_id)
            )
        """)
        # One row per LLM provider call that returned (llm.chat_ex records
        # them). mission_id is NULL for calls outside a mission, and is not a
        # foreign key: a call can finish after its mission was deleted.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                mission_id TEXT,
                purpose TEXT,
                tier TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                duration_ms INTEGER,
                created_at TEXT
            )
        """)
        # --- Chunks, dense embeddings, corpus export state ---
        # chunks: a document's body split by chunking.document_chunks, one
        # row per chunk, replaced whole on every upsert_document. chunk_id is
        # "<document_id>#<idx>" and content_sha256 the chunk text's digest.
        # chunk_embeddings: one vector per (chunk, model), little-endian
        # float32 bytes; storage never decodes them. Kept across a re-crawl
        # for every chunk whose (chunk_id, content_sha256) did not change.
        # document_exports: what a vector collection last received for a
        # document, so an unchanged one is skipped next time.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                section TEXT,
                text TEXT NOT NULL,
                char_len INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                UNIQUE(document_id, idx)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS document_exports (
                document_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                embed_model TEXT NOT NULL,
                n_chunks INTEGER NOT NULL DEFAULT 0,
                exported_at TEXT NOT NULL,
                PRIMARY KEY (document_id, collection)
            )
        """)
        # Lookup indexes for the hot filters and joins: the Library by
        # collection and unfiltered (both paginate by crawled_at), extractions
        # per document, a mission's requirements, reverse document-to-mission
        # links, an agent's in-flight missions, and a mission's LLM usage.
        for _stmt in (
            "CREATE INDEX IF NOT EXISTS idx_documents_query_crawled "
            "ON documents(search_query, crawled_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_crawled "
            "ON documents(crawled_at)",
            "CREATE INDEX IF NOT EXISTS idx_extractions_document "
            "ON extractions(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_requirements_mission "
            "ON requirements(mission_id)",
            "CREATE INDEX IF NOT EXISTS idx_mission_documents_document "
            "ON mission_documents(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_missions_agent_status "
            "ON missions(agent_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_mission "
            "ON llm_calls(mission_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document "
            "ON chunks(document_id)",
        ):
            await db.execute(_stmt)

        async with db.execute("SELECT COUNT(*) FROM documents") as c:
            docs_ct = (await c.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM documents_fts") as c:
            fts_ct = (await c.fetchone())[0]
        if fts_ct != docs_ct:
            await db.execute("DELETE FROM documents_fts")
            await db.execute("""
                INSERT INTO documents_fts(doc_id, title, domain, url, content)
                SELECT id, COALESCE(title, ''), domain, url, COALESCE(content_markdown, '')
                FROM documents
            """)
        await db.commit()


async def insert_document(doc: Document) -> str:
    """Store a crawled document and return the id that is authoritative for
    its (url, search_query): the existing row's id on a re-crawl, not doc.id.
    Kept as a name for callers; it is exactly upsert_document and, like it,
    raises on failure instead of printing and reporting success."""
    return await upsert_document(doc)


def _build_fts_query(text: str) -> str:
    parts = []
    for tok in text.split():
        tok = tok.replace('"', '""')
        if tok:
            parts.append(f'"{tok}"')
    return " ".join(parts)


async def search_documents_fts(query: str, search_filter: Optional[str] = None) -> list[Document]:
    fts_q = _build_fts_query(query)
    if not fts_q:
        return []
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if search_filter:
            sql = """SELECT d.* FROM documents d
                     JOIN documents_fts f ON f.doc_id = d.id
                     WHERE documents_fts MATCH ? AND d.search_query = ?
                     ORDER BY rank LIMIT 100"""
            args = (fts_q, search_filter)
        else:
            sql = """SELECT d.* FROM documents d
                     JOIN documents_fts f ON f.doc_id = d.id
                     WHERE documents_fts MATCH ?
                     ORDER BY rank LIMIT 100"""
            args = (fts_q,)
        try:
            async with db.execute(sql, args) as cursor:
                rows = await cursor.fetchall()
                return [Document(**dict(row)) for row in rows]
        except Exception as e:
            print(f"FTS query error: {e}")
            return []


async def insert_extraction(ext: ExtractedData) -> bool:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute(
                """INSERT INTO extractions
                   (id, document_id, model, extracted_at, prompt, data_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ext.id, ext.document_id, ext.model, ext.extracted_at,
                 ext.prompt, ext.data_json)
            )
            await db.commit()
            return True
        except Exception as e:
            print(f"Error inserting extraction: {e}")
            return False


async def insert_search(record: SearchRecord) -> bool:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute(
                """INSERT INTO searches (id, query, executed_at, result_count, job_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (record.id, record.query, record.executed_at, record.result_count, record.job_id)
            )
            await db.commit()
            return True
        except Exception as e:
            print(f"Error inserting search: {e}")
            return False


async def get_document(doc_id: str) -> Optional[Document]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return Document(**dict(row))
    return None


# Every documents column except the two (potentially huge) markdown bodies.
_DOC_META_COLUMNS = (
    "id, url, domain, title, search_query, crawled_at, word_count, "
    "links_internal, links_external, metadata_json"
)


async def _list_documents(where: str, where_args: tuple, limit: Optional[int],
                          offset: int, preview_chars: Optional[int]) -> list[Document]:
    """Shared body of get_all_documents / get_documents_by_search.

    preview_chars set: content_fit carries only the first preview_chars of
    (content_fit, else content_markdown) and content_markdown is None, so a
    listing page does not pull every full page body out of SQLite. The crawler
    stores a missing fit_markdown as '' rather than NULL, hence the nullif."""
    args: list = []
    if preview_chars is None:
        cols = "*"
    else:
        cols = (f"{_DOC_META_COLUMNS}, "
                "substr(coalesce(nullif(content_fit, ''), content_markdown), 1, ?) "
                "AS content_fit, "
                "NULL AS content_markdown")
        args.append(max(0, int(preview_chars)))
    sql = f"SELECT {cols} FROM documents"
    if where:
        sql += f" WHERE {where}"
        args.extend(where_args)
    # id breaks crawled_at ties so consecutive pages neither overlap nor skip.
    sql += " ORDER BY crawled_at DESC, id"
    offset = max(0, int(offset or 0))
    if limit is not None or offset:
        # SQLite needs a LIMIT to take an OFFSET; -1 means "no limit".
        sql += " LIMIT ? OFFSET ?"
        args.extend((-1 if limit is None else max(0, int(limit)), offset))
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, args) as cursor:
            rows = await cursor.fetchall()
            return [Document(**dict(row)) for row in rows]


async def get_documents_by_search(query: str, limit: Optional[int] = None, offset: int = 0,
                                  preview_chars: Optional[int] = None) -> list[Document]:
    return await _list_documents("search_query = ?", (query,), limit, offset, preview_chars)


async def get_all_documents(limit: Optional[int] = None, offset: int = 0,
                            preview_chars: Optional[int] = None) -> list[Document]:
    return await _list_documents("", (), limit, offset, preview_chars)


async def get_extractions_for_document(doc_id: str) -> list[ExtractedData]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM extractions WHERE document_id = ? ORDER BY extracted_at DESC",
            (doc_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [ExtractedData(**dict(row)) for row in rows]


async def get_search_history() -> list[SearchRecord]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM searches ORDER BY executed_at DESC LIMIT 50") as cursor:
            rows = await cursor.fetchall()
            return [SearchRecord(**dict(row)) for row in rows]


async def count_documents() -> int:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM documents") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def count_searches() -> int:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM searches") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def count_extractions() -> int:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM extractions") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def count_domains() -> int:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(DISTINCT domain) FROM documents") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def get_doc_ids_with_extractions() -> set[str]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT DISTINCT document_id FROM extractions") as cursor:
            rows = await cursor.fetchall()
            return {row[0] for row in rows}


async def get_search_history_enriched() -> list[dict]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT s.id, s.query, s.executed_at, s.result_count, s.job_id,
                      (SELECT COUNT(DISTINCT d.id)
                       FROM documents d
                       JOIN extractions e ON e.document_id = d.id
                       WHERE d.search_query = s.query) AS extracted_count
               FROM searches s
               ORDER BY s.executed_at DESC
               LIMIT 50"""
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_related_documents(doc_id: str, search_query: str, domain: str, limit: int = 3) -> list[Document]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT * FROM documents
               WHERE id != ?
                 AND (search_query = ? OR domain = ?)
               ORDER BY CASE WHEN search_query = ? THEN 0 ELSE 1 END, crawled_at DESC
               LIMIT ?""",
            (doc_id, search_query, domain, search_query, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [Document(**dict(row)) for row in rows]


async def upsert_document(doc: Document) -> str:
    """Insert a document, or if one already exists for (url, search_query),
    refresh its content in place and keep its existing id. Returns the id that
    is now authoritative for that (url, search_query), so mission links and
    extractions never point at an orphaned id (INSERT OR REPLACE would mint a
    new one on conflict).

    One INSERT ... ON CONFLICT DO UPDATE statement, not select-then-write: two
    concurrent crawls of the same URL cannot both miss the row and then collide
    on the UNIQUE constraint. The FTS row is replaced in the same transaction,
    and so are the document's chunks (under the authoritative id, so a
    re-crawl keeps the embeddings of every chunk whose text did not change)."""
    db_path = get_db_path()
    chunks = chunking.document_chunks(doc)
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """INSERT INTO documents
               (id, url, domain, title, search_query, crawled_at,
                content_markdown, content_fit, word_count,
                links_internal, links_external, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(url, search_query) DO UPDATE SET
                   domain=excluded.domain, title=excluded.title,
                   crawled_at=excluded.crawled_at,
                   content_markdown=excluded.content_markdown,
                   content_fit=excluded.content_fit,
                   word_count=excluded.word_count,
                   links_internal=excluded.links_internal,
                   links_external=excluded.links_external,
                   metadata_json=excluded.metadata_json
               RETURNING id""",
            (doc.id, doc.url, doc.domain, doc.title, doc.search_query,
             doc.crawled_at, doc.content_markdown, doc.content_fit,
             doc.word_count, doc.links_internal, doc.links_external,
             doc.metadata_json),
        ) as cur:
            doc_id = (await cur.fetchone())[0]
        await db.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        await db.execute(
            """INSERT INTO documents_fts(doc_id, title, domain, url, content)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, doc.title or "", doc.domain, doc.url, doc.content_markdown or ""),
        )
        await _replace_document_chunks(db, doc_id, chunks)
        await db.commit()
        return doc_id


# --- Agents ---

async def insert_agent(agent: Agent) -> bool:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO agents
               (id, name, expertise, persona_prompt, default_max_passes,
                default_max_sources, default_per_req_attempts, schedule_cron,
                schedule_question, active, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (agent.id, agent.name, agent.expertise, agent.persona_prompt,
             agent.default_max_passes, agent.default_max_sources,
             agent.default_per_req_attempts, agent.schedule_cron,
             agent.schedule_question, agent.active, agent.created_at),
        )
        await db.commit()
        return True


async def get_agent(agent_id: str) -> Optional[Agent]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)) as cur:
            row = await cur.fetchone()
            return Agent(**dict(row)) if row else None


async def update_agent(agent_id: str, **fields) -> None:
    if not fields:
        return
    db_path = get_db_path()
    cols = _validated_columns(fields, _AGENT_COLUMNS, "agents")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE agents SET {cols} WHERE id = ?",
            (*fields.values(), agent_id),
        )
        await db.commit()


async def delete_agent(agent_id: str) -> None:
    """Delete the agent row only. Its past missions, requirements, and the
    crawled documents are preserved (missions render gracefully without their
    agent)."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        await db.commit()


async def list_agents(active_only: bool = False) -> list[Agent]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        sql = "SELECT * FROM agents"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY created_at DESC"
        async with db.execute(sql) as cur:
            rows = await cur.fetchall()
            return [Agent(**dict(row)) for row in rows]


async def list_scheduled_agents() -> list[Agent]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM agents WHERE active = 1 AND schedule_cron IS NOT NULL AND schedule_cron != ''"
        ) as cur:
            rows = await cur.fetchall()
            return [Agent(**dict(row)) for row in rows]


# --- Missions ---

async def insert_mission(mission: Mission) -> bool:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO missions
               (id, agent_id, question, status, plan_json, budget_json,
                brief_markdown, brief_sources_json, brief_warnings_json, job_id,
                parent_mission_id, error, created_at, started_at, finished_at,
                stop_reason, resume_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mission.id, mission.agent_id, mission.question, mission.status,
             mission.plan_json, mission.budget_json, mission.brief_markdown,
             mission.brief_sources_json, mission.brief_warnings_json,
             mission.job_id, mission.parent_mission_id, mission.error,
             mission.created_at, mission.started_at, mission.finished_at,
             mission.stop_reason, mission.resume_count),
        )
        await db.commit()
        return True


async def update_mission(mission_id: str, **fields) -> None:
    if not fields:
        return
    db_path = get_db_path()
    cols = _validated_columns(fields, _MISSION_COLUMNS, "missions")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE missions SET {cols} WHERE id = ?",
            (*fields.values(), mission_id),
        )
        await db.commit()


async def claim_mission_status(mission_id: str, from_status: str, to_status: str) -> bool:
    """Atomic compare-and-set on a mission's status. True only for the one
    caller whose UPDATE actually moved it from `from_status`, so a double
    submit (two approve clicks, approve racing retask) starts one worker."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "UPDATE missions SET status = ? WHERE id = ? AND status = ?",
            (to_status, mission_id, from_status),
        )
        await db.commit()
        return cur.rowcount == 1


async def agent_has_active_mission(agent_id: str) -> bool:
    """True while any of the agent's missions has a live worker (planning,
    collecting, synthesizing). `awaiting_approval` does not count."""
    db_path = get_db_path()
    marks = ",".join("?" * len(_IN_FLIGHT_MISSION_STATUSES))
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            f"SELECT 1 FROM missions WHERE agent_id = ? AND status IN ({marks}) LIMIT 1",
            (agent_id, *_IN_FLIGHT_MISSION_STATUSES),
        ) as cur:
            return await cur.fetchone() is not None


async def get_mission(mission_id: str) -> Optional[Mission]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)) as cur:
            row = await cur.fetchone()
            return Mission(**dict(row)) if row else None


async def list_missions(limit: int = 50) -> list[Mission]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM missions ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [Mission(**dict(row)) for row in rows]


async def reconcile_interrupted_missions() -> int:
    """A mission's worker lives in a daemon thread, so a restart or crash kills
    it while the DB row still says it is running — the mission view then polls a
    status that will never change. Called once at startup, when no worker can be
    running yet, so anything in an in-flight state is provably orphaned.

    `awaiting_approval` is deliberately excluded: it is a legitimate resting
    state that waits on the user, not on a thread.
    """
    stuck = _IN_FLIGHT_MISSION_STATUSES
    note = "Interrupted by a server restart — the collection worker did not survive."
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"""UPDATE missions SET status = 'error', error = ?, finished_at = ?
                WHERE status IN ({','.join('?' * len(stuck))})""",
            (note, datetime.now(timezone.utc).isoformat(), *stuck),
        )
        await db.commit()
        return cur.rowcount or 0


async def delete_mission(mission_id: str) -> None:
    """Delete a mission, its plan and its LLM usage rows. Crawled documents
    are left in the library — they are shared with searches and other
    missions."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM mission_documents WHERE mission_id = ?", (mission_id,))
        await db.execute("DELETE FROM requirements WHERE mission_id = ?", (mission_id,))
        await db.execute("DELETE FROM llm_calls WHERE mission_id = ?", (mission_id,))
        await db.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
        await db.commit()


async def get_agent_track_records() -> dict[str, dict]:
    """Per-agent stats for the agents index: missions run, % of requirements
    satisfied, distinct sources gathered, and last run. Keyed by agent id."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT m.agent_id AS agent_id,
                      COUNT(DISTINCT m.id) AS missions,
                      MAX(m.created_at) AS last_run,
                      (SELECT COUNT(*) FROM requirements r
                       JOIN missions m2 ON m2.id = r.mission_id
                       WHERE m2.agent_id = m.agent_id) AS req_total,
                      (SELECT COUNT(*) FROM requirements r
                       JOIN missions m2 ON m2.id = r.mission_id
                       WHERE m2.agent_id = m.agent_id AND r.status = 'satisfied') AS req_satisfied,
                      (SELECT COUNT(DISTINCT md.document_id) FROM mission_documents md
                       JOIN missions m3 ON m3.id = md.mission_id
                       WHERE m3.agent_id = m.agent_id) AS sources
               FROM missions m GROUP BY m.agent_id"""
        ) as cur:
            rows = await cur.fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        total = d.get("req_total") or 0
        d["req_pct"] = round(100 * (d.get("req_satisfied") or 0) / total) if total else None
        out[d["agent_id"]] = d
    return out


async def get_missions_enriched(limit: int = 50) -> list[dict]:
    """Missions with requirement coverage + document counts, for the History
    timeline and the Library collection selector."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT m.id, m.question, m.status, m.created_at, m.agent_id, m.job_id,
                      (SELECT COUNT(*) FROM requirements r WHERE r.mission_id = m.id) AS req_total,
                      (SELECT COUNT(*) FROM requirements r WHERE r.mission_id = m.id
                       AND r.status = 'satisfied') AS req_satisfied,
                      (SELECT COUNT(*) FROM requirements r WHERE r.mission_id = m.id
                       AND r.status = 'unmet') AS req_unmet,
                      (SELECT COUNT(DISTINCT md.document_id) FROM mission_documents md
                       WHERE md.mission_id = m.id) AS doc_count
               FROM missions m ORDER BY m.created_at DESC LIMIT ?""",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_distinct_search_queries(limit: int = 100) -> list[dict]:
    """Distinct one-shot search queries (from the searches table, so this
    excludes agentic missions). Used by the Library collection selector."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT query, COUNT(*) AS runs, MAX(executed_at) AS last_run
               FROM searches GROUP BY query ORDER BY last_run DESC LIMIT ?""",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_latest_finished_mission(agent_id: str, question: str, before_mission_id: str) -> Optional[Mission]:
    """Most recent done mission for the same agent+question, excluding the given
    mission. Used for the Phase 2 delta ('what's new since last run')."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT * FROM missions
               WHERE agent_id = ? AND question = ? AND status = 'done' AND id != ?
               ORDER BY finished_at DESC LIMIT 1""",
            (agent_id, question, before_mission_id),
        ) as cur:
            row = await cur.fetchone()
            return Mission(**dict(row)) if row else None


# --- Requirements ---

async def insert_requirement(req: Requirement) -> bool:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO requirements
               (id, mission_id, title, description, rationale, status,
                attempts, next_queries_json, satisfied_doc_ids_json,
                assessment_missing, assessment_confidence, accepted_by_user,
                search_stats_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (req.id, req.mission_id, req.title, req.description, req.rationale,
             req.status, req.attempts, req.next_queries_json,
             req.satisfied_doc_ids_json, req.assessment_missing,
             req.assessment_confidence, req.accepted_by_user,
             req.search_stats_json),
        )
        await db.commit()
        return True


async def update_requirement(req_id: str, **fields) -> None:
    if not fields:
        return
    db_path = get_db_path()
    cols = _validated_columns(fields, _REQUIREMENT_COLUMNS, "requirements")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE requirements SET {cols} WHERE id = ?",
            (*fields.values(), req_id),
        )
        await db.commit()


async def delete_requirement(req_id: str) -> None:
    """Drop a requirement (used when the user cuts one from the plan at the
    approval gate). Also clears its document links so no orphan rows remain."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM mission_documents WHERE requirement_id = ?", (req_id,))
        await db.execute("DELETE FROM requirements WHERE id = ?", (req_id,))
        await db.commit()


async def get_requirements_for_mission(mission_id: str) -> list[Requirement]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM requirements WHERE mission_id = ? ORDER BY rowid", (mission_id,)
        ) as cur:
            rows = await cur.fetchall()
            return [Requirement(**dict(row)) for row in rows]


# --- Mission ↔ document links ---

async def link_mission_document(mission_id: str, requirement_id: str, document_id: str) -> None:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT OR IGNORE INTO mission_documents
               (mission_id, requirement_id, document_id) VALUES (?, ?, ?)""",
            (mission_id, requirement_id, document_id),
        )
        await db.commit()


async def get_mission_documents(mission_id: str) -> list[Document]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT DISTINCT d.* FROM documents d
               JOIN mission_documents m ON m.document_id = d.id
               WHERE m.mission_id = ?
               ORDER BY d.crawled_at DESC""",
            (mission_id,),
        ) as cur:
            rows = await cur.fetchall()
            return [Document(**dict(row)) for row in rows]


async def get_mission_pair_documents(mission_id: str, parent_id: str
                                     ) -> tuple[list[Document], list[Document]]:
    """The documents of a mission and of its parent run, for the compare
    view. Each list is exactly what get_mission_documents returns for that
    mission (same query, same ordering); a missing mission yields []."""
    return (await get_mission_documents(mission_id),
            await get_mission_documents(parent_id))


async def get_requirement_documents(mission_id: str, requirement_id: str) -> list[Document]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT d.* FROM documents d
               JOIN mission_documents m ON m.document_id = d.id
               WHERE m.mission_id = ? AND m.requirement_id = ?
               ORDER BY d.crawled_at DESC""",
            (mission_id, requirement_id),
        ) as cur:
            rows = await cur.fetchall()
            return [Document(**dict(row)) for row in rows]


async def get_prior_mission_urls(mission_id: str) -> set[str]:
    """All document URLs collected by any OTHER mission, for delta detection."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """SELECT DISTINCT d.url FROM documents d
               JOIN mission_documents m ON m.document_id = d.id
               WHERE m.mission_id != ?""",
            (mission_id,),
        ) as cur:
            rows = await cur.fetchall()
            return {row[0] for row in rows}


# --- LLM call telemetry ---

async def insert_llm_call(call: LlmCall) -> None:
    """Record one LLM provider call. Raises on failure; the recording site
    (llm.chat_ex) decides that telemetry must never fail the call itself."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO llm_calls
               (id, mission_id, purpose, tier, model, prompt_tokens,
                completion_tokens, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (call.id, call.mission_id, call.purpose, call.tier, call.model,
             call.prompt_tokens, call.completion_tokens, call.duration_ms,
             call.created_at),
        )
        await db.commit()


async def get_mission_llm_usage(mission_id: str,
                                exclude_purposes: tuple[str, ...] = ()) -> dict:
    """A mission's LLM usage: call count and token sums overall and per
    purpose. A mission with no recorded calls (or no such mission) gets zeros
    and an empty by_purpose; NULL token columns count as 0.

    `exclude_purposes` leaves those purposes out of the top-level totals
    (the token budget skips embed/rerank rows by default) while by_purpose
    still lists every purpose, so telemetry shows the whole spend."""
    excluded = frozenset(exclude_purposes)
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """SELECT purpose, COUNT(*),
                      COALESCE(SUM(prompt_tokens), 0),
                      COALESCE(SUM(completion_tokens), 0)
               FROM llm_calls WHERE mission_id = ?
               GROUP BY purpose ORDER BY purpose""",
            (mission_id,),
        ) as cur:
            rows = await cur.fetchall()
    by_purpose = {
        purpose: {"calls": int(calls), "prompt_tokens": int(prompt),
                  "completion_tokens": int(completion)}
        for purpose, calls, prompt, completion in rows
    }
    counted = [p for purpose, p in by_purpose.items() if purpose not in excluded]
    return {
        "calls": sum(p["calls"] for p in counted),
        "prompt_tokens": sum(p["prompt_tokens"] for p in counted),
        "completion_tokens": sum(p["completion_tokens"] for p in counted),
        "by_purpose": by_purpose,
    }


async def get_requirement_document_map(mission_id: str) -> dict[str, list[str]]:
    """requirement_id -> [document_id] for a mission, from mission_documents
    in link order. Every requirement of the mission has a key (an empty list
    when nothing was linked to it), in plan order; a link whose requirement
    row is gone still appears under its id, after those."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT id FROM requirements WHERE mission_id = ? ORDER BY rowid",
            (mission_id,),
        ) as cur:
            out: dict[str, list[str]] = {row[0]: [] for row in await cur.fetchall()}
        async with db.execute(
            """SELECT requirement_id, document_id FROM mission_documents
               WHERE mission_id = ? ORDER BY rowid""",
            (mission_id,),
        ) as cur:
            for req_id, doc_id in await cur.fetchall():
                out.setdefault(req_id, []).append(doc_id)
    return out


# --- Chunks, embeddings, export state ---

def _batches(items: list, size: int = _IN_BATCH) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _unique(ids: Iterable[str]) -> list[str]:
    """Input order, duplicates dropped."""
    return list(dict.fromkeys(ids))


def chunk_rows_for(document_id: str, chunks) -> list[ChunkRow]:
    """The `chunks` rows a document's chunking.Chunk list becomes."""
    rows = []
    for chunk in chunks:
        text = chunk.text
        rows.append(ChunkRow(
            chunk_id=f"{document_id}#{chunk.idx}", document_id=document_id,
            idx=int(chunk.idx), section=chunk.section, text=text,
            char_len=len(text), content_sha256=chunking.chunk_sha256(text)))
    return rows


async def _replace_document_chunks(db, document_id: str, chunks) -> None:
    """Replace a document's chunk rows on an open connection, inside the
    caller's transaction. Embeddings survive for every chunk whose
    (chunk_id, content_sha256) is unchanged; those of a chunk that vanished
    or whose text changed are deleted, so no vector can describe text that
    is no longer there."""
    rows = chunk_rows_for(document_id, chunks)
    async with db.execute(
        "SELECT chunk_id, content_sha256 FROM chunks WHERE document_id = ?",
        (document_id,),
    ) as cur:
        old = {cid: sha for cid, sha in await cur.fetchall()}
    unchanged = {r.chunk_id for r in rows if old.get(r.chunk_id) == r.content_sha256}
    stale = [cid for cid in _unique([*old, *(r.chunk_id for r in rows)])
             if cid not in unchanged]
    for batch in _batches(stale):
        await db.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({_marks(len(batch))})",
            batch)
    await db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    await db.executemany(
        """INSERT INTO chunks
           (chunk_id, document_id, idx, section, text, char_len, content_sha256)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(r.chunk_id, r.document_id, r.idx, r.section, r.text, r.char_len,
          r.content_sha256) for r in rows],
    )


async def replace_document_chunks(document_id: str, chunks) -> None:
    """Rebuild a document's chunks from a chunking.Chunk list in one
    transaction (see _replace_document_chunks for what happens to its
    embeddings). upsert_document does this itself; this is for rebuilding
    chunks of documents stored before chunking existed."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await _replace_document_chunks(db, document_id, chunks)
        await db.commit()


_CHUNK_COLUMNS = "chunk_id, document_id, idx, section, text, char_len, content_sha256"


def _chunk_row(row) -> ChunkRow:
    return ChunkRow(**dict(row))


async def get_document_chunks(document_id: str) -> list[ChunkRow]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE document_id = ? ORDER BY idx",
            (document_id,),
        ) as cur:
            return [_chunk_row(r) for r in await cur.fetchall()]


async def get_chunks_for_documents(document_ids: Iterable[str]) -> dict[str, list[ChunkRow]]:
    """document_id -> its chunks in idx order. Every requested id has a key
    (an empty list for a document without chunks, or no such document), in
    input order."""
    ids = _unique(document_ids)
    out: dict[str, list[ChunkRow]] = {doc_id: [] for doc_id in ids}
    if not ids:
        return out
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        for batch in _batches(ids):
            async with db.execute(
                f"""SELECT {_CHUNK_COLUMNS} FROM chunks
                    WHERE document_id IN ({_marks(len(batch))})
                    ORDER BY document_id, idx""",
                batch,
            ) as cur:
                for row in await cur.fetchall():
                    chunk = _chunk_row(row)
                    out[chunk.document_id].append(chunk)
    return out


async def get_chunks_missing_embeddings(model: str,
                                        document_ids: Optional[Iterable[str]] = None,
                                        limit: int = 2000) -> list[ChunkRow]:
    """Chunks with no chunk_embeddings row for `model`, optionally only of
    the given documents, in (document_id, idx) order, at most `limit`."""
    limit = max(0, int(limit))
    if limit == 0:
        return []
    ids = None if document_ids is None else _unique(document_ids)
    if ids is not None and not ids:
        return []
    base = f"""SELECT {", ".join("c." + col.strip() for col in _CHUNK_COLUMNS.split(","))}
               FROM chunks c
               LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
               WHERE e.chunk_id IS NULL"""
    out: list[ChunkRow] = []
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if ids is None:
            async with db.execute(
                base + " ORDER BY c.document_id, c.idx LIMIT ?", (model, limit)
            ) as cur:
                return [_chunk_row(r) for r in await cur.fetchall()]
        for batch in _batches(ids):
            remaining = limit - len(out)
            if remaining <= 0:
                break
            async with db.execute(
                base + f" AND c.document_id IN ({_marks(len(batch))})"
                       " ORDER BY c.document_id, c.idx LIMIT ?",
                (model, *batch, remaining),
            ) as cur:
                out.extend(_chunk_row(r) for r in await cur.fetchall())
    return out


async def upsert_chunk_embeddings(rows: list[tuple[str, str, int, bytes]]) -> None:
    """Store vectors as (chunk_id, model, dim, vector) tuples, replacing any
    existing row for the same (chunk_id, model). `vector` is little-endian
    float32 bytes, so it must be exactly dim * 4 bytes long; callers convert
    (storage never imports numpy)."""
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    params = []
    for chunk_id, model, dim, vector in rows:
        dim = int(dim)
        vector = bytes(vector)
        if dim <= 0 or len(vector) != dim * 4:
            raise ValueError(
                f"embedding for {chunk_id!r}: {len(vector)} bytes is not "
                f"{dim} little-endian float32 values")
        params.append((chunk_id, model, dim, vector, now))
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.executemany(
            """INSERT INTO chunk_embeddings (chunk_id, model, dim, vector, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(chunk_id, model) DO UPDATE SET
                   dim=excluded.dim, vector=excluded.vector,
                   created_at=excluded.created_at""",
            params,
        )
        await db.commit()


async def get_chunk_embeddings(model: str, document_ids: Optional[Iterable[str]] = None
                               ) -> list[tuple[str, str, bytes]]:
    """(chunk_id, document_id, vector) for every embedded chunk of `model`,
    optionally only of the given documents, in (document_id, idx) order.
    Only chunks that still exist are returned."""
    ids = None if document_ids is None else _unique(document_ids)
    if ids is not None and not ids:
        return []
    base = """SELECT e.chunk_id, c.document_id, e.vector
              FROM chunk_embeddings e
              JOIN chunks c ON c.chunk_id = e.chunk_id
              WHERE e.model = ?"""
    order = " ORDER BY c.document_id, c.idx"
    out: list[tuple[str, str, bytes]] = []
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        if ids is None:
            async with db.execute(base + order, (model,)) as cur:
                return [(cid, did, bytes(vec)) for cid, did, vec in await cur.fetchall()]
        for batch in _batches(ids):
            async with db.execute(
                base + f" AND c.document_id IN ({_marks(len(batch))})" + order,
                (model, *batch),
            ) as cur:
                out.extend((cid, did, bytes(vec)) for cid, did, vec in await cur.fetchall())
    return out


async def get_embedding_coverage(model: str) -> dict:
    """How much of the library `model` has embedded: document and chunk
    counts, how many chunks have a vector, how many documents have every
    chunk embedded, and how many chunks still need one."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM documents") as cur:
            documents = (await cur.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT document_id) FROM chunks"
        ) as cur:
            chunks, documents_chunked = await cur.fetchone()
        async with db.execute(
            """SELECT COUNT(*) FROM chunk_embeddings e
               JOIN chunks c ON c.chunk_id = e.chunk_id WHERE e.model = ?""",
            (model,),
        ) as cur:
            embedded = (await cur.fetchone())[0]
        async with db.execute(
            """SELECT COUNT(*) FROM (
                   SELECT c.document_id FROM chunks c
                   LEFT JOIN chunk_embeddings e
                     ON e.chunk_id = c.chunk_id AND e.model = ?
                   GROUP BY c.document_id
                   HAVING COUNT(e.chunk_id) = COUNT(*))""",
            (model,),
        ) as cur:
            documents_embedded = (await cur.fetchone())[0]
    return {
        "model": model,
        "documents": int(documents),
        "documents_chunked": int(documents_chunked),
        "documents_embedded": int(documents_embedded),
        "chunks": int(chunks),
        "embedded": int(embedded),
        "missing": int(chunks) - int(embedded),
    }


_EXPORT_COLUMNS = "document_id, collection, content_sha256, embed_model, n_chunks, exported_at"


async def get_document_exports(document_ids: Iterable[str], collection: str
                               ) -> dict[str, DocumentExport]:
    """document_id -> its export row for `collection`; documents never
    exported there have no key."""
    ids = _unique(document_ids)
    out: dict[str, DocumentExport] = {}
    if not ids:
        return out
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        for batch in _batches(ids):
            async with db.execute(
                f"""SELECT {_EXPORT_COLUMNS} FROM document_exports
                    WHERE collection = ? AND document_id IN ({_marks(len(batch))})""",
                (collection, *batch),
            ) as cur:
                for row in await cur.fetchall():
                    exp = DocumentExport(**dict(row))
                    out[exp.document_id] = exp
    return out


async def list_exported_document_ids(collection: str) -> list[str]:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT document_id FROM document_exports WHERE collection = ? ORDER BY document_id",
            (collection,),
        ) as cur:
            return [row[0] for row in await cur.fetchall()]


async def upsert_document_export(row: DocumentExport) -> None:
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO document_exports
               (document_id, collection, content_sha256, embed_model, n_chunks, exported_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(document_id, collection) DO UPDATE SET
                   content_sha256=excluded.content_sha256,
                   embed_model=excluded.embed_model,
                   n_chunks=excluded.n_chunks,
                   exported_at=excluded.exported_at""",
            (row.document_id, row.collection, row.content_sha256, row.embed_model,
             int(row.n_chunks), row.exported_at),
        )
        await db.commit()


async def delete_document_exports(document_ids: Iterable[str], collection: str) -> None:
    ids = _unique(document_ids)
    if not ids:
        return
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        for batch in _batches(ids):
            await db.execute(
                f"""DELETE FROM document_exports
                    WHERE collection = ? AND document_id IN ({_marks(len(batch))})""",
                (collection, *batch),
            )
        await db.commit()


async def get_documents_by_ids(doc_ids: Iterable[str]) -> list[Document]:
    """Full documents (bodies included) in the order the ids were given,
    duplicates dropped, unknown ids skipped."""
    ids = _unique(doc_ids)
    if not ids:
        return []
    found: dict[str, Document] = {}
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        for batch in _batches(ids):
            async with db.execute(
                f"SELECT * FROM documents WHERE id IN ({_marks(len(batch))})", batch
            ) as cur:
                for row in await cur.fetchall():
                    doc = Document(**dict(row))
                    found[doc.id] = doc
    return [found[doc_id] for doc_id in ids if doc_id in found]


async def get_document_mission_ids(document_ids: Iterable[str]) -> dict[str, list[str]]:
    """document_id -> the ids of every mission that linked it, in link
    order, duplicates dropped. Every requested id has a key (an empty list
    for a document no mission collected). The vector export stamps these on
    each point's payload."""
    ids = _unique(document_ids)
    out: dict[str, list[str]] = {doc_id: [] for doc_id in ids}
    if not ids:
        return out
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        for batch in _batches(ids):
            async with db.execute(
                f"""SELECT document_id, mission_id FROM mission_documents
                    WHERE document_id IN ({_marks(len(batch))}) ORDER BY rowid""",
                batch,
            ) as cur:
                for doc_id, mission_id in await cur.fetchall():
                    if mission_id not in out[doc_id]:
                        out[doc_id].append(mission_id)
    return out
