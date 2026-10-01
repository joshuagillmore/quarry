import os
import json
import aiosqlite
from datetime import datetime, timezone
from typing import Optional
from models import (Document, ExtractedData, SearchRecord, Agent, Mission,
                    Requirement, LlmCall)

DB_PATH = None

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
    on the UNIQUE constraint. The FTS row is replaced in the same transaction."""
    db_path = get_db_path()
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


async def get_mission_llm_usage(mission_id: str) -> dict:
    """A mission's LLM usage: call count and token sums overall and per
    purpose. A mission with no recorded calls (or no such mission) gets zeros
    and an empty by_purpose; NULL token columns count as 0."""
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
    return {
        "calls": sum(p["calls"] for p in by_purpose.values()),
        "prompt_tokens": sum(p["prompt_tokens"] for p in by_purpose.values()),
        "completion_tokens": sum(p["completion_tokens"] for p in by_purpose.values()),
        "by_purpose": by_purpose,
    }
