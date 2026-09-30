import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str


class Agent(BaseModel):
    """A saved, reusable expert persona that plans and runs collection."""
    id: str
    name: str
    expertise: str
    persona_prompt: str
    default_max_passes: int = 4
    default_max_sources: int = 30
    default_per_req_attempts: int = 3
    # Cron expression + the standing question an unattended run researches.
    schedule_cron: Optional[str] = None
    schedule_question: Optional[str] = None
    active: int = 1
    created_at: str


class Requirement(BaseModel):
    """An EEI — one collection requirement within a mission's plan."""
    id: str
    mission_id: str
    title: str
    description: str = ""
    rationale: str = ""
    status: str = "pending"  # pending | satisfied | unmet
    attempts: int = 0
    next_queries_json: Optional[str] = None  # JSON list[str] of queries to run next
    satisfied_doc_ids_json: Optional[str] = None  # JSON list[str]
    # The assessor's own reasoning about this requirement, surfaced in the UI.
    assessment_missing: Optional[str] = None   # what the sources still lack
    assessment_confidence: Optional[str] = None  # high | medium | low
    # Set when the user overrides the assessor and accepts a gap as-is, so the
    # UI can say so instead of implying the assessor was satisfied.
    accepted_by_user: int = 0
    # JSON list of {"pass": int, "query": str, "engine": str|null,
    # "results": int}, one entry per search run for this requirement.
    search_stats_json: Optional[str] = None


class Mission(BaseModel):
    """One execution of an agent against a question."""
    id: str
    agent_id: str
    question: str
    status: str = "planning"  # planning|awaiting_approval|collecting|synthesizing|done|error
    plan_json: Optional[str] = None
    budget_json: Optional[str] = None
    brief_markdown: Optional[str] = None
    # JSON list[str] of document ids in the exact order the brief's [n]
    # citations were numbered, so the source rail always matches the text.
    brief_sources_json: Optional[str] = None
    # JSON list of {"kind": str, "detail": str}: quality checks run on the
    # brief after synthesis (uncited paragraphs, junk citations, ...).
    brief_warnings_json: Optional[str] = None
    job_id: Optional[str] = None
    parent_mission_id: Optional[str] = None  # Phase 2 delta lineage
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class Document(BaseModel):
    id: str
    url: str
    domain: str
    title: Optional[str] = None
    search_query: str
    crawled_at: str
    content_markdown: Optional[str] = None
    content_fit: Optional[str] = None
    word_count: int = 0
    links_internal: int = 0
    links_external: int = 0
    metadata_json: Optional[str] = None


class ExtractedData(BaseModel):
    id: str
    document_id: str
    model: str
    extracted_at: str
    prompt: str
    data_json: str


class SearchRecord(BaseModel):
    id: str
    query: str
    executed_at: str
    result_count: int
    job_id: Optional[str] = None


@dataclass(kw_only=True)
class LlmCall:
    """One LLM provider call that returned: what it was for, which model
    actually answered, the tokens the provider reported, and how long it took.
    `mission_id` is None for calls outside a mission (one-shot extraction).
    Keyword-only, so a positional call cannot silently swap two fields."""
    purpose: str   # plan | assess | brief | extract | chat
    tier: str      # reasoning | fast (the tier requested, not who answered)
    model: str     # the model id that actually produced the response
    mission_id: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
