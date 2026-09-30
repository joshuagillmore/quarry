"""Stage 1 of agentic collection: turn (agent persona + question) into a
collection plan — a list of requirements (EEIs), each seeded with search queries.
"""
import json
import uuid

from llm import chat_json
from models import Agent, Requirement
from prompt_templates import build_plan_prompt


def _text(value) -> str:
    """A model-supplied text field, or "" when it is not a string (a list or
    number would otherwise crash .strip() or be stringified into the plan)."""
    return value.strip() if isinstance(value, str) else ""


def build_collection_plan(agent: Agent, mission_id: str, question: str) -> list[Requirement]:
    """Call the LLM to decompose the question into requirements. Returns
    Requirement objects (not yet persisted). Raises ValueError if the model
    produced nothing usable."""
    parsed, _raw = chat_json(agent.persona_prompt, build_plan_prompt(question), max_tokens=1500)

    items = []
    if parsed and isinstance(parsed.get("requirements"), list):
        items = parsed["requirements"]
    if not items:
        raise ValueError("planner returned no requirements")

    requirements: list[Requirement] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"))
        if not title:
            continue
        raw_queries = item.get("queries")
        queries = ([q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
                   if isinstance(raw_queries, list) else [])
        if not queries:
            queries = [f"{question} {title}"]
        requirements.append(Requirement(
            id=str(uuid.uuid4()),
            mission_id=mission_id,
            title=title[:200],
            description=_text(item.get("description"))[:1000],
            rationale=_text(item.get("rationale"))[:1000],
            status="pending",
            attempts=0,
            next_queries_json=json.dumps(queries[:3]),
        ))

    if not requirements:
        raise ValueError("planner returned no usable requirements")
    return requirements
