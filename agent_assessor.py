"""Gap analysis: judge whether the documents collected for a requirement
satisfy it, and if not, propose refined queries aimed at the gap.
"""
import sys
from dataclasses import dataclass

from llm import chat_json
from models import Document, Requirement
from prompt_templates import build_assess_prompt


@dataclass
class Assessment:
    satisfied: bool
    confidence: str
    missing: str
    next_queries: list[str]


from content_quality import is_usable  # noqa: F401  (re-exported for brief.py)


def _sources_block(docs: list[Document], max_docs: int = 6, excerpt_chars: int = 2000) -> str:
    # Deprioritize junk (captcha/near-empty) rather than dropping it: every
    # collected doc can still feed the assessment when slots remain, so a
    # short-but-valid page is never invisible to the grader.
    ranked = sorted(docs, key=lambda d: not is_usable(d))  # stable: usable first
    lines = []
    for i, d in enumerate(ranked[:max_docs], 1):
        body = (d.content_fit or d.content_markdown or "")[:excerpt_chars]
        lines.append(f"[{i}] {d.title or d.domain} ({d.url})\n{body}")
    return "\n\n".join(lines)


def _is_yes(value) -> bool:
    """Strict reading of the model's verdict. bool("false") is True, so a
    model that answers with a string (or 1, or a list) must not silently
    satisfy a requirement: only JSON true or "true"/"yes" count."""
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in {"true", "yes"}


def assess_requirement(requirement: Requirement, docs: list[Document]) -> Assessment:
    """Returns an Assessment. On LLM/parse failure, returns a not-satisfied
    assessment with no new queries (caller's attempt cap will still advance)."""
    prompt = build_assess_prompt(requirement.title, requirement.description, _sources_block(docs))
    # The persona isn't needed for grading; a tight system message keeps it cheap.
    try:
        parsed, _raw = chat_json(
            "You are a meticulous research analyst grading source coverage.",
            prompt, max_tokens=600,
        )
    except Exception as e:  # noqa: BLE001
        # A provider hiccup must not destroy a mission that has already paid to
        # crawl. Treat it as "not assessed": the requirement stays open and the
        # loop moves on, spending an attempt rather than the whole run.
        print(f"[ASSESS] call failed for {requirement.title!r}: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return Assessment(False, "unknown",
                          f"assessment could not be completed ({type(e).__name__})", [])

    if not parsed:
        return Assessment(False, "low", "could not assess", [])

    raw_next = parsed.get("next_queries")
    next_q = ([q.strip() for q in raw_next if isinstance(q, str) and q.strip()]
              if isinstance(raw_next, list) else [])
    return Assessment(
        satisfied=_is_yes(parsed.get("satisfied")),
        confidence=str(parsed.get("confidence") or "low"),
        missing=str(parsed.get("missing") or ""),
        next_queries=next_q[:3],
    )
