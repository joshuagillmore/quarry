"""Gap analysis: judge whether the documents collected for a requirement
satisfy it, and if not, propose refined queries aimed at the gap.
"""
import re
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


# Function words long enough to pass the 5-letter cut; they say nothing
# about what a requirement is about.
_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "along", "among", "around",
    "because", "before", "being", "below", "between", "beyond", "could",
    "during", "either", "every", "might", "neither", "other", "others",
    "should", "since", "their", "theirs", "there", "these", "those",
    "though", "through", "under", "until", "where", "whether", "which",
    "while", "whose", "within", "without", "would",
})
# Words of 5+ letters (letters only: digits and underscores split words).
_TERM_RE = re.compile(r"[^\W\d_]{5,}")

NO_TERM_OVERLAP = "no collected source mentions the requirement's key terms"
_CLAUSE_END_RE = re.compile(r"[.;:?!\n]")


def key_terms(requirement: Requirement) -> set[str]:
    """The words that say what a requirement is about: lowercased words of 5+
    letters from its title and description, minus a few function words. May
    be empty (a title of short words only), in which case nothing can be
    judged from it."""
    text = f"{requirement.title or ''} {requirement.description or ''}".lower()
    return set(_TERM_RE.findall(text)) - _STOPWORDS


def _requery(requirement: Requirement) -> list[str]:
    """A fresh query for a requirement nothing on-topic was found for: its
    title plus the first clause of its description (when that adds
    anything), so the next pass does not repeat the queries that missed."""
    title = (requirement.title or "").strip()
    clause = _CLAUSE_END_RE.split(requirement.description or "", maxsplit=1)[0].strip()
    query = f"{title} {clause}" if clause and clause.lower() not in title.lower() else title
    query = query[:200].strip()
    return [query] if query else []


def _mentions_any(docs: list[Document], terms: set[str]) -> bool:
    for d in docs:
        body = (d.content_fit or d.content_markdown or "").lower()
        if any(term in body for term in terms):
            return True
    return False


def assess_requirement(requirement: Requirement, docs: list[Document],
                       search_note: str = "") -> Assessment:
    """Returns an Assessment. On LLM/parse failure, returns a not-satisfied
    assessment with no new queries (caller's attempt cap will still advance).

    When the requirement has key terms and no collected source mentions any
    of them (or nothing was collected), the answer is already known: not
    satisfied, with no LLM call, re-tasked with a query built from the
    requirement itself (_requery). A requirement without key terms is always
    sent to the LLM. `search_note` summarises this pass's searches for the
    prompt."""
    terms = key_terms(requirement)
    if terms and not _mentions_any(docs, terms):
        print(f"[ASSESS] skipped LLM for {requirement.title!r}: none of "
              f"{len(docs)} source(s) mentions its key terms",
              file=sys.stderr, flush=True)
        return Assessment(False, "low", NO_TERM_OVERLAP, _requery(requirement))

    prompt = build_assess_prompt(requirement.title, requirement.description,
                                 _sources_block(docs), search_note=search_note)
    # The persona isn't needed for grading; a tight system message keeps it cheap.
    try:
        parsed, _raw = chat_json(
            "You are a meticulous research analyst grading source coverage.",
            prompt, max_tokens=600,
            purpose="assess", mission_id=requirement.mission_id,
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
