"""Final stage: synthesize a Markdown brief from a mission's requirements and
collected documents, with [n] citations. Supports a delta block ("what's new
since last run") for scheduled runs.
"""
import json
import re

from models import Document, Mission, Requirement
from llm import chat
from content_quality import is_usable
from prompt_templates import build_brief_prompt


def _coverage_block(requirements: list[Requirement]) -> str:
    lines = []
    for r in requirements:
        mark = {"satisfied": "[x]", "unmet": "[ ] UNMET", "pending": "[ ] pending"}.get(r.status, r.status)
        lines.append(f"- {mark} {r.title}")
    return "\n".join(lines) or "(no requirements)"


MAX_BRIEF_SOURCES = 20


def ordered_sources(docs: list[Document], max_docs: int = MAX_BRIEF_SOURCES) -> list[Document]:
    """The exact source ordering the brief's `[n]` markers refer to: usable
    (non-junk) docs first, never dropping one while slots remain, capped.
    The mission page numbers its source rail with this so citations bind to the
    right document."""
    return sorted(docs, key=lambda d: not is_usable(d))[:max_docs]


def _sources_block(docs: list[Document], max_docs: int = MAX_BRIEF_SOURCES,
                   excerpt_chars: int = 2200) -> str:
    lines = []
    for i, d in enumerate(ordered_sources(docs, max_docs), 1):
        body = (d.content_fit or d.content_markdown or "")[:excerpt_chars]
        lines.append(f"[{i}] {d.title or d.domain} — {d.url}\n{body}")
    return "\n\n".join(lines)


def ordered_sources_for_mission(mission: Mission, docs: list[Document]) -> list[Document]:
    """The source-rail ordering for a mission's page. When the mission stored
    the order its brief was written with (`brief_sources_json`), use exactly
    that — ids no longer present are skipped — followed by any documents the
    brief did not number (e.g. added by a later retask), in ordered_sources
    order. Without a usable stored order, this is ordered_sources(docs)."""
    stored = None
    raw = getattr(mission, "brief_sources_json", None) if mission else None
    if raw:
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            stored = None
    if not isinstance(stored, list):
        return ordered_sources(docs)

    by_id = {d.id: d for d in docs}
    out: list[Document] = []
    seen: set[str] = set()
    for doc_id in stored:
        if isinstance(doc_id, str) and doc_id in by_id and doc_id not in seen:
            out.append(by_id[doc_id])
            seen.add(doc_id)
    rest = [d for d in docs if d.id not in seen]
    out.extend(ordered_sources(rest, max_docs=len(rest)))
    return out


_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
_TAG_NAME_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)")
_LITERAL_TAGS = ("code", "pre")


def linkify_citations(html: str, max_n: int) -> str:
    """Turn `[n]` markers in already-sanitized brief HTML into citation
    controls. Runs AFTER markdown_render.render_markdown (never instead of it):
    the only thing injected is markup built from an integer we re-serialize
    ourselves, so no LLM/page content reaches the DOM unescaped. Out-of-range
    numbers are left as plain text.

    Only text is touched: never the inside of a tag (an href or title that
    contains "[1]" would otherwise get a <button> spliced into the attribute
    value), and never text inside <code>/<pre>, where [1] is literal. The
    input is sanitizer output, so `>` inside attribute values is already
    escaped and splitting on tags is exact."""
    def repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if not 1 <= n <= max_n:
            return m.group(0)
        return (f'<button type="button" class="cite" data-cite="{n}" '
                f'aria-label="Source {n}">{n}</button>')

    parts = _TAG_SPLIT_RE.split(html or "")
    literal_depth = 0
    for i, part in enumerate(parts):
        if i % 2:  # a tag (split() puts captured separators at odd indexes)
            m = _TAG_NAME_RE.match(part)
            if m and m.group(2).lower() in _LITERAL_TAGS:
                if m.group(1):
                    literal_depth = max(0, literal_depth - 1)
                elif not part.rstrip(">").rstrip().endswith("/"):
                    literal_depth += 1
        elif part and not literal_depth:
            parts[i] = _CITE_RE.sub(repl, part)
    return "".join(parts)


def _delta_block(docs: list[Document], new_urls: set[str], max_items: int = 10) -> str:
    """What this run found that its previous run did not, referred to by the
    same [n] the sources block uses plus the domain. Never the page title:
    it is page-controlled text (a prompt-injection vector) and duplicates
    what the numbered source already shows."""
    if not new_urls:
        return ""
    numbered = [(n, d) for n, d in enumerate(ordered_sources(docs), 1) if d.url in new_urls]
    if not numbered:
        return ""
    lines = ["NEW SINCE LAST RUN — these sources were not in the previous run; "
             "lead the brief with what they add, citing them by number:"]
    for n, d in numbered[:max_items]:
        lines.append(f"- [{n}] {d.domain}")
    unlisted = len(new_urls & {d.url for d in docs}) - min(len(numbered), max_items)
    if unlisted > 0:
        lines.append(f"- (+{unlisted} more new source(s) not listed)")
    return "\n".join(lines)


def synthesize_brief(
    mission: Mission,
    requirements: list[Requirement],
    docs: list[Document],
    new_urls: set[str] | None = None,
) -> str:
    """Return Markdown. On LLM failure, return a minimal fallback brief built
    from coverage so a run always produces something readable."""
    coverage = _coverage_block(requirements)
    delta = _delta_block(docs, new_urls or set())
    prompt = build_brief_prompt(mission.question, coverage, _sources_block(docs), delta)
    persona = "You are an expert analyst writing a concise, source-grounded research brief."
    try:
        return chat(persona, prompt, max_tokens=2000, tier="fast")
    except Exception as e:  # noqa: BLE001 - brief must degrade gracefully
        n_sat = sum(1 for r in requirements if r.status == "satisfied")
        lines = [
            f"## Summary",
            f"_Automated brief generation failed ({type(e).__name__}); showing coverage only._",
            "",
            f"Collected {len(docs)} sources. Requirement coverage "
            f"{n_sat}/{len(requirements)} satisfied.",
            "",
            "## Coverage & Gaps",
            coverage,
        ]
        return "\n".join(lines)
