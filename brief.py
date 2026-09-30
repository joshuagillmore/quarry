"""Final stage: synthesize a Markdown brief from a mission's requirements and
collected documents, with [n] citations. Supports a delta block ("what's new
since last run") for scheduled runs.
"""
import json
import re

from models import Document, Mission, Requirement
from llm import chat
from agent_assessor import key_terms
from content_quality import is_usable, looks_like_block_page
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


def ordered_sources_for_mission(mission: Mission,
                                docs: list[Document]) -> list[Document | None]:
    """The source-rail ordering for a mission's page. When the mission stored
    the order its brief was written with (`brief_sources_json`), use exactly
    that, followed by any documents the brief did not number (e.g. added by
    a later retask), in ordered_sources order. Without a usable stored
    order, this is ordered_sources(docs) and holds no None.

    Numbering is positional: slot i is the brief's [i+1]. Every stored entry
    keeps its slot, so an id whose document is gone (deleted since), or an
    entry that cannot be used (a repeat, a non-string), is a None slot
    rather than a shift of every later number. Consumers must therefore:
    render a None slot as "source removed"; skip None slots when mapping a
    document to its number (doc_number); and bound linkify_citations by
    position, counting only up to the last non-None slot, never by the
    count of non-None slots, or a document after a removed one would lose
    its link."""
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
    out: list[Document | None] = []
    seen: set[str] = set()
    for doc_id in stored:
        if isinstance(doc_id, str) and doc_id in by_id and doc_id not in seen:
            out.append(by_id[doc_id])
            seen.add(doc_id)
        else:
            out.append(None)
    rest = [d for d in docs if d.id not in seen]
    out.extend(ordered_sources(rest, max_docs=len(rest)))
    return out


# One marker or a comma-separated group: models write both "[2]" and
# "[2,14]" / "[2, 6, 10]"; each number becomes its own control.
_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
# A whole tag, quote-aware: a `>` inside a quoted attribute value does not end
# it. The alternatives are disjoint on their first character, so no
# backtracking blow-up.
_TAG_SPLIT_RE = re.compile(r"""(<(?:[^>"']|"[^"]*"|'[^']*')*>)""")
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
    value), and never text inside <code>/<pre>, where [1] is literal. bleach
    escapes `<` in text but leaves `>` raw inside attribute values
    (title="a > [1]"), so tags are matched quote-aware rather than ending at
    the first `>`."""
    def button(n: int) -> str:
        return (f'<button type="button" class="cite" data-cite="{n}" '
                f'aria-label="Source {n}">{n}</button>')

    def repl(m: "re.Match[str]") -> str:
        nums = [int(x) for x in m.group(1).split(",")]
        if not any(1 <= n <= max_n for n in nums):
            return m.group(0)
        # In-range numbers become controls; out-of-range ones stay as digits
        # so a partly-valid group is never silently dropped.
        return ", ".join(button(n) if 1 <= n <= max_n else str(n) for n in nums)

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


# --- Brief quality checks ---

UNCITED_MIN_CHARS = 80
_HEADING_RE = re.compile(r"#{1,6}\s")
_ITEM_RE = re.compile(r"(?:[-*+]|\d{1,3}[.)])\s+")
_FENCE_PREFIXES = ("```", "~~~")


def _brief_blocks(brief_md: str) -> list[tuple[str, str]]:
    """(section heading, text) for each paragraph and each list item of a
    Markdown brief, in order. Headings themselves and fenced code are not
    blocks; a list item's continuation lines belong to the item."""
    blocks: list[tuple[str, str]] = []
    section = ""
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current:
            blocks.append((section, " ".join(current)))
            current.clear()

    for raw in (brief_md or "").splitlines():
        line = raw.strip()
        if line.startswith(_FENCE_PREFIXES):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line:
            flush()
        elif _HEADING_RE.match(line):
            flush()
            section = line.lstrip("#").strip()
        elif _ITEM_RE.match(line):
            flush()
            current.append(_ITEM_RE.sub("", line, count=1))
        else:
            current.append(line)
    flush()
    return blocks


def _excerpt(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _is_coverage_section(heading: str) -> bool:
    """The brief's "Coverage & Gaps" section talks about the requirements,
    not about what a source says, so it is not held to citations."""
    return heading.lower().startswith("coverage")


def brief_warnings(mission: Mission, requirements: list[Requirement],
                   docs: list[Document], brief_md: str) -> list[dict]:
    """Quality checks on a written brief, as {"kind", "detail"} dicts:

    - "uncited_paragraph": a paragraph or list item of UNCITED_MIN_CHARS+
      characters with no [n] (outside the Coverage & Gaps section);
    - "junk_citation": a cited [n] whose source is a block/near-empty page
      (not is_usable), once per number;
    - "requirement_unmentioned": a requirement none of whose key terms (as
      the assessor computes them) appears in the brief; a requirement with
      no key terms is never flagged.

    [n] is resolved with ordered_sources_for_mission(mission, docs), so pass
    the mission with the brief_sources_json the brief was numbered with. A
    number that is out of range or names a removed source (None slot) is
    not a junk citation. Details carry brief/page text: render escaped."""
    warnings: list[dict] = []

    for section, text in _brief_blocks(brief_md):
        if (len(text) >= UNCITED_MIN_CHARS and not _CITE_RE.search(text)
                and not _is_coverage_section(section)):
            warnings.append({"kind": "uncited_paragraph", "detail": _excerpt(text)})

    slots = ordered_sources_for_mission(mission, docs)
    flagged: set[int] = set()
    for m in _CITE_RE.finditer(brief_md or ""):
        for n in (int(x) for x in m.group(1).split(",")):
            if n in flagged or not 1 <= n <= len(slots):
                continue
            doc = slots[n - 1]
            if doc is None or is_usable(doc):
                continue
            flagged.add(n)
            _junk, why = looks_like_block_page(doc.title, doc.word_count or 0)
            warnings.append({"kind": "junk_citation",
                             "detail": f"[{n}] {doc.domain}: {why}"})

    text = (brief_md or "").lower()
    for r in requirements:
        terms = key_terms(r)
        if terms and not any(t in text for t in terms):
            warnings.append({"kind": "requirement_unmentioned", "detail": r.title})
    return warnings


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
        return chat(persona, prompt, max_tokens=2000, tier="fast",
                    purpose="brief", mission_id=mission.id)
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
