"""Split a stored document's markdown into the chunks that get embedded.

This is a port of Coherence's LONG_FORM chunk policy
(``coherence/ingest/chunking.py``: ``chunk_spans`` under ``LONG_FORM``) onto the
markdown Quarry stores, so a document exported into a Coherence collection is
cut the same way Coherence cuts its own corpus. The boundary rule is copied
verbatim; the only Quarry additions are the markdown-to-span parse at the top
and a hard cap on a single giant paragraph at the bottom.

THE RULES (what does not change between here and Coherence)

  Spans are atomic. A span is a markdown paragraph (a run of non-blank lines),
  a single list item, or a whole fenced code block. A span is never split, so a
  citation into a chunk resolves to a passage someone can read and check.

  Section boundaries are chunk boundaries. An ATX heading (``# ...`` to
  ``###### ...``) sets the current section and is not itself a span. A chunk
  spanning two sections would carry two topics and its section attribution
  would be a lie.

  The section heading is prepended to each chunk (``f"{section}\\n\\n{body}"``)
  when one is set, because a mid-document paragraph is often nearly
  contentless on its own and retrievable under its heading.

  A boundary is otherwise a size decision: a span is appended to the open
  chunk unless that would take the chunk past ``target_chars`` and the chunk
  already holds something.

THE ``indexed_text`` INVARIANT

  ``indexed_text(title, text)`` is the ONE string that is embedded for a chunk:
  title, blank line, chunk text. The chunk text itself (section-prefixed, no
  title) is what gets stored and quoted back. Every path that embeds a chunk --
  the in-app index, the Qdrant export, any future sparse arm -- must build its
  string through this function and nothing else, so that what one arm indexes
  is exactly what another arm indexed. Coherence learned that lesson the hard
  way (five f-strings, two of which disagreed); there is one here.

Stdlib only: this module is imported by storage on every document write.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Coherence LONG_FORM: about 500 tokens, large enough for a complete argument,
# small enough that one embedding is not averaged across several.
CHUNK_TARGET_CHARS = 2000

# Quarry-only guard. Coherence's spans come from an HTML parser that already
# bounds a paragraph at a <p>; crawl4ai markdown can hand us a single "paragraph"
# of tens of thousands of characters (a minified page, a table dumped as one
# line, a transcript with no blank lines). Atomicity is kept up to this size;
# past it a span is cut at the last newline or space before the limit so no
# chunk is unboundedly large. This never fires on normal prose and is the one
# place the port deliberately departs from Coherence.
HARD_MAX_CHARS = 8000

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_TRAILING_HASHES_RE = re.compile(r"\s+#+\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of a document.

    ``text`` is the section-prefixed body with no title: what is stored and
    quoted. The string that is embedded is ``indexed_text(title, text)``.
    """

    idx: int
    section: str | None
    text: str


def indexed_text(title: str, text: str) -> str:
    """The one string that is embedded for a chunk: title, blank line, chunk.

    The title is indexed but is not part of ``Chunk.text``: a chunk from the
    middle of a page rarely names its own subject, so the title makes it
    findable, while keeping it out of the stored text keeps it out of what a
    brief quotes back. Every embedding path must go through this function;
    see the module docstring.
    """
    return f"{title}\n\n{text}"


def document_body(doc) -> str:
    """The markdown that is chunked: ``content_fit``, else ``content_markdown``.

    An empty string counts as missing, because the crawler stores a missing
    ``fit_markdown`` as ``''`` rather than ``NULL`` (see CLAUDE.md).
    """
    fit = getattr(doc, "content_fit", None)
    if fit:
        return fit
    return getattr(doc, "content_markdown", None) or ""


def document_title(doc) -> str:
    """The title used in ``indexed_text``: the page title, else its domain."""
    title = (getattr(doc, "title", None) or "").strip()
    return title or (getattr(doc, "domain", None) or "")


def document_digest(doc) -> str:
    """sha256 of ``indexed_text(title, body)``: what an export compares to
    decide whether a document's vectors are stale. Title and body both feed
    the embedding, so a changed title changes the digest."""
    return hashlib.sha256(
        indexed_text(document_title(doc), document_body(doc)).encode("utf-8")
    ).hexdigest()


def chunk_sha256(text: str) -> str:
    """sha256 of one chunk's stored text; the per-chunk change key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _heading_text(line: str) -> str | None:
    """The section title if ``line`` is an ATX heading, else None.

    Leading hashes are stripped by the match; an optional closing run of
    hashes (``## Title ##``) is stripped too, as CommonMark reads it.
    """
    m = _HEADING_RE.match(line)
    if not m:
        return None
    return _TRAILING_HASHES_RE.sub("", m.group(1)).strip()


def _cap_span(text: str) -> list[str]:
    """Cut a span longer than HARD_MAX_CHARS into pieces no longer than that,
    each cut at the last newline or space before the limit (a hard cut only
    if a run has no whitespace at all). Below the cap the span is returned
    unchanged, so atomicity holds for every normal paragraph."""
    pieces: list[str] = []
    rest = text
    while len(rest) > HARD_MAX_CHARS:
        window = rest[:HARD_MAX_CHARS]
        cut = max(window.rfind("\n"), window.rfind(" "))
        if cut <= 0:
            cut = HARD_MAX_CHARS
        head, rest = rest[:cut].strip(), rest[cut:].strip()
        if head:
            pieces.append(head)
    if rest:
        pieces.append(rest)
    return pieces


def _spans(markdown: str) -> list[tuple[str | None, str]]:
    """Parse markdown into ``(section, text)`` spans, in document order.

    A span is a paragraph (consecutive non-blank lines), one list item (a
    line starting with ``-``/``*``/``+`` or ``1.``/``1)`` plus its continuation
    lines), or one fenced code block kept whole even across blank lines. A
    heading line sets the section for what follows and is not a span.
    """
    spans: list[tuple[str | None, str]] = []
    section: str | None = None
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            for piece in _cap_span(text):
                spans.append((section, piece))
        buf = []

    for line in markdown.splitlines():
        if in_fence:
            buf.append(line)
            if _FENCE_RE.match(line):
                in_fence = False
                flush()
            continue
        if _FENCE_RE.match(line):
            flush()
            in_fence = True
            buf.append(line)
            continue
        if not line.strip():
            flush()
            continue
        heading = _heading_text(line)
        if heading is not None:
            flush()
            section = heading
            continue
        if _LIST_ITEM_RE.match(line):
            flush()
        buf.append(line)
    flush()
    return spans


def chunk_markdown(markdown: str,
                   target_chars: int = CHUNK_TARGET_CHARS) -> list[Chunk]:
    """Chunk a markdown body under the LONG_FORM rule.

    Empty (or whitespace-only) body -> ``[]``. A body that yields no spans at
    all (headings only, say) -> one chunk of the whole body, as Coherence's
    ``chunk_paragraphs`` falls back to ``[body]``.
    """
    if not markdown or not markdown.strip():
        return []

    spans = _spans(markdown)

    # --- Coherence chunk_spans, LONG_FORM, verbatim -------------------------
    # Every Quarry span is kind "body" (there is no footnote typing in
    # markdown), so Coherence's `kind != cur_kind` term is True only on the
    # first span, when buf is empty and it cannot force a flush. It is left
    # out; the other two terms are copied as written.
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    cur_section: str | None = None

    def flush() -> None:
        if not buf:
            return
        body = "\n\n".join(buf)
        if cur_section:
            body = f"{cur_section}\n\n{body}"
        chunks.append(Chunk(idx=len(chunks), section=cur_section, text=body))

    for section, text in spans:
        boundary = (
            section != cur_section
            or (size + len(text) > target_chars and buf)
        )
        if boundary and buf:
            flush()
            buf, size = [], 0
        cur_section = section
        buf.append(text)
        size += len(text)
    flush()
    # -----------------------------------------------------------------------

    if not chunks:
        return [Chunk(idx=0, section=None, text=markdown.strip())]
    return chunks


def document_chunks(doc) -> list[Chunk]:
    """The chunks of a stored document: ``chunk_markdown(document_body(doc))``."""
    return chunk_markdown(document_body(doc))
