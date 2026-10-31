#!/usr/bin/env python3
"""Freeze what Coherence's chunker makes of Quarry's chunking fixture.

    COHERENCE_SRC=/home/user/coherence/src python tests/tools/freeze_coherence_expected.py
    COHERENCE_SRC=/home/user/coherence/src python tests/tools/freeze_coherence_expected.py --check

Quarry's ``chunking.chunk_markdown`` is a port of Coherence's
``chunk_spans(spans, LONG_FORM)``. The port is tested without importing
Coherence (``tests/test_chunking.py`` runs in CI, where Coherence is not
checked out), so this script does the import once, here, and writes the
result to ``tests/fixtures/coherence_chunks_expected.json``. The test then
compares the port's output to that file, chunk text for chunk text.

Spans are built from ``tests/fixtures/chunking_fixture.md`` the way
Coherence's realistic indexer gets them from ``corpus/realistic/parse.py``:
one span per paragraph, one per list item (``<li>`` there), with the section
set by the heading in force, as ``{"span_id", "type", "text", "section",
"page", "bbox"}``. The parse here is written on its own and deliberately
does not import ``chunking.py`` from Quarry: if it did, the frozen file would
only prove the port agrees with itself.

``--check`` regenerates into a temporary directory and diffs against the
committed file, exiting non-zero on any difference. Not collected by pytest.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE_MD = REPO / "tests" / "fixtures" / "chunking_fixture.md"
EXPECTED_JSON = REPO / "tests" / "fixtures" / "coherence_chunks_expected.json"
DEFAULT_COHERENCE_SRC = "/home/user/coherence/src"

_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_CLOSING_HASHES = re.compile(r"\s+#+\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*```")


def load_coherence_chunking():
    src = os.environ.get("COHERENCE_SRC", DEFAULT_COHERENCE_SRC)
    if not (Path(src) / "coherence" / "ingest" / "chunking.py").is_file():
        sys.exit(f"COHERENCE_SRC={src!r} has no coherence/ingest/chunking.py")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module("coherence.ingest.chunking")


def markdown_spans(markdown: str) -> list[dict]:
    """Realistic-indexer-shaped spans from a markdown body.

    Mirrors what parse.py's traversal yields for HTML: a heading sets the
    section and emits nothing; a <p> is a span; each <li> is its own span
    (typed ``list_item``); and, since markdown has one thing HTML prose does
    not, a fenced code block is one span kept whole across blank lines.
    """
    spans: list[dict] = []
    section: str | None = None
    lines: list[str] = []
    kind = "paragraph"
    in_fence = False

    def emit() -> None:
        nonlocal lines, kind
        text = "\n".join(lines).strip()
        if text:
            spans.append({
                "span_id": f"s{len(spans) + 1:04d}",
                "type": kind,
                "text": text,
                "section": section,
                "page": None,
                "bbox": None,
            })
        lines, kind = [], "paragraph"

    for line in markdown.splitlines():
        if in_fence:
            lines.append(line)
            if _FENCE.match(line):
                in_fence = False
                emit()
            continue
        if _FENCE.match(line):
            emit()
            in_fence = True
            lines.append(line)
            continue
        if not line.strip():
            emit()
            continue
        m = _HEADING.match(line)
        if m:
            emit()
            section = _CLOSING_HASHES.sub("", m.group(1)).strip()
            continue
        if _LIST_ITEM.match(line):
            emit()
            kind = "list_item"
        lines.append(line)
    emit()
    return spans


def build_expected(chunking_mod, markdown: str) -> dict:
    policy = chunking_mod.LONG_FORM
    spans = markdown_spans(markdown)
    chunks = chunking_mod.chunk_spans(spans, policy)
    return {
        "source": "coherence.ingest.chunking.chunk_spans",
        "policy": {
            "name": policy.name,
            "target_chars": policy.target_chars,
            "respect_sections": policy.respect_sections,
            "prepend_section": policy.prepend_section,
            "segregate_footnotes": policy.segregate_footnotes,
        },
        "fixture": FIXTURE_MD.relative_to(REPO).as_posix(),
        "fixture_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "n_spans": len(spans),
        "spans": [{"span_id": s["span_id"], "type": s["type"],
                   "section": s["section"], "chars": len(s["text"])}
                  for s in spans],
        "chunks": [
            {"idx": i, "section": c.section, "kind": c.kind,
             "span_ids": list(c.span_ids), "text": c.text}
            for i, c in enumerate(chunks)
        ],
    }


def render(expected: dict) -> str:
    return json.dumps(expected, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate to a temp dir and diff against the "
                         "committed fixture; exit 1 on any difference")
    args = ap.parse_args(argv)

    chunking_mod = load_coherence_chunking()
    markdown = FIXTURE_MD.read_text(encoding="utf-8")
    text = render(build_expected(chunking_mod, markdown))

    if not args.check:
        EXPECTED_JSON.write_text(text, encoding="utf-8")
        n = text.count('"idx":')
        print(f"wrote {EXPECTED_JSON.relative_to(REPO).as_posix()} ({n} chunks)")
        return 0

    with tempfile.TemporaryDirectory(prefix="quarry-chunk-check-") as tmp:
        fresh = Path(tmp) / EXPECTED_JSON.name
        fresh.write_text(text, encoding="utf-8")
        if not EXPECTED_JSON.is_file():
            print(f"missing {EXPECTED_JSON}; run without --check to create it")
            return 1
        committed = EXPECTED_JSON.read_text(encoding="utf-8")
        regenerated = fresh.read_text(encoding="utf-8")
    if committed == regenerated:
        print(f"ok: {EXPECTED_JSON.relative_to(REPO).as_posix()} matches "
              f"Coherence chunk_spans output")
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(keepends=True), regenerated.splitlines(keepends=True),
        fromfile="committed", tofile="regenerated")
    sys.stdout.writelines(diff)
    print("DIFF: committed fixture differs from Coherence output; regenerate "
          "without --check and review the change")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
