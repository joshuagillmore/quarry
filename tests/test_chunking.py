"""chunking.py: the LONG_FORM port, checked against frozen Coherence output.

``coherence_chunks_expected.json`` is what Coherence's own ``chunk_spans``
(LONG_FORM) produced for ``chunking_fixture.md``, frozen by
``tests/tools/freeze_coherence_expected.py``. Nothing here imports Coherence.
"""
import hashlib
import json
import os
import sys

import pytest

import chunking
from chunking import (
    CHUNK_TARGET_CHARS, HARD_MAX_CHARS, Chunk, chunk_markdown, chunk_sha256,
    document_body, document_chunks, document_digest, document_title,
    indexed_text,
)
from models import Document

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_markdown() -> str:
    with open(os.path.join(_FIXTURES, "chunking_fixture.md"), encoding="utf-8") as f:
        return f.read()


def _expected() -> dict:
    with open(os.path.join(_FIXTURES, "coherence_chunks_expected.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _doc(**kw) -> Document:
    base = dict(id="d1", url="https://example.org/a", domain="example.org",
                title="A Title", search_query="q", crawled_at="2026-01-01T00:00:00+00:00")
    base.update(kw)
    return Document(**base)


# --- parity with Coherence -------------------------------------------------

def test_fixture_is_the_one_the_expected_file_was_frozen_from():
    md = _fixture_markdown()
    exp = _expected()
    assert exp["fixture_sha256"] == hashlib.sha256(md.encode("utf-8")).hexdigest(), \
        "chunking_fixture.md changed; rerun tests/tools/freeze_coherence_expected.py"
    assert exp["policy"]["name"] == "long_form"
    assert exp["policy"]["target_chars"] == CHUNK_TARGET_CHARS
    assert exp["policy"]["respect_sections"] and exp["policy"]["prepend_section"]


def test_chunk_spans_rule_matches_coherence():
    exp = _expected()
    got = chunk_markdown(_fixture_markdown())
    assert [c.text for c in got] == [c["text"] for c in exp["chunks"]]
    assert [c.section for c in got] == [c["section"] for c in exp["chunks"]]
    assert [c.idx for c in got] == list(range(len(exp["chunks"])))


def test_fixture_exercises_every_rule():
    """The parity test only proves something if the fixture hits each rule."""
    exp = _expected()
    chunks = exp["chunks"]
    assert len(chunks) >= 6
    assert chunks[0]["section"] is None                       # text before any heading
    assert len({c["section"] for c in chunks if c["section"]}) == 3
    assert any(len(c["text"]) > CHUNK_TARGET_CHARS for c in chunks)   # oversize paragraph
    assert any(len(c["span_ids"]) > 1 for c in chunks)         # spans grouped by size
    assert "Section boundaries are chunk boundaries" in {c["section"] for c in chunks}  # closing hashes stripped
    assert any(s["type"] == "list_item" for s in exp["spans"])
    code = [c for c in chunks if "```python" in c["text"]]
    assert len(code) == 1 and "```python\ndef indexed_text" in code[0]["text"]


# --- the rules, one at a time ---------------------------------------------

def test_heading_sets_section_and_is_prepended():
    out = chunk_markdown("intro para\n\n# Alpha\n\nfirst\n\nsecond")
    assert [c.section for c in out] == [None, "Alpha"]
    assert out[0].text == "intro para"
    assert out[1].text == "Alpha\n\nfirst\n\nsecond"


def test_heading_closing_hashes_are_stripped_but_inline_hash_kept():
    out = chunk_markdown("## Title ##\n\nbody")
    assert out[0].section == "Title"
    out = chunk_markdown("## C# basics\n\nbody")
    assert out[0].section == "C# basics"


def test_heading_needs_space_after_hashes():
    out = chunk_markdown("#hashtag line\n\nbody")
    assert out[0].section is None
    assert out[0].text == "#hashtag line\n\nbody"


def test_section_change_forces_boundary_even_when_short():
    out = chunk_markdown("# A\n\nx\n\n# B\n\ny")
    assert [(c.section, c.text) for c in out] == [("A", "A\n\nx"), ("B", "B\n\ny")]


def test_size_rule_groups_until_target_then_flushes():
    p = "w" * 900
    out = chunk_markdown("\n\n".join([p, p, p]))
    # 900 + 900 <= 2000 fits; the third would make 2700 > 2000 -> new chunk.
    assert [c.text for c in out] == [f"{p}\n\n{p}", p]


def test_paragraph_over_target_is_never_split():
    p = "x" * 3000
    out = chunk_markdown(f"a\n\n{p}\n\nb")
    assert [c.text for c in out] == ["a", p, "b"]


def test_oversize_paragraph_only_hits_hard_max():
    words = ("word " * 5000).strip()         # ~25k chars, one paragraph
    assert len(words) > 3 * HARD_MAX_CHARS
    out = chunk_markdown(words)
    assert len(out) > 1
    assert all(len(c.text) <= HARD_MAX_CHARS for c in out)
    for c in out:                            # cut on whitespace, never mid-word
        assert not c.text.startswith(" ") and not c.text.endswith(" ")
        assert set(c.text.split()) == {"word"}
    assert " ".join(c.text for c in out).split() == words.split()   # nothing lost
    # Below the cap, nothing is cut, however far over the target it is.
    assert len(chunk_markdown("y" * HARD_MAX_CHARS)) == 1


def test_oversize_paragraph_cuts_at_newline_when_present():
    line = "z" * 5000
    out = chunk_markdown(f"{line}\n{line}")
    assert [c.text for c in out] == [line, line]


def test_empty_body_gives_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n \n") == []


def test_body_with_no_spans_is_one_whole_chunk():
    out = chunk_markdown("# Only\n\n## Headings\n")
    assert out == [Chunk(idx=0, section=None, text="# Only\n\n## Headings")]


def test_list_items_are_separate_spans():
    out = chunk_markdown("- one\n- two\n  continued\n* three\n1. four\n2) five")
    assert out[0].text == "- one\n\n- two\n  continued\n\n* three\n\n1. four\n\n2) five"
    big = "- " + "i" * 1500
    out = chunk_markdown("\n".join([big, big]))
    assert [c.text for c in out] == [big, big]      # divided at the item boundary


def test_code_fence_is_one_span_across_blank_lines():
    md = "before\n\n```\nline one\n\nline two\n```\n\nafter"
    out = chunk_markdown(md)
    assert out[0].text == "before\n\n```\nline one\n\nline two\n```\n\nafter"
    big = "p" * 1995                        # + 13 chars of fence > target
    out = chunk_markdown(f"{big}\n\n```\na\n\nb\n```")
    assert [c.text for c in out] == [big, "```\na\n\nb\n```"]


def test_unclosed_fence_runs_to_end():
    out = chunk_markdown("```\na\n\nb")
    assert out[0].text == "```\na\n\nb"


def test_custom_target_is_honoured():
    out = chunk_markdown("aaaa\n\nbbbb\n\ncccc", target_chars=8)
    assert [c.text for c in out] == ["aaaa\n\nbbbb", "cccc"]


def test_chunk_is_frozen_with_sequential_idx():
    out = chunk_markdown("# S\n\na\n\n# T\n\nb")
    assert [c.idx for c in out] == [0, 1]
    with pytest.raises(Exception):
        out[0].idx = 5


# --- document helpers -----------------------------------------------------

def test_indexed_text_shape():
    assert indexed_text("Title", "chunk text") == "Title\n\nchunk text"
    assert indexed_text("", "t") == "\n\nt"


def test_document_body_prefers_fit_and_treats_empty_as_missing():
    assert document_body(_doc(content_fit="fit", content_markdown="raw")) == "fit"
    assert document_body(_doc(content_fit="", content_markdown="raw")) == "raw"
    assert document_body(_doc(content_fit=None, content_markdown="raw")) == "raw"
    assert document_body(_doc(content_fit="", content_markdown=None)) == ""


def test_document_title_falls_back_to_domain():
    assert document_title(_doc(title="A Title")) == "A Title"
    assert document_title(_doc(title="  ")) == "example.org"
    assert document_title(_doc(title=None)) == "example.org"


def test_document_digest_is_sha256_of_indexed_text_and_tracks_title_and_body():
    d = _doc(content_fit="body")
    want = hashlib.sha256("A Title\n\nbody".encode("utf-8")).hexdigest()
    assert document_digest(d) == want
    assert document_digest(_doc(content_fit="body", title="Other")) != want
    assert document_digest(_doc(content_fit="other")) != want
    assert document_digest(_doc(content_fit="", content_markdown="body")) == want


def test_chunk_sha256():
    assert chunk_sha256("abc") == hashlib.sha256(b"abc").hexdigest()
    assert chunk_sha256("é") == hashlib.sha256("é".encode("utf-8")).hexdigest()


def test_document_chunks_uses_document_body():
    d = _doc(content_fit="# H\n\npara", content_markdown="ignored")
    assert document_chunks(d) == [Chunk(idx=0, section="H", text="H\n\npara")]
    assert document_chunks(_doc(content_fit="", content_markdown=None)) == []


def test_module_is_stdlib_only():
    src_dir = os.path.dirname(os.path.abspath(chunking.__file__))
    with open(os.path.join(src_dir, "chunking.py"), encoding="utf-8") as f:
        lines = [ln.split()[1] for ln in f
                 if ln.startswith("import ") or ln.startswith("from ")]
    stdlib = set(sys.stdlib_module_names)
    assert lines and all(name.split(".")[0] in stdlib for name in lines), lines
