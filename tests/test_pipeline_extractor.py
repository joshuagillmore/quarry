"""extractor.py parses model output with llm._extract_json: fenced or
prose-wrapped objects are kept as structure, anything else is wrapped."""
import json

import extractor
from models import Document


def _doc():
    return Document(id="d1", url="http://x/1", domain="x", title="t",
                    search_query="q", crawled_at="t", content_markdown="body text")


def _extract(monkeypatch, text):
    monkeypatch.setattr(extractor, "chat_ex", lambda *a, **k: (text, "model-x"))
    ext = extractor.extract_from_document(_doc())
    assert ext is not None and ext.model == "model-x"
    return json.loads(ext.data_json)


def test_fenced_json_is_kept_as_structure(monkeypatch):
    assert _extract(monkeypatch, '```json\n{"summary": "s"}\n```') == {"summary": "s"}


def test_prose_wrapped_json_is_kept_as_structure(monkeypatch):
    assert _extract(monkeypatch, 'Here you go: {"summary": "s"} hope it helps') == {"summary": "s"}


def test_plain_object(monkeypatch):
    assert _extract(monkeypatch, '{"a": 1}') == {"a": 1}


def test_non_json_output_is_wrapped(monkeypatch):
    assert _extract(monkeypatch, "no json here") == {"raw_response": "no json here"}
    # A bare JSON scalar is not structured data worth keeping as such.
    assert _extract(monkeypatch, "42") == {"raw_response": "42"}


def test_json_arrays_are_kept_whole(monkeypatch):
    """Whole-text or fenced arrays are stored as-is: taking the first object
    out of `[{...}, {...}]` would silently truncate the extraction."""
    rows = [{"a": 1}, {"b": 2}]
    assert _extract(monkeypatch, json.dumps(rows)) == rows
    assert _extract(monkeypatch, "```json\n" + json.dumps(rows) + "\n```") == rows
    assert _extract(monkeypatch, "[1, 2]") == [1, 2]
