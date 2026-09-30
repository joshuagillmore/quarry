from llm import _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_prose():
    assert _extract_json('Sure, here: {"a": 1} — done') == {"a": 1}


def test_no_json():
    assert _extract_json("no json at all") is None


# Only a JSON *object* is a usable answer: callers do parsed.get(...).

def test_list_is_not_an_object():
    assert _extract_json("[1, 2]") is None


def test_fenced_list_is_not_an_object():
    assert _extract_json("```json\n[1, 2]\n```") is None


def test_scalars_are_not_objects():
    assert _extract_json('"just a string"') is None
    assert _extract_json("42") is None
    assert _extract_json("") is None


def test_stray_braces_before_the_object():
    assert _extract_json('Use {curly} braces, then {"a": 1} ok') == {"a": 1}


def test_list_before_the_object():
    assert _extract_json('[1] and then {"ok": true}') == {"ok": True}


def test_nested_object_is_returned_whole():
    assert _extract_json('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}


def test_array_of_objects_is_not_scanned_into():
    """A valid JSON array is an answer in its own right; picking its first
    element would silently drop the rest."""
    assert _extract_json('[{"a": 1}]') is None
    assert _extract_json('[{"a": 1}, {"b": 2}]') is None
    assert _extract_json('```json\n[{"a": 1}, {"b": 2}]\n```') is None


def test_invalid_fence_then_object_in_prose_still_found():
    assert _extract_json('```\nnot json\n```\nanswer: {"a": 1}') == {"a": 1}
