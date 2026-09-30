"""Guard static/style.css against corruption.

A stray shell line once got appended into the stylesheet; browsers then treat it
as a malformed selector and silently swallow every rule after it, so the page
renders with missing styles while the file still "contains" them. These checks
catch that class of breakage without a browser.
"""
import re

import pytest

CSS_PATH = "static/style.css"


@pytest.fixture(scope="module")
def css():
    with open(CSS_PATH, encoding="utf-8") as f:
        return f.read()


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def test_braces_balanced(css):
    body = _strip_comments(css)
    assert body.count("{") == body.count("}"), "unbalanced braces in style.css"


def test_comments_closed(css):
    assert css.count("/*") == css.count("*/"), "unterminated CSS comment"


def test_no_shell_or_heredoc_junk(css):
    for marker in ("grep -c", "cat >>", "printf ", "<<'", "EOF", "#!/"):
        assert marker not in css, f"shell fragment {marker!r} leaked into style.css"


def test_every_declaration_block_is_reachable(css):
    """Outside a block, the only legal things are selectors, at-rules and
    comments. A line with no selector-ish characters that never reaches a brace
    means the parser is off the rails."""
    body = _strip_comments(css)
    depth = 0
    buf = ""
    for ch in body:
        if ch == "{":
            selector = buf.strip()
            if depth == 0 and selector:
                assert not re.search(r"^\s*(grep|cat|printf|echo|python)\b", selector), \
                    f"shell command parsed as selector: {selector[:60]!r}"
            depth += 1
            buf = ""
        elif ch == "}":
            depth = max(0, depth - 1)
            buf = ""
        else:
            buf += ch
    assert depth == 0, "style.css ends inside an unclosed block"
    # Trailing text after the last block must be blank/comments only.
    assert not buf.strip(), f"stray text after final rule: {buf.strip()[:60]!r}"


def test_key_rules_present(css):
    """Spot-check rules the mission surfaces depend on."""
    for sel in (".req-open[hidden]", ".telemetry", ".tele-cell", ".srail",
                ".cite", ".mrow", ".agent-budget", ".coverage",
                ".qadd[hidden]", ".btn:disabled", ".lib-pager",
                '.stage[data-state="error"]'):
        assert sel in css, f"missing rule {sel}"
