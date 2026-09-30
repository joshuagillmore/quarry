"""render_markdown is the XSS control for every crawled page and LLM brief:
markdown -> bleach allowlist -> linkify. These pin what it must strip."""
import re

from markdown_render import render_markdown


def test_script_tag_removed():
    out = render_markdown("hello <script>alert(1)</script> world")
    assert "<script" not in out.lower()


def test_event_handler_attributes_removed():
    out = render_markdown('<img src="https://x.example/a.png" onerror="alert(1)">')
    assert "onerror" not in out.lower()
    assert "alert(1)" not in out


def test_javascript_links_neutralised_in_any_case():
    for scheme in ("javascript", "JaVaScRiPt", "JAVASCRIPT"):
        out = render_markdown(f"[click]({scheme}:alert(1))")
        assert "javascript:" not in out.lower(), scheme
        assert "href" not in out, scheme


def test_data_uri_images_dropped():
    out = render_markdown("![x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)")
    assert "data:" not in out


def test_iframe_removed():
    out = render_markdown('<iframe src="https://evil.example"></iframe>')
    assert "<iframe" not in out.lower()


def test_class_stripped_from_arbitrary_tags():
    # A crawled page must not be able to style itself as app chrome.
    out = render_markdown('<div class="flash error">Session expired, sign in again</div>')
    assert "class" not in out
    assert "Session expired" in out


def test_fenced_code_keeps_language_class():
    out = render_markdown("```python\nprint('hi')\n```")
    assert 'class="language-python"' in out


def test_code_class_must_be_a_language_class():
    out = render_markdown('<code class="flash error">x</code>')
    assert "class" not in out
    out = render_markdown('<code class="language-js flash">x</code>')
    assert "flash" not in out


def test_links_are_hardened():
    out = render_markdown("[site](https://example.com/page)")
    tag = re.search(r"<a [^>]*>", out).group(0)
    assert 'href="https://example.com/page"' in tag
    assert 'rel="noopener nofollow"' in tag
    assert 'target="_blank"' in tag


def test_bare_urls_are_linkified_and_hardened():
    out = render_markdown("see https://example.com/x for more")
    tag = re.search(r"<a [^>]*>", out).group(0)
    assert 'rel="noopener nofollow"' in tag
    assert 'target="_blank"' in tag
