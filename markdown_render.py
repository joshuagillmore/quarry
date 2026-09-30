import re

import bleach
import markdown as _md

_ALLOWED_TAGS = [
    "p", "br", "hr",
    "strong", "em", "b", "i", "u", "s", "del", "ins",
    "code", "pre", "blockquote",
    "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "a",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "img",
    "span", "div",
]

_LANGUAGE_CLASS = re.compile(r"language-[A-Za-z0-9_+#.-]+")


def _code_class(tag, name, value):
    """`class` survives only as a fenced-code language marker
    (`language-python`). Anywhere else a crawled page could borrow the app's
    own classes (`flash error`, `btn`) and dress itself up as app chrome."""
    return name == "class" and bool(_LANGUAGE_CLASS.fullmatch(value or ""))


_ALLOWED_ATTRS = {
    "a": ["href", "title", "rel", "target"],
    "img": ["src", "alt", "title"],
    "code": _code_class,
}

_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def _harden_links(attrs, new=False):
    href = attrs.get((None, "href"), "")
    if href.startswith(("http://", "https://")):
        attrs[(None, "rel")] = "noopener nofollow"
        attrs[(None, "target")] = "_blank"
    return attrs


def render_markdown(text: str) -> str:
    if not text:
        return ""
    html = _md.markdown(
        text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    cleaned = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
    return bleach.linkify(cleaned, callbacks=[_harden_links])


# Leading site chrome that crawls pick up before the real content starts. A
# library card exists to help you recognise a page, so a preview of
# "Jump to content Main menu move to sidebar hide" is worse than useless.
_NAV_NOISE = re.compile(
    r"^(jump to (content|navigation).*|skip to (main )?content.*|open main menu|"
    r"main menu.*|move to sidebar.*|toggle .*|navigation|contents|menu|search|"
    r"hide|show|home|log ?in|sign ?in|create account|donate|edit|read|view source|"
    r"view history|tools|appearance|personal tools|uncategorized|\[.*\])$",
    re.IGNORECASE)


# Furniture that reads like a sentence and so survives the prose test. These
# are site-wide, so without this the same preview appears on every page from
# that domain — the opposite of what a card is for.
_CHROME_PHRASES = re.compile(
    r"the content is as wide as possible"          # Wikipedia appearance panel
    r"|toggle the table of contents"
    r"|from wikipedia, the free encyclopedia"      # precedes the real first line
    r"|skip to (main |to )?content"
    r"|open (main )?menu\b|open navigation"
    r"|enable javascript|your browser (does not|doesn't) support"
    r"|accept (all )?cookies|we use cookies",
    re.IGNORECASE)


def _looks_like_prose(line: str) -> bool:
    """Does this line read like article text rather than site furniture?

    Blacklisting nav words is endless whack-a-mole (every site words its menu
    differently), so recognise prose instead: real sentences are long, contain
    sentence punctuation, and are mostly lowercase — menus are short Title Case
    fragments.
    """
    words = line.split()
    if len(words) < 8:
        return False
    if not re.search(r"[.!?](\s|$)", line):
        return False
    if _CHROME_PHRASES.search(line):
        return False
    # A feed/listing strip ("3 min read ... 4 days ago ... 7 min read ...")
    # repeats its unit; prose does not.
    if len(re.findall(r"\bmin read\b|\b\d+ (days?|hours?|minutes?) ago\b", line)) >= 2:
        return False
    capitalised = sum(1 for w in words if w[:1].isupper())
    return capitalised / len(words) < 0.5


def snippet(text: str, limit: int = 170) -> str:
    """A short, human-readable preview of crawled markdown.

    Raw `content_markdown` opens with link syntax and a navigation menu (and
    `content_fit` is often empty), so showing its first 150 characters renders
    as source code or as a menu identical on every page from that site. This
    strips the markup and starts at the first line that actually reads like
    prose.
    """
    plain = to_plain_text(text or "")
    if not plain:
        return ""

    lines = [ln.strip(" \t#*->|_=") for ln in plain.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    start = next((i for i, ln in enumerate(lines) if _looks_like_prose(ln)), None)
    if start is None:
        # No sentence anywhere (link farm, table, stub): fall back to the first
        # line that is at least not obvious chrome, so the card still says
        # something. If every line is chrome, say nothing rather than show a
        # menu — the caller renders "No readable text extracted".
        start = next((i for i, ln in enumerate(lines)
                      if len(ln) >= 3 and not _NAV_NOISE.match(ln)), None)
        if start is None:
            return ""

    kept, budget = [], limit * 3
    for ln in lines[start:]:
        kept.append(ln)
        if sum(len(k) for k in kept) >= budget:
            break

    out = re.sub(r"\s+", " ", " ".join(kept)).strip()

    # Emphasis markers and backslash escapes are markup, not words. An opening
    # ** often sits on a line we skipped, so unpaired leftovers matter too.
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out)
    out = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", out)   # spare snake_case
    out = out.replace("**", "")
    out = re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()

    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return out


def to_plain_text(text: str) -> str:
    """Strip images, link URLs, raw HTML, and reference defs from a markdown
    source. Headings, lists, and paragraph breaks are preserved as text."""
    if not text:
        return ""
    # Adjacent links — a nav bar is [a](u)[b](u)[c](u) — would otherwise fuse
    # into "abc" once the markup is stripped.
    out = re.sub(r"\)\s*\[", ") [", text)
    # Image syntax: ![alt](url) and ![alt][ref] → drop entirely
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)
    out = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", "", out)
    # Link syntax: [text](url) and [text][ref] → keep just text.
    # The label may be empty: removing an image from a linked image
    # ([![alt](img)](url)) leaves [](url), which must disappear too.
    out = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", out)
    # Reference link definitions: "[ref]: https://..." on their own line
    out = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", "", out, flags=re.MULTILINE)
    # Raw HTML tags (crawled markdown sometimes embeds them)
    out = re.sub(r"<[^>]+>", "", out)
    # Collapse runs of blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
