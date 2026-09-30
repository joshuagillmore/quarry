"""Is a crawled page real content, or a block/interstitial page?

Crawlers routinely "succeed" on captcha walls, cookie interstitials and empty
shells. Those pages are worthless to the assessor and the brief, and — worse —
they consume the mission's source budget and crowd real content out of the
LLM's limited view. This is the one place that judgement lives, shared by the
crawler (pre-storage) and the assessor/brief (pre-prompt).
"""

_BLOCK_TITLE_MARKERS = (
    "checking your browser", "just a moment", "recaptcha", "captcha",
    "are you a robot", "are you human", "access denied", "403 forbidden",
    "attention required", "verify you are human", "bot verification",
    "enable javascript", "please enable cookies", "ddos protection",
)

# Below this a page carries no usable information even if it "loaded".
MIN_CONTENT_WORDS = 60

# A title marker only condemns a short page. Real interstitials are a few
# dozen words; an article *about* the error ("How to Fix a 403 Forbidden
# Error") is thousands, and is exactly what a troubleshooting question needs.
BLOCK_TITLE_MAX_WORDS = 300


def looks_like_block_page(title: str | None, word_count: int,
                          min_words: int = MIN_CONTENT_WORDS) -> tuple[bool, str]:
    """Return (is_junk, human-readable reason). Reason is surfaced in the live
    log so a rejected page is explained rather than silently dropped."""
    t = (title or "").strip().lower()
    if word_count < BLOCK_TITLE_MAX_WORDS:
        for marker in _BLOCK_TITLE_MARKERS:
            if marker in t:
                return True, f"anti-bot/interstitial page ({marker})"
    if word_count < min_words:
        return True, f"only {word_count} words of content"
    return False, ""


def is_usable(doc, min_words: int = MIN_CONTENT_WORDS) -> bool:
    """Document-level view of the same judgement."""
    junk, _ = looks_like_block_page(getattr(doc, "title", ""),
                                    getattr(doc, "word_count", 0) or 0,
                                    min_words)
    return not junk
