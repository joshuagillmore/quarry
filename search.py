"""Web search with engine fallback.

`ddgs` fronts several engines, and they fail independently — the duckduckgo
backend in particular returns "No results found" under light load while brave
and bing answer in under a second. A single-engine search therefore reports
"no search results" for questions the web can clearly answer, which silently
starves a mission. So: try engines in order until one produces results, backing
off on rate limits.
"""
import sys
import time
from urllib.parse import urlparse

from ddgs import DDGS

from config import settings
from models import SearchResult

try:  # ddgs raises a dedicated error for throttling; tolerate older builds
    from ddgs.exceptions import RatelimitException
except Exception:  # pragma: no cover
    class RatelimitException(Exception):
        pass


def _backends() -> list[str]:
    configured = [b.strip() for b in (settings.search_backends or "").split(",") if b.strip()]
    return configured or ["auto", "brave", "bing", "duckduckgo"]


def _rows(query: str, max_results: int, backend: str) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results, backend=backend))


_WEB_SCHEMES = ("http", "https")
_ATTEMPTS_PER_ENGINE = 2


def _is_web_url(url: str) -> bool:
    """Only http(s) results are ever crawled: a file://, javascript: or ftp:
    'result' would point the headless browser somewhere it has no business
    going."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    return parts.scheme in _WEB_SCHEMES and bool(parts.netloc)


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """First engine that returns anything wins. Returns [] only when every
    engine fails or genuinely has nothing."""
    last_error = None
    for backend in _backends():
        for attempt in range(_ATTEMPTS_PER_ENGINE):
            try:
                rows = _rows(query, max_results, backend)
            except RatelimitException as e:
                last_error = e
                # Back off before retrying this engine — but not after its
                # last attempt, when the next engine is a fresh service.
                if attempt + 1 < _ATTEMPTS_PER_ENGINE:
                    time.sleep(1.5 * (attempt + 1))
                continue
            except Exception as e:  # engine-specific failure: fall through
                last_error = e
                break

            results = [
                SearchResult(
                    url=r.get("href") or r.get("url") or "",
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                )
                for r in rows
            ]
            results = [r for r in results if _is_web_url(r.url)]
            if results:
                if backend != _backends()[0]:
                    print(f"[SEARCH] '{query[:60]}' served by fallback engine "
                          f"'{backend}' ({len(results)} results)",
                          file=sys.stderr, flush=True)
                return results
            break  # engine answered with nothing; try the next one

    print(f"[SEARCH] no results for '{query[:60]}' from any engine "
          f"({type(last_error).__name__ if last_error else 'all empty'})",
          file=sys.stderr, flush=True)
    return []
