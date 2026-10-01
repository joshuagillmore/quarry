import json
import os
import sys
import tempfile
import threading
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# utf-8-sig strips a BOM (what PowerShell's `-Encoding utf8` writes) so the
# first key is not read with a stray U+FEFF prefix.
load_dotenv(encoding="utf-8-sig")


class Settings(BaseSettings):
    # extra="ignore": .env may legitimately hold keys this class does not
    # declare (OPENAI_API_KEY and friends, which LiteLLM reads itself).
    # env_ignore_empty: `CRAWL_TIMEOUT=` means "use the default", not "fail to
    # parse an empty int".
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        extra="ignore",
        env_ignore_empty=True,
    )

    # API key for whichever hosted LLM provider you point LLM_PROVIDER at.
    # Provider-agnostic: LLM_API_KEY is the name to use. COHERE_API_KEY is kept
    # working because early versions assumed Cohere. Local Ollama needs no key.
    llm_api_key: str = ""
    cohere_api_key: str = ""  # deprecated alias for llm_api_key
    llm_provider: str = "cohere/command-a-03-2025"
    # Optional "fast" tier for summarization-style calls (brief, extraction).
    # Empty -> those calls fall back to llm_provider. For a local Ollama model
    # use e.g. "ollama_chat/qwen2.5:14b" with ollama_api_base set.
    llm_provider_fast: str = ""
    ollama_api_base: str = "http://host.docker.internal:11434"
    search_max_results: int = 5
    # Engines tried in order until one returns results (ddgs backends).
    # The duckduckgo backend alone frequently returns nothing under load.
    search_backends: str = "auto,brave,bing,duckduckgo"
    db_path: str = "data/research.db"
    crawl_timeout: int = 30000
    # When headless Chromium fails a page or gets a block/captcha wall, retry
    # it once with a plain HTTP fetch and convert that HTML instead. Some bot
    # walls only fingerprint the browser; set false to skip the second fetch.
    crawl_fallback: bool = True
    # Per-call LLM request timeout in seconds. Without one, a hung provider
    # holds a mission worker (and its job slot) forever.
    llm_timeout_s: int = 120
    # Default per-mission LLM token budget (prompt + completion across every
    # call the mission makes). 0 = unlimited. Used when a mission's
    # budget_json does not set its own max_llm_tokens; collection stops at the
    # next pass/requirement boundary once the budget is exceeded.
    max_llm_tokens: int = 0
    # Loopback by default: the dev server must not be network-reachable
    # unless asked (the Docker image sets its own bind address).
    flask_host: str = "127.0.0.1"
    flask_port: int = 5000
    flask_debug: bool = False
    flask_secret_key: str = ""
    # Optional login password. Empty = auth disabled (localhost-only posture).
    # Accepts plaintext or a Werkzeug hash (pbkdf2:/scrypt: prefix).
    # Deliberately NOT in OVERRIDE_KEYS: the Settings page must never be able
    # to set or clear it, or an unauthenticated visitor could.
    quarry_password: str = ""
    # Set true when serving behind a TLS-terminating reverse proxy: applies
    # ProxyFix (real client IPs for the login rate limiter) and marks the
    # session cookie Secure. Off for plain localhost HTTP.
    quarry_behind_proxy: bool = False
    # Mirrors the compose-level publish interface (env_file passes it through)
    # so the app can warn when it is exposed beyond loopback without a password.
    quarry_bind: str = "127.0.0.1"
    # Comma-separated extra hostnames accepted in the Host header, on top of
    # localhost / 127.0.0.1 / [::1]. Always enforced while auth is off (a DNS
    # rebinding guard); with QUARRY_PASSWORD set, enforced only if non-empty.
    # Add the name you browse to if it is not localhost.
    quarry_trusted_hosts: str = ""


settings = Settings()


# --- UI-editable overrides (Settings page) ---
# Persisted as JSON in the data volume so they survive container recreate
# (unlike the baked .env), layered over the env-derived defaults at startup,
# and mutated live on save. These take precedence over .env.
OVERRIDE_KEYS = (
    "llm_provider", "llm_provider_fast", "ollama_api_base",
    "llm_api_key", "cohere_api_key", "search_max_results",
)


def active_api_key() -> str:
    """The key handed to hosted providers. Prefers the provider-agnostic
    LLM_API_KEY, falling back to the legacy COHERE_API_KEY so existing installs
    keep working. Empty is fine for Ollama, and for hosted providers LiteLLM
    will still read its own env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)."""
    return (settings.llm_api_key or settings.cohere_api_key or "").strip()

# Seed suggestions for the Settings model dropdowns; user-used models are added.
DEFAULT_KNOWN_MODELS = [
    "cohere/command-a-03-2025",
    "ollama_chat/qwen2.5:14b",
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-4o-mini",
]


def _overrides_path() -> str:
    data_dir = os.path.dirname(settings.db_path) or "."
    return os.path.join(data_dir, "settings.json")


def load_overrides() -> None:
    path = _overrides_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except json.JSONDecodeError as e:
        # Don't silently fall back to .env defaults — a truncated file would
        # otherwise make UI-saved settings (incl. an API key) vanish unnoticed.
        print(f"[CONFIG] WARNING: {path} is corrupt ({e}); ignoring saved "
              f"settings and preserving the file as {path}.corrupt",
              file=sys.stderr, flush=True)
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            pass
        return
    for k in OVERRIDE_KEYS:
        if data.get(k) is not None:
            setattr(settings, k, data[k])


# Serialises save_overrides' read-modify-write: two concurrent Settings saves
# must not each start from the same file and drop the other's change.
_overrides_lock = threading.Lock()


def save_overrides(values: dict) -> None:
    path = _overrides_path()
    with _overrides_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        for k in OVERRIDE_KEYS:
            if k in values and values[k] is not None:
                current[k] = values[k]
                setattr(settings, k, values[k])
        # Remember every model that's been used so it stays in the dropdown
        # even after you overwrite a field.
        models = list(current.get("known_models", []))
        for m in (current.get("llm_provider"), current.get("llm_provider_fast")):
            if m and m not in models:
                models.append(m)
        current["known_models"] = models
        data_dir = os.path.dirname(path) or "."
        os.makedirs(data_dir, exist_ok=True)
        # Atomic write (unique temp file + rename) so a crash mid-dump can't
        # truncate the live file and take UI-saved settings (incl. the API
        # key) with it. mkstemp creates the file 0600 from the start, so the
        # key is never briefly world-readable.
        fd, tmp = tempfile.mkstemp(dir=data_dir, prefix="settings.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def persistent_secret_key() -> str:
    """A stable Flask secret key. Without one, sessions die on every restart —
    tolerable for flash messages, unacceptable once login sessions exist.
    Precedence: FLASK_SECRET_KEY env var, else a key generated once and kept in
    the data volume (0600)."""
    if settings.flask_secret_key:
        return settings.flask_secret_key
    import secrets as _secrets
    path = os.path.join(os.path.dirname(settings.db_path) or ".", "secret_key")
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key
    except FileNotFoundError:
        pass
    key = _secrets.token_hex(32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # Created 0600 (not chmod-ed afterwards) so the key is never briefly
    # readable by other users.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
                 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key)
    try:
        os.chmod(tmp, 0o600)  # a stale tmp from a crash keeps its old mode
    except OSError:
        pass  # best-effort (e.g. Windows bind mounts)
    os.replace(tmp, path)
    return key


def known_models() -> list[str]:
    """Model ids to suggest in the Settings dropdowns: seeds + any ever used +
    the currently active ones."""
    models = list(DEFAULT_KNOWN_MODELS)
    # Same lock as save_overrides: on Windows, replacing a file another thread
    # holds open fails, so a render must not overlap a save.
    with _overrides_lock:
        try:
            with open(_overrides_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
    for m in list(data.get("known_models", [])) + [settings.llm_provider, settings.llm_provider_fast]:
        if m and m not in models:
            models.append(m)
    return models


load_overrides()
