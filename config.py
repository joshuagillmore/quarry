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
    # --- Embeddings & rerank (LiteLLM ids, like llm_provider) ---
    # Dense model every stored document chunk is embedded with (indexing).
    # Empty = embeddings off: ranking paths keep today's behaviour.
    embed_model: str = "cohere/embed-v5.0-pro"
    # Model for query-side embeddings. Embed 5 Pro (index) and Fast (query)
    # share one vector space; empty = same model as embed_model.
    embed_query_model: str = "cohere/embed-v5.0-fast"
    # dimensions= passed to cohere/ models (Matryoshka 256-2048). Other
    # providers return their native size.
    embed_dim: int = 1024
    # Reranker applied to search candidates, assessor sources and the brief's
    # source order. Empty = rerank off.
    rerank_model: str = "cohere/rerank-v5.0-pro"
    # Per-call timeout for embed/rerank requests, in seconds.
    embed_timeout_s: int = 60
    # Embed/rerank calls are recorded in telemetry either way; by default they
    # are excluded from a mission's max_llm_tokens budget. True counts them.
    embed_tokens_count_toward_budget: bool = False
    # --- Corpus export (Qdrant collection shaped for Coherence/Fathom) ---
    # From a container on Docker Desktop, a host Qdrant is reachable at
    # host.docker.internal; the compose `qdrant` profile uses http://qdrant:6333.
    qdrant_url: str = "http://host.docker.internal:6333"
    qdrant_api_key: str = ""
    # Collection name; letters, digits, '_' and '-' only.
    qdrant_export_collection: str = "quarry"
    # Markings stamped on every exported point (Coherence's ACL filter reads
    # them). compartments/releasability are comma-separated lists.
    export_clearance_rank: int = 0
    export_classification: str = "NATO UNCLASSIFIED"
    export_compartments: str = ""
    export_releasability: str = ""
    # Export a mission's documents (and run the corpus self-test) as soon as
    # its brief is written.
    export_on_mission_done: bool = False
    # Where manifests and self-test eval files go. Empty = <dir of
    # DB_PATH>/export (see export_dir()).
    export_dir: str = ""
    # Corpus self-test: recall@k is reported at this k, and the check passes
    # when recall >= the floor. 0.0 = report only, never "fail".
    corpus_check_k: int = 10
    corpus_check_floor: float = 0.0
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
    "embed_model", "embed_query_model", "rerank_model",
    "qdrant_url", "qdrant_api_key", "qdrant_export_collection",
    "export_on_mission_done",
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
DEFAULT_KNOWN_EMBED_MODELS = [
    "cohere/embed-v5.0-pro",
    "cohere/embed-v5.0-fast",
    "cohere/embed-v4.0",
    "ollama/nomic-embed-text",
]
DEFAULT_KNOWN_RERANK_MODELS = [
    "cohere/rerank-v5.0-pro",
    "cohere/rerank-v5.0-fast",
    "cohere/rerank-v4.0-pro",
    "cohere/rerank-v3.5",
]


def export_dir() -> str:
    """Directory for export manifests and corpus self-test eval files:
    EXPORT_DIR, else `export/` next to the SQLite file (so it lands in the
    data volume under Docker)."""
    if settings.export_dir:
        return settings.export_dir
    return os.path.join(os.path.dirname(settings.db_path) or ".", "export")


def csv_list(value: str | None) -> list[str]:
    """Split a comma-separated setting (export_compartments, ...) into its
    non-empty, stripped items. "" -> []."""
    return [p.strip() for p in (value or "").split(",") if p.strip()]


_TRUE_WORDS = ("1", "true", "yes", "on")


def _coerce_override(key: str, value):
    """settings.json is hand-editable, so a bool or int key may come back as
    a string ("true", "7"). Cast to the field's declared type; a value that
    cannot be cast is ignored (None) rather than poisoning the singleton."""
    field = Settings.model_fields.get(key)
    if field is None:
        return value
    ann = field.annotation
    if ann is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in _TRUE_WORDS
        return None
    if ann is int:
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if ann is str:
        return value if isinstance(value, str) else str(value)
    return value


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
            value = _coerce_override(k, data[k])
            if value is not None:
                setattr(settings, k, value)


# settings.json list -> the override keys whose values accumulate into it.
_KNOWN_MODEL_LISTS = (
    ("known_models", ("llm_provider", "llm_provider_fast")),
    ("known_embed_models", ("embed_model", "embed_query_model")),
    ("known_rerank_models", ("rerank_model",)),
)

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
                value = _coerce_override(k, values[k])
                if value is None:
                    continue
                current[k] = value
                setattr(settings, k, value)
        # Remember every model that's been used so it stays in the dropdown
        # even after you overwrite a field.
        for list_key, model_keys in _KNOWN_MODEL_LISTS:
            models = list(current.get(list_key, []))
            for m in (current.get(k) for k in model_keys):
                if m and m not in models:
                    models.append(m)
            current[list_key] = models
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


def _known(seed: list[str], list_key: str, active: tuple[str, ...]) -> list[str]:
    models = list(seed)
    # Same lock as save_overrides: on Windows, replacing a file another thread
    # holds open fails, so a render must not overlap a save.
    with _overrides_lock:
        try:
            with open(_overrides_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
    for m in list(data.get(list_key, [])) + list(active):
        if m and m not in models:
            models.append(m)
    return models


def known_models() -> list[str]:
    """Model ids to suggest in the Settings dropdowns: seeds + any ever used +
    the currently active ones."""
    return _known(DEFAULT_KNOWN_MODELS, "known_models",
                  (settings.llm_provider, settings.llm_provider_fast))


def known_embed_models() -> list[str]:
    """Embedding model ids for the Settings dropdowns (same rule as
    known_models, accumulated under `known_embed_models`)."""
    return _known(DEFAULT_KNOWN_EMBED_MODELS, "known_embed_models",
                  (settings.embed_model, settings.embed_query_model))


def known_rerank_models() -> list[str]:
    """Rerank model ids for the Settings dropdowns (accumulated under
    `known_rerank_models`)."""
    return _known(DEFAULT_KNOWN_RERANK_MODELS, "known_rerank_models",
                  (settings.rerank_model,))


load_overrides()
