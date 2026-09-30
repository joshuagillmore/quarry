"""Suite-wide test isolation.

Module import time (before any app module is imported): pin six environment
keys over whatever the developer's shell exported (an exported QUARRY_PASSWORD,
say), then reset the `config.settings` singleton to its declared defaults plus
those pins, so values from the repo's .env never reach the code under test.
DB_PATH -- and with it data/settings.json and data/secret_key -- points at a
throwaway session directory.

Not covered: os.environ itself still carries whatever config's load_dotenv()
read from .env. That only matters to code that builds a fresh Settings() or
reads os.environ directly; tests doing that clear the keys they exercise
(see test_config.py).

Per test (autouse fixture): a fresh SQLite file, a clean job store and login
rate limiter, zero retry/backoff delays, and the settings singleton restored
afterwards so a test that saves overrides cannot leak them into the next one.
"""
import atexit
import os
import shutil
import sys
import tempfile
import time as _real_time

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
_STUBS_DIR = os.path.join(_TESTS_DIR, "stubs")


def _on_sys_path(path: str) -> bool:
    want = os.path.normcase(os.path.abspath(path))
    return any(os.path.normcase(os.path.abspath(p or ".")) == want for p in sys.path)


# pytest.ini's `pythonpath` normally supplies both; this keeps a bare
# `pytest tests/...` from another working directory importable too.
for _p in (_STUBS_DIR, _REPO_ROOT):
    if not _on_sys_path(_p):
        sys.path.insert(0, _p)

_SESSION_DIR = tempfile.mkdtemp(prefix="quarry-tests-")
atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)

# Hard assignment, not setdefault: the point is to override whatever the
# shell exported.
_PINNED_ENV = {
    "QUARRY_PASSWORD": "",
    "QUARRY_BIND": "127.0.0.1",
    "QUARRY_BEHIND_PROXY": "false",
    "FLASK_SECRET_KEY": "test-secret-key-0123456789abcdef0123456789abcdef",
    "DB_PATH": os.path.join(_SESSION_DIR, "research.db"),
    "LLM_PROVIDER_FAST": "",
}
os.environ.update(_PINNED_ENV)

import config  # noqa: E402  -- must follow the environment pinning above

# The singleton was built from env + the repo's .env (a dev box's provider,
# fast model, API key...). Settings also ignores empty env values
# (env_ignore_empty), so an empty pin alone would fall through to .env. Start
# every field from its declared default, then apply the pins.
for _name, _field in config.Settings.model_fields.items():
    setattr(config.settings, _name, _field.get_default(call_default_factory=True))
config.settings.quarry_password = ""
config.settings.quarry_bind = "127.0.0.1"
config.settings.quarry_behind_proxy = False
config.settings.flask_secret_key = _PINNED_ENV["FLASK_SECRET_KEY"]
config.settings.db_path = _PINNED_ENV["DB_PATH"]
config.settings.llm_provider_fast = ""


class _NoSleepTime:
    """Stands in for the `time` module inside search.py only. Patching
    `search.time.sleep` directly would patch the real, shared `time` module and
    turn every sleep in the process (the SSE poll loop, background workers)
    into a busy-wait."""

    @staticmethod
    def sleep(_seconds=0):
        return None

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    import auth
    import jobs
    import llm
    import search
    import storage

    settings_snapshot = config.settings.model_dump()

    db_path = str(tmp_path / "research.db")
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    monkeypatch.setattr(config.settings, "db_path", db_path)

    with jobs._lock:
        saved_store = dict(jobs._store)
        saved_recent = list(jobs._recent_job_ids)
        jobs._store.clear()
        jobs._recent_job_ids.clear()
    with auth._lock:
        saved_failures = {ip: list(ts) for ip, ts in auth._failures.items()}
        auth._failures.clear()

    monkeypatch.setattr(llm, "RETRY_DELAY_S", 0)
    monkeypatch.setattr(search, "time", _NoSleepTime())

    # The Flask app is module-global across tests; make its lazy DB init run
    # again against this test's fresh database.
    app_mod = sys.modules.get("app")
    if app_mod is not None and hasattr(app_mod, "app"):
        monkeypatch.setattr(app_mod.app, "_db_initialized", False, raising=False)

    yield

    for key, value in settings_snapshot.items():
        setattr(config.settings, key, value)
    with jobs._lock:
        jobs._store.clear()
        jobs._store.update(saved_store)
        jobs._recent_job_ids[:] = saved_recent
    with auth._lock:
        auth._failures.clear()
        auth._failures.update(saved_failures)
