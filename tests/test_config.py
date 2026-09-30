"""Settings loading is tolerant of real-world .env files, and the overrides
and secret-key files are written safely."""
import json
import os
import stat
import threading

import pytest

import config

# Keys the tests below read from a temp .env. The repo's own .env has already
# been loaded into os.environ by config's load_dotenv(), and real env vars
# outrank the env file, so each test clears the ones it exercises.
_KEYS = ("CRAWL_TIMEOUT", "OPENAI_API_KEY", "LLM_TIMEOUT_S",
         "QUARRY_TRUSTED_HOSTS", "FLASK_HOST")


@pytest.fixture()
def clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def _env_file(tmp_path, text: str, bom: bool = False):
    p = tmp_path / "test.env"
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return str(p)


def test_env_file_with_bom_and_unknown_key_loads(tmp_path, clean_env):
    """A BOM (what PowerShell's -Encoding utf8 writes) must not corrupt the
    first key, and a vendor key LiteLLM reads itself must not be rejected."""
    path = _env_file(tmp_path, "CRAWL_TIMEOUT=12345\nOPENAI_API_KEY=x\n", bom=True)
    s = config.Settings(_env_file=path)
    assert s.crawl_timeout == 12345
    assert not hasattr(s, "openai_api_key")


def test_empty_value_keeps_default(tmp_path, clean_env):
    path = _env_file(tmp_path, "CRAWL_TIMEOUT=\n")
    s = config.Settings(_env_file=path)
    assert s.crawl_timeout == 30000


def test_new_settings_and_safe_defaults(tmp_path, clean_env):
    s = config.Settings(_env_file=None)
    assert s.flask_host == "127.0.0.1"
    assert s.llm_timeout_s == 120
    assert s.quarry_trusted_hosts == ""

    path = _env_file(tmp_path, "LLM_TIMEOUT_S=45\nQUARRY_TRUSTED_HOSTS=a.example,b.example\n")
    s = config.Settings(_env_file=path)
    assert s.llm_timeout_s == 45
    assert s.quarry_trusted_hosts == "a.example,b.example"


def _data_dir() -> str:
    return os.path.dirname(config.settings.db_path)


def test_save_overrides_writes_and_leaves_no_temp_files():
    config.save_overrides({"llm_provider": "openai/gpt-4o-mini", "search_max_results": 7})
    with open(os.path.join(_data_dir(), "settings.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["llm_provider"] == "openai/gpt-4o-mini"
    assert saved["search_max_results"] == 7
    assert "openai/gpt-4o-mini" in saved["known_models"]
    assert config.settings.llm_provider == "openai/gpt-4o-mini"
    assert "openai/gpt-4o-mini" in config.known_models()
    assert [n for n in os.listdir(_data_dir()) if n.endswith(".tmp")] == []


def test_concurrent_save_overrides_lose_no_update():
    """Each writer owns one key; after all writers finish, every key must hold
    its writer's final value. An unlocked read-modify-write loses some."""
    keys = ("llm_provider", "llm_provider_fast", "ollama_api_base", "llm_api_key")
    rounds = 40
    errors = []
    start = threading.Barrier(len(keys))

    def writer(key):
        try:
            start.wait()
            for i in range(rounds):
                config.save_overrides({key: f"{key}-{i}"})
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    with open(os.path.join(_data_dir(), "settings.json"), encoding="utf-8") as f:
        saved = json.load(f)
    for k in keys:
        assert saved[k] == f"{k}-{rounds - 1}"
    assert [n for n in os.listdir(_data_dir()) if n.endswith(".tmp")] == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_save_overrides_file_is_owner_only():
    config.save_overrides({"llm_api_key": "k-secret"})
    mode = stat.S_IMODE(os.stat(os.path.join(_data_dir(), "settings.json")).st_mode)
    assert mode == 0o600


def test_persistent_secret_key_is_generated_once(monkeypatch):
    monkeypatch.setattr(config.settings, "flask_secret_key", "")
    first = config.persistent_secret_key()
    assert len(first) >= 32
    assert config.persistent_secret_key() == first
    assert [n for n in os.listdir(_data_dir()) if n.endswith(".tmp")] == []
    if os.name != "nt":
        mode = stat.S_IMODE(os.stat(os.path.join(_data_dir(), "secret_key")).st_mode)
        assert mode == 0o600
