"""Embedding, rerank and corpus-export settings: the Cohere 5.0 defaults, the
Settings-page overrides (and what they must never do: echo or clear the
Qdrant key), the known-model accumulation and the small helpers."""
import json
import os

import pytest

import config

_ENV_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".env.example")

_NEW_KEYS = ("EMBED_MODEL", "EMBED_QUERY_MODEL", "EMBED_DIM", "RERANK_MODEL",
             "EMBED_TIMEOUT_S", "EMBED_TOKENS_COUNT_TOWARD_BUDGET",
             "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_EXPORT_COLLECTION",
             "EXPORT_CLEARANCE_RANK", "EXPORT_CLASSIFICATION",
             "EXPORT_COMPARTMENTS", "EXPORT_RELEASABILITY",
             "EXPORT_ON_MISSION_DONE", "EXPORT_DIR",
             "CORPUS_CHECK_K", "CORPUS_CHECK_FLOOR")


@pytest.fixture()
def clean_env(monkeypatch):
    # The repo's .env is already in os.environ (config's load_dotenv) and real
    # env vars outrank an env file, so clear the keys these tests read.
    for k in _NEW_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def client():
    import app as app_mod
    app_mod.initialize()
    return app_mod.app.test_client()


def _data_dir() -> str:
    return os.path.dirname(config.settings.db_path)


def _saved() -> dict:
    with open(os.path.join(_data_dir(), "settings.json"), encoding="utf-8") as f:
        return json.load(f)


# --- defaults -------------------------------------------------------------

def test_default_models_are_cohere_v5(clean_env):
    s = config.Settings(_env_file=None)
    assert s.embed_model == "cohere/embed-v5.0-pro"
    assert s.embed_query_model == "cohere/embed-v5.0-fast"
    assert s.rerank_model == "cohere/rerank-v5.0-pro"
    assert s.embed_dim == 1024
    assert s.embed_timeout_s == 60
    assert s.embed_tokens_count_toward_budget is False
    # The suite's singleton starts from the same defaults (conftest).
    assert config.settings.embed_model == "cohere/embed-v5.0-pro"
    assert config.settings.rerank_model == "cohere/rerank-v5.0-pro"


def test_export_defaults(clean_env):
    s = config.Settings(_env_file=None)
    assert s.qdrant_url == "http://host.docker.internal:6333"
    assert s.qdrant_api_key == ""
    assert s.qdrant_export_collection == "quarry"
    assert s.export_clearance_rank == 0
    assert s.export_classification == "NATO UNCLASSIFIED"
    assert s.export_compartments == "" and s.export_releasability == ""
    assert s.export_on_mission_done is False
    assert s.export_dir == ""
    assert s.corpus_check_k == 10
    assert s.corpus_check_floor == 0.0


def test_env_file_sets_the_new_keys(tmp_path, clean_env):
    p = tmp_path / "t.env"
    p.write_text("EMBED_MODEL=\nRERANK_MODEL=ollama/bge-reranker\nEMBED_DIM=512\n"
                 "EMBED_TOKENS_COUNT_TOWARD_BUDGET=true\nEXPORT_ON_MISSION_DONE=1\n"
                 "EXPORT_COMPARTMENTS=A, B ,,C\nCORPUS_CHECK_FLOOR=0.5\n", encoding="utf-8")
    s = config.Settings(_env_file=str(p))
    # An empty value means "use the default" (env_ignore_empty), so turning
    # embeddings off is a Settings-page action, not an empty .env line.
    assert s.embed_model == "cohere/embed-v5.0-pro"
    assert s.rerank_model == "ollama/bge-reranker"
    assert s.embed_dim == 512
    assert s.embed_tokens_count_toward_budget is True
    assert s.export_on_mission_done is True
    assert config.csv_list(s.export_compartments) == ["A", "B", "C"]
    assert s.corpus_check_floor == 0.5


def test_env_example_documents_every_new_key_with_its_default(clean_env):
    with open(_ENV_EXAMPLE, encoding="utf-8") as f:
        keys = {line.split("=", 1)[0] for line in f.read().splitlines()
                if "=" in line and not line.lstrip().startswith("#")}
    assert set(_NEW_KEYS) <= keys
    s = config.Settings(_env_file=_ENV_EXAMPLE)
    d = config.Settings(_env_file=None)
    for name in (k.lower() for k in _NEW_KEYS):
        assert getattr(s, name) == getattr(d, name), name


# --- overrides ------------------------------------------------------------

def test_override_keys_cover_the_settings_page_fields():
    for k in ("embed_model", "embed_query_model", "rerank_model", "qdrant_url",
              "qdrant_api_key", "qdrant_export_collection", "export_on_mission_done"):
        assert k in config.OVERRIDE_KEYS, k
    # Budget accounting and markings stay .env-only (not UI-editable).
    for k in ("embed_dim", "embed_tokens_count_toward_budget", "export_clearance_rank",
              "export_classification", "export_dir", "corpus_check_floor"):
        assert k not in config.OVERRIDE_KEYS, k


def test_save_overrides_handles_bool_and_blank_model():
    config.save_overrides({"embed_model": "", "rerank_model": "cohere/rerank-v5.0-fast",
                           "export_on_mission_done": True})
    assert config.settings.embed_model == ""
    assert config.settings.rerank_model == "cohere/rerank-v5.0-fast"
    assert config.settings.export_on_mission_done is True
    saved = _saved()
    assert saved["embed_model"] == ""
    assert saved["export_on_mission_done"] is True
    config.save_overrides({"export_on_mission_done": False})
    assert config.settings.export_on_mission_done is False
    assert _saved()["export_on_mission_done"] is False


def test_load_overrides_coerces_hand_edited_types(monkeypatch):
    """settings.json is hand-editable: a bool written as "true" and an int as
    "7" must land typed, and garbage must not reach the singleton."""
    os.makedirs(_data_dir(), exist_ok=True)
    with open(os.path.join(_data_dir(), "settings.json"), "w", encoding="utf-8") as f:
        json.dump({"export_on_mission_done": "true", "search_max_results": "7",
                   "embed_model": "ollama/nomic-embed-text",
                   "qdrant_export_collection": 12}, f)
    config.load_overrides()
    assert config.settings.export_on_mission_done is True
    assert config.settings.search_max_results == 7
    assert config.settings.embed_model == "ollama/nomic-embed-text"
    assert config.settings.qdrant_export_collection == "12"

    with open(os.path.join(_data_dir(), "settings.json"), "w", encoding="utf-8") as f:
        json.dump({"export_on_mission_done": "no", "search_max_results": "lots"}, f)
    config.load_overrides()
    assert config.settings.export_on_mission_done is False
    assert config.settings.search_max_results == 7   # unparseable int ignored


def test_known_embed_and_rerank_models_accumulate():
    assert config.known_embed_models()[:4] == ["cohere/embed-v5.0-pro", "cohere/embed-v5.0-fast",
                                               "cohere/embed-v4.0", "ollama/nomic-embed-text"]
    assert config.known_rerank_models()[:4] == ["cohere/rerank-v5.0-pro", "cohere/rerank-v5.0-fast",
                                                "cohere/rerank-v4.0-pro", "cohere/rerank-v3.5"]
    config.save_overrides({"embed_model": "openai/text-embedding-3-small",
                           "embed_query_model": "ollama/mxbai-embed-large",
                           "rerank_model": "ollama/bge-reranker-v2-m3"})
    # Overwrite again: the earlier ids must survive in the dropdown.
    config.save_overrides({"embed_model": "cohere/embed-v5.0-pro",
                           "rerank_model": "cohere/rerank-v5.0-pro"})
    saved = _saved()
    assert "openai/text-embedding-3-small" in saved["known_embed_models"]
    assert "ollama/mxbai-embed-large" in saved["known_embed_models"]
    assert "ollama/bge-reranker-v2-m3" in saved["known_rerank_models"]
    assert "openai/text-embedding-3-small" in config.known_embed_models()
    assert "ollama/bge-reranker-v2-m3" in config.known_rerank_models()
    # The lists don't bleed into each other or into the chat-model list.
    assert "ollama/bge-reranker-v2-m3" not in config.known_embed_models()
    assert "openai/text-embedding-3-small" not in config.known_models()
    assert len(config.known_embed_models()) == len(set(config.known_embed_models()))


# --- helpers --------------------------------------------------------------

def test_export_dir_defaults_next_to_the_db(monkeypatch):
    monkeypatch.setattr(config.settings, "db_path", os.path.join("some", "dir", "research.db"))
    monkeypatch.setattr(config.settings, "export_dir", "")
    assert config.export_dir() == os.path.join("some", "dir", "export")
    monkeypatch.setattr(config.settings, "db_path", "research.db")
    assert config.export_dir() == os.path.join(".", "export")
    monkeypatch.setattr(config.settings, "export_dir", "/srv/quarry-export")
    assert config.export_dir() == "/srv/quarry-export"


def test_csv_list():
    assert config.csv_list("") == []
    assert config.csv_list(None) == []
    assert config.csv_list("NATO, FVEY ,,") == ["NATO", "FVEY"]
    assert config.csv_list("  one  ") == ["one"]


# --- Settings page --------------------------------------------------------

_FORM = {
    "llm_provider": "cohere/command-a-03-2025", "llm_provider_fast": "",
    "ollama_api_base": "http://host.docker.internal:11434", "search_max_results": "5",
    "embed_model": "cohere/embed-v5.0-pro", "embed_query_model": "cohere/embed-v5.0-fast",
    "rerank_model": "cohere/rerank-v5.0-pro",
    "qdrant_url": "http://host.docker.internal:6333", "qdrant_export_collection": "quarry",
}


def test_settings_page_saves_models_and_keeps_qdrant_key_when_blank(client, monkeypatch):
    monkeypatch.setattr(config.settings, "qdrant_api_key", "")
    r = client.post("/settings", data={**_FORM, "qdrant_api_key": "qk-secret",
                                       "embed_model": "ollama/nomic-embed-text",
                                       "embed_query_model": "",
                                       "rerank_model": "cohere/rerank-v5.0-fast",
                                       "qdrant_url": "http://qdrant:6333",
                                       "qdrant_export_collection": "quarry_v2",
                                       "export_on_mission_done": "on"})
    assert r.status_code == 302
    s = config.settings
    assert s.embed_model == "ollama/nomic-embed-text"
    assert s.embed_query_model == ""
    assert s.rerank_model == "cohere/rerank-v5.0-fast"
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.qdrant_export_collection == "quarry_v2"
    assert s.export_on_mission_done is True
    assert s.qdrant_api_key == "qk-secret"
    saved = _saved()
    assert saved["qdrant_api_key"] == "qk-secret"
    assert saved["export_on_mission_done"] is True
    assert "ollama/nomic-embed-text" in saved["known_embed_models"]
    assert "cohere/rerank-v5.0-fast" in saved["known_rerank_models"]

    # A second save with the key field blank, and the checkbox unticked
    # (absent from the form), keeps the key and turns the export off.
    r = client.post("/settings", data={**_FORM, "qdrant_api_key": "   "})
    assert r.status_code == 302
    assert config.settings.qdrant_api_key == "qk-secret"
    assert _saved()["qdrant_api_key"] == "qk-secret"
    assert config.settings.export_on_mission_done is False


def test_settings_page_blank_models_turn_features_off(client):
    r = client.post("/settings", data={**_FORM, "embed_model": "", "rerank_model": ""})
    assert r.status_code == 302
    assert config.settings.embed_model == ""
    assert config.settings.rerank_model == ""


def test_settings_page_never_echoes_the_qdrant_key(client, monkeypatch):
    monkeypatch.setattr(config.settings, "qdrant_api_key", "qk-do-not-leak")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    html = client.get("/settings").get_data(as_text=True)
    assert "qk-do-not-leak" not in html
    assert 'name="qdrant_api_key"' in html and 'type="password"' in html
    assert "set — leave blank to keep" in html
    # Every new field renders with its current value.
    for name in ("embed_model", "embed_query_model", "rerank_model", "qdrant_url",
                 "qdrant_export_collection", "export_on_mission_done"):
        assert f'name="{name}"' in html, name
    assert 'value="cohere/embed-v5.0-pro"' in html
    assert 'id="embed-model-list"' in html and 'id="rerank-model-list"' in html


def test_settings_page_bounds_and_collection_name(client):
    long_model = "m" * 260
    bad_names = ("my collection", "quarry/evil", 'q"uote', "../x", "")
    for bad in bad_names:
        r = client.post("/settings", data={**_FORM, "qdrant_export_collection": bad,
                                           "embed_model": long_model,
                                           "qdrant_url": "http://h:6333/" + "p" * 400})
        assert r.status_code == 302
        assert config.settings.qdrant_export_collection == "quarry", bad
    assert len(config.settings.embed_model) == 200
    assert len(config.settings.qdrant_url) == 300
    with client.session_transaction() as sess:
        msgs = [m for _c, m in sess.get("_flashes", [])]
    assert any("Collection name not saved" in m for m in msgs)

    # Over-long names are truncated to 100 chars like every other field, then
    # validated: a 101-char name of valid characters is saved at 100.
    r = client.post("/settings", data={**_FORM, "qdrant_export_collection": "a" * 101})
    assert r.status_code == 302
    assert config.settings.qdrant_export_collection == "a" * 100
    # Blank URL keeps the current one.
    r = client.post("/settings", data={**_FORM, "qdrant_url": ""})
    assert config.settings.qdrant_url == "http://host.docker.internal:6333"
