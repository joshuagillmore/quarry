"""The API key is provider-agnostic, and no key is baked into the repo."""
import pathlib

import config
import llm


def test_llm_api_key_preferred(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_api_key", "new-style")
    monkeypatch.setattr(config.settings, "cohere_api_key", "legacy")
    assert config.active_api_key() == "new-style"


def test_legacy_cohere_key_still_honoured(monkeypatch):
    """Existing installs set COHERE_API_KEY; they must keep working."""
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "cohere_api_key", "legacy")
    assert config.active_api_key() == "legacy"


def test_no_key_configured(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "cohere_api_key", "")
    assert config.active_api_key() == ""


def test_hosted_provider_gets_the_key(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_api_key", "k-123")
    monkeypatch.setattr(config.settings, "cohere_api_key", "")
    for model in ("cohere/command-a-03-2025", "openai/gpt-4o-mini",
                  "anthropic/claude-sonnet-4-5"):
        assert llm._provider_kwargs(model) == {"api_key": "k-123"}


def test_ollama_never_gets_a_key(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_api_key", "k-123")
    monkeypatch.setattr(config.settings, "ollama_api_base", "http://h:11434")
    kwargs = llm._provider_kwargs("ollama_chat/qwen2.5:14b")
    assert kwargs == {"api_base": "http://h:11434"}
    assert "api_key" not in kwargs


def test_no_key_means_litellm_falls_back_to_vendor_env(monkeypatch):
    """With no key configured we must pass no api_key at all, so LiteLLM can
    read OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment."""
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "cohere_api_key", "")
    assert llm._provider_kwargs("openai/gpt-4o-mini") == {}


def test_repo_ships_no_real_api_key():
    """.env.example must contain only blanks/placeholders — never a live key."""
    # Resolved from this file, not the cwd, so the check runs from anywhere.
    example = pathlib.Path(__file__).resolve().parent.parent / ".env.example"
    text = example.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip().endswith("API_KEY"):
            assert value.strip() == "", f"{name} must ship empty, got {value!r}"
