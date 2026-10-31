"""embeddings.rerank / try_rerank (WP1) against the litellm stub: Cohere
Rerank 5 on the wire, the result shape, a zero-token LlmCall per call, and
the degrade rule (None plus one log line, never an exception, from
try_rerank).

Written before embeddings.py existed (WP0): a ModuleNotFoundError on
`import embeddings` means WP1 has not landed, not that these are wrong."""
import asyncio
import sqlite3

import litellm
import pytest

import embeddings
import llm
import storage
from config import settings

RERANK = "cohere/rerank-v5.0-pro"


class AuthenticationError(Exception):
    """Named like litellm's: llm._NON_RETRYABLE matches on class name, so
    the retry loop raises at once and writes no 'retrying' line."""


class RateLimitError(Exception):
    """Named like litellm's transient error: retried."""


@pytest.fixture(autouse=True)
def _cohere_v5(monkeypatch):
    litellm.reset_calls()
    monkeypatch.setattr(settings, "rerank_model", RERANK)
    monkeypatch.setattr(settings, "embed_timeout_s", 33)
    monkeypatch.setattr(settings, "llm_api_key", "test-rerank-key")
    monkeypatch.setattr(settings, "cohere_api_key", "")
    monkeypatch.setattr(llm, "RETRY_DELAY_S", 0)
    yield
    litellm.reset_calls()


def _sql(query, args=()):
    db = sqlite3.connect(storage.DB_PATH)
    try:
        return db.execute(query, args).fetchall()
    finally:
        db.close()


DOCS = [
    "dogs chase cars on the road",              # 0: no overlap
    "cats sleep on warm mats all day",          # 1: cats, mats
    "the cats and the mats and the sun",        # 2: cats, mats (tie -> lower index first)
    "mats are made of rubber",                  # 3: mats
]
QUERY = "cats mats"


def test_rerank_sends_v5_pro_with_max_tokens_per_doc_2048():
    assert embeddings.RERANK_MAX_TOKENS_PER_DOC == 2048
    embeddings.rerank(QUERY, DOCS, top_n=2)
    [kw] = litellm.rerank_calls
    assert kw["model"] == RERANK
    assert kw["query"] == QUERY
    assert kw["documents"] == DOCS
    assert kw["top_n"] == 2
    assert kw["max_tokens_per_doc"] == 2048
    assert kw["timeout"] == settings.embed_timeout_s == 33
    assert kw["api_key"] == "test-rerank-key"
    assert "api_base" not in kw
    assert "relevant_spans" not in kw, "not used by Quarry"
    assert "input_type" not in kw and "dimensions" not in kw, "embedding-only params"


def test_rerank_no_top_n_sends_none_and_returns_every_document():
    ranked = embeddings.rerank(QUERY, DOCS)
    assert litellm.last_rerank_kwargs["top_n"] is None
    assert sorted(i for i, _s in ranked) == [0, 1, 2, 3]


def test_results_sorted_desc_index_score():
    ranked = embeddings.rerank(QUERY, DOCS)
    assert isinstance(ranked, list) and all(isinstance(r, tuple) and len(r) == 2 for r in ranked)
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in ranked)
    assert [i for i, _s in ranked] == [1, 2, 3, 0]
    scores = [s for _i, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == scores[1] == 1.0 and scores[2] == 0.5 and scores[3] == 0.0

    top = embeddings.rerank(QUERY, DOCS, top_n=2)
    assert top == [(1, 1.0), (2, 1.0)]


def test_zero_token_llmcall_recorded_with_purpose_rerank():
    asyncio.run(storage.init_db())
    embeddings.rerank(QUERY, DOCS, top_n=3, mission_id="m1")
    usage = asyncio.run(storage.get_mission_llm_usage("m1"))
    assert usage["calls"] == 1
    assert usage["by_purpose"] == {
        "rerank": {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0}}
    rows = _sql("SELECT purpose, tier, model, mission_id, prompt_tokens, "
                "completion_tokens, duration_ms FROM llm_calls")
    [(purpose, tier, model, mission_id, p, c, ms)] = rows
    assert (purpose, tier, model, mission_id, p, c) == (
        "rerank", "rerank", RERANK, "m1", 0, 0)
    assert isinstance(ms, int) and ms >= 0

    embeddings.rerank(QUERY, DOCS)
    assert _sql("SELECT mission_id FROM llm_calls ORDER BY rowid")[-1] == (None,)


def test_try_rerank_returns_ranking_when_it_works():
    assert embeddings.try_rerank(QUERY, DOCS, top_n=2) == [(1, 1.0), (2, 1.0)]
    assert litellm.last_rerank_kwargs["top_n"] == 2


def test_try_rerank_none_on_disabled_or_failure_with_one_log_line(monkeypatch, capsys):
    asyncio.run(storage.init_db())

    # Disabled: empty model.
    monkeypatch.setattr(settings, "rerank_model", "")
    assert embeddings.rerank_enabled() is False
    assert issubclass(embeddings.RerankDisabled, RuntimeError)
    with pytest.raises(embeddings.RerankDisabled):
        embeddings.rerank(QUERY, DOCS)
    capsys.readouterr()
    assert embeddings.try_rerank(QUERY, DOCS, top_n=2) is None
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1, lines
    assert "rerank" in lines[0].lower()
    assert litellm.rerank_calls == [], "no provider call when disabled"

    # Enabled but the provider raises a non-retryable error: None, one line,
    # nothing recorded, and the exception never reaches the caller.
    monkeypatch.setattr(settings, "rerank_model", RERANK)
    assert embeddings.rerank_enabled() is True
    attempts = []

    def auth_fails(**kw):
        attempts.append(kw["model"])
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", auth_fails)
    capsys.readouterr()
    assert embeddings.try_rerank(QUERY, DOCS, top_n=2) is None
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1, lines
    assert "rerank" in lines[0].lower() and "AuthenticationError" in lines[0]
    assert len(attempts) == 1
    assert _sql("SELECT COUNT(*) FROM llm_calls") == [(0,)]

    # A transient error is retried, then still degrades to None.
    attempts.clear()

    def always_limited(**kw):
        attempts.append(kw["model"])
        raise RateLimitError("429")

    monkeypatch.setattr(litellm, "rerank", always_limited)
    capsys.readouterr()
    assert embeddings.try_rerank(QUERY, DOCS) is None
    assert len(attempts) == llm.RETRY_ATTEMPTS
    assert "RateLimitError" in capsys.readouterr().err
    assert _sql("SELECT COUNT(*) FROM llm_calls") == [(0,)]


def test_rerank_raises_on_failure_where_try_rerank_degrades(monkeypatch):
    def auth_fails(**kw):
        raise AuthenticationError("bad key")

    monkeypatch.setattr(litellm, "rerank", auth_fails)
    with pytest.raises(AuthenticationError):
        embeddings.rerank(QUERY, DOCS)


def test_rerank_model_override_and_purpose_forwarded(monkeypatch):
    asyncio.run(storage.init_db())
    embeddings.rerank(QUERY, DOCS, model="cohere/rerank-v5.0-fast",
                      max_tokens_per_doc=512, purpose="rerank", mission_id="m2")
    kw = litellm.last_rerank_kwargs
    assert (kw["model"], kw["max_tokens_per_doc"]) == ("cohere/rerank-v5.0-fast", 512)
    assert _sql("SELECT model, purpose, mission_id FROM llm_calls") == [
        ("cohere/rerank-v5.0-fast", "rerank", "m2")]
