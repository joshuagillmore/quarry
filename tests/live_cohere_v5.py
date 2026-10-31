"""Live check of the Cohere 5.0 ids Quarry targets, against the real API.

Not collected by pytest (no `test_` prefix). Needs COHERE_API_KEY (LLM_API_KEY
is accepted as the provider-agnostic alias); without one it prints SKIP and
exits 0 so an overseer can tell "not run" from "failed". Run from the repo's
venv (real litellm, not tests/stubs):

    COHERE_API_KEY=... python tests/live_cohere_v5.py

Measures and checks:
  1. embed one string with embed-v5.0-pro (search_document) and
     embed-v5.0-fast (search_query) at dimensions=1024 -> both length 1024,
     cosine between them > 0.9 (shared space) and < 1 (not the same vector);
  2. rerank 3 documents with rerank-v5.0-pro (max_tokens_per_doc=2048) ->
     the on-topic document ranks first.
Exit status: 0 pass, 0 skip (key missing), 1 fail.
"""
import math
import os
import sys
import time

EMBED_PRO = "cohere/embed-v5.0-pro"
EMBED_FAST = "cohere/embed-v5.0-fast"
RERANK_PRO = "cohere/rerank-v5.0-pro"
DIM = 1024

TEXT = ("The James Webb Space Telescope observes in the infrared from the "
        "second Lagrange point, about 1.5 million kilometres from Earth.")
QUERY = "Where does the James Webb Space Telescope orbit?"
DOCS = [
    "A recipe for sourdough bread needs a mature starter and a long cold proof.",
    "The James Webb Space Telescope sits at the Sun-Earth L2 point, 1.5 million km away.",
    "Quarterly revenue rose eight percent on stronger demand in Europe.",
]
ON_TOPIC = 1

failures: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {label}")
    if not cond:
        failures.append(label)


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    key = (os.environ.get("COHERE_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    if not key:
        print("SKIP: COHERE_API_KEY not set; live Cohere 5.0 check not run (n=0)")
        return 0

    import litellm  # noqa: E402  -- must be the real one, not tests/stubs
    if os.path.basename(os.path.dirname(getattr(litellm, "__file__", "") or "")) == "stubs":
        print("FAIL: imported the litellm test stub; run from the repo venv, not pytest's path")
        return 1

    print(f"embed: {EMBED_PRO} (search_document) vs {EMBED_FAST} (search_query) at {DIM}")
    vectors = {}
    for model, input_type in ((EMBED_PRO, "search_document"), (EMBED_FAST, "search_query")):
        t0 = time.monotonic()
        resp = litellm.embedding(model=model, input=[TEXT], input_type=input_type,
                                 dimensions=DIM, api_key=key, timeout=60,
                                 drop_params=True)
        ms = int((time.monotonic() - t0) * 1000)
        vec = resp.data[0]["embedding"]
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        print(f"  {model}: len={len(vec)} prompt_tokens={tokens} {ms} ms "
              f"first={vec[0]:+.5f} norm={math.sqrt(sum(x * x for x in vec)):.4f}")
        check(len(vec) == DIM, f"{model} returns {DIM} floats")
        check(all(isinstance(x, float) for x in vec), f"{model} values are floats")
        vectors[model] = vec

    cos = cosine(vectors[EMBED_PRO], vectors[EMBED_FAST])
    print(f"  cosine(pro, fast) = {cos:.4f}")
    check(cos > 0.9, "pro and fast share a space (cosine > 0.9)")
    check(cos < 0.9999, "pro and fast are different models (cosine < 1)")

    print(f"rerank: {RERANK_PRO}, {len(DOCS)} documents, max_tokens_per_doc=2048")
    t0 = time.monotonic()
    resp = litellm.rerank(model=RERANK_PRO, query=QUERY, documents=DOCS,
                          top_n=len(DOCS), max_tokens_per_doc=2048,
                          api_key=key, timeout=60)
    ms = int((time.monotonic() - t0) * 1000)
    results = list(resp.results)
    for r in results:
        print(f"  index={r['index']} score={r['relevance_score']:.4f}  {DOCS[r['index']][:60]}")
    print(f"  {ms} ms")
    check(len(results) == len(DOCS), "one result per document")
    check(results and results[0]["index"] == ON_TOPIC, "on-topic document ranks first")
    scores = [r["relevance_score"] for r in results]
    check(scores == sorted(scores, reverse=True), "results come back in descending score")

    if failures:
        print(f"FAIL: {len(failures)} check(s) failed: {failures}")
        return 1
    print("PASS: Cohere Embed 5 (pro/fast, 1024 dims, shared space) and Rerank 5 pro verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
