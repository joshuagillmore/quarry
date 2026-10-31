"""Minimal stub so modules that `import litellm` are importable in unit tests.

`completion` is a placeholder: real chat calls are monkeypatched in the tests
and it is never invoked. `embedding` and `rerank` are deterministic fakes
(embeddings.py calls them for real) with the attribute shapes litellm's own
response objects have: `resp.data[i]["embedding"]`, `resp.usage.prompt_tokens`,
`resp.results[i]["index"]` / `["relevance_score"]`. Every call is recorded in
`last_embedding_kwargs` / `last_rerank_kwargs` and appended to the
`embedding_calls` / `rerank_calls` histories (reset them in a test that counts).
"""
import hashlib
import math
import re


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    """`usage` is only set when given, so a response without one (some
    providers omit it) can be modelled too."""
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        if usage is not None:
            self.usage = usage


def completion(**kwargs):
    return _Resp("{}")


# ---------------------------------------------------------------------------
# embedding / rerank fakes
# ---------------------------------------------------------------------------

last_embedding_kwargs: dict | None = None
last_rerank_kwargs: dict | None = None
embedding_calls: list[dict] = []
rerank_calls: list[dict] = []


def reset_calls() -> None:
    """Forget every recorded embedding/rerank call."""
    global last_embedding_kwargs, last_rerank_kwargs
    last_embedding_kwargs = None
    last_rerank_kwargs = None
    embedding_calls.clear()
    rerank_calls.clear()


class _EmbedUsage:
    """litellm's `Usage` for an embedding response: no completion tokens."""
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 0
        self.total_tokens = prompt_tokens


class _EmbeddingResponse:
    """Shape of litellm.EmbeddingResponse: `data` is a list of dicts
    ({"object", "index", "embedding"}), `usage` has `prompt_tokens`."""
    def __init__(self, model, vectors, prompt_tokens):
        self.object = "list"
        self.model = model
        self.data = [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)]
        self.usage = _EmbedUsage(prompt_tokens)


class _RerankResponse:
    """Shape of litellm.RerankResponse: `results` is a list of dicts
    ({"index", "relevance_score"}) in descending score order."""
    def __init__(self, results):
        self.id = "stub-rerank"
        self.results = results
        self.meta = {}


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


def _word_count(text) -> int:
    return len(str(text or "").split())


def _unit_vector(text, dim: int) -> list[float]:
    """A deterministic unit vector seeded from `text`: the same text always
    embeds to the same vector, different texts to different ones."""
    components = []
    counter = 0
    seed = str(text or "").encode("utf-8")
    while len(components) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(digest) - 1, 2):
            if len(components) >= dim:
                break
            raw = int.from_bytes(digest[i:i + 2], "big")   # 0..65535
            components.append(raw / 32767.5 - 1.0)         # -1..1
        counter += 1
    norm = math.sqrt(sum(c * c for c in components)) or 1.0
    return [c / norm for c in components]


def _as_texts(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def embedding(model, input, **kw):
    """Deterministic fake of litellm.embedding. `dim = kw["dimensions"]`
    (default 8, so a call that did not ask for a size is visible by the
    vector length); `usage.prompt_tokens` is the input's total word count."""
    global last_embedding_kwargs
    call = {"model": model, "input": input, **kw}
    last_embedding_kwargs = call
    embedding_calls.append(call)
    texts = _as_texts(input)
    dim = int(kw.get("dimensions") or 8)
    vectors = [_unit_vector(t, dim) for t in texts]
    return _EmbeddingResponse(model, vectors, sum(_word_count(t) for t in texts))


def _document_text(doc) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text", ""))
    return str(doc)


def rerank(model, query, documents, top_n=None, **kw):
    """Deterministic fake of litellm.rerank: each document scored by the
    share of the query's tokens it contains, results in descending score
    (ties by index), cut to `top_n` when given."""
    global last_rerank_kwargs
    call = {"model": model, "query": query, "documents": documents,
            "top_n": top_n, **kw}
    last_rerank_kwargs = call
    rerank_calls.append(call)
    q_tokens = set(_tokens(query))
    denominator = len(q_tokens) or 1
    scored = []
    for index, doc in enumerate(documents):
        overlap = len(q_tokens & set(_tokens(_document_text(doc))))
        scored.append({"index": index, "relevance_score": overlap / denominator})
    scored.sort(key=lambda r: (-r["relevance_score"], r["index"]))
    if top_n is not None:
        scored = scored[:int(top_n)]
    return _RerankResponse(scored)
