import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

from llm import chat_ex, model_for, json_candidates, _extract_json
from models import Document, ExtractedData

DEFAULT_PROMPT = (
    "Analyze this document and extract the following in JSON format:\n"
    "1. summary: A 2-3 sentence summary of the content\n"
    "2. key_facts: A list of the most important facts or claims\n"
    "3. entities: Named entities (people, organizations, locations, dates)\n"
    "4. topics: Main topics or themes covered\n"
    "5. sentiment: Overall sentiment (positive, negative, neutral, mixed)\n"
)


def _parse_output(text: str):
    """The model's JSON answer, or None. A whole-text or fenced JSON object
    *or array* is kept as-is (an extraction prompt may well ask for a list,
    and keeping only its first element would silently truncate it); failing
    that, an object embedded in prose; else None and the caller stores the
    raw text."""
    for candidate in json_candidates(text):
        try:
            value = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, (dict, list)):
            return value
    return _extract_json(text)


def extract_from_document(
    doc: Document,
    custom_prompt: str = "",
) -> Optional[ExtractedData]:
    prompt = custom_prompt or DEFAULT_PROMPT

    content = doc.content_fit or doc.content_markdown
    if not content:
        print(f"[EXTRACT] No content for {doc.url}", file=sys.stderr, flush=True)
        return None

    # Truncate to avoid exceeding context limits (~20k chars)
    if len(content) > 20000:
        content = content[:20000] + "\n\n[...truncated...]"

    model = model_for("fast")
    print(f"[EXTRACT] {model} on {doc.url} ({len(content)} chars)", file=sys.stderr, flush=True)

    system = (
        "You are a data extraction assistant. Extract structured information "
        "from web documents. Always respond with valid JSON."
    )
    user = f"{prompt}\n\n---\nDOCUMENT SOURCE: {doc.url}\n---\n\n{content}"

    try:
        # Summarization-style work -> "fast" tier (e.g. local Ollama qwen2.5:14b).
        # chat_ex reports which model actually answered (the fast tier falls
        # back to the reasoning model when e.g. Ollama is down) so the stored
        # extraction row is labeled with the true producer.
        result_text, model = chat_ex(system, user, temperature=0.0, max_tokens=2000, tier="fast")
        print(f"[EXTRACT] Got response ({len(result_text)} chars)", file=sys.stderr, flush=True)

        parsed = _parse_output(result_text)
        if parsed is None:
            result_json = json.dumps({"raw_response": result_text}, indent=2)
        else:
            result_json = json.dumps(parsed, indent=2)

        return ExtractedData(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            model=model,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            prompt=prompt[:500],
            data_json=result_json,
        )

    except Exception as e:
        print(f"[EXTRACT] ERROR for {doc.url}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None
