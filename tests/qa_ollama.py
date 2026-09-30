"""Rerun the qwen-dependent path against a LIVE (containerized) Ollama.
Validates: fast-tier extraction + brief actually run on qwen (not the Cohere
fallback), and a fully-local mission (reasoning=qwen) plans + completes.
"""
import json, re, sqlite3, time, urllib.parse, urllib.request
BASE = "http://127.0.0.1:5000"; DB = "/app/data/research.db"
FAST = "ollama_chat/qwen2.5:14b"; OLLAMA = "http://host.docker.internal:11434"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r: return r.read().decode("utf-8", "replace")
def post(path, data):
    req = urllib.request.Request(BASE + path, data=urllib.parse.urlencode(data).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=120) as r: return r.geturl()
def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r: return json.load(r)
def settings(reasoning, fast):
    post("/settings", {"llm_provider": reasoning, "llm_provider_fast": fast,
                       "ollama_api_base": OLLAMA, "search_max_results": "5"})

def _field(html, name, default):
    # The settings form's <input> tags wrap onto a second line, so the match
    # has to span lines (re.S) to find id="..." and value="..." together.
    m = re.search(rf'<input\b[^>]*\bid="{name}"[^>]*>', html, re.S)
    if not m:
        return default
    v = re.search(r'value="([^"]*)"', m.group(0))
    return v.group(1) if v else default

def read_live_settings():
    """Snapshot the settings actually in effect before this driver touches
    them, so they can be put back at the end instead of overwritten with the
    hard-coded Cohere/qwen QA config."""
    html = get("/settings")
    return {
        "llm_provider": _field(html, "llm_provider", "cohere/command-a-03-2025"),
        "llm_provider_fast": _field(html, "llm_provider_fast", ""),
        "ollama_api_base": _field(html, "ollama_api_base", OLLAMA),
        "search_max_results": _field(html, "search_max_results", "5"),
    }
def poll(mid, terminal, timeout):
    end = time.time() + timeout
    while time.time() < end:
        s = get_json(f"/api/mission/{mid}")
        if s["status"] in terminal: return s
        time.sleep(3)
    return get_json(f"/api/mission/{mid}")
con = sqlite3.connect(DB)

def run(name, exp, q, extract, reasoning="cohere/command-a-03-2025", fast=FAST):
    settings(reasoning, fast)
    post("/agents/new", {"name": name, "expertise": exp, "max_sources": "3", "max_passes": "1", "per_req_attempts": "1"})
    aid = con.execute("SELECT id FROM agents WHERE name=? ORDER BY created_at DESC LIMIT 1", (name,)).fetchone()[0]
    form = {"query": q, "max_sources": "3", "max_passes": "1", "per_req_attempts": "1"}
    if extract: form["extract"] = "on"; form["extract_prompt"] = ""
    mid = post(f"/agents/{aid}/run", form).rstrip("/").split("/")[-1]
    s = poll(mid, {"awaiting_approval", "error"}, 150)
    reqs = len(s.get("requirements", []))
    if s["status"] == "awaiting_approval":
        post(f"/missions/{mid}/approve", {}); s = poll(mid, {"done", "error"}, 400)
    ndocs = con.execute("SELECT COUNT(DISTINCT document_id) FROM mission_documents WHERE mission_id=?", (mid,)).fetchone()[0]
    nx = con.execute("SELECT COUNT(*) FROM extractions e JOIN mission_documents md ON md.document_id=e.document_id WHERE md.mission_id=?", (mid,)).fetchone()[0]
    brief = con.execute("SELECT brief_markdown FROM missions WHERE id=?", (mid,)).fetchone()[0] or ""
    xmodel = con.execute("SELECT e.model FROM extractions e JOIN mission_documents md ON md.document_id=e.document_id WHERE md.mission_id=? LIMIT 1", (mid,)).fetchone()
    print(f"{name}: status={s['status']} reqs={reqs} docs={ndocs} extractions={nx} "
          f"brief={len(brief)} stub={'Automated brief generation failed' in brief} "
          f"xmodel={xmodel[0] if xmodel else None}", flush=True)
    post(f"/agents/{aid}/delete", {})

try:
    print("== capturing the user's real settings (restored on exit) ==", flush=True)
    _user_settings = read_live_settings()

    print("== live-Ollama qwen path ==", flush=True)
    run("QAO-Vax", "immunology", "How do vaccines work?", True)
    run("QAO-Econ", "macroeconomics", "What causes inflation?", True)
    run("QAO-Local", "general science", "What is photosynthesis?", False,
        reasoning=FAST, fast=FAST)   # fully local: planning+assessment on qwen
finally:
    post("/settings", _user_settings)  # restore the user's real config
print("QA_OLLAMA_DONE", flush=True)
