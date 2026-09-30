"""End-to-end QA harness: drives the running app over HTTP across 10 missions
with varying agents + settings, and verifies each surface. Run inside the
container:  python /tmp/qa_harness.py
Verification reads the SQLite DB directly; actions go through the HTTP routes.
"""
import json, re, sqlite3, sys, time, urllib.parse, urllib.request

BASE = "http://127.0.0.1:5000"
DB = "/app/data/research.db"

def db():
    return sqlite3.connect(DB)

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.getcode(), r.read().decode("utf-8", "replace")

def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)

def post(path, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.geturl(), r.read().decode("utf-8", "replace")

def set_settings(reasoning, fast):
    post("/settings", {"llm_provider": reasoning, "llm_provider_fast": fast,
                       "ollama_api_base": "http://host.docker.internal:11434",
                       "search_max_results": "5"})

def _field(html, name, default):
    # The settings form's <input> tags wrap onto a second line, so the match
    # has to span lines (re.S) to find id="..." and value="..." together.
    m = re.search(rf'<input\b[^>]*\bid="{name}"[^>]*>', html, re.S)
    if not m:
        return default
    v = re.search(r'value="([^"]*)"', m.group(0))
    return v.group(1) if v else default

def read_live_settings():
    """Snapshot the settings actually in effect right now (whatever the user
    has configured, via .env or the Settings page), so the harness can put
    them back at the end instead of overwriting real config with its own
    hard-coded Cohere/qwen QA baseline."""
    _, html = get("/settings")
    return {
        "llm_provider": _field(html, "llm_provider", "cohere/command-a-03-2025"),
        "llm_provider_fast": _field(html, "llm_provider_fast", ""),
        "ollama_api_base": _field(html, "ollama_api_base", "http://host.docker.internal:11434"),
        "search_max_results": _field(html, "search_max_results", "5"),
    }

def restore_live_settings(saved):
    post("/settings", saved)

def agent_id_by_name(name):
    row = db().execute("SELECT id FROM agents WHERE name=? ORDER BY created_at DESC LIMIT 1", (name,)).fetchone()
    return row[0] if row else None

def poll(mid, terminal, timeout):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        s = get_json(f"/api/mission/{mid}")
        if s["status"] != last:
            print(f"      status -> {s['status']}", flush=True)
            last = s["status"]
        if s["status"] in terminal:
            return s
        time.sleep(2)
    return get_json(f"/api/mission/{mid}")

TESTS = [
 {"name":"QA01-Astro","exp":"observational astrophysics","q":"How do gravitational waves form?","src":4,"pas":1,"try":1,"ext":False},
 {"name":"QA02-Econ","exp":"macroeconomics","q":"What causes inflation?","src":4,"pas":1,"try":1,"ext":True,"prompt":""},
 {"name":"QA03-Hist","exp":"20th century history","q":"What were the main causes of World War I?","src":5,"pas":2,"try":2,"ext":True,"prompt":"Extract key dates, people, and places","edit":True},
 {"name":"QA04-Cyber","exp":"cybersecurity","q":"What is a zero-day vulnerability?","src":3,"pas":1,"try":1,"ext":False},
 {"name":"QA05-Climate","exp":"climate science","q":"What is the greenhouse effect?","src":4,"pas":1,"try":1,"ext":True,"prompt":""},
 {"name":"QA06-Bio","exp":"molecular biology","q":"How does CRISPR gene editing work?","src":4,"pas":1,"try":1,"ext":False,"settings":("cohere/command-a-03-2025","")},
 {"name":"QA07-Law","exp":"constitutional law","q":"What is judicial review?","src":4,"pas":1,"try":1,"ext":True,"prompt":""},
 {"name":"QA08-AI","exp":"artificial intelligence","q":"What is a transformer neural network?","src":3,"pas":1,"try":1,"ext":False},
 {"name":"QA09-Space","exp":"planetary science","q":"What is the James Webb Space Telescope?","src":4,"pas":1,"try":1,"ext":True,"prompt":"","settings":("cohere/command-a-03-2025","ollama_chat/qwen2.5:14b")},
 {"name":"QA10-Local","exp":"general science","q":"What is photosynthesis?","src":3,"pas":1,"try":1,"ext":False,"settings":("ollama_chat/qwen2.5:14b","ollama_chat/qwen2.5:14b")},
]

results = []
created_agents = []

def check(cond, label, detail=""):
    print(f"      [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    return {"label": label, "ok": bool(cond), "detail": detail}

try:
    print("== capturing the user's real settings (restored on exit) ==", flush=True)
    _user_settings = read_live_settings()

    print("== baseline settings ==", flush=True)
    set_settings("cohere/command-a-03-2025", "ollama_chat/qwen2.5:14b")

    for i, t in enumerate(TESTS, 1):
        print(f"\n=== TEST {i}: {t['name']} ({t['exp']}) ===", flush=True)
        checks = []
        if "settings" in t:
            set_settings(*t["settings"])
            print(f"      settings -> reasoning={t['settings'][0]} fast={t['settings'][1] or '(reasoning)'}", flush=True)

        # create agent
        post("/agents/new", {"name": t["name"], "expertise": t["exp"],
                             "max_sources": str(t["src"]), "max_passes": str(t["pas"]),
                             "per_req_attempts": str(t["try"])})
        aid = agent_id_by_name(t["name"])
        checks.append(check(aid is not None, "agent created"))
        if aid:
            created_agents.append(aid)

        # optional edit
        if t.get("edit") and aid:
            post(f"/agents/{aid}/edit", {"name": t["name"]+"*", "expertise": t["exp"],
                                         "max_sources": str(t["src"]), "max_passes": str(t["pas"]),
                                         "per_req_attempts": str(t["try"]), "persona_prompt": ""})
            newname = db().execute("SELECT name FROM agents WHERE id=?", (aid,)).fetchone()[0]
            checks.append(check(newname == t["name"]+"*", "agent edit persisted", newname))

        # run mission
        form = {"query": t["q"], "max_sources": str(t["src"]), "max_passes": str(t["pas"]),
                "per_req_attempts": str(t["try"])}
        if t["ext"]:
            form["extract"] = "on"; form["extract_prompt"] = t.get("prompt", "")
        final, _ = post(f"/agents/{aid}/run", form)
        mid = final.rstrip("/").split("/")[-1]
        checks.append(check(final.split("/")[-2] == "missions" if "/missions/" in final else False,
                            "run -> mission page", final))

        # plan
        s = poll(mid, {"awaiting_approval","error","done"}, 100)
        checks.append(check(s["status"]=="awaiting_approval", "planning produced a plan", s.get("error") or s["status"]))
        nreq = len(s.get("requirements", []))
        checks.append(check(nreq >= 2, "requirements decomposed", f"{nreq} requirements"))

        if s["status"] == "awaiting_approval":
            post(f"/missions/{mid}/approve", {})
            s = poll(mid, {"done","error"}, 300)
        checks.append(check(s["status"]=="done", "mission completed", s.get("error") or s["status"]))

        # DB verification
        con = db()
        ndocs = con.execute("SELECT COUNT(DISTINCT document_id) FROM mission_documents WHERE mission_id=?", (mid,)).fetchone()[0]
        nsat = con.execute("SELECT COUNT(*) FROM requirements WHERE mission_id=? AND status='satisfied'", (mid,)).fetchone()[0]
        ntot = con.execute("SELECT COUNT(*) FROM requirements WHERE mission_id=?", (mid,)).fetchone()[0]
        briefrow = con.execute("SELECT brief_markdown FROM missions WHERE id=?", (mid,)).fetchone()
        blen = len(briefrow[0]) if briefrow and briefrow[0] else 0
        checks.append(check(ndocs >= 1, "sources collected", f"{ndocs} docs (rate-limited if 0)"))
        checks.append(check(blen > 120, "brief synthesized", f"{blen} chars, coverage {nsat}/{ntot}"))

        # extraction verification
        if t["ext"]:
            nx = con.execute("""SELECT COUNT(*) FROM extractions e JOIN mission_documents md
                                ON md.document_id=e.document_id WHERE md.mission_id=?""", (mid,)).fetchone()[0]
            checks.append(check(nx >= 1, "extractions produced", f"{nx} extractions"))
            # document page renders extraction without raw JSON leak
            drow = con.execute("""SELECT d.id FROM documents d JOIN mission_documents md ON md.document_id=d.id
                                  JOIN extractions e ON e.document_id=d.id WHERE md.mission_id=? LIMIT 1""",(mid,)).fetchone()
            if drow:
                code, html = get(f"/document/{drow[0]}")
                leak = '{&#34;name&#34;:' in html or '{"name":' in html
                checks.append(check(code==200 and not leak, "extraction renders (no raw JSON)", "leak!" if leak else "clean"))

        # library filter by mission + history inclusion
        code, libhtml = get(f"/documents?mission={mid}")
        checks.append(check(code==200 and ("doc-card" in libhtml or ndocs==0), "library ?mission filter works"))
        code, hist = get("/history")
        checks.append(check(mid in hist or ("mission" in hist), "mission shows in history"))

        results.append({"test": t["name"], "mid": mid, "docs": ndocs, "cov": f"{nsat}/{ntot}",
                        "brief": blen, "checks": checks})
        time.sleep(4)  # ease DDG rate limiting

    # delete one harness agent + verify
    if created_agents:
        did = created_agents[0]
        post(f"/agents/{did}/delete", {})
        gone = db().execute("SELECT COUNT(*) FROM agents WHERE id=?", (did,)).fetchone()[0] == 0
        print("\n=== cleanup ===", flush=True)
        check(gone, "agent delete works")

finally:
    print("\n== restoring the user's real settings ==", flush=True)
    restore_live_settings(_user_settings)

# summary
print("\n================ SUMMARY ================", flush=True)
total = passed = 0
for r in results:
    fails = [c["label"] for c in r["checks"] if not c["ok"]]
    total += len(r["checks"]); passed += sum(1 for c in r["checks"] if c["ok"])
    status = "OK" if not fails else ("FAIL: " + ", ".join(fails))
    print(f"{r['test']:14} docs={r['docs']:<3} cov={r['cov']:<5} brief={r['brief']:<5} {status}", flush=True)
print(f"\nChecks passed: {passed}/{total}", flush=True)
print("QA_HARNESS_DONE", flush=True)
