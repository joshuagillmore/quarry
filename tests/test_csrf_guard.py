"""The origin-check CSRF guard: cross-site browser POSTs are rejected, while
same-origin form posts and non-browser clients pass through."""
import storage


def _client(tmp_path):
    storage.DB_PATH = str(tmp_path / "t.db")
    import app as app_mod
    app_mod.app._db_initialized = False  # re-init against this test's temp DB
    return app_mod.app.test_client()


def test_cross_site_origin_blocked(tmp_path):
    client = _client(tmp_path)
    r = client.post("/search", data={"query": "x"},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_null_origin_blocked(tmp_path):
    # A sandboxed-iframe form post sends "Origin: null"; a legitimate
    # same-origin form post never does (audit finding).
    client = _client(tmp_path)
    r = client.post("/search", data={"query": "x"}, headers={"Origin": "null"})
    assert r.status_code == 403


def test_cross_site_fetch_metadata_blocked(tmp_path):
    client = _client(tmp_path)
    r = client.post("/search", data={"query": "x"},
                    headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_origin_and_headerless_posts_pass(tmp_path):
    client = _client(tmp_path)
    # Non-browser client (no Origin / Sec-Fetch-Site): passes the guard and
    # reaches the route. The route's own "unknown agent" redirect to /agents
    # proves it ran -- a bare 302 could just as well be the login gate.
    r = client.post("/agents/nonexistent/run", data={"question": "q"})
    assert r.status_code == 302
    assert r.headers["Location"] == "/agents"
    # Same-origin browser post: Origin matches Host.
    r = client.post("/agents/nonexistent/run", data={"question": "q"},
                    headers={"Origin": "http://localhost",
                             "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 302
    assert r.headers["Location"] == "/agents"
    # Same-site (a sibling subdomain) and user-initiated ("none") also pass.
    for sfs in ("same-site", "none"):
        r = client.post("/agents/nonexistent/run", data={"question": "q"},
                        headers={"Sec-Fetch-Site": sfs})
        assert r.headers["Location"] == "/agents", sfs
    # GETs are never blocked.
    assert client.get("/").status_code == 200


def test_mismatched_origin_port_blocked(tmp_path):
    # Same hostname, different port is a different origin.
    client = _client(tmp_path)
    r = client.post("/search", data={"query": "x"},
                    headers={"Origin": "http://localhost:6666"})
    assert r.status_code == 403
