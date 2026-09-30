"""The optional password gate: off by default, airtight when on."""
import pytest

import auth
import storage


@pytest.fixture()
def client(tmp_path, monkeypatch):
    storage.DB_PATH = str(tmp_path / "t.db")
    monkeypatch.setattr(auth.settings, "db_path", str(tmp_path / "t.db"))
    import app as app_mod
    # The Flask app is module-global across test files; force ensure_db to
    # re-run against this test's fresh temp DB.
    app_mod.app._db_initialized = False
    auth._failures.clear()
    return app_mod.app.test_client()


def _enable(monkeypatch, password="hunter2-quarry"):
    monkeypatch.setattr(auth.settings, "quarry_password", password)


def test_auth_off_means_open(client, monkeypatch):
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    assert client.get("/").status_code == 200
    assert client.get("/agents").status_code == 200
    # /login just bounces home when auth is off
    r = client.get("/login")
    assert r.status_code == 302 and r.headers["Location"].endswith("/")


def test_auth_on_gates_everything(client, monkeypatch):
    _enable(monkeypatch)
    for path in ("/", "/agents", "/missions", "/settings", "/documents", "/history"):
        r = client.get(path)
        assert r.status_code == 302, path
        assert "/login" in r.headers["Location"], path
    # APIs answer 401 JSON, not a redirect the fetch() would silently follow
    r = client.get("/api/mission/nope")
    assert r.status_code == 401
    # POST routes are gated too
    r = client.post("/search", data={"query": "x"})
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_login_page_is_reachable_and_standalone(client, monkeypatch):
    _enable(monkeypatch)
    r = client.get("/login")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # Must not leak the workspace shell (doc counts, recent queries, models)
    assert "sidebar" not in html
    assert "credits-card" not in html


def test_wrong_password_rejected_right_password_admits(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/login", data={"password": "wrong"})
    assert r.status_code == 401
    r = client.post("/login", data={"password": "hunter2-quarry"})
    assert r.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/api/mission/nope").status_code == 404  # authed: real 404 now


def test_hashed_password_supported(client, monkeypatch):
    from werkzeug.security import generate_password_hash
    _enable(monkeypatch, generate_password_hash("s3cret"))
    assert client.post("/login", data={"password": "s3cret"}).status_code == 302
    client.post("/logout")
    assert client.post("/login", data={"password": "wrong"}).status_code == 401


def test_rate_limit_locks_out(client, monkeypatch):
    _enable(monkeypatch)
    for _ in range(auth._MAX_FAILURES):
        client.post("/login", data={"password": "wrong"})
    r = client.post("/login", data={"password": "hunter2-quarry"})  # even correct
    assert r.status_code == 429


def test_logout_ends_the_session(client, monkeypatch):
    _enable(monkeypatch)
    client.post("/login", data={"password": "hunter2-quarry"})
    assert client.get("/").status_code == 200
    client.post("/logout")
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_no_open_redirect(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/login", data={"password": "hunter2-quarry",
                                    "next": "https://evil.example/phish"})
    assert r.status_code == 302
    assert "evil.example" not in r.headers["Location"]
    client.post("/logout")
    r = client.post("/login", data={"password": "hunter2-quarry", "next": "//evil.example"})
    assert "evil.example" not in r.headers["Location"]


def test_insecure_exposure_flag(client, monkeypatch):
    import app as app_mod
    # exposed + no password -> flagged
    monkeypatch.setattr(auth.settings, "quarry_bind", "0.0.0.0")
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    assert app_mod.insecure_exposure() is True
    # exposed + password -> fine
    monkeypatch.setattr(auth.settings, "quarry_password", "pw")
    assert app_mod.insecure_exposure() is False
    # loopback + no password -> fine (the default posture)
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    monkeypatch.setattr(auth.settings, "quarry_bind", "127.0.0.1")
    assert app_mod.insecure_exposure() is False


def test_security_headers_present(client, monkeypatch):
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    r = client.get("/")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_insecure_exposure_behind_proxy_without_password(client, monkeypatch):
    import app as app_mod
    monkeypatch.setattr(auth.settings, "quarry_bind", "127.0.0.1")
    monkeypatch.setattr(auth.settings, "quarry_behind_proxy", True)
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    # A TLS proxy publishes the app however loopback the bind looks.
    assert app_mod.insecure_exposure() is True
    monkeypatch.setattr(auth.settings, "quarry_password", "pw")
    assert app_mod.insecure_exposure() is False


def test_dev_server_exposure_check(monkeypatch):
    import app as app_mod
    monkeypatch.setattr(auth.settings, "quarry_password", "")
    assert app_mod.dev_server_exposed("0.0.0.0") is True
    assert app_mod.dev_server_exposed("192.168.1.20") is True
    for host in ("127.0.0.1", "localhost", "::1"):
        assert app_mod.dev_server_exposed(host) is False, host
    monkeypatch.setattr(auth.settings, "quarry_password", "pw")
    assert app_mod.dev_server_exposed("0.0.0.0") is False


# --- password verification -------------------------------------------------

def test_non_ascii_password_verifies(monkeypatch):
    _enable(monkeypatch, "pässwörd-日本")
    assert auth.verify_password("pässwörd-日本") is True
    assert auth.verify_password("passwort") is False
    assert auth.verify_password("pässwörd-日") is False


def test_non_ascii_password_logs_in(client, monkeypatch):
    _enable(monkeypatch, "pässwörd-日本")
    r = client.post("/login", data={"password": "wröng"})
    assert r.status_code == 401          # not a 500 from compare_digest
    r = client.post("/login", data={"password": "pässwörd-日本"})
    assert r.status_code == 302
    assert client.get("/").status_code == 200


def test_malformed_hash_is_a_failed_login_not_a_crash(client, monkeypatch):
    _enable(monkeypatch, "pbkdf2:sha256:garbage-without-separators")
    assert auth.verify_password("anything") is False
    r = client.post("/login", data={"password": "anything"})
    assert r.status_code == 401


def test_argon2_prefix_is_not_treated_as_a_hash(monkeypatch):
    # Werkzeug cannot verify argon2; the value is compared as plaintext
    # rather than crashing inside check_password_hash.
    _enable(monkeypatch, "argon2:literal")
    assert auth.verify_password("argon2:literal") is True
    assert auth.verify_password("literal") is False


# --- session binding -----------------------------------------------------

def test_session_token_tracks_password_and_key(monkeypatch):
    _enable(monkeypatch, "one")
    t1 = auth.session_token()
    assert len(t1) == 16 and int(t1, 16) >= 0
    _enable(monkeypatch, "two")
    assert auth.session_token() != t1


def test_old_cookie_rejected_after_password_change(client, monkeypatch):
    _enable(monkeypatch)
    client.post("/login", data={"password": "hunter2-quarry"})
    assert client.get("/").status_code == 200
    _enable(monkeypatch, "rotated-password")
    r = client.get("/")
    assert r.status_code == 302 and r.headers["Location"].startswith("/login")
    assert client.get("/api/mission/nope").status_code == 401


def test_legacy_boolean_session_is_rejected(client, monkeypatch):
    _enable(monkeypatch)
    with client.session_transaction() as sess:
        sess["authed"] = True       # what pre-binding versions stored
    r = client.get("/")
    assert r.status_code == 302 and r.headers["Location"].startswith("/login")


# --- rate limiter --------------------------------------------------------

def test_reserve_attempt_counts_before_verification():
    ip = "10.0.0.9"
    for _ in range(auth._MAX_FAILURES):
        ok, wait = auth.reserve_attempt(ip)
        assert ok is True and wait == 0
    ok, wait = auth.reserve_attempt(ip)
    assert ok is False and wait > 0
    # A refused attempt is not itself recorded.
    assert len(auth._failures[ip]) == auth._MAX_FAILURES


def test_clear_failures_and_empty_entries_are_dropped():
    auth.reserve_attempt("10.0.0.1")
    auth.clear_failures("10.0.0.1")
    assert "10.0.0.1" not in auth._failures
    # Entries whose attempts have all aged out are deleted, not kept empty.
    with auth._lock:
        auth._failures["10.0.0.2"] = [0.0]
    assert auth.is_locked_out("10.0.0.2") == (False, 0)
    assert "10.0.0.2" not in auth._failures


def test_concurrent_attempts_all_count(client, monkeypatch):
    # Attempts in flight (reserved, not yet verified) count toward the lockout.
    _enable(monkeypatch)
    for _ in range(auth._MAX_FAILURES):
        auth.reserve_attempt("127.0.0.1")
    r = client.post("/login", data={"password": "hunter2-quarry"})
    assert r.status_code == 429


def test_successful_login_clears_the_counter(client, monkeypatch):
    _enable(monkeypatch)
    client.post("/login", data={"password": "wrong"})
    client.post("/login", data={"password": "wrong"})
    assert client.post("/login", data={"password": "hunter2-quarry"}).status_code == 302
    assert "127.0.0.1" not in auth._failures


# --- login `next` ----------------------------------------------------------

def _login_location(client, nxt, via="form"):
    if via == "form":
        r = client.post("/login", data={"password": "hunter2-quarry", "next": nxt})
    else:
        r = client.post("/login", query_string={"next": nxt},
                        data={"password": "hunter2-quarry"})
    client.post("/logout")
    assert r.status_code == 302
    return r.headers["Location"]


def test_next_rejects_browser_normalised_offsite_targets(client, monkeypatch):
    _enable(monkeypatch)
    for nxt in ("/\t/evil.example/", "/\n/evil.example", "/\\evil.example",
                "/\\/evil.example", "\\\\evil.example", "//evil.example",
                "/x\\evil.example", "/ /evil.example",
                "https://evil.example/", "javascript:alert(1)", "evil.example",
                "/ok path", ""):
        for via in ("form", "query"):
            assert _login_location(client, nxt, via) == "/", (nxt, via)


def test_next_percent_encoded_tab_redirects_home(client, monkeypatch):
    _enable(monkeypatch)
    r = client.post("/login?next=/%09/evil.example/", data={"password": "hunter2-quarry"})
    assert r.status_code == 302 and r.headers["Location"] == "/"


def test_next_same_site_path_is_honoured(client, monkeypatch):
    _enable(monkeypatch)
    assert _login_location(client, "/documents?q=x") == "/documents?q=x"
    assert _login_location(client, "/missions/abc", via="query") == "/missions/abc"


def test_next_survives_a_failed_attempt(client, monkeypatch):
    _enable(monkeypatch)
    html = client.get("/login?next=/documents").get_data(as_text=True)
    assert 'name="next" value="/documents"' in html
    r = client.post("/login?next=/documents", data={"password": "wrong"})
    assert r.status_code == 401
    assert 'name="next" value="/documents"' in r.get_data(as_text=True)
