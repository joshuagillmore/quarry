"""Optional password authentication for Quarry.

Single-user app → single password, no user table. Auth is OFF when
QUARRY_PASSWORD is unset (the compose file publishes on 127.0.0.1 by default,
so localhost-only remains the default posture) and ON the moment a password is
configured — which is what makes exposing the app on a LAN or behind Tailscale
reasonable.

QUARRY_PASSWORD accepts either a plaintext password or a Werkzeug hash
(pbkdf2:/scrypt: prefix) for those who don't want plaintext in .env:
    python -c "from werkzeug.security import generate_password_hash as g; print(g('...'))"
Anything else, including an `argon2:` string (Werkzeug cannot verify argon2),
is compared as plaintext.

A login stores a token derived from the secret key and the password, not a
bare flag, so changing QUARRY_PASSWORD (or the secret key) signs every
existing session out.

Deliberately NOT a UI-editable setting: an unauthenticated visitor must never
be able to set (or clear) the password through the Settings page.
"""
import hashlib
import hmac
import threading
import time

from werkzeug.security import check_password_hash

from config import settings

_HASH_PREFIXES = ("pbkdf2:", "scrypt:")

# Set once by app.py at import (auth.configure(app.secret_key)).
_secret_key = ""

# --- login rate limiting (in-memory; single process by design) ---
_MAX_FAILURES = 5
_WINDOW_S = 15 * 60
_SWEEP_OVER = 1024      # sweep aged-out IPs once this many are tracked
_failures: dict[str, list[float]] = {}
_lock = threading.Lock()


def configure(secret_key: str) -> None:
    global _secret_key
    _secret_key = secret_key or ""


def _password() -> str:
    return (settings.quarry_password or "").strip()


def enabled() -> bool:
    return bool(_password())


def verify_password(candidate: str) -> bool:
    secret = _password()
    if not secret:
        return False
    candidate = candidate or ""
    if secret.startswith(_HASH_PREFIXES):
        try:
            return bool(check_password_hash(secret, candidate))
        except Exception:  # noqa: BLE001 - a malformed hash is a failed login
            return False
    # Bytes, not str: compare_digest rejects non-ASCII str arguments.
    return hmac.compare_digest(secret.encode(), candidate.encode())


def session_token() -> str:
    """What a logged-in session carries. Bound to the secret key and the
    current password, so rotating either invalidates every old cookie."""
    return hashlib.sha256((_secret_key + _password()).encode()).hexdigest()[:16]


def _recent_locked(ip: str, now: float) -> list[float]:
    """Caller holds _lock. The IP's attempts still inside the window; an IP
    with none left is dropped rather than kept as an empty entry."""
    recent = [t for t in _failures.get(ip, ()) if now - t < _WINDOW_S]
    if recent:
        _failures[ip] = recent
    else:
        _failures.pop(ip, None)
    return recent


def _sweep_locked(now: float) -> None:
    """Caller holds _lock. Bound memory: forget IPs whose attempts all aged
    out, so a spray of one-off source addresses cannot grow the dict."""
    if len(_failures) <= _SWEEP_OVER:
        return
    for ip in [ip for ip, ts in _failures.items() if not ts or now - ts[-1] >= _WINDOW_S]:
        del _failures[ip]


def is_locked_out(ip: str) -> tuple[bool, int]:
    """(locked, seconds_remaining). Sliding window: N attempts in the window
    lock that IP until the oldest one ages out."""
    now = time.time()
    with _lock:
        recent = _recent_locked(ip, now)
        if len(recent) >= _MAX_FAILURES:
            return True, int(_WINDOW_S - (now - recent[0])) + 1
    return False, 0


def reserve_attempt(ip: str) -> tuple[bool, int]:
    """(allowed, seconds_until_allowed). Check the lockout and count this
    attempt in one step under the lock, so concurrent guesses cannot all slip
    past a check-then-record gap. A successful login calls clear_failures, so
    only failed attempts survive in the window."""
    now = time.time()
    with _lock:
        recent = _recent_locked(ip, now)
        if len(recent) >= _MAX_FAILURES:
            return False, int(_WINDOW_S - (now - recent[0])) + 1
        _sweep_locked(now)
        _failures[ip] = recent + [now]
    return True, 0


def clear_failures(ip: str) -> None:
    with _lock:
        _failures.pop(ip, None)
