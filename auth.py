"""
auth.py
=======
Password hashing, JWT issue/verify, and the FastAPI dependencies that guard
routes.

WHY AUTH MOVED TO THE BACKEND
-----------------------------
It used to live in apps/web/lib/auth/dev-store.ts: an in-memory Map with
PLAINTEXT PASSWORDS and opaque random tokens. That store died on every Next.js
restart, could not be shared across processes, and the FastAPI service -- the
one that actually holds conversations -- had no idea who anyone was.

Now there is one source of truth. Users live in the same database as their
conversations, so a conversation can belong to a person.

PASSWORD HASHING
----------------
hashlib.scrypt, from the standard library. Memory-hard, so GPU cracking is
expensive, and it saves a dependency. Parameters (n=2**14, r=8, p=1) are the
usual interactive-login settings -- roughly 100ms per hash on a laptop, which
is slow enough to make brute force impractical and fast enough that nobody
notices logging in.

Each hash carries its own salt and parameters, so raising the cost later does
not invalidate existing passwords.

TOKENS
------
Signed JWTs (HS256) rather than opaque random strings, so the API can verify a
token without a database round trip. Signed with SERENITY_JWT_SECRET.

Losing that secret logs everyone out. Changing it is how you force that.

ANONYMOUS ACCESS IS DELIBERATE
------------------------------
`optional_user` allows an unauthenticated conversation. This is a mental-health
support tool: putting a signup wall in front of someone who is about to
disclose suicidal ideation is the wrong trade. Anonymous conversations work and
are simply not tied to an account -- so they cannot be revisited, which is a
cost the user pays knowingly rather than a door closed in their face.

Use `require_user` for anything that touches stored history.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets

from fastapi import Depends, Header, HTTPException, status

JWT_SECRET = os.environ.get("SERENITY_JWT_SECRET", "")
JWT_TTL_HOURS = int(os.environ.get("SERENITY_JWT_TTL_HOURS", "168"))   # 7 days
_PROD = os.environ.get("SERENITY_ENV", "dev").lower() in ("prod", "production")

if not JWT_SECRET:
    if _PROD:
        raise RuntimeError(
            "SERENITY_JWT_SECRET is not set. Refusing to start in production.")
    JWT_SECRET = "dev-only-insecure-secret-change-me"

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    """Returns 'scrypt$n$r$p$salt_b64$hash_b64'. Self-describing on purpose."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return "$".join(["scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
                     base64.b64encode(salt).decode(),
                     base64.b64encode(dk).decode()])


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"),
                            salt=base64.b64decode(salt_b64),
                            n=int(n), r=int(r), p=int(p), dklen=32)
        # compare_digest, not ==, so the comparison does not leak timing
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- tokens
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_token(user_id: str, email: str) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"},
                             separators=(",", ":")).encode())
    now = dt.datetime.now(dt.timezone.utc)
    payload = _b64(json.dumps({
        "sub": user_id, "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(hours=JWT_TTL_HOURS)).timestamp()),
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64(sig)}"


def decode_token(token: str) -> dict | None:
    """Returns the payload, or None if the token is invalid or expired."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        expected = hmac.new(JWT_SECRET.encode(),
                            f"{header_b64}.{payload_b64}".encode(),
                            hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _unb64(sig_b64)):
            return None
        payload = json.loads(_unb64(payload_b64))
        if payload.get("exp", 0) < dt.datetime.now(dt.timezone.utc).timestamp():
            return None
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- deps
def _token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


async def optional_user(authorization: str | None = Header(default=None)) -> dict | None:
    """
    Identify the caller if they are signed in; return None if not.

    Used on /chat. A signup wall in front of someone about to disclose
    suicidal ideation is the wrong trade -- anonymous conversations work, they
    just are not tied to an account and so cannot be revisited later.
    """
    token = _token_from_header(authorization)
    return decode_token(token) if token else None


async def require_user(authorization: str | None = Header(default=None)) -> dict:
    """Reject anonymous callers. For anything touching stored history."""
    token = _token_from_header(authorization)
    payload = decode_token(token) if token else None
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Sign in to continue.",
                            headers={"WWW-Authenticate": "Bearer"})
    return payload


# ---------------------------------------------------------------- admin
# /dashboard shows user message text. It is a developer tool and must not be
# publicly reachable. Off entirely unless ADMIN_TOKEN is set; when set, callers
# must present it.
ADMIN_TOKEN = os.environ.get("SERENITY_ADMIN_TOKEN", "")


async def require_admin(x_admin_token: str | None = Header(default=None)) -> bool:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found.")          # 404, not 403: do not confirm it exists
    if not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Forbidden.")
    return True


def secrets_ok() -> dict:
    return {
        "jwt_secret_set": JWT_SECRET != "dev-only-insecure-secret-change-me",
        "admin_token_set": bool(ADMIN_TOKEN),
        "token_ttl_hours": JWT_TTL_HOURS,
    }
