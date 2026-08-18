"""
test_auth.py
============
Exercises hashing, tokens, and the route guards without a server.

    venv\\Scripts\\python.exe test_auth.py

Checks the things that actually matter for auth:
  1. passwords are never recoverable, and identical passwords hash differently
  2. tokens verify, and tampering with any part invalidates them
  3. expiry is enforced
  4. anonymous callers are allowed on /chat but rejected on protected routes
  5. /dashboard is 404 when no admin token is configured, 403 with a wrong one
  6. attach_user never reassigns an already-owned conversation
"""

import asyncio
import base64
import json
import os
import shutil
import tempfile

TMP = tempfile.mkdtemp(prefix="serenity_auth_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(TMP, 'auth.db')}"
os.environ["SERENITY_ENC_KEY"] = "sVh0kQ2Zp7XcJ9nR4tB6yU8wE1oA3iL5mF7gD0hK2vM="
os.environ["SERENITY_JWT_SECRET"] = "test-secret-not-for-production"

import auth  # noqa: E402
import db  # noqa: E402
import sessions  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


print("=" * 62)
print("1. password hashing")
print("=" * 62)
pw = "correct horse battery staple"
h1 = auth.hash_password(pw)
h2 = auth.hash_password(pw)
check("plaintext never appears in the hash", pw not in h1)
check("same password -> different hashes (salted)", h1 != h2)
check("correct password verifies", auth.verify_password(pw, h1))
check("wrong password rejected", not auth.verify_password("wrong", h1))
check("empty password rejected", not auth.verify_password("", h1))
check("corrupted hash fails safely", not auth.verify_password(pw, "garbage"))
check("hash is self-describing", h1.startswith("scrypt$16384$8$1$"))

print()
print("=" * 62)
print("2. tokens")
print("=" * 62)
tok = auth.create_token("user-123", "a@b.com")
payload = auth.decode_token(tok)
check("round trips", payload is not None and payload["sub"] == "user-123")
check("carries email", payload and payload["email"] == "a@b.com")

head, body, sig = tok.split(".")
check("tampered payload rejected",
      auth.decode_token(f"{head}.{body[:-2]}XX.{sig}") is None)
check("tampered signature rejected",
      auth.decode_token(f"{head}.{body}.{sig[:-2]}XX") is None)
check("garbage rejected", auth.decode_token("not-a-token") is None)
check("empty rejected", auth.decode_token("") is None)

# forge a payload and re-encode without a valid signature
forged = base64.urlsafe_b64encode(
    json.dumps({"sub": "admin", "email": "x", "exp": 9999999999}).encode()
).rstrip(b"=").decode()
check("unsigned forgery rejected",
      auth.decode_token(f"{head}.{forged}.{sig}") is None)

print()
print("=" * 62)
print("3. expiry")
print("=" * 62)
old_ttl = auth.JWT_TTL_HOURS
auth.JWT_TTL_HOURS = -1                       # already expired
expired = auth.create_token("u", "e@x.com")
auth.JWT_TTL_HOURS = old_ttl
check("expired token rejected", auth.decode_token(expired) is None)
check("fresh token still accepted",
      auth.decode_token(auth.create_token("u", "e@x.com")) is not None)

print()
print("=" * 62)
print("4. route guards")
print("=" * 62)
check("optional_user allows anonymous",
      run(auth.optional_user(authorization=None)) is None)
check("optional_user identifies a bearer token",
      (run(auth.optional_user(authorization=f"Bearer {tok}")) or {}).get("sub") == "user-123")
check("optional_user ignores a bad token",
      run(auth.optional_user(authorization="Bearer nonsense")) is None)

try:
    run(auth.require_user(authorization=None))
    check("require_user rejects anonymous", False)
except HTTPException as e:
    check("require_user rejects anonymous", e.status_code == 401, "(401)")

check("require_user accepts a valid token",
      run(auth.require_user(authorization=f"Bearer {tok}"))["sub"] == "user-123")

print()
print("=" * 62)
print("5. admin route")
print("=" * 62)
auth.ADMIN_TOKEN = ""
try:
    run(auth.require_admin(x_admin_token=None))
    check("404 when admin token unset", False)
except HTTPException as e:
    check("404 when admin token unset", e.status_code == 404,
          "(hides the route's existence)")

auth.ADMIN_TOKEN = "s3cret"
try:
    run(auth.require_admin(x_admin_token="wrong"))
    check("403 on wrong admin token", False)
except HTTPException as e:
    check("403 on wrong admin token", e.status_code == 403)
check("admin allowed with correct token",
      run(auth.require_admin(x_admin_token="s3cret")) is True)

print()
print("=" * 62)
print("6. conversation ownership")
print("=" * 62)
db.init_db()
s = sessions.get_or_create(None)
sessions.attach_user(s.session_id, "user-A")
with db.SessionLocal() as dbs:
    check("conversation bound to the user",
          dbs.get(db.Conversation, s.session_id).user_id == "user-A")
sessions.attach_user(s.session_id, "user-B")     # attempted takeover
with db.SessionLocal() as dbs:
    check("ownership never reassigned",
          dbs.get(db.Conversation, s.session_id).user_id == "user-A",
          "(a leaked session id cannot be claimed)")

print()
shutil.rmtree(TMP, ignore_errors=True)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    raise SystemExit(1)
print("ALL PASS -- auth layer is sound")
