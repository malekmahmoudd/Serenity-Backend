"""
config.py
=========
Single place secrets and settings are read from, validated once at import.

WHY THIS EXISTS SEPARATELY FROM os.environ.get() SCATTERED EVERYWHERE
-----------------------------------------------------------------------
Before this file, six modules each read their own env vars with their own
defaults and their own (inconsistent) production checks. A typo in a variable
name failed silently -- os.environ.get("SERENITY_ENC_KEY", "") just returns ""
and the ephemeral-key warning fires instead of a clear "you misspelled this".

Now every setting is declared once, in one place, and `validate()` runs at
startup and refuses to boot in production with anything missing or insecure.

LOCAL DEVELOPMENT: A .env FILE
-------------------------------
Copy .env.example to .env and fill it in. python-dotenv loads it before
anything else reads os.environ. .env is gitignored -- check that it actually
is before your first commit, since a leaked .env is a leaked database key,
JWT secret, and Groq key all at once.

PRODUCTION: NOT A .env FILE
----------------------------
.env works locally because only you have the machine. In production, secrets
belong in whatever your host provides:

    Render / Railway / Fly    -> the platform's "Environment" / "Secrets" UI,
                                  injected at deploy time, never in the repo
    AWS                       -> Secrets Manager or Parameter Store (SecureString)
    Docker Compose             -> docker secrets, not environment: in the compose file
    Kubernetes                -> Secret objects mounted as env vars, not ConfigMaps

The common thread: secrets should never be visible in `docker inspect`,
platform dashboards' plaintext env list (where avoidable), or git history.

ROTATION
--------
SERENITY_ENC_KEY: see crypto.py's MultiFernet mechanism -- put the new key
    first, old key in SERENITY_ENC_KEY_OLD, redeploy, re-encrypt at leisure.
SERENITY_JWT_SECRET: rotating logs out every signed-in user immediately.
    There's no graceful version for JWTs without a lot more infrastructure
    (key ID in the header, multiple accepted signing keys) -- for this
    product's scale, an all-at-once rotation is an acceptable simplification.
GROQ_API_KEY: rotate directly in the Groq console; no code change needed.
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()          # no-op if .env does not exist; safe in production
except ImportError:
    pass  # python-dotenv is a dev convenience, not a hard dependency

ENV = os.environ.get("SERENITY_ENV", "dev").lower()
IS_PROD = ENV in ("prod", "production")

# ---------------------------------------------------------------- required
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./serenity.db")
SERENITY_ENC_KEY = os.environ.get("SERENITY_ENC_KEY", "")
SERENITY_ENC_KEY_OLD = os.environ.get("SERENITY_ENC_KEY_OLD", "")
SERENITY_JWT_SECRET = os.environ.get("SERENITY_JWT_SECRET", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------- optional
SERENITY_ADMIN_TOKEN = os.environ.get("SERENITY_ADMIN_TOKEN", "")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))
STORE_TEXT = os.environ.get("STORE_TEXT", "true").lower() == "true"


def validate() -> list[str]:
    """
    Returns a list of problems. In production, ANY problem is fatal -- see
    main.py, which calls this at startup and refuses to serve traffic with a
    misconfigured deployment rather than serving it badly.
    """
    problems = []

    if IS_PROD:
        if not SERENITY_ENC_KEY:
            problems.append(
                "SERENITY_ENC_KEY is not set. Stored conversations would be "
                "encrypted with a throwaway key that dies with the process.")
        if not SERENITY_JWT_SECRET:
            problems.append(
                "SERENITY_JWT_SECRET is not set. Auth tokens would be signed "
                "with the shared dev secret -- anyone could forge a login.")
        if DATABASE_URL.startswith("sqlite"):
            problems.append(
                "DATABASE_URL is SQLite in production. Most hosts (Render, "
                "Railway, Fly) wipe the filesystem on every deploy -- every "
                "conversation would be lost on the next release. Point this "
                "at Postgres.")
        if not GROQ_API_KEY:
            problems.append(
                "GROQ_API_KEY is not set. Every reply will be a hand-written "
                "template rather than a real generation.")
        if "*" in ALLOWED_ORIGINS:
            problems.append(
                "ALLOWED_ORIGINS includes '*' in production. Any website "
                "could call this API using a signed-in user's browser session.")
    else:
        # In dev, missing secrets are a warning, not a boot failure -- the
        # whole point of the degraded-mode design in classifier.py and llm.py
        # is that you can run this locally before every piece is configured.
        if not SERENITY_ENC_KEY:
            problems.append(
                "(dev) SERENITY_ENC_KEY not set -- stored text will be "
                "unreadable after this process exits.")
        if not GROQ_API_KEY:
            problems.append(
                "(dev) GROQ_API_KEY not set -- replies will use templates.")

    return problems


def summary() -> dict:
    """What /health exposes. No secret VALUES ever appear here, only whether
    each one is configured -- this function is safe to expose over HTTP."""
    return {
        "env": ENV,
        "database": DATABASE_URL.split("://", 1)[0],
        "encryption_key_set": bool(SERENITY_ENC_KEY),
        "jwt_secret_set": bool(SERENITY_JWT_SECRET),
        "groq_key_set": bool(GROQ_API_KEY),
        "admin_token_set": bool(SERENITY_ADMIN_TOKEN),
        "retention_days": RETENTION_DAYS,
        "storing_text": STORE_TEXT,
    }


def _print_and_check() -> None:
    problems = validate()
    for p in problems:
        print(f"{'FATAL' if IS_PROD else 'WARN'}: {p}", file=sys.stderr, flush=True)
    if IS_PROD and problems:
        print(f"\nRefusing to start: {len(problems)} configuration "
              f"problem(s) in production. Fix the above and restart.",
              file=sys.stderr, flush=True)
        sys.exit(1)


_print_and_check()
