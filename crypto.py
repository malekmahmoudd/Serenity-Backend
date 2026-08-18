"""
crypto.py
=========
Envelope encryption for message text.

WHAT THIS PROTECTS AGAINST
--------------------------
A leaked database dump, a stolen backup, a misconfigured cloud console, a
support engineer browsing tables. In all of those the rows are unreadable
without SERENITY_ENC_KEY, which does not live in the database.

WHAT IT DOES NOT PROTECT AGAINST
--------------------------------
A compromised application server. The app needs the key to function, so anyone
who owns the process owns the plaintext. Encryption at rest is not a substitute
for not storing something.

Be honest about this in your privacy notice. "Encrypted at rest" is true;
"we cannot read your messages" would not be.

KEY MANAGEMENT
--------------
Set SERENITY_ENC_KEY to a urlsafe base64 32-byte key. Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If it is missing, dev mode generates an ephemeral key and warns -- data written
in that session becomes unreadable on restart, which is loud and obvious rather
than silent. Production refuses to start without one; losing the key means
losing every stored conversation, and that should not happen by accident.

ROTATION
--------
MultiFernet decrypts with any key in the list and encrypts with the first. To
rotate: put the new key first, keep the old one, redeploy, re-encrypt at
leisure, then drop the old key.

    SERENITY_ENC_KEY=<new>
    SERENITY_ENC_KEY_OLD=<previous>
"""

from __future__ import annotations

import os
import warnings

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

_PROD = os.environ.get("SERENITY_ENV", "dev").lower() in ("prod", "production")


def _build() -> MultiFernet:
    keys = []
    primary = os.environ.get("SERENITY_ENC_KEY", "").strip()
    old = os.environ.get("SERENITY_ENC_KEY_OLD", "").strip()

    if primary:
        keys.append(Fernet(primary.encode()))
    if old:
        keys.append(Fernet(old.encode()))

    if not keys:
        if _PROD:
            raise RuntimeError(
                "SERENITY_ENC_KEY is not set. Refusing to start in production: "
                "an ephemeral key would make every stored conversation "
                "unreadable after the next restart.")
        warnings.warn(
            "SERENITY_ENC_KEY not set -- using an EPHEMERAL key. Anything "
            "written now becomes unreadable when this process exits. Set the "
            "variable for anything you intend to keep.", stacklevel=2)
        keys.append(Fernet(Fernet.generate_key()))

    return MultiFernet(keys)


_fernet = _build()


def encrypt(plaintext: str | None) -> bytes | None:
    if plaintext is None:
        return None
    return _fernet.encrypt(plaintext.encode("utf-8"))


def decrypt(token: bytes | None) -> str | None:
    """
    Returns None on failure rather than raising.

    A row encrypted under a key we no longer hold is not an error worth taking
    the request down for -- the conversation continues, that message is simply
    unreadable. Raising here would turn a rotation mistake into an outage.
    """
    if token is None:
        return None
    try:
        return _fernet.decrypt(token).decode("utf-8")
    except (InvalidToken, TypeError, ValueError):
        return None


def key_configured() -> bool:
    return bool(os.environ.get("SERENITY_ENC_KEY", "").strip())
