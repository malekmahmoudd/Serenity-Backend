"""
sessions.py
===========
Conversation state, backed by the database.

WHY THE SERVER NEEDS THIS AT ALL
--------------------------------
The frontend keeps its own message list for rendering, but classification runs
on the ACCUMULATED conversation, not the latest message -- the model was
trained on ~114-token posts, so a four-word opener is out of distribution.
Rebuilding that accumulation is what this module does.

WHAT CHANGED FROM THE IN-MEMORY VERSION
---------------------------------------
Previously a module-level dict. It died on every restart and could not survive
more than one uvicorn worker -- mid-conversation, a reload meant the next
message was treated as turn one.

Now every turn is persisted. Message text is encrypted (crypto.py); metadata is
not. The public API is unchanged, so main.py needed almost no edits.

A SMALL IN-PROCESS CACHE SITS IN FRONT
--------------------------------------
Rehydrating and decrypting every turn on every message would mean N decrypt
calls per request. The cache holds live conversations; the database is the
truth, and a cache miss simply reloads.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field

import crypto
from db import STORE_TEXT, Conversation, SessionLocal, Turn

CACHE_TTL = 60 * 60          # drop from memory after an hour idle
MAX_CACHED = 2000


@dataclass
class Session:
    session_id: str
    turns: list = field(default_factory=list)      # user messages, plaintext
    history: list = field(default_factory=list)    # full transcript for the LLM
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    resolved_labels: list = field(default_factory=list)
    crisis_flagged: bool = False
    last_action: str = ""

    @property
    def accumulated(self) -> str:
        """Every user message so far, joined. This is what gets classified."""
        return " ".join(self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)


_cache: dict[str, Session] = {}
_lock = threading.Lock()


def _evict() -> None:
    now = time.time()
    for k in [k for k, s in _cache.items() if now - s.touched > CACHE_TTL]:
        _cache.pop(k, None)
    if len(_cache) > MAX_CACHED:
        for k, _ in sorted(_cache.items(),
                           key=lambda kv: kv[1].touched)[:len(_cache) - MAX_CACHED]:
            _cache.pop(k, None)


def _hydrate(session_id: str) -> Session | None:
    """Rebuild a Session from the database. Returns None if unknown."""
    with SessionLocal() as s:
        conv = s.get(Conversation, session_id)
        if conv is None:
            return None
        sess = Session(session_id=conv.id)
        sess.crisis_flagged = bool(conv.crisis_flagged)
        sess.last_action = conv.last_action or ""
        sess.resolved_labels = [l for l in (conv.resolved_labels or "").split(",") if l]
        for t in conv.turns:
            user_text = crypto.decrypt(t.user_text_enc)
            reply = crypto.decrypt(t.reply_enc)
            # A row we cannot decrypt (key rotated away) still counts toward
            # turn_count, but contributes nothing to the classified text. The
            # conversation degrades rather than breaking.
            sess.turns.append(user_text or "")
            if user_text:
                sess.history.append({"role": "user", "content": user_text})
            if reply:
                sess.history.append({"role": "assistant", "content": reply})
        return sess


def get_or_create(session_id: str | None) -> Session:
    with _lock:
        _evict()
        if session_id:
            if session_id in _cache:
                s = _cache[session_id]
                s.touched = time.time()
                return s
            revived = _hydrate(session_id)
            if revived is not None:
                _cache[session_id] = revived
                return revived

        sid = session_id or str(uuid.uuid4())
        sess = Session(session_id=sid)
        _cache[sid] = sess
        with SessionLocal() as db:
            if db.get(Conversation, sid) is None:
                db.add(Conversation(id=sid))
                db.commit()
        return sess


def persist_turn(sess: Session, user_text: str, reply: str, action: str,
                 labels: list, scores: dict, tokens: int, reason: str,
                 llm_used: bool, crisis_patterns: list) -> None:
    """Write one exchange. Called after the reply is generated."""
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3] if scores else []
    with SessionLocal() as db:
        conv = db.get(Conversation, sess.session_id)
        if conv is None:
            conv = Conversation(id=sess.session_id)
            db.add(conv)
        conv.turn_count = sess.turn_count
        conv.crisis_flagged = sess.crisis_flagged
        conv.last_action = action
        conv.resolved_labels = ",".join(sess.resolved_labels or [])

        db.add(Turn(
            conversation_id=sess.session_id,
            turn_index=sess.turn_count,
            # STORE_TEXT=false gives operational-metadata-only mode: the
            # dashboard and analytics keep working, nothing anyone said is kept.
            user_text_enc=crypto.encrypt(user_text) if STORE_TEXT else None,
            reply_enc=crypto.encrypt(reply) if STORE_TEXT else None,
            action=action,
            labels=",".join(labels or []),
            top_scores=json.dumps([[k, round(v, 4)] for k, v in top]),
            evidence_tokens=tokens,
            reason=reason[:300],
            llm_used=bool(llm_used),
            crisis_patterns=",".join(crisis_patterns or []),
        ))
        db.commit()


def attach_user(session_id: str, user_id: str) -> None:
    """
    Bind a conversation to an account. Idempotent, and it never reassigns:
    once a conversation belongs to someone it stays theirs, so a session id
    that leaked cannot be claimed by a second account.
    """
    with SessionLocal() as db_s:
        conv = db_s.get(Conversation, session_id)
        if conv is not None and not conv.user_id:
            conv.user_id = user_id
            db_s.commit()


def reset(session_id: str) -> None:
    """Forget a conversation entirely -- cache and database."""
    with _lock:
        _cache.pop(session_id, None)
    with SessionLocal() as db:
        conv = db.get(Conversation, session_id)
        if conv is not None:
            db.delete(conv)     # cascade removes the turns
            db.commit()


def count() -> int:
    return len(_cache)
