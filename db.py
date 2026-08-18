"""
db.py
=====
Database engine, models, and the retention sweep.

WHY SQLALCHEMY AND NOT RAW SQL
------------------------------
So the move from SQLite to Postgres is a connection string and nothing else:

    DATABASE_URL=sqlite:///./serenity.db                    # local
    DATABASE_URL=postgresql+psycopg://user:pw@host/db       # deployed

SQLite is genuinely fine for development and small single-server deployments.
It is NOT fine on Render/Railway/Fly without a mounted volume -- those
filesystems are ephemeral and the file disappears on every deploy.

WHAT IS ENCRYPTED AND WHAT IS NOT
---------------------------------
Encrypted (crypto.py):   message text, assistant replies
Plaintext:               action, labels, scores, token counts, timestamps

That split is deliberate. The metadata is what you need for the dashboard and
for spotting a degrading system -- crisis fire rate, abstention rate, score
distributions -- and none of it is meaningful about a person on its own.
"depression 0.79" attached to a random UUID is not a disclosure. The text is.

RETENTION
---------
Rows older than RETENTION_DAYS are deleted by `sweep_expired()`, called at
startup and hourly. Default 90 days.

Whatever number you set has to match what the consent screen tells users. A
retention policy nobody was told about is not a policy.
"""

from __future__ import annotations

import datetime as dt
import os

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Integer,
                        LargeBinary, String, Text, create_engine, delete,
                        func, select)
from sqlalchemy.orm import DeclarativeBase, Session as OrmSession, relationship, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./serenity.db")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))
# Store message text at all? False gives you the operational-metadata-only
# mode -- restart-safe, no history feature, minimal exposure.
STORE_TEXT = os.environ.get("STORE_TEXT", "true").lower() == "true"

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, future=True,
                       pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    """
    An account. Passwords are scrypt hashes (auth.py) -- never plaintext, which
    is what the previous Next.js dev-store held.

    Email is stored in the clear: it is the login identifier and has to be
    queryable. That is a deliberate exception to the encrypt-everything rule,
    and it means a database leak reveals WHO used the service even though it
    does not reveal WHAT they said. Worth stating plainly in the privacy notice.
    """
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    email = Column(String(320), unique=True, index=True, nullable=False)
    name = Column(String(120), default="")
    password_hash = Column(String(300), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    last_login = Column(DateTime(timezone=True), nullable=True)
    consented_at = Column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    """One chat session. Keyed by the UUID the API hands the frontend."""
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True, nullable=True)   # once auth exists
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=_utcnow,
                        onupdate=_utcnow, index=True)

    turn_count = Column(Integer, default=0)
    crisis_flagged = Column(Boolean, default=False, index=True)
    last_action = Column(String(16), default="")
    resolved_labels = Column(String(200), default="")          # comma-separated

    turns = relationship("Turn", back_populates="conversation",
                         cascade="all, delete-orphan",
                         order_by="Turn.turn_index")


class Turn(Base):
    """
    One exchange. `user_text_enc` and `reply_enc` are Fernet ciphertext; every
    other column is plaintext metadata.
    """
    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id",
                                                    ondelete="CASCADE"),
                             index=True)
    turn_index = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)

    user_text_enc = Column(LargeBinary, nullable=True)
    reply_enc = Column(LargeBinary, nullable=True)

    action = Column(String(16), index=True)
    labels = Column(String(200), default="")
    top_scores = Column(Text, default="")        # JSON, for the dashboard
    evidence_tokens = Column(Integer, default=0)
    reason = Column(String(300), default="")
    llm_used = Column(Boolean, default=False)
    crisis_patterns = Column(String(200), default="")

    conversation = relationship("Conversation", back_populates="turns")


def init_db() -> None:
    Base.metadata.create_all(engine)


def sweep_expired() -> int:
    """
    Delete conversations older than RETENTION_DAYS. Returns rows removed.

    Cutoff is on updated_at, not created_at: a conversation someone returned to
    last week should not vanish because it started in March.
    """
    cutoff = _utcnow() - dt.timedelta(days=RETENTION_DAYS)
    with SessionLocal() as s:
        ids = s.scalars(select(Conversation.id)
                        .where(Conversation.updated_at < cutoff)).all()
        if not ids:
            return 0
        s.execute(delete(Turn).where(Turn.conversation_id.in_(ids)))
        s.execute(delete(Conversation).where(Conversation.id.in_(ids)))
        s.commit()
        return len(ids)


def stats() -> dict:
    with SessionLocal() as s:
        return {
            "conversations": s.scalar(select(func.count()).select_from(Conversation)) or 0,
            "turns": s.scalar(select(func.count()).select_from(Turn)) or 0,
            "crisis_conversations": s.scalar(
                select(func.count()).select_from(Conversation)
                .where(Conversation.crisis_flagged.is_(True))) or 0,
            "backend": DATABASE_URL.split("://", 1)[0],
            "retention_days": RETENTION_DAYS,
            "storing_text": STORE_TEXT,
        }
