"""
test_persistence.py
===================
Exercises the storage layer without FastAPI, TensorFlow, or a network.

    venv\\Scripts\\python.exe test_persistence.py

Checks:
  1. text is genuinely unreadable in the database file
  2. a session survives a "restart" (cache cleared, rehydrated from disk)
  3. metadata stays queryable in plaintext
  4. reset() actually deletes, cascade included
  5. the retention sweep removes old rows and spares recent ones
  6. STORE_TEXT=false keeps metadata and stores no text at all

Uses a throwaway database and a fixed key so runs are reproducible.
"""

import os
import shutil
import tempfile

TMP = tempfile.mkdtemp(prefix="serenity_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(TMP, 'test.db')}"
os.environ["SERENITY_ENC_KEY"] = "sVh0kQ2Zp7XcJ9nR4tB6yU8wE1oA3iL5mF7gD0hK2vM="
os.environ["RETENTION_DAYS"] = "90"
os.environ["STORE_TEXT"] = "true"

import datetime as dt  # noqa: E402
import sqlite3  # noqa: E402

import crypto  # noqa: E402
import db  # noqa: E402
import sessions  # noqa: E402

SECRET = "i can't get out of bed most mornings and nothing matters anymore"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


print("=" * 62)
print("1. round trip and encryption at rest")
print("=" * 62)
db.init_db()
s = sessions.get_or_create(None)
sid = s.session_id
s.turns.append(SECRET)
s.history.append({"role": "user", "content": SECRET})
sessions.persist_turn(s, user_text=SECRET, reply="I hear you.", action="RESOLVE",
                      labels=["depression"], scores={"depression": 0.79, "bpd": 0.15},
                      tokens=21, reason="depression cleared threshold",
                      llm_used=True, crisis_patterns=[])

raw = open(os.path.join(TMP, "test.db"), "rb").read()
check("plaintext absent from the db file", SECRET.encode() not in raw)
check("a distinctive phrase is absent too", b"out of bed" not in raw)

con = sqlite3.connect(os.path.join(TMP, "test.db"))
blob, action, labels = con.execute(
    "SELECT user_text_enc, action, labels FROM turns LIMIT 1").fetchone()
con.close()
check("stored text is Fernet ciphertext", bytes(blob).startswith(b"gAAAAA"))
check("metadata is plaintext and queryable",
      action == "RESOLVE" and labels == "depression", f"({action}/{labels})")
check("decrypts back to the original", crypto.decrypt(bytes(blob)) == SECRET)

print()
print("=" * 62)
print("2. survives a restart")
print("=" * 62)
sessions._cache.clear()          # simulate the process dying
revived = sessions.get_or_create(sid)
check("session rehydrated", revived.session_id == sid)
check("accumulated text restored", SECRET in revived.accumulated)
check("turn count restored", revived.turn_count == 1, f"({revived.turn_count})")
check("history restored for the LLM", len(revived.history) == 2,
      f"({len(revived.history)} messages)")

print()
print("=" * 62)
print("3. crisis flag persists")
print("=" * 62)
c = sessions.get_or_create(None)
c.turns.append("i don't want to be here anymore")
c.crisis_flagged = True
sessions.persist_turn(c, user_text="i don't want to be here anymore",
                      reply="resources", action="SAFETY", labels=[], scores={},
                      tokens=6, reason="crisis language detected by rule",
                      llm_used=False, crisis_patterns=["not_here"])
cid = c.session_id
sessions._cache.clear()
check("crisis flag survives restart", sessions.get_or_create(cid).crisis_flagged)
check("crisis conversations countable in stats",
      db.stats()["crisis_conversations"] == 1)

print()
print("=" * 62)
print("4. reset deletes")
print("=" * 62)
before = db.stats()
sessions.reset(cid)
after = db.stats()
check("conversation removed", after["conversations"] == before["conversations"] - 1)
check("turns cascade-deleted", after["turns"] < before["turns"],
      f"({before['turns']} -> {after['turns']})")

print()
print("=" * 62)
print("5. retention sweep")
print("=" * 62)
old = sessions.get_or_create(None)
sessions.persist_turn(old, user_text="ancient", reply="x", action="ASK",
                      labels=[], scores={}, tokens=1, reason="", llm_used=False,
                      crisis_patterns=[])
with db.SessionLocal() as dbs:
    conv = dbs.get(db.Conversation, old.session_id)
    conv.updated_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=120)
    dbs.commit()
kept_before = db.stats()["conversations"]
removed = db.sweep_expired()
check("expired conversation removed", removed == 1, f"(removed {removed})")
check("recent conversations spared",
      db.stats()["conversations"] == kept_before - 1)

print()
print("=" * 62)
print("6. STORE_TEXT=false -- metadata only")
print("=" * 62)
db.STORE_TEXT = False
sessions.STORE_TEXT = False
n = sessions.get_or_create(None)
sessions.persist_turn(n, user_text="this must not be stored", reply="nor this",
                      action="ASK", labels=[], scores={"anxiety": 0.3},
                      tokens=5, reason="test", llm_used=True, crisis_patterns=[])
con = sqlite3.connect(os.path.join(TMP, "test.db"))
row = con.execute("SELECT user_text_enc, reply_enc, action, evidence_tokens "
                  "FROM turns ORDER BY id DESC LIMIT 1").fetchone()
con.close()
check("no user text stored", row[0] is None)
check("no reply stored", row[1] is None)
check("metadata still recorded", row[2] == "ASK" and row[3] == 5)

print()
print("=" * 62)
print(f"stats: {db.stats()}")
print("=" * 62)
shutil.rmtree(TMP, ignore_errors=True)
if FAILURES:
    print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
    raise SystemExit(1)
print("\nALL PASS -- persistence layer is sound")
