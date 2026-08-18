"""
main.py
=======
Serenity API.

    POST /chat      one conversational turn
    POST /reset     clear a session
    GET  /health    what is loaded and what is not

REQUEST FLOW
------------
    message
      -> append to session, rebuild accumulated text
      -> decision.decide()          crisis rule, then classifier, then policy
      -> llm.respond()              structured state in, prose out
      -> response

The classifier and the LLM are both optional at runtime. Missing model? The
decision layer degrades to rule-only and the API still answers. Missing API
key? Hand-written fallbacks. The server should boot and serve before every
piece is in place, so you can exercise the request path early.

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

# config.py MUST import before every other backend module, for the same
# reason .env has to load first: it calls load_dotenv() itself, and it is
# where the "refuse to boot misconfigured" check lives. Importing anything
# else first risks the exact bug that shipped earlier -- a secret silently
# read as empty because something else's import ran before the environment
# was populated.
import config     # noqa: F401  (import solely for its side effects)

import datetime as dt
import os
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

import auth
import classifier
import crypto
import dashboard
import db
import decision as decision_layer
import llm
import metrics
import ratelimit
import safety
import sessions

app = FastAPI(title="Serenity API", version="1.0.0")

print(f"[config] env={config.ENV}  {config.summary()}", flush=True)

# Tables, plus one retention sweep at boot. Rows past RETENTION_DAYS are
# deleted -- and whatever that number is, the consent screen has to say it. A
# retention policy nobody was told about is not a policy.
db.init_db()
_swept = db.sweep_expired()
if _swept:
    print(f"retention sweep: removed {_swept} expired conversations", flush=True)

ALLOWED_ORIGINS = config.ALLOWED_ORIGINS.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last resort. Without this, an unexpected error anywhere in the request
    path returns FastAPI's default response, and depending on deployment
    config that can include a raw traceback -- which for this app can mean a
    stack frame containing part of what a user just typed appearing in a
    browser dev console. Log the real error server-side; tell the client
    nothing more than "something went wrong, try again."
    """
    print(f"UNHANDLED: {request.method} {request.url.path}  "
          f"{type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our end. Please try again."},
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    gpt_response: str          # kept: the existing frontend reads this key
    session_id: str
    action: str
    labels: list = []
    crisis: bool = False
    turn: int = 0
    evidence_tokens: int = 0


class ResetRequest(BaseModel):
    session_id: str


class RegisterRequest(BaseModel):
    name: str = ""
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


def _user_public(u) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name or ""}


def _auth_rate_guard(request: Request) -> None:
    """
    Harder limit on the auth endpoints. This is where credential stuffing and
    account enumeration land, and unlike /chat there is no reason a human hits
    these more than a few times a minute.
    """
    allowed, retry = ratelimit.auth_ip.check(ratelimit.client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many attempts. Please wait a moment.",
                            headers={"Retry-After": str(int(retry) + 1)})


@app.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest, request: Request) -> AuthResponse:
    _auth_rate_guard(request)
    email = (req.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "A valid email is required.")
    if len(req.password or "") < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    with db.SessionLocal() as s:
        if s.scalar(select(db.User).where(db.User.email == email)):
            raise HTTPException(409, "An account with this email already exists.")
        user = db.User(id=str(uuid.uuid4()), email=email,
                       name=(req.name or "").strip(),
                       password_hash=auth.hash_password(req.password))
        s.add(user)
        s.commit()
        return AuthResponse(token=auth.create_token(user.id, user.email),
                            user=_user_public(user))


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest, request: Request) -> AuthResponse:
    _auth_rate_guard(request)
    email = (req.email or "").strip().lower()
    with db.SessionLocal() as s:
        user = s.scalar(select(db.User).where(db.User.email == email))
        # One message for both "no such user" and "wrong password", so the
        # endpoint cannot be used to enumerate which emails have accounts.
        if user is None or not auth.verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid email or password.")
        user.last_login = dt.datetime.now(dt.timezone.utc)
        s.commit()
        return AuthResponse(token=auth.create_token(user.id, user.email),
                            user=_user_public(user))


@app.get("/auth/me")
async def me(claims: dict = Depends(auth.require_user)) -> dict:
    with db.SessionLocal() as s:
        user = s.get(db.User, claims["sub"])
        if user is None:
            raise HTTPException(401, "Account no longer exists.")
        return _user_public(user)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "env": config.ENV,
        "classifier": classifier.status(),
        "llm": {"configured": llm.available(), "model": llm.LLM_MODEL},
        "active_sessions": sessions.count(),
        "crisis_rule": "loaded",
        "storage": {**db.stats(), "encryption_key_set": crypto.key_configured()},
        "rate_limits": ratelimit.stats(),
        "secrets": auth.secrets_ok(),
        "metrics": metrics.snapshot(),
    }


@app.get("/metrics")
def metrics_endpoint(_: bool = Depends(auth.require_admin)) -> dict:
    """
    Same numbers as health()["metrics"], as their own route.

    Admin-gated like /dashboard, even though nothing here is user text -- the
    crisis fire count and week-over-week ratio are still sensitive in
    aggregate (they reveal how many people using this app are in crisis on a
    given day), so this stays behind the same X-Admin-Token as the turn log.
    """
    return metrics.snapshot()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(_: bool = Depends(auth.require_admin)) -> str:
    """
    Human-readable /health plus a live turn log.

    Admin-only: the log contains user message text. Returns 404 unless
    SERENITY_ADMIN_TOKEN is set, and 403 without the X-Admin-Token header --
    404 rather than 403 for the unset case so the route's existence is not
    confirmed to a stranger.
    """
    return dashboard.render(health())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request,
               claims: dict | None = Depends(auth.optional_user)) -> ChatResponse:
    _t0 = time.monotonic()
    text = (req.message or "").strip()
    if not text:
        return ChatResponse(
            gpt_response="I'm listening whenever you're ready.",
            session_id=req.session_id or "", action="ASK")

    # Rate limit -- but never for someone in crisis. Telling a distressed
    # person to slow down is a worse failure than the abuse this prevents.
    # Bypassable in principle by including crisis words; that trade is
    # deliberate.
    if not safety.is_crisis(text):
        if claims:
            allowed, retry = ratelimit.chat_user.check(claims["sub"])
        else:
            allowed, retry = ratelimit.chat_anon.check(ratelimit.client_ip(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="You're sending messages very quickly. Take a moment, "
                       "then try again.",
                headers={"Retry-After": str(int(retry) + 1)})

    sess = sessions.get_or_create(req.session_id)
    # Tie the conversation to the account when one is present. Anonymous
    # conversations still work -- see auth.optional_user for why a signup wall
    # in front of a distressed person is the wrong trade.
    if claims:
        sessions.attach_user(sess.session_id, claims["sub"])

    # Capture BEFORE this turn is judged.
    #   crisis_before -- carries context for tone; must never re-route a later
    #                    unrelated message.
    #   prev_fired    -- whether the previous turn itself triggered SAFETY. If
    #                    it did, decide() will not re-scan it, so one
    #                    disclosure cannot fire twice.
    crisis_before = sess.crisis_flagged
    prev_turn = sess.turns[-1] if sess.turns else None
    prev_fired = sess.last_action == "SAFETY"

    sess.turns.append(text)
    sess.history.append({"role": "user", "content": text})

    d = decision_layer.decide(
        accumulated=sess.accumulated,
        latest=text,
        turn=sess.turn_count,
        prev_turn=prev_turn,
        prev_fired=prev_fired,
        crisis_flagged=crisis_before,
    )

    sess.last_action = d.action
    if d.action == "SAFETY":
        sess.crisis_flagged = True
    if d.action == "RESOLVE":
        # Tell the LLM whether it has already named this category. Without
        # this it reintroduces "this overlaps with depression" on every
        # subsequent turn, which reads as the app diagnosing repeatedly and
        # buries whatever the user actually just said.
        d.already_introduced = bool(sess.resolved_labels)
        sess.resolved_labels = d.labels

    # One line per turn in the uvicorn console. None of this is visible in the
    # chat bubble: whether the classifier fired, what it scored, or whether the
    # reply came from the LLM or a hand-written template.
    top3 = sorted(d.scores.items(), key=lambda kv: -kv[1])[:3] if d.scores else []
    print(f"[{sess.session_id[:8]}] t{d.turn} {d.action:<8} "
          f"tok={d.evidence_tokens:<4} "
          f"labels={d.labels or '-'} "
          f"top3={[(k, round(v, 3)) for k, v in top3]} "
          f"| {text[:50]}", flush=True)

    reply = llm.respond(d.to_state(), sess.history, question=d.question)
    sess.history.append({"role": "assistant", "content": reply})

    # Persist. Message text encrypted, metadata in the clear -- see db.py for
    # why the split is drawn there.
    sessions.persist_turn(sess, user_text=text, reply=reply, action=d.action,
                          labels=d.labels, scores=d.scores,
                          tokens=d.evidence_tokens, reason=d.reason,
                          llm_used=llm.last_call_used_api,
                          crisis_patterns=d.crisis_patterns)

    # Feed the /dashboard turn log. llm.last_call_used_api tells us whether
    # this reply was generated or came from a hand-written fallback -- not
    # something you can tell from the chat bubble.
    dashboard.record(session_id=sess.session_id, action=d.action, text=text,
                     labels=d.labels, scores=d.scores,
                     tokens=d.evidence_tokens, turn=d.turn,
                     llm_used=llm.last_call_used_api, reason=d.reason)

    # Feed /metrics and /health -- the queryable counters ESCALATION_POLICY.md
    # relies on for the crisis-review cadence. llm_attempted is False only
    # when no key is configured at all (a deployment fact); when a key IS
    # configured but the call still failed, that is llm_attempted=True,
    # llm_used=False, which is what makes llm_failure_rate meaningful rather
    # than always reading zero on a dev machine with no key set.
    metrics.record_turn(
        action=d.action,
        llm_used=llm.last_call_used_api,
        llm_attempted=llm.available(),
        latency_ms=(time.monotonic() - _t0) * 1000,
    )

    return ChatResponse(
        gpt_response=reply,
        session_id=sess.session_id,
        action=d.action,
        labels=d.labels,
        crisis=(d.action == "SAFETY"),
        turn=d.turn,
        evidence_tokens=d.evidence_tokens,
    )


@app.post("/reset")
async def reset(req: ResetRequest) -> dict:
    sessions.reset(req.session_id)
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"service": "Serenity API", "docs": "/docs", "health": "/health"}
