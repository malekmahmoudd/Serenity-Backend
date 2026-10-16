from __future__ import annotations

import os
import time
import uuid
import datetime as dt

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

import config     # must load first
import auth, classifier, crypto, dashboard, db
import decision as decision_layer
import llm, metrics, ratelimit, safety, sessions

# FastAPI app
app = FastAPI(title="Serenity API", version="1.0.0")

print(f"[config] env={config.ENV} {config.summary()}", flush=True)

# DB init + retention sweep
db.init_db()
_swept = db.sweep_expired()
if _swept:
    print(f"retention sweep: removed {_swept} expired conversations", flush=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    print(f"UNHANDLED: {request.method} {request.url.path} {type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our end. Please try again."})

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    gpt_response: str
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

# Helpers
def _user_public(u) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name or ""}

def _auth_rate_guard(request: Request) -> None:
    allowed, retry = ratelimit.auth_ip.check(ratelimit.client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many attempts. Please wait a moment.",
                            headers={"Retry-After": str(int(retry) + 1)})

# Routes
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
    return metrics.snapshot()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(_: bool = Depends(auth.require_admin)) -> str:
    return dashboard.render(health())

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request,
               claims: dict | None = Depends(auth.optional_user)) -> ChatResponse:
    t0 = time.monotonic()
    text = (req.message or "").strip()
    if not text:
        return ChatResponse(gpt_response="I'm listening whenever you're ready.",
                            session_id=req.session_id or "", action="ASK")
    if not safety.is_crisis(text):
        if claims:
            allowed, retry = ratelimit.chat_user.check(claims["sub"])
        else:
            allowed, retry = ratelimit.chat_anon.check(ratelimit.client_ip(request))
        if not allowed:
            raise HTTPException(429, "You're sending messages very quickly. Take a moment, then try again.",
                                headers={"Retry-After": str(int(retry) + 1)})
    sess = sessions.get_or_create(req.session_id)
    if claims:
        sessions.attach_user(sess.session_id, claims["sub"])
    crisis_before = sess.crisis_flagged
    prev_turn = sess.turns[-1] if sess.turns else None
    prev_fired = sess.last_action == "SAFETY"
    sess.turns.append(text)
    sess.history.append({"role": "user", "content": text})
    d = decision_layer.decide(accumulated=sess.accumulated, latest=text,
                              turn=sess.turn_count, prev_turn=prev_turn,
                              prev_fired=prev_fired, crisis_flagged=crisis_before,
                              asked=sess.asked_questions)
    sess.last_action = d.action
    # Record the question BEFORE the LLM rewords it. The bank entry is what
    # pick_question matches against, so the reworded version would not exclude
    # anything on the next turn.
    if d.question:
        sess.asked_questions.append(d.question)
    if d.action == "SAFETY":
        sess.crisis_flagged = True
    elif d.action == "RESOLVE":
        d.already_introduced = bool(sess.resolved_labels)
        sess.resolved_labels = d.labels
    reply = llm.respond(d.to_state(), sess.history, question=d.question)
    sess.history.append({"role": "assistant", "content": reply})
    sessions.persist_turn(sess, user_text=text, reply=reply, action=d.action,
                          labels=d.labels, scores=d.scores, tokens=d.evidence_tokens,
                          reason=d.reason, llm_used=llm.last_call_used_api,
                          crisis_patterns=d.crisis_patterns)
    dashboard.record(session_id=sess.session_id, action=d.action, text=text,
                     labels=d.labels, scores=d.scores, tokens=d.evidence_tokens,
                     turn=d.turn, llm_used=llm.last_call_used_api, reason=d.reason)
    metrics.record_turn(action=d.action, llm_used=llm.last_call_used_api,
                        llm_attempted=llm.available(),
                        latency_ms=(time.monotonic() - t0) * 1000)
    return ChatResponse(gpt_response=reply, session_id=sess.session_id,
                        action=d.action, labels=d.labels,
                        crisis=d.action == "SAFETY", turn=d.turn,
                        evidence_tokens=d.evidence_tokens)

@app.post("/reset")
async def reset(req: ResetRequest):
    sessions.reset(req.session_id)
    return {"ok": True}

@app.get("/")
def root():
    return {"service": "Serenity API", "docs": "/docs", "health": "/health"}

# Startup block: binds to $PORT if the platform sets it, else 8000
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
