<<<<<<< HEAD
<<<<<<< HEAD
---
title: Serenity API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Serenity — backend API

FastAPI service: LSTM classifier (ONNX) + rule-based crisis detection +
Groq-backed conversational replies. See `main.py` for the request flow and
`ESCALATION_POLICY.md` for the safety design this API implements.

Frontend lives separately (Next.js on Vercel) and calls this Space's URL.

## Configuration

Set these under **Settings → Variables and secrets** on this Space — never in
the repo:

| name | required | notes |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string (Neon, etc.) |
| `SERENITY_ENC_KEY` | yes | `Fernet.generate_key()` -- see `crypto.py` |
| `SERENITY_JWT_SECRET` | yes | any long random string |
| `GROQ_API_KEY` | yes | groq.com |
| `SERENITY_ADMIN_TOKEN` | yes | protects `/dashboard` and `/metrics` |
| `SERENITY_ENV` | yes | set to `prod` |
| `ALLOWED_ORIGINS` | yes | your Vercel domain, comma-separated if more than one |

Health check: `/health`. This Space's code is public by default -- nothing in
it is a secret (see `text_cleaning.py` etc. for the reasoning behind every
design choice); the actual secrets above are the only things that must never
appear in the repo itself.
=======
# Serenity
>>>>>>> 4db25003a65b62f54b4f0cb2a950aac078c6e696
=======
# Serenity-Backend
>>>>>>> 662b00fe660081e66ea3edbf604e8e3399c96835
