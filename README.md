# Serenity — backend API

FastAPI service: LSTM classifier (ONNX) + rule-based crisis detection +
Groq-backed conversational replies. See `main.py` for the request flow and
`ESCALATION_POLICY.md` for the safety design this API implements.

The frontend lives separately (Next.js on Vercel) and calls this service's URL.

## Deployment (Dockhost)

The service is a single Docker container built from the `Dockerfile` in this
repo.

- Container port: `8000` (the app also honours `$PORT` if the platform sets it)
- Health check path: `/health`
- Instances: run exactly **one**. Chat sessions and rate limits are held in
  memory, so multiple replicas would each have their own copy.

## Configuration

Set these as environment variables on the hosting platform — never in the repo:

| name | required | notes |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string, e.g. `postgresql+psycopg://user:pw@host/db` |
| `SERENITY_ENC_KEY` | yes | `Fernet.generate_key()` -- see `crypto.py`. Keep the same key when moving hosts, or stored history becomes unreadable |
| `SERENITY_JWT_SECRET` | yes | any long random string |
| `GROQ_API_KEY` | yes | groq.com |
| `SERENITY_ADMIN_TOKEN` | yes | protects `/dashboard` and `/metrics` |
| `SERENITY_ENV` | yes | set to `prod` |
| `ALLOWED_ORIGINS` | yes | your frontend domain, comma-separated if more than one |

In `prod` the app refuses to start if a required value is missing, if
`DATABASE_URL` is SQLite, or if `ALLOWED_ORIGINS` contains `*`.
