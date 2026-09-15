# Continuum M0

First runnable vertical slice of Continuum.

## What works

- Next.js text journal UI
- FastAPI REST API
- PostgreSQL persistence
- Alembic migrations
- Redis + Celery async processing
- Mock Context-MoDE analysis
- Journal status polling
- Health/readiness endpoints
- Docker Compose local environment

## Golden path

1. User submits a text journal from the web app.
2. `POST /v1/journals` stores it as `QUEUED`.
3. API enqueues `process_journal` in Celery.
4. Worker marks it `PROCESSING`.
5. Mock Context-MoDE deterministically creates energy/stress values.
6. Worker persists a `state_observation` and marks the journal `COMPLETED`.
7. Web UI polls `GET /v1/journals/{id}` and renders the result.

## Run

Prerequisite: Docker Desktop / Docker Engine with Compose.

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Web: http://localhost:3000
- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health
- API readiness: http://localhost:8000/ready

## Test through curl

```bash
curl -X POST http://localhost:8000/v1/journals \
  -H 'Content-Type: application/json' \
  -d '{"entry_type":"TEXT","text":"Work was exhausting today but I am hopeful about tomorrow."}'
```

Then request the returned id:

```bash
curl http://localhost:8000/v1/journals/<journal-id>
```

## M0 scope

M0 intentionally uses a mock AI implementation behind a service interface. No diagnostic or clinical inference is performed. The next milestones replace the mock progressively without changing the application contract.
