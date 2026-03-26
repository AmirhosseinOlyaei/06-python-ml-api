# Python Async ML Inference API

**Feature:** End-to-end async sentiment analysis pipeline — synchronous inference, batch inference, and an async fire-and-poll job queue, all backed by a HuggingFace transformer model served without blocking the event loop.

## Stack
- Python 3.12+
- FastAPI 0.115 + Uvicorn (ASGI)
- Pydantic v2 + `pydantic-settings` (env-driven config)
- HuggingFace `transformers` — `distilbert-base-uncased-finetuned-sst-2-english`
- `structlog` — structured JSON logging
- `ThreadPoolExecutor` — CPU-bound inference off the event loop

## Architecture
```
src/
  models.py    ← Pydantic v2 request/response schemas + Settings (env-configurable)
  pipeline.py  ← SentimentPipeline: lazy load, async lock, ThreadPoolExecutor offload,
                  single + batch predict, structured latency logging
  api.py       ← FastAPI app factory:
                  POST /v1/sentiment          ← sync inference
                  POST /v1/sentiment/batch    ← batch (up to 32 texts)
                  POST /v1/sentiment/async    ← submit job → 202 Accepted
                  GET  /v1/sentiment/async/{id} ← poll result
                  DELETE /v1/sentiment/async/{id} ← cleanup
                  GET  /health                ← model status + queue depth
  main.py      ← uvicorn entrypoint + structlog JSON configuration
```

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.main
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

## Configuration (env vars with ML_API_ prefix)
| Variable | Default | Description |
|----------|---------|-------------|
| `ML_API_MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` | HF model id |
| `ML_API_DEVICE` | `cpu` | `cpu` or `cuda` |
| `ML_API_MAX_BATCH_SIZE` | `32` | Max texts per batch request |
| `ML_API_WORKER_CONCURRENCY` | `4` | Thread pool size |
| `ML_API_QUEUE_MAXSIZE` | `1000` | Max async jobs in memory |
| `ML_API_LOG_LEVEL` | `INFO` | Logging verbosity |

## API examples

```bash
# Sync inference
curl -X POST http://localhost:8000/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This library is absolutely fantastic!"}'

# Batch
curl -X POST http://localhost:8000/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible experience.", "It was okay"]}'

# Async submit
JOB=$(curl -s -X POST http://localhost:8000/v1/sentiment/async \
  -H "Content-Type: application/json" \
  -d '{"text":"The async pattern works beautifully here"}' | jq -r .job_id)

# Poll result
curl http://localhost:8000/v1/sentiment/async/$JOB

# Health check
curl http://localhost:8000/health
```

## Key design decisions
- **Lazy model loading** — model loads on first request, not at startup, protected by `asyncio.Lock` against thundering-herd duplicate loads
- **Event loop preservation** — `run_in_executor(ThreadPoolExecutor)` ensures the HF pipeline's CPU/GPU work never blocks the asyncio loop
- **Pydantic v2 validators** — `@field_validator` strips whitespace; annotated `Field(min_length, max_length)` enforces invariants at the boundary
- **Async job store** — in-memory dict keyed by UUID; production would swap for Redis with TTL
- **Structured JSON logging** — every request, inference, and error emits a structlog JSON line with latency, model name, batch size
- **Global exception handler** — catches unhandled errors and returns `500` without leaking stack traces to clients
