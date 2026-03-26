from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .models import (
    AsyncJobResponse,
    AsyncJobResult,
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    JobStatus,
    Settings,
)
from .pipeline import SentimentPipeline

logger = structlog.get_logger(__name__)


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title="Async ML Inference API",
        description="Sentiment analysis with async job queue and batch support.",
        version="1.0.0",
    )

    ml_pipeline = SentimentPipeline(settings)
    job_store: dict[str, AsyncJobResult] = {}

    @app.on_event("startup")
    async def startup() -> None:
        logger.info("api_startup", model=settings.model_name)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await ml_pipeline.shutdown()
        logger.info("api_shutdown")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        t0 = time.perf_counter()
        response = await call_next(request)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency,
            request_id=request_id,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency)
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=ml_pipeline.is_loaded,
            model_name=settings.model_name,
            queue_depth=len([j for j in job_store.values() if j.status == JobStatus.PENDING]),
            device=settings.device,
        )

    @app.post("/v1/sentiment", response_model=InferenceResponse, tags=["inference"])
    async def predict_sync(req: InferenceRequest) -> InferenceResponse:
        request_id = req.request_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            result = await ml_pipeline.predict(req.text)
        except Exception as e:
            logger.exception("predict_error", request_id=request_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        latency = round((time.perf_counter() - t0) * 1000, 1)
        return InferenceResponse(
            request_id=request_id,
            result=result,
            model_name=settings.model_name,
            latency_ms=latency,
        )

    @app.post("/v1/sentiment/batch", response_model=BatchInferenceResponse, tags=["inference"])
    async def predict_batch(req: BatchInferenceRequest) -> BatchInferenceResponse:
        request_id = req.request_id or str(uuid.uuid4())
        if len(req.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(req.texts)} exceeds limit of {settings.max_batch_size}",
            )
        t0 = time.perf_counter()
        try:
            results = await ml_pipeline.predict_batch(req.texts)
        except Exception as e:
            logger.exception("predict_batch_error", request_id=request_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        latency = round((time.perf_counter() - t0) * 1000, 1)
        return BatchInferenceResponse(
            request_id=request_id,
            results=results,
            model_name=settings.model_name,
            latency_ms=latency,
            batch_size=len(results),
        )

    @app.post("/v1/sentiment/async", response_model=AsyncJobResponse,
              status_code=status.HTTP_202_ACCEPTED, tags=["async"])
    async def submit_async_job(
        req: InferenceRequest,
        background_tasks: BackgroundTasks,
    ) -> AsyncJobResponse:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if len(job_store) >= settings.queue_maxsize:
            raise HTTPException(status_code=503, detail="Job queue is full; try again later")

        job_store[job_id] = AsyncJobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
        )

        background_tasks.add_task(_run_async_job, job_id, req.text, ml_pipeline, job_store)
        logger.info("async_job_submitted", job_id=job_id)

        return AsyncJobResponse(job_id=job_id, status=JobStatus.PENDING, created_at=now)

    @app.get("/v1/sentiment/async/{job_id}", response_model=AsyncJobResult, tags=["async"])
    async def get_async_job(job_id: str) -> AsyncJobResult:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return job

    @app.delete("/v1/sentiment/async/{job_id}", status_code=204, tags=["async"])
    async def delete_async_job(job_id: str) -> None:
        if job_id not in job_store:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        del job_store[job_id]

    return app


async def _run_async_job(
    job_id: str,
    text: str,
    pipeline: SentimentPipeline,
    job_store: dict[str, AsyncJobResult],
) -> None:
    job = job_store[job_id]
    job_store[job_id] = AsyncJobResult(
        **{**job.model_dump(), "status": JobStatus.RUNNING}
    )

    t0 = time.perf_counter()
    try:
        result = await pipeline.predict(text)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        job_store[job_id] = AsyncJobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result=result,
            created_at=job.created_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency,
        )
        logger.info("async_job_completed", job_id=job_id, latency_ms=latency)
    except Exception as e:
        job_store[job_id] = AsyncJobResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
            created_at=job.created_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.exception("async_job_failed", job_id=job_id, error=str(e))
