from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_API_", env_file=".env")

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    device: str = "cpu"
    max_batch_size: int = 32
    max_text_length: int = 512
    queue_maxsize: int = 1000
    worker_concurrency: int = 4
    log_level: str = "INFO"


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class InferenceRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=2048)]
    request_id: str | None = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class BatchInferenceRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=32)]
    request_id: str | None = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v]
        if any(len(t) == 0 for t in cleaned):
            raise ValueError("Texts must not be empty strings")
        return cleaned


class SentimentScore(BaseModel):
    label: Sentiment
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    text_length: int


class InferenceResponse(BaseModel):
    request_id: str
    result: SentimentScore
    model_name: str
    latency_ms: float


class BatchInferenceResponse(BaseModel):
    request_id: str
    results: list[SentimentScore]
    model_name: str
    latency_ms: float
    batch_size: int


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class AsyncJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str


class AsyncJobResult(BaseModel):
    job_id: str
    status: JobStatus
    result: SentimentScore | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    queue_depth: int
    device: str
