from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import structlog
import torch
from transformers import pipeline as hf_pipeline, Pipeline

from .models import Sentiment, SentimentScore, Settings

logger = structlog.get_logger(__name__)


class SentimentPipeline:
    """
    Wraps a HuggingFace sentiment pipeline with:
    - Lazy loading on first inference
    - Thread-pool offloading so the asyncio event loop is never blocked
    - Batched inference support
    - Structured latency logging
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipe: Pipeline | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_concurrency,
            thread_name_prefix="ml-worker",
        )
        self._load_lock = asyncio.Lock()
        self._loaded = False

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            loop = asyncio.get_running_loop()
            logger.info("loading_model", model=self._settings.model_name, device=self._settings.device)
            t0 = time.perf_counter()
            self._pipe = await loop.run_in_executor(
                self._executor, self._load_model
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("model_loaded", model=self._settings.model_name, load_ms=round(elapsed, 1))
            self._loaded = True

    def _load_model(self) -> Pipeline:
        device = 0 if self._settings.device == "cuda" and torch.cuda.is_available() else -1
        return hf_pipeline(
            "sentiment-analysis",
            model=self._settings.model_name,
            device=device,
            truncation=True,
            max_length=self._settings.max_text_length,
        )

    async def predict(self, text: str) -> SentimentScore:
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        raw: list[dict[str, Any]] = await loop.run_in_executor(
            self._executor,
            lambda: self._pipe([text]),  # type: ignore[misc]
        )

        latency = (time.perf_counter() - t0) * 1000
        result = raw[0]
        score = SentimentScore(
            label=Sentiment(result["label"]),
            score=round(float(result["score"]), 6),
            text_length=len(text),
        )
        logger.debug("predict_single",
                     label=score.label, score=score.score,
                     text_length=len(text), latency_ms=round(latency, 1))
        return score

    async def predict_batch(self, texts: list[str]) -> list[SentimentScore]:
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        raw: list[dict[str, Any]] = await loop.run_in_executor(
            self._executor,
            lambda: self._pipe(texts),  # type: ignore[misc]
        )

        latency = (time.perf_counter() - t0) * 1000
        results = [
            SentimentScore(
                label=Sentiment(r["label"]),
                score=round(float(r["score"]), 6),
                text_length=len(t),
            )
            for r, t in zip(raw, texts)
        ]
        logger.info("predict_batch",
                    batch_size=len(texts), latency_ms=round(latency, 1))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        logger.info("pipeline_shutdown")
