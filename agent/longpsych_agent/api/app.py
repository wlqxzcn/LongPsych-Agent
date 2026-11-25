from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from ..config import settings

from ..reranking.qr_future_pipeline import QRFutureRetrievalPipeline


LOGGER = logging.getLogger("longpsych_agent.api")




class SearchRequest(BaseModel):
    query: str | list[str] = Field(
        ...,
        description="A single query string or a batch of query strings.",
    )
    mode: Literal["dense", "orca", "qr_future"] = "qr_future"
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=5000)


class SearchResponseEnvelope(BaseModel):
    status: Literal["ok"] = "ok"
    data: dict | list[dict]


def _validate_request(payload: SearchRequest) -> None:
    if isinstance(payload.query, str):
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query cannot be empty.")
    elif isinstance(payload.query, list):
        if not payload.query:
            raise HTTPException(status_code=422, detail="query batch cannot be empty.")
        if any(not isinstance(item, str) or not item.strip() for item in payload.query):
            raise HTTPException(
                status_code=422,
                detail="Every query in the batch must be a non-empty string.",
            )

    if payload.mode != "qr_future" and payload.candidate_k is not None:
        raise HTTPException(
            status_code=422,
            detail="candidate_k is only valid when mode='qr_future'.",
        )
    if (
        payload.mode == "qr_future"
        and payload.candidate_k is not None
        and payload.candidate_k < payload.top_k
    ):
        raise HTTPException(
            status_code=422,
            detail="candidate_k must be greater than or equal to top_k.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info(
        "Loading LongPsychAgent pipeline from %s",
        settings.root,
    )

    pipeline = QRFutureRetrievalPipeline.from_pretrained(
        model_name_or_path=str(settings.model_path),
        index_dir=settings.index_dir,
        qr_future_project_root=settings.root,
        default_candidate_k=settings.candidate_k,
        device=settings.device,
        batch_size=settings.batch_size,
        show_progress_bar=False,
        verify_hashes=settings.verify_hashes,
    )

    app.state.pipeline = pipeline
    app.state.search_lock = threading.RLock()

    LOGGER.info(
        "LongPsychAgent pipeline is ready: %s",
        pipeline.health(),
    )

    try:
        yield
    finally:
        app.state.pipeline = None
        app.state.search_lock = None
        LOGGER.info("LongPsychAgent API shutdown complete.")


app = FastAPI(
    title="LongPsychAgent Retrieval API",
    version="1.0.0",
    description=(
        "Dense BGE retrieval, ORCA k=2 retrieval, and "
        "Dense candidate recall followed by five-fold QR-Future reranking."
    ),
    lifespan=lifespan,
)


@app.get("/", tags=["system"])
def root():
    return {
        "service": "LongPsychAgent Retrieval API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["system"])
def health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready.")
    return pipeline.health()


@app.post(
    "/v1/search",
    tags=["retrieval"],
    response_model=SearchResponseEnvelope,
)
def search(payload: SearchRequest, request: Request):
    _validate_request(payload)

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready.")

    lock = getattr(request.app.state, "search_lock", None)
    if lock is None:
        raise HTTPException(status_code=503, detail="Search service is not ready.")

    try:
        with lock:
            result = pipeline.search(
                query=payload.query,
                top_k=payload.top_k,
                mode=payload.mode,
                candidate_k=payload.candidate_k,
            )
        return {"status": "ok", "data": result}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Retrieval request failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal retrieval error. Check the server log.",
        ) from exc
